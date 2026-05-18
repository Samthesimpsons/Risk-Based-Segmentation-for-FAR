"""Transaction-level return regression on Buy events."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.regression import coefficients_dataframe, fit_ols_clustered
from src.config.settings import DataPaths
from src.data.loading import (
    load_assets,
    load_close_prices,
    load_customers,
    load_transactions,
)
from src.utils.profile_coherence import (
    build_asset_risk_classes,
    build_customer_profile_lookup,
    compute_annualised_volatility,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/transaction_return_regression")
HORIZON_MONTHS = 6
PRICE_LOOKUP_TOLERANCE_DAYS = 7
FORMULA = (
    "realised_return ~ is_coherent + asset_volatility + C(customer_type) + C(year)"
)
CLUSTER_COLUMN = "customerID"


def _compute_realised_returns(
    *,
    buys: pd.DataFrame,
    close_prices: pd.DataFrame,
    horizon_months: int,
    tolerance_days: int,
) -> pd.DataFrame:
    """Join each Buy to its start price and the close price horizon_months later."""
    tolerance = pd.Timedelta(days=tolerance_days)
    sorted_prices = close_prices[["ISIN", "timestamp", "closePrice"]].sort_values(
        "timestamp"
    )

    sorted_buys = buys.sort_values("timestamp").reset_index(drop=True)
    with_start = pd.merge_asof(
        sorted_buys,
        sorted_prices.rename(columns={"closePrice": "start_price"}),
        on="timestamp",
        by="ISIN",
        direction="backward",
        tolerance=tolerance,
    )

    with_start["target_date"] = with_start["timestamp"] + pd.DateOffset(
        months=horizon_months
    )
    end_lookup = sorted_prices.rename(
        columns={"closePrice": "end_price", "timestamp": "end_timestamp"}
    )
    with_end = pd.merge_asof(
        with_start.sort_values("target_date"),
        end_lookup,
        left_on="target_date",
        right_on="end_timestamp",
        by="ISIN",
        direction="backward",
        tolerance=tolerance,
    )

    priced = with_end.dropna(subset=["start_price", "end_price"]).copy()
    priced = priced[priced["start_price"] > 0.0]
    priced["realised_return"] = (priced["end_price"] - priced["start_price"]) / priced[
        "start_price"
    ]
    return priced


def _attach_controls(
    *,
    realised_returns: pd.DataFrame,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
    volatility_lookup: pd.Series,
) -> pd.DataFrame:
    """Attach coherence flag, asset volatility, customer type, and transaction year."""
    panel = realised_returns.copy()
    panel["customer_band"] = panel["customerID"].map(
        lambda customer_id: (
            profile.risk_band
            if (profile := customer_profiles.get(customer_id)) is not None
            else None
        )
    )
    panel["asset_band"] = panel["ISIN"].map(asset_risk_classes)
    panel["customer_type"] = panel["customerID"].map(
        lambda customer_id: (
            profile.customer_type
            if (profile := customer_profiles.get(customer_id)) is not None
            else None
        )
    )
    panel["asset_volatility"] = panel["ISIN"].map(volatility_lookup)
    panel["year"] = panel["timestamp"].dt.year

    panel = panel.dropna(
        subset=[
            "customer_band",
            "asset_band",
            "customer_type",
            "asset_volatility",
            "realised_return",
        ]
    )
    panel["customer_band"] = panel["customer_band"].astype(int)
    panel["asset_band"] = panel["asset_band"].astype(int)
    panel["discordance"] = (panel["customer_band"] - panel["asset_band"]).abs()
    panel["is_coherent"] = (panel["discordance"] <= 1).astype(int)
    return panel


def _build_panel(
    *,
    data_paths: DataPaths,
    horizon_months: int,
    price_lookup_tolerance_days: int,
) -> pd.DataFrame:
    """Assemble one row per Buy transaction with return, coherence, and controls."""
    print("Loading raw data...")
    transactions = load_transactions(
        data_paths.data_directory / data_paths.transactions_file
    )
    close_prices = load_close_prices(
        data_paths.data_directory / data_paths.close_prices_file
    )
    customers = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    assets = load_assets(data_paths.data_directory / data_paths.asset_information_file)

    customer_profiles = build_customer_profile_lookup(customers)
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)
    volatility_lookup = compute_annualised_volatility(close_prices)

    buys = transactions[transactions["transactionType"] == "Buy"].copy()
    print(f"  Loaded {len(buys)} Buy transactions; computing forward returns...")

    realised_returns = _compute_realised_returns(
        buys=buys,
        close_prices=close_prices,
        horizon_months=horizon_months,
        tolerance_days=price_lookup_tolerance_days,
    )
    return _attach_controls(
        realised_returns=realised_returns,
        customer_profiles=customer_profiles,
        asset_risk_classes=asset_risk_classes,
        volatility_lookup=volatility_lookup,
    )


def _build_summary(
    *,
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    horizon_months: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Build the machine-readable headline summary."""
    coherent_row = coefficients.loc[coefficients["term"] == "is_coherent"]
    coherent_coefficient = (
        coherent_row.iloc[0].to_dict() if not coherent_row.empty else {}
    )
    coherent_panel = panel[panel["is_coherent"] == 1]
    discordant_panel = panel[panel["is_coherent"] == 0]
    return {
        "horizon_months": horizon_months,
        "output_directory": str(output_directory),
        "panel_rows": int(len(panel)),
        "unique_customers": int(panel["customerID"].nunique()),
        "coherent_share": float((panel["is_coherent"] == 1).mean()),
        "mean_realised_return_overall": float(panel["realised_return"].mean()),
        "mean_realised_return_coherent": (
            float(coherent_panel["realised_return"].mean())
            if not coherent_panel.empty
            else None
        ),
        "mean_realised_return_discordant": (
            float(discordant_panel["realised_return"].mean())
            if not discordant_panel.empty
            else None
        ),
        "is_coherent_coefficient": coherent_coefficient,
    }


def _print_headline(summary: dict[str, Any]) -> None:
    """Print the key coefficient and slice means to stdout."""
    print(
        f"\nPanel: {summary['panel_rows']} transactions, "
        f"{summary['unique_customers']} unique customers, "
        f"coherent share {summary['coherent_share']:.3f}"
    )
    print(
        f"Raw slice means: coherent {summary['mean_realised_return_coherent']:.4f}, "
        f"discordant {summary['mean_realised_return_discordant']:.4f}"
    )
    coefficient = summary["is_coherent_coefficient"]
    if coefficient:
        print(
            f"is_coherent coefficient (cluster-robust SE on customerID): "
            f"{coefficient['estimate']:+.4f} "
            f"(SE {coefficient['std_error']:.4f}, "
            f"95% CI [{coefficient['ci_lower']:+.4f}, {coefficient['ci_upper']:+.4f}], "
            f"p = {coefficient['p_value']:.3g})"
        )


def run_transaction_return_regression(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    data_paths: DataPaths | None = None,
    horizon_months: int = HORIZON_MONTHS,
    price_lookup_tolerance_days: int = PRICE_LOOKUP_TOLERANCE_DAYS,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Fit the transaction-level OLS and save coefficients, summary, and panel."""
    data_paths = data_paths or DataPaths()
    panel = _build_panel(
        data_paths=data_paths,
        horizon_months=horizon_months,
        price_lookup_tolerance_days=price_lookup_tolerance_days,
    )
    if panel.empty:
        raise RuntimeError(
            "Empty panel after joining forward prices; check tolerance and horizon."
        )

    print(f"Fitting OLS: {FORMULA}")
    model_fit = fit_ols_clustered(FORMULA, panel, CLUSTER_COLUMN)
    coefficients = coefficients_dataframe(model_fit)

    output_directory = output_root / (
        run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    coefficients.to_csv(output_directory / "coefficients.csv", index=False)

    summary = _build_summary(
        panel=panel,
        coefficients=coefficients,
        horizon_months=horizon_months,
        output_directory=output_directory,
    )
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))

    _print_headline(summary)
    print(f"\nTransaction-return regression artefacts saved to {output_directory}")
    return summary
