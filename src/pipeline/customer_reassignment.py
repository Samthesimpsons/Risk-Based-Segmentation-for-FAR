"""Regroup customers onto their revealed risk band and write a regrouped data copy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.schemas import CustomerProfile
from src.config.settings import DataPaths
from src.data.loading import (
    load_assets,
    load_close_prices,
    load_customers,
    load_transactions,
)
from src.utils.profile_coherence import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    INCOME,
    RISK_BAND_NAMES,
    build_asset_risk_classes,
    build_customer_profile_lookup,
)

DEFAULT_OUTPUT_DATA_DIRECTORY = Path("data/regrouped")
DEFAULT_SUMMARY_DIRECTORY = Path("outputs/reassignment")

BAND_ORDER = [CONSERVATIVE, INCOME, BALANCED, AGGRESSIVE]
BAND_LABELS = [RISK_BAND_NAMES[band] for band in BAND_ORDER]


def compute_revealed_bands(
    buy_transactions: pd.DataFrame,
    asset_risk_classes: dict[str, int],
) -> dict[str, int]:
    """Return `customer_id -> revealed band`, the lower median of bought-asset bands.

    Buys whose asset has no assigned band are dropped. With an even number of bands the
    lower of the two middle values is taken, so ties round down toward Conservative.
    """
    bought = buy_transactions.assign(
        asset_band=buy_transactions["ISIN"].astype(str).map(asset_risk_classes)
    ).dropna(subset=["asset_band"])

    revealed_bands: dict[str, int] = {}
    for customer_id, group in bought.groupby("customerID"):
        sorted_bands = np.sort(group["asset_band"].to_numpy(dtype=int))
        lower_median = int(sorted_bands[(len(sorted_bands) - 1) // 2])
        revealed_bands[str(customer_id)] = lower_median
    return revealed_bands


def reassign_customer_bands(
    customers: pd.DataFrame,
    revealed_bands: dict[str, int],
    original_profiles: dict[str, CustomerProfile],
) -> pd.DataFrame:
    """Return a copy of `customers` with `riskLevel` replaced by the revealed band.

    Every banded customer with a revealed band has their `riskLevel` overwritten by the
    canonical band name; customers with no declared band or no scoreable Buys are left
    unchanged. The original value is preserved in `originalRiskLevel` and a boolean
    `reassigned` flags rows whose band actually changed.
    """
    result = customers.copy()
    result["originalRiskLevel"] = result["riskLevel"]

    new_risk_levels: list[Any] = []
    reassigned_flags: list[bool] = []
    for _, row in result.iterrows():
        customer_id = str(row["customerID"])
        profile = original_profiles.get(customer_id)
        original_band = profile.risk_band if profile is not None else None
        revealed_band = revealed_bands.get(customer_id)

        if original_band is not None and revealed_band is not None:
            new_risk_levels.append(RISK_BAND_NAMES[revealed_band])
            reassigned_flags.append(original_band != revealed_band)
        else:
            new_risk_levels.append(row["riskLevel"])
            reassigned_flags.append(False)

    result["riskLevel"] = new_risk_levels
    result["reassigned"] = reassigned_flags
    return result


def _band_population(bands: list[int]) -> dict[str, int]:
    """Count customers per band name across the supplied ordinal bands."""
    population = {label: 0 for label in BAND_LABELS}
    for band in bands:
        population[RISK_BAND_NAMES[band]] += 1
    return population


def _discordance_stats(
    buy_transactions: pd.DataFrame,
    asset_risk_classes: dict[str, int],
    customer_bands: dict[str, int],
) -> dict[str, float]:
    """Mean discordance and discordant share (d>=2) over scoreable Buys for given bands."""
    scoreable = buy_transactions.assign(
        asset_band=buy_transactions["ISIN"].astype(str).map(asset_risk_classes),
        customer_band=buy_transactions["customerID"].astype(str).map(customer_bands),
    ).dropna(subset=["asset_band", "customer_band"])

    if scoreable.empty:
        return {"mean_discordance": 0.0, "discordant_share": 0.0}

    discordance = (
        scoreable["customer_band"].astype(int) - scoreable["asset_band"].astype(int)
    ).abs()
    return {
        "mean_discordance": float(discordance.mean()),
        "discordant_share": float((discordance >= 2).mean()),
    }


def _build_summary(
    buy_transactions: pd.DataFrame,
    original_profiles: dict[str, CustomerProfile],
    revealed_bands: dict[str, int],
    asset_risk_classes: dict[str, int],
    reassigned_customers: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble reassignment counts, per-band population shift, and discordance change."""
    original_bands = {
        customer_id: profile.risk_band
        for customer_id, profile in original_profiles.items()
        if profile.risk_band is not None
    }
    revealed_for_banded = {
        customer_id: revealed_bands[customer_id]
        for customer_id in original_bands
        if customer_id in revealed_bands
    }
    new_bands = {
        customer_id: revealed_for_banded.get(customer_id, original_band)
        for customer_id, original_band in original_bands.items()
    }

    return {
        "populations": {
            "total_customers": int(len(original_profiles)),
            "banded_customers": int(len(original_bands)),
            "banded_customers_with_scoreable_buys": int(len(revealed_for_banded)),
            "customers_reassigned": int(reassigned_customers["reassigned"].sum()),
        },
        "band_population_before": _band_population(list(original_bands.values())),
        "band_population_after": _band_population(list(new_bands.values())),
        "discordance_before": _discordance_stats(
            buy_transactions, asset_risk_classes, original_bands
        ),
        "discordance_after": _discordance_stats(
            buy_transactions, asset_risk_classes, new_bands
        ),
    }


def run_customer_reassignment(
    output_data_directory: Path = DEFAULT_OUTPUT_DATA_DIRECTORY,
    data_paths: DataPaths | None = None,
    summary_directory: Path = DEFAULT_SUMMARY_DIRECTORY,
) -> dict[str, Any]:
    """Reassign banded customers to their revealed band and write the regrouped data copy."""
    data_paths = data_paths or DataPaths()
    output_data_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)

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

    buy_transactions = transactions[transactions["transactionType"] == "Buy"].copy()

    print("Building original profiles and asset risk classes...")
    original_profiles = build_customer_profile_lookup(customers)
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print("Computing revealed bands from purchase history...")
    revealed_bands = compute_revealed_bands(buy_transactions, asset_risk_classes)

    print("Reassigning customer bands...")
    reassigned_customers = reassign_customer_bands(
        customers, revealed_bands, original_profiles
    )

    print(f"Writing regrouped data copy to {output_data_directory}/...")
    reassigned_customers.to_csv(
        output_data_directory / data_paths.customer_information_file, index=False
    )
    shutil.copy2(
        data_paths.data_directory / data_paths.asset_information_file,
        output_data_directory / data_paths.asset_information_file,
    )

    print("Computing reassignment summary...")
    summary = _build_summary(
        buy_transactions,
        original_profiles,
        revealed_bands,
        asset_risk_classes,
        reassigned_customers,
    )
    summary_path = summary_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    populations = summary["populations"]
    before = summary["discordance_before"]
    after = summary["discordance_after"]
    print(
        f"\nReassignment complete. Regrouped data in: {output_data_directory}\n"
        f"  Banded customers: {populations['banded_customers']} "
        f"({populations['customers_reassigned']} reassigned)\n"
        f"  Mean discordance: {before['mean_discordance']:.4f} -> "
        f"{after['mean_discordance']:.4f}\n"
        f"  Discordant share (d>=2): {before['discordant_share']:.4f} -> "
        f"{after['discordant_share']:.4f}\n"
        f"  Summary: {summary_path}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regroup customers onto their revealed risk band."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIRECTORY,
        help="Directory to write the regrouped data copy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DataPaths().data_directory,
        help="Directory of the original FAR-Trans CSV files.",
    )
    args = parser.parse_args()
    run_customer_reassignment(
        output_data_directory=args.output_dir,
        data_paths=DataPaths(data_directory=args.data_dir),
    )
