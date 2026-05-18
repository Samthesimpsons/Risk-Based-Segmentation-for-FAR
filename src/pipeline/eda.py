"""Profile-coherence dataset audit: writes summary.json consumed by findings.ipynb."""

from __future__ import annotations

import json
from collections import Counter
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
    NUMBER_OF_RISK_BANDS,
    RISK_BAND_NAMES,
    build_asset_risk_classes,
    build_customer_profile_lookup,
    compute_pairwise_discordance,
    is_profile_coherent,
)

DEFAULT_OUTPUT_DIRECTORY = Path("outputs/eda")

BAND_ORDER = [CONSERVATIVE, INCOME, BALANCED, AGGRESSIVE]
BAND_LABELS = [RISK_BAND_NAMES[band] for band in BAND_ORDER]


def _annotate_transactions_with_discordance(
    transactions: pd.DataFrame,
    customer_profiles: dict[str, CustomerProfile],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Return a copy of transactions with discordance and band columns attached."""
    result = transactions.copy()
    customer_bands: list[float] = []
    asset_bands: list[float] = []
    discordances: list[float] = []
    coherence_flags: list[bool] = []

    for _, row in result.iterrows():
        customer_id = str(row["customerID"])
        asset_id = str(row["ISIN"])
        profile = customer_profiles.get(customer_id)
        customer_band = profile.risk_band if profile is not None else None
        asset_band = asset_risk_classes.get(asset_id)

        discordance = compute_pairwise_discordance(customer_band, asset_band)

        customer_bands.append(
            float(customer_band) if customer_band is not None else float("nan")
        )
        asset_bands.append(
            float(asset_band) if asset_band is not None else float("nan")
        )
        discordances.append(
            float(discordance) if discordance is not None else float("nan")
        )
        coherence_flags.append(is_profile_coherent(discordance))

    result["customer_band"] = customer_bands
    result["asset_band"] = asset_bands
    result["discordance"] = discordances
    result["is_profile_coherent"] = coherence_flags
    return result


def _asset_band_distribution(
    asset_information: pd.DataFrame, asset_risk_classes: dict[str, int]
) -> dict[str, dict[str, int]]:
    """Return `{category -> {band_name -> count}}` for every assetCategory."""
    distribution: dict[str, dict[str, int]] = {}
    for _, row in asset_information.iterrows():
        category = str(row["assetCategory"])
        isin = str(row["ISIN"])
        band = asset_risk_classes.get(isin)
        if band is None:
            continue
        distribution.setdefault(category, {label: 0 for label in BAND_LABELS})
        distribution[category][RISK_BAND_NAMES[band]] += 1
    return distribution


def _transaction_discordance_summary(
    annotated_transactions: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate discordance counts and rates across all transactions."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"])
    discordance_values = valid["discordance"].astype(int).to_list()
    counter = Counter(discordance_values)
    total = sum(counter.values())
    distribution = {
        str(distance): counter.get(distance, 0)
        for distance in range(NUMBER_OF_RISK_BANDS)
    }
    coherent_count = sum(1 for distance in discordance_values if distance <= 1)
    strict_count = sum(1 for distance in discordance_values if distance == 0)
    return {
        "total_transactions": int(len(annotated_transactions)),
        "transactions_with_both_bands": int(total),
        "discordance_counts": distribution,
        "fraction_coherent_default": float(coherent_count / total) if total else 0.0,
        "fraction_coherent_strict": float(strict_count / total) if total else 0.0,
        "mean_discordance": float(np.mean(discordance_values)) if total else 0.0,
    }


def _customer_self_discordance(
    annotated_transactions: pd.DataFrame,
) -> dict[str, float | list[float] | list[int]]:
    """Per-customer fraction of profile-discordant transactions, summarised as a histogram."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"])
    grouped = valid.groupby("customerID")
    discordant_share = grouped.apply(
        lambda group: 1.0 - float(group["is_profile_coherent"].mean()),
        include_groups=False,
    )
    values = np.asarray(discordant_share.to_numpy(dtype=float))
    histogram_edges = [round(edge / 10, 1) for edge in range(11)]
    if values.size == 0:
        return {
            "mean_discordant_share": 0.0,
            "fraction_fully_coherent": 0.0,
            "fraction_fully_discordant": 0.0,
            "discordant_share_histogram_edges": histogram_edges,
            "discordant_share_histogram_counts": [0] * 10,
        }
    counts, _ = np.histogram(values, bins=np.array(histogram_edges))
    return {
        "mean_discordant_share": float(values.mean()),
        "fraction_fully_coherent": float((values == 0.0).mean()),
        "fraction_fully_discordant": float((values == 1.0).mean()),
        "discordant_share_histogram_edges": histogram_edges,
        "discordant_share_histogram_counts": [int(count) for count in counts],
    }


def _discordance_by_year(annotated_transactions: pd.DataFrame) -> dict[str, float]:
    """Mean discordance per calendar year of the transaction timestamp."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"]).copy()
    valid["year"] = valid["timestamp"].dt.year
    means = valid.groupby("year")["discordance"].mean()
    return {str(year): float(value) for year, value in means.items()}


def _buy_coverage_by_year(
    buy_transactions: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Per-year Buy counts plus the last observed Buy date (used to flag partial years)."""
    by_year = buy_transactions.groupby(buy_transactions["timestamp"].dt.year)
    return {
        str(year): {
            "count": int(len(group)),
            "first_date": group["timestamp"].min().date().isoformat(),
            "last_date": group["timestamp"].max().date().isoformat(),
        }
        for year, group in by_year
    }


def _discordance_by_risk_level(
    annotated_transactions: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Distribution of discordance bins per declared MiFID band."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"]).copy()
    valid["customer_band_name"] = (
        valid["customer_band"].astype(int).map(RISK_BAND_NAMES)
    )
    valid["discordance_int"] = valid["discordance"].astype(int)

    distribution: dict[str, dict[str, int]] = {}
    for band_name, group in valid.groupby("customer_band_name"):
        counts = group["discordance_int"].value_counts().to_dict()
        distribution[str(band_name)] = {
            str(distance): int(counts.get(distance, 0))
            for distance in range(NUMBER_OF_RISK_BANDS)
        }
    return distribution


def run_eda(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    data_paths: DataPaths | None = None,
) -> dict[str, Any]:
    """Compute the profile-coherence dataset audit and write summary.json."""
    data_paths = data_paths or DataPaths()
    output_directory.mkdir(parents=True, exist_ok=True)

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

    print("Building customer profile lookup...")
    profiles = build_customer_profile_lookup(customers)

    print("Assigning asset risk classes (hierarchical mapping)...")
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print("Annotating transactions with profile-discordance...")
    annotated_buys = _annotate_transactions_with_discordance(
        buy_transactions, profiles, asset_risk_classes
    )

    print("Computing summary statistics...")
    asset_distribution = _asset_band_distribution(assets, asset_risk_classes)
    transaction_summary = _transaction_discordance_summary(annotated_buys)
    self_summary = _customer_self_discordance(annotated_buys)
    year_means = _discordance_by_year(annotated_buys)
    buy_coverage_by_year = _buy_coverage_by_year(buy_transactions)
    risk_level_distribution = _discordance_by_risk_level(annotated_buys)

    summary: dict[str, Any] = {
        "populations": {
            "total_customers": int(len(profiles)),
            "customers_with_band": int(
                sum(1 for p in profiles.values() if p.risk_band is not None)
            ),
            "customers_with_predicted_band": int(
                sum(
                    1
                    for p in profiles.values()
                    if p.risk_band is not None and p.risk_band_is_predicted
                )
            ),
            "total_assets": int(len(assets)),
            "total_buy_transactions": int(len(buy_transactions)),
            "transactions_with_both_bands": int(
                annotated_buys.dropna(subset=["customer_band", "asset_band"]).shape[0]
            ),
        },
        "asset_band_distribution_by_category": asset_distribution,
        "transaction_discordance_summary": transaction_summary,
        "customer_self_discordance_summary": self_summary,
        "buy_coverage_by_year": buy_coverage_by_year,
        "mean_discordance_by_year": year_means,
        "transaction_discordance_by_risk_level": risk_level_distribution,
    }

    print("Writing summary.json...")
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nEDA complete. Outputs in: {output_directory}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the profile-coherence dataset audit"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory to write summary.json",
    )
    args = parser.parse_args()
    run_eda(output_directory=args.output_dir)
