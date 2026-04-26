"""Profile-coherence dataset audit: writes summary.json and band/discordance figures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    compute_annualised_volatility,
    compute_pairwise_discordance,
    is_profile_coherent,
)

DEFAULT_OUTPUT_DIRECTORY = Path("outputs/eda")

BAND_ORDER = [CONSERVATIVE, INCOME, BALANCED, AGGRESSIVE]
BAND_LABELS = [RISK_BAND_NAMES[band] for band in BAND_ORDER]


def annotate_transactions_with_discordance(
    transactions: pd.DataFrame,
    customer_profiles: dict[str, CustomerProfile],
    asset_risk_classes: dict[str, int],
    *,
    squared: bool = False,
) -> pd.DataFrame:
    """Return a copy of `transactions` with discordance and band columns attached."""
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

        discordance = compute_pairwise_discordance(
            customer_band, asset_band, squared=squared
        )

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


def _build_volatility_only_risk_classes(
    asset_information: pd.DataFrame, close_prices: pd.DataFrame
) -> dict[str, int]:
    """Map every asset to a volatility-quartile risk band over the full universe."""
    volatility_lookup = compute_annualised_volatility(close_prices)
    if volatility_lookup.empty:
        return {str(isin): BALANCED for isin in asset_information["ISIN"]}

    quartiles = np.quantile(volatility_lookup.to_numpy(), [0.25, 0.5, 0.75])

    risk_classes: dict[str, int] = {}
    for _, row in asset_information.iterrows():
        isin = str(row["ISIN"])
        if isin not in volatility_lookup.index:
            risk_classes[isin] = BALANCED
            continue
        volatility = float(volatility_lookup.loc[isin])
        if volatility <= quartiles[0]:
            risk_classes[isin] = CONSERVATIVE
        elif volatility <= quartiles[1]:
            risk_classes[isin] = INCOME
        elif volatility <= quartiles[2]:
            risk_classes[isin] = BALANCED
        else:
            risk_classes[isin] = AGGRESSIVE
    return risk_classes


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


def _customer_band_distribution(
    profiles: dict[str, Any],
) -> dict[str, Any]:
    """Return per-band counts split by declared vs predicted provenance."""
    distribution: dict[str, Any] = {
        "declared": {label: 0 for label in BAND_LABELS},
        "predicted": {label: 0 for label in BAND_LABELS},
        "none_count": 0,
    }
    for profile in profiles.values():
        if profile.risk_band is None:
            distribution["none_count"] += 1
            continue
        provenance = "predicted" if profile.risk_band_is_predicted else "declared"
        distribution[provenance][RISK_BAND_NAMES[profile.risk_band]] += 1
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
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-customer fraction of profile-discordant transactions plus a summary dict."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"])
    grouped = valid.groupby("customerID")
    discordant_share = grouped.apply(
        lambda group: 1.0 - float(group["is_profile_coherent"].mean()),
        include_groups=False,
    )
    values = np.asarray(discordant_share.to_numpy(dtype=float))
    if values.size == 0:
        return values, {
            "mean_discordant_share": 0.0,
            "median_discordant_share": 0.0,
            "fraction_with_majority_discordance": 0.0,
            "fraction_fully_coherent": 0.0,
            "fraction_fully_discordant": 0.0,
        }
    summary = {
        "mean_discordant_share": float(values.mean()),
        "median_discordant_share": float(np.median(values)),
        "fraction_with_majority_discordance": float((values > 0.5).mean()),
        "fraction_fully_coherent": float((values == 0.0).mean()),
        "fraction_fully_discordant": float((values == 1.0).mean()),
    }
    return values, summary


def _discordance_by_segment(
    annotated_transactions: pd.DataFrame,
    customer_segment_lookup: dict[str, str | None],
) -> dict[str, float]:
    """Mean discordance per `customerType`, restricted to transactions with both bands."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"]).copy()
    valid["segment"] = valid["customerID"].map(customer_segment_lookup)
    means = valid.groupby("segment")["discordance"].mean()
    return {str(segment): float(value) for segment, value in means.items()}


def _discordance_by_year(annotated_transactions: pd.DataFrame) -> dict[str, float]:
    """Mean discordance per calendar year of the transaction timestamp."""
    valid = annotated_transactions.dropna(subset=["customer_band", "asset_band"]).copy()
    valid["year"] = valid["timestamp"].dt.year
    means = valid.groupby("year")["discordance"].mean()
    return {str(year): float(value) for year, value in means.items()}


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


def _save_asset_band_distribution_plot(
    distribution: dict[str, dict[str, int]], output_path: Path
) -> None:
    """Stacked bar of asset-band counts by category."""
    categories = sorted(distribution.keys())
    bottom = np.zeros(len(categories))
    figure, axis = plt.subplots(figsize=(8, 5))
    for label in BAND_LABELS:
        values = np.array([distribution[c].get(label, 0) for c in categories])
        axis.bar(categories, values, bottom=bottom, label=label)
        bottom += values
    axis.set_xlabel("Asset Category")
    axis.set_ylabel("Number of Assets")
    axis.set_title("Asset Distribution Across MiFID Risk Bands by Category")
    axis.legend(title="MiFID Band")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_customer_band_distribution_plot(
    distribution: dict[str, dict[str, int]], output_path: Path
) -> None:
    """Grouped bar showing declared vs predicted band counts."""
    indices = np.arange(len(BAND_LABELS))
    bar_width = 0.4
    figure, axis = plt.subplots(figsize=(8, 5))
    declared_values = [distribution["declared"][label] for label in BAND_LABELS]
    predicted_values = [distribution["predicted"][label] for label in BAND_LABELS]
    axis.bar(indices - bar_width / 2, declared_values, bar_width, label="Declared")
    axis.bar(indices + bar_width / 2, predicted_values, bar_width, label="Predicted")
    axis.set_xticks(indices)
    axis.set_xticklabels(BAND_LABELS)
    axis.set_xlabel("Declared MiFID Band")
    axis.set_ylabel("Number of Customers")
    axis.set_title("Customer Risk-Band Distribution (Declared vs Predicted)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_transaction_discordance_plot(
    discordance_counts: dict[str, int], output_path: Path
) -> None:
    """Bar chart of transaction-level discordance."""
    ordered_keys: list[str] = [
        str(distance) for distance in range(NUMBER_OF_RISK_BANDS)
    ]
    values = [discordance_counts.get(key, 0) for key in ordered_keys]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar([f"d={key}" for key in ordered_keys], values, color="#4472C4")
    axis.set_xlabel("Discordance |b_user - b_asset|")
    axis.set_ylabel("Number of Transactions")
    axis.set_title("Transaction-Level Profile Discordance Distribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_self_discordance_histogram(
    discordant_share: np.ndarray, output_path: Path
) -> None:
    """Histogram of per-customer share of discordant transactions."""
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(discordant_share, bins=20, color="#A0522D", edgecolor="white")
    axis.set_xlabel("Fraction of Customer's Transactions That Are Profile-Discordant")
    axis.set_ylabel("Number of Customers")
    axis.set_title("Per-Customer Self-Discordance Distribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_segment_discordance_plot(means: dict[str, float], output_path: Path) -> None:
    """Bar chart of mean discordance per customer segment."""
    segments = list(means.keys())
    values = [means[segment] for segment in segments]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(segments, values, color="#5B9BD5")
    axis.set_xlabel("Customer Type")
    axis.set_ylabel("Mean Discordance |b_user - b_asset|")
    axis.set_title("Mean Profile Discordance by Customer Segment")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_year_discordance_plot(means: dict[str, float], output_path: Path) -> None:
    """Line plot of mean discordance over calendar years."""
    sorted_pairs = sorted(means.items(), key=lambda item: int(item[0]))
    years = [pair[0] for pair in sorted_pairs]
    values = [pair[1] for pair in sorted_pairs]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(years, values, marker="o", color="#ED7D31")
    axis.set_xlabel("Year")
    axis.set_ylabel("Mean Discordance")
    axis.set_title("Profile Discordance Over Time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_risk_level_discordance_plot(
    distribution: dict[str, dict[str, int]], output_path: Path
) -> None:
    """Stacked bar of discordance bins per declared MiFID band."""
    bands = [label for label in BAND_LABELS if label in distribution]
    bottom = np.zeros(len(bands))
    figure, axis = plt.subplots(figsize=(8, 5))
    for distance in range(NUMBER_OF_RISK_BANDS):
        values = np.array([distribution[band].get(str(distance), 0) for band in bands])
        axis.bar(bands, values, bottom=bottom, label=f"d={distance}")
        bottom += values
    axis.set_xlabel("Declared MiFID Band")
    axis.set_ylabel("Number of Transactions")
    axis.set_title("Transaction Discordance Decomposition by Declared MiFID Band")
    axis.legend(title="Discordance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def run_eda(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    data_paths: DataPaths | None = None,
) -> dict[str, Any]:
    """Compute the profile-coherence dataset audit, save figures, return the summary dict."""
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
    customer_segment_lookup = {
        customer_id: profile.customer_type for customer_id, profile in profiles.items()
    }

    print("Assigning asset risk classes (hierarchical mapping)...")
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print("Annotating transactions with profile-discordance...")
    annotated_buys = annotate_transactions_with_discordance(
        buy_transactions, profiles, asset_risk_classes
    )

    print("Computing summary statistics...")
    asset_distribution = _asset_band_distribution(assets, asset_risk_classes)
    customer_distribution = _customer_band_distribution(profiles)
    transaction_summary = _transaction_discordance_summary(annotated_buys)
    discordant_share, self_summary = _customer_self_discordance(annotated_buys)
    segment_means = _discordance_by_segment(annotated_buys, customer_segment_lookup)
    year_means = _discordance_by_year(annotated_buys)
    risk_level_distribution = _discordance_by_risk_level(annotated_buys)

    print("Computing sensitivity: pure-volatility risk classes...")
    volatility_only_classes = _build_volatility_only_risk_classes(assets, close_prices)
    volatility_only_summary = _transaction_discordance_summary(
        annotate_transactions_with_discordance(
            buy_transactions, profiles, volatility_only_classes
        )
    )

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
        "customer_band_distribution": customer_distribution,
        "transaction_discordance_summary": transaction_summary,
        "customer_self_discordance_summary": self_summary,
        "mean_discordance_by_segment": segment_means,
        "mean_discordance_by_year": year_means,
        "transaction_discordance_by_risk_level": risk_level_distribution,
        "sensitivity_volatility_only_summary": volatility_only_summary,
    }

    print("Writing summary.json...")
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Writing plots...")
    _save_asset_band_distribution_plot(
        asset_distribution, output_directory / "asset_band_distribution.png"
    )
    _save_customer_band_distribution_plot(
        customer_distribution, output_directory / "customer_band_distribution.png"
    )
    _save_transaction_discordance_plot(
        transaction_summary["discordance_counts"],
        output_directory / "transaction_discordance_distribution.png",
    )
    _save_self_discordance_histogram(
        discordant_share, output_directory / "customer_self_discordance_histogram.png"
    )
    _save_segment_discordance_plot(
        segment_means, output_directory / "discordance_by_segment.png"
    )
    _save_year_discordance_plot(
        year_means, output_directory / "discordance_by_year.png"
    )
    _save_risk_level_discordance_plot(
        risk_level_distribution, output_directory / "discordance_by_risk_level.png"
    )

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
        help="Directory to write figures and summary.json",
    )
    args = parser.parse_args()
    run_eda(output_directory=args.output_dir)
