"""Post-evaluation decomposition: best-trial selection, coherent/discordant ROI breakdown, scatter plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import DataPaths
from src.data.loading import load_assets, load_close_prices, load_customers
from src.pipeline.baseline_evaluation import (
    GRID_SPECS,
    GridSpec,
    _PRIMARY_METRIC_TO_KEY,
)
from src.profile_coherence.customer_profile import build_customer_profile_lookup
from src.profile_coherence.risk_classification import build_asset_risk_classes

DEFAULT_EVALUATION_DIRECTORY = Path("outputs/results/evaluation")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/baseline_decomposition")

_PRIMARY_METRIC_TO_PER_SPLIT_COLUMN: dict[str, str] = {
    "average_ndcg": "ndcg_at_k",
    "average_roi": "roi_at_k",
    "average_recall": "recall_at_k",
    "average_profile_coherence": "profile_coherence_at_k",
}

_DISPLAY_MODEL_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "light_gcn": "LightGCN",
    "profile_coherent_light_gcn": "Profile-Coherent LightGCN",
}


def _discover_run_directories(
    evaluation_directory: Path,
    explicit_run_timestamp: str | None,
) -> dict[str, Path]:
    """Map each model name to the run-timestamp directory that the analysis should use.

    When `explicit_run_timestamp` is None we pick each model's latest by
    lexicographic timestamp ordering (which matches chronological since the
    timestamps are `YYYYMMDD_HHMMSS`).
    """
    chosen: dict[str, Path] = {}
    if not evaluation_directory.exists():
        return chosen

    for model_directory in sorted(evaluation_directory.iterdir()):
        if not model_directory.is_dir():
            continue
        run_dirs = [path for path in model_directory.iterdir() if path.is_dir()]
        if not run_dirs:
            continue

        if explicit_run_timestamp is not None:
            candidate = model_directory / explicit_run_timestamp
            if not candidate.is_dir():
                continue
            chosen[model_directory.name] = candidate
            continue

        chosen[model_directory.name] = sorted(run_dirs, key=lambda p: p.name)[-1]
    return chosen


def _load_per_split_metrics(run_directory: Path) -> pd.DataFrame:
    """Concatenate every trial's `per_split_metrics.csv` under one run directory."""
    frames: list[pd.DataFrame] = []
    for trial_directory in sorted(run_directory.iterdir()):
        metrics_path = trial_directory / "per_split_metrics.csv"
        if not metrics_path.exists():
            continue
        frames.append(pd.read_csv(metrics_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _average_per_trial(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reduce a per-split metrics frame to one row per trial with means and stds."""
    if metrics_dataframe.empty:
        return metrics_dataframe

    aggregations: dict[str, list[str]] = {
        column: ["mean", "std"]
        for column in (
            "ndcg_at_k",
            "roi_at_k",
            "recall_at_k",
            "profile_coherence_at_k",
        )
    }
    aggregations["split_index"] = ["count"]
    aggregated = metrics_dataframe.groupby("trial_id").agg(aggregations)
    aggregated.columns = ["_".join(column).rstrip("_") for column in aggregated.columns]
    return aggregated.reset_index().rename(columns={"split_index_count": "split_count"})


def _select_best_trial_id(
    aggregated_metrics: pd.DataFrame, primary_metric_key: str
) -> str | None:
    """Return the trial_id with the highest mean of the primary metric, or None."""
    if aggregated_metrics.empty:
        return None
    column = f"{_PRIMARY_METRIC_TO_PER_SPLIT_COLUMN[primary_metric_key]}_mean"
    if column not in aggregated_metrics.columns:
        return None
    return str(aggregated_metrics.loc[aggregated_metrics[column].idxmax(), "trial_id"])


def _annotate_recommendations(
    recommendations: pd.DataFrame,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Attach customer_band, asset_band, discordance, is_coherent columns."""
    annotated = recommendations.copy()

    annotated["customer_band"] = annotated["customer_id"].map(
        lambda customer_id: (
            profile.risk_band
            if (profile := customer_profiles.get(customer_id)) is not None
            else None
        )
    )
    annotated["asset_band"] = annotated["asset_id"].map(asset_risk_classes)

    customer_band_array = annotated["customer_band"].astype("Float64")
    asset_band_array = annotated["asset_band"].astype("Float64")
    discordance = (customer_band_array - asset_band_array).abs()
    annotated["discordance"] = discordance
    annotated["is_coherent"] = (discordance <= 1).astype("boolean")
    annotated["is_strictly_coherent"] = (discordance == 0).astype("boolean")
    return annotated


def _build_main_results_row(
    model_name: str,
    aggregated: pd.DataFrame,
    best_trial_id: str,
    primary_metric_key: str,
) -> dict[str, Any]:
    """Pull the best trial's row out of the aggregated frame."""
    best = aggregated.loc[aggregated["trial_id"] == best_trial_id].iloc[0]
    return {
        "model": model_name,
        "display_name": _DISPLAY_MODEL_NAMES.get(model_name, model_name),
        "best_trial_id": best_trial_id,
        "primary_metric": primary_metric_key,
        "split_count": int(best["split_count"]),
        "ndcg_at_k_mean": float(best["ndcg_at_k_mean"]),
        "ndcg_at_k_std": float(best["ndcg_at_k_std"]),
        "roi_at_k_mean": float(best["roi_at_k_mean"]),
        "roi_at_k_std": float(best["roi_at_k_std"]),
        "recall_at_k_mean": float(best["recall_at_k_mean"]),
        "recall_at_k_std": float(best["recall_at_k_std"]),
        "profile_coherence_at_k_mean": float(best["profile_coherence_at_k_mean"]),
        "profile_coherence_at_k_std": float(best["profile_coherence_at_k_std"]),
    }


def _decomposition_row(model_name: str, annotated: pd.DataFrame) -> dict[str, Any]:
    """ROI breakdown by coherence flag for one model's best trial."""
    total_recommendations = len(annotated)
    coherent = annotated[annotated["is_coherent"].fillna(False)]
    discordant = annotated[~annotated["is_coherent"].fillna(False)]
    strict_coherent = annotated[annotated["is_strictly_coherent"].fillna(False)]
    strict_discordant = annotated[~annotated["is_strictly_coherent"].fillna(False)]

    return {
        "model": model_name,
        "display_name": _DISPLAY_MODEL_NAMES.get(model_name, model_name),
        "total_recommendations": total_recommendations,
        "coherent_recommendations": int(len(coherent)),
        "discordant_recommendations": int(len(discordant)),
        "coherent_share": float(len(coherent) / total_recommendations)
        if total_recommendations
        else 0.0,
        "mean_monthly_return_overall": float(annotated["monthly_return"].mean())
        if total_recommendations
        else 0.0,
        "mean_monthly_return_coherent": float(coherent["monthly_return"].mean())
        if len(coherent)
        else 0.0,
        "mean_monthly_return_discordant": float(discordant["monthly_return"].mean())
        if len(discordant)
        else 0.0,
        "strict_coherent_recommendations": int(len(strict_coherent)),
        "mean_monthly_return_strict_coherent": float(
            strict_coherent["monthly_return"].mean()
        )
        if len(strict_coherent)
        else 0.0,
        "mean_monthly_return_strict_discordant": float(
            strict_discordant["monthly_return"].mean()
        )
        if len(strict_discordant)
        else 0.0,
    }


def _save_scatter(
    main_results: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Render a single labelled scatter (one point per baseline)."""
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for _, row in main_results.iterrows():
        axis.scatter(row[x_column], row[y_column], s=120)
        axis.annotate(
            row["display_name"],
            (row[x_column], row[y_column]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _print_main_results_table(main_results: pd.DataFrame, top_k: int = 10) -> None:
    """Pretty-print the headline N-row table to stdout for the cluster log."""
    if main_results.empty:
        print("(no main results to display)")
        return
    header = (
        f"{'Model':<28} {'nDCG@' + str(top_k):<14} {'ROI@' + str(top_k):<14}"
        f" {'Recall@' + str(top_k):<14} {'PC@' + str(top_k):<10}"
    )
    width = len(header)
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'-' * width}")
    for _, row in main_results.iterrows():
        print(
            f"{row['display_name']:<28}"
            f" {row['ndcg_at_k_mean']:<14.4f}"
            f" {row['roi_at_k_mean']:<14.6f}"
            f" {row['recall_at_k_mean']:<14.4f}"
            f" {row['profile_coherence_at_k_mean']:<10.4f}"
        )
    print(f"{'=' * width}")


def _print_decomposition_table(decomposition: pd.DataFrame) -> None:
    """Pretty-print the decomposition table to stdout."""
    if decomposition.empty:
        print("(no decomposition rows to display)")
        return
    header = (
        f"{'Model':<28} {'Coherent share':<16}"
        f" {'ROI overall':<14} {'ROI coherent':<14} {'ROI discordant':<14}"
    )
    width = len(header)
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'-' * width}")
    for _, row in decomposition.iterrows():
        print(
            f"{row['display_name']:<28}"
            f" {row['coherent_share']:<16.4f}"
            f" {row['mean_monthly_return_overall']:<14.6f}"
            f" {row['mean_monthly_return_coherent']:<14.6f}"
            f" {row['mean_monthly_return_discordant']:<14.6f}"
        )
    print(f"{'=' * width}")


def run_decomposition(
    evaluation_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
    top_k: int = 10,
    extra_grid_specs: dict[str, GridSpec] | None = None,
) -> dict[str, Any]:
    """Run the post-evaluation decomposition; returns the headline summary dict."""
    data_paths = data_paths or DataPaths()
    all_grid_specs: dict[str, GridSpec] = {**GRID_SPECS, **(extra_grid_specs or {})}
    print("Loading customer profiles and asset risk classes...")
    customers = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    customer_profiles = build_customer_profile_lookup(customers)
    assets = load_assets(data_paths.data_directory / data_paths.asset_information_file)
    close_prices = load_close_prices(
        data_paths.data_directory / data_paths.close_prices_file
    )
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print(f"Discovering evaluation runs in {evaluation_directory} ...")
    runs = _discover_run_directories(evaluation_directory, run_timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No evaluation runs found under {evaluation_directory}. "
            "Run `uv run poe evaluate-baselines` first."
        )

    main_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    chosen_run_timestamp = run_timestamp

    for model_name, run_directory in runs.items():
        if model_name not in all_grid_specs:
            print(f"  Skipping unknown model directory '{model_name}'")
            continue

        if chosen_run_timestamp is None:
            chosen_run_timestamp = run_directory.name

        primary_metric = all_grid_specs[model_name].primary_metric
        primary_metric_key = _PRIMARY_METRIC_TO_KEY[primary_metric]
        print(
            f"\n[{model_name}] reading {run_directory}"
            f" (best trial by {primary_metric_key})"
        )

        per_split_metrics = _load_per_split_metrics(run_directory)
        if per_split_metrics.empty:
            print("  No per_split_metrics.csv found; skipping.")
            continue
        aggregated = _average_per_trial(per_split_metrics)
        best_trial_id = _select_best_trial_id(aggregated, primary_metric_key)
        if best_trial_id is None:
            print("  Could not determine best trial; skipping.")
            continue
        print(f"  Best trial: {best_trial_id}")

        main_rows.append(
            _build_main_results_row(
                model_name, aggregated, best_trial_id, primary_metric_key
            )
        )

        parquet_path = run_directory / best_trial_id / "recommendations.parquet"
        if not parquet_path.exists():
            print(
                f"  Missing recommendations parquet at {parquet_path}; skipping decomposition."
            )
            continue
        recommendations = pd.read_parquet(parquet_path)
        annotated = _annotate_recommendations(
            recommendations, customer_profiles, asset_risk_classes
        )
        decomposition_rows.append(_decomposition_row(model_name, annotated))

    main_results = pd.DataFrame(main_rows)
    decomposition = pd.DataFrame(decomposition_rows)

    output_directory = output_root / (chosen_run_timestamp or "latest")
    output_directory.mkdir(parents=True, exist_ok=True)

    main_results.to_csv(output_directory / "main_results.csv", index=False)
    decomposition.to_csv(output_directory / "decomposition.csv", index=False)

    if not main_results.empty:
        _save_scatter(
            main_results,
            x_column="ndcg_at_k_mean",
            y_column="profile_coherence_at_k_mean",
            x_label=f"nDCG@{top_k}",
            y_label=f"PC@{top_k}",
            title="Preference accuracy vs profile coherence",
            output_path=output_directory / "scatter_ndcg_vs_pc.png",
        )
        _save_scatter(
            main_results,
            x_column="profile_coherence_at_k_mean",
            y_column="roi_at_k_mean",
            x_label=f"PC@{top_k}",
            y_label=f"ROI@{top_k} (monthly)",
            title="Profile coherence vs realised ROI",
            output_path=output_directory / "scatter_pc_vs_roi.png",
        )

    summary = {
        "run_timestamp": chosen_run_timestamp,
        "evaluation_directory": str(evaluation_directory),
        "output_directory": str(output_directory),
        "models": [row["model"] for row in main_rows],
        "main_results": main_rows,
        "decomposition": decomposition_rows,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))

    _print_main_results_table(main_results, top_k=top_k)
    _print_decomposition_table(decomposition)
    print(f"\nDecomposition outputs saved to {output_directory}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-evaluation decomposition of FAR-Trans baselines."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIRECTORY,
        help="Directory containing per-model per-run evaluation outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write decomposition artefacts (default: outputs/analysis/baseline_decomposition).",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help=(
            "Specific run timestamp (matches directory name under each model). "
            "Default: pick each model's latest."
        ),
    )
    arguments = parser.parse_args()

    run_decomposition(
        evaluation_directory=arguments.evaluation_dir,
        output_root=arguments.output_root,
        run_timestamp=arguments.run_timestamp,
    )
