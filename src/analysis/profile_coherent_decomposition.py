"""Ablation + lambda-sensitivity tables for Profile-Coherent LightGCN trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.baseline_decomposition import (
    _annotate_recommendations,
    _average_per_trial,
    _load_per_split_metrics,
)
from src.config.settings import DataPaths
from src.data.loading import load_assets, load_close_prices, load_customers
from src.pipeline.profile_coherent_evaluation import MODEL_NAME, _DEFAULT_LAMBDA
from src.profile_coherence.customer_profile import build_customer_profile_lookup
from src.profile_coherence.risk_classification import build_asset_risk_classes

DEFAULT_EVALUATION_DIRECTORY = Path("outputs/results/evaluation")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/profile_coherent_decomposition")

_METRIC_COLUMNS: tuple[str, ...] = (
    "ndcg_at_k_mean",
    "ndcg_at_k_std",
    "roi_at_k_mean",
    "roi_at_k_std",
    "recall_at_k_mean",
    "recall_at_k_std",
    "profile_coherence_at_k_mean",
    "profile_coherence_at_k_std",
)


def _resolve_run_directory(
    evaluation_directory: Path, run_timestamp: str | None
) -> Path:
    """Return the PC-LGCN run directory to analyse (latest if not specified)."""
    model_directory = evaluation_directory / MODEL_NAME
    if not model_directory.exists():
        raise FileNotFoundError(
            f"No PC-LGCN evaluation directory at {model_directory}. "
            "Run `uv run poe evaluate-profile-coherent` first."
        )
    run_dirs = [path for path in model_directory.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped runs under {model_directory}.")
    if run_timestamp is not None:
        candidate = model_directory / run_timestamp
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"Requested run {run_timestamp} not found under {model_directory}."
            )
        return candidate
    return sorted(run_dirs, key=lambda p: p.name)[-1]


def _load_per_trial_dataframe(run_directory: Path) -> pd.DataFrame:
    """Average each trial's per-split metrics and merge in the cell hyperparameters."""
    per_split = _load_per_split_metrics(run_directory)
    if per_split.empty:
        return per_split

    aggregated = _average_per_trial(per_split)
    hyperparameter_columns = [
        "profile_embedding_enabled",
        "profile_coherence_enabled",
        "profile_coherence_lambda",
    ]
    cell_keys = per_split[["trial_id", *hyperparameter_columns]].drop_duplicates(
        "trial_id"
    )
    return aggregated.merge(cell_keys, on="trial_id", how="left")


def _split_ablation_and_sensitivity(
    per_trial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bucket each trial as either an ablation cell or a lambda-sensitivity point."""
    if per_trial.empty:
        return per_trial, per_trial

    is_default_lambda = per_trial["profile_coherence_lambda"] == _DEFAULT_LAMBDA
    ablation_mask = is_default_lambda
    sensitivity_mask = (
        per_trial["profile_embedding_enabled"] & per_trial["profile_coherence_enabled"]
    )

    ablation = (
        per_trial[ablation_mask]
        .sort_values(["profile_embedding_enabled", "profile_coherence_enabled"])
        .reset_index(drop=True)
    )
    sensitivity = (
        per_trial[sensitivity_mask]
        .sort_values("profile_coherence_lambda")
        .reset_index(drop=True)
    )
    return ablation, sensitivity


def _decomposition_for_trial(
    run_directory: Path,
    trial_id: str,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> dict[str, Any]:
    """Per-trial coherent/discordant ROI breakdown."""
    parquet_path = run_directory / trial_id / "recommendations.parquet"
    if not parquet_path.exists():
        return {}
    recommendations = pd.read_parquet(parquet_path)
    annotated = _annotate_recommendations(
        recommendations, customer_profiles, asset_risk_classes
    )
    coherent = annotated[annotated["is_coherent"].fillna(False)]
    discordant = annotated[~annotated["is_coherent"].fillna(False)]
    total = len(annotated)
    return {
        "coherent_share": float(len(coherent) / total) if total else 0.0,
        "mean_monthly_return_overall": float(annotated["monthly_return"].mean())
        if total
        else 0.0,
        "mean_monthly_return_coherent": float(coherent["monthly_return"].mean())
        if len(coherent)
        else 0.0,
        "mean_monthly_return_discordant": float(discordant["monthly_return"].mean())
        if len(discordant)
        else 0.0,
    }


def _attach_decomposition(
    table: pd.DataFrame,
    run_directory: Path,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Merge per-trial coherent/discordant ROI breakdown onto the table."""
    if table.empty:
        return table
    rows = []
    for _, row in table.iterrows():
        decomposition = _decomposition_for_trial(
            run_directory,
            str(row["trial_id"]),
            customer_profiles,
            asset_risk_classes,
        )
        rows.append({"trial_id": row["trial_id"], **decomposition})
    return table.merge(pd.DataFrame(rows), on="trial_id", how="left")


def _print_table(title: str, table: pd.DataFrame, columns: list[str]) -> None:
    """Pretty-print a small table to stdout for the cluster log."""
    if table.empty:
        print(f"({title}: empty)")
        return
    print(f"\n=== {title} ===")
    display = table[columns].copy()
    print(display.to_string(index=False))


def run_profile_coherent_decomposition(
    evaluation_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Produce the ablation and lambda-sensitivity tables for one PC-LGCN run."""
    data_paths = data_paths or DataPaths()
    run_directory = _resolve_run_directory(evaluation_directory, run_timestamp)
    chosen_run_timestamp = run_directory.name
    print(f"PC-LGCN run: {run_directory}")

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

    per_trial = _load_per_trial_dataframe(run_directory)
    if per_trial.empty:
        raise FileNotFoundError(
            f"No per_split_metrics.csv files under {run_directory}."
        )

    ablation, sensitivity = _split_ablation_and_sensitivity(per_trial)
    ablation = _attach_decomposition(
        ablation, run_directory, customer_profiles, asset_risk_classes
    )
    sensitivity = _attach_decomposition(
        sensitivity, run_directory, customer_profiles, asset_risk_classes
    )

    output_directory = output_root / chosen_run_timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(output_directory / "ablation.csv", index=False)
    sensitivity.to_csv(output_directory / "lambda_sensitivity.csv", index=False)

    summary_columns = [
        "profile_embedding_enabled",
        "profile_coherence_enabled",
        "profile_coherence_lambda",
        "ndcg_at_k_mean",
        "roi_at_k_mean",
        "recall_at_k_mean",
        "profile_coherence_at_k_mean",
        "coherent_share",
    ]
    _print_table(
        "PC-LGCN ablation (lambda fixed)",
        ablation,
        [c for c in summary_columns if c in ablation.columns],
    )
    _print_table(
        "PC-LGCN lambda sensitivity",
        sensitivity,
        [c for c in summary_columns if c in sensitivity.columns],
    )

    summary = {
        "run_timestamp": chosen_run_timestamp,
        "run_directory": str(run_directory),
        "output_directory": str(output_directory),
        "metric_columns": list(_METRIC_COLUMNS),
        "ablation_rows": ablation.to_dict(orient="records"),
        "sensitivity_rows": sensitivity.to_dict(orient="records"),
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDecomposition outputs saved to {output_directory}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile-Coherent LightGCN ablation + sensitivity decomposition."
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
        help="Where to write decomposition artefacts.",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Specific PC-LGCN run timestamp. Default: latest.",
    )
    arguments = parser.parse_args()

    run_profile_coherent_decomposition(
        evaluation_directory=arguments.evaluation_dir,
        output_root=arguments.output_root,
        run_timestamp=arguments.run_timestamp,
    )
