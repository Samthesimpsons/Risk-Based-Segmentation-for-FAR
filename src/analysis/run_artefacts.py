"""Discover and load per-trial artefacts written by the grid search stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.registry import PRIMARY_METRIC_TO_PER_SPLIT_COLUMN


def discover_run_directories(
    evaluation_directory: Path,
    explicit_run_timestamp: str | None,
) -> dict[str, Path]:
    """Map each model name to the run-timestamp directory the analysis should use."""
    chosen: dict[str, Path] = {}
    if not evaluation_directory.exists():
        return chosen

    for model_directory in sorted(evaluation_directory.iterdir()):
        if not model_directory.is_dir():
            continue
        run_directories = [path for path in model_directory.iterdir() if path.is_dir()]
        if not run_directories:
            continue

        if explicit_run_timestamp is not None:
            candidate = model_directory / explicit_run_timestamp
            if not candidate.is_dir():
                continue
            chosen[model_directory.name] = candidate
            continue

        chosen[model_directory.name] = sorted(
            run_directories, key=lambda path: path.name
        )[-1]
    return chosen


def load_per_split_metrics(run_directory: Path) -> pd.DataFrame:
    """Concatenate every trial's per_split_metrics.csv under one run directory."""
    frames: list[pd.DataFrame] = []
    for trial_directory in sorted(run_directory.iterdir()):
        metrics_path = trial_directory / "per_split_metrics.csv"
        if not metrics_path.exists():
            continue
        frames.append(pd.read_csv(metrics_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def average_per_trial(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reduce a per-split metrics frame to one row per trial with means and stds."""
    if metrics_dataframe.empty:
        return metrics_dataframe

    aggregations: dict[str, list[str]] = {
        column: ["mean", "std"]
        for column in (
            "ndcg_at_k",
            "roi_at_k",
            "profile_coherence_at_k",
            "profile_coherence_lift_at_k",
        )
        if column in metrics_dataframe.columns
    }
    aggregations["split_index"] = ["count"]
    aggregated = metrics_dataframe.groupby("trial_id").agg(aggregations)
    aggregated.columns = ["_".join(column).rstrip("_") for column in aggregated.columns]
    return aggregated.reset_index().rename(columns={"split_index_count": "split_count"})


def select_best_trial_id(
    aggregated_metrics: pd.DataFrame, primary_metric_key: str
) -> str | None:
    """Return the trial id with the highest mean of the primary metric, or None."""
    if aggregated_metrics.empty:
        return None
    column = f"{PRIMARY_METRIC_TO_PER_SPLIT_COLUMN[primary_metric_key]}_mean"
    if column not in aggregated_metrics.columns:
        return None
    return str(aggregated_metrics.loc[aggregated_metrics[column].idxmax(), "trial_id"])


def best_trial_recommendations(
    run_directory: Path, primary_metric_key: str
) -> tuple[str, pd.DataFrame] | None:
    """Return the best trial id and its recommendations parquet under one run directory."""
    per_split_metrics = load_per_split_metrics(run_directory)
    if per_split_metrics.empty:
        return None
    aggregated = average_per_trial(per_split_metrics)
    best_trial_id = select_best_trial_id(aggregated, primary_metric_key)
    if best_trial_id is None:
        return None
    parquet_path = run_directory / best_trial_id / "recommendations.parquet"
    if not parquet_path.exists():
        return None
    return best_trial_id, pd.read_parquet(parquet_path)
