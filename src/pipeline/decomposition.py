"""2-model (RF + LightGCN) best-trial headline table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.run_artefacts import (
    average_per_trial,
    discover_run_directories,
    load_per_split_metrics,
    select_best_trial_id,
)
from src.config.registry import (
    DISPLAY_MODEL_NAMES,
    GRID_SPECS,
    PRIMARY_METRIC_TO_KEY,
)
from src.config.settings import DataPaths
from src.utils.metrics import compute_balance

DEFAULT_EVALUATION_DIRECTORY = Path("outputs/results/evaluation")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/baseline_decomposition")


def _build_main_results_row(
    model_name: str,
    aggregated: pd.DataFrame,
    best_trial_id: str,
    primary_metric_key: str,
) -> dict[str, Any]:
    """Pull the best trial's row out of the aggregated metrics frame."""
    best = aggregated.loc[aggregated["trial_id"] == best_trial_id].iloc[0]
    row: dict[str, Any] = {
        "model": model_name,
        "display_name": DISPLAY_MODEL_NAMES.get(model_name, model_name),
        "best_trial_id": best_trial_id,
        "primary_metric": primary_metric_key,
        "split_count": int(best["split_count"]),
        "ndcg_at_k_mean": float(best["ndcg_at_k_mean"]),
        "ndcg_at_k_std": float(best["ndcg_at_k_std"]),
        "roi_at_k_mean": float(best["roi_at_k_mean"]),
        "roi_at_k_std": float(best["roi_at_k_std"]),
        "profile_coherence_at_k_mean": float(best["profile_coherence_at_k_mean"]),
        "profile_coherence_at_k_std": float(best["profile_coherence_at_k_std"]),
        "balance": compute_balance(
            float(best["roi_at_k_mean"]),
            float(best["ndcg_at_k_mean"]),
            float(best["profile_coherence_at_k_mean"]),
        ),
    }
    if "profile_coherence_lift_at_k_mean" in best.index:
        row["profile_coherence_lift_at_k_mean"] = float(
            best["profile_coherence_lift_at_k_mean"]
        )
        row["profile_coherence_lift_at_k_std"] = float(
            best["profile_coherence_lift_at_k_std"]
        )
    return row


def _print_main_results_table(main_results: pd.DataFrame, top_k: int) -> None:
    """Pretty-print the headline results table to stdout."""
    if main_results.empty:
        print("(no main results to display)")
        return
    has_lift = "profile_coherence_lift_at_k_mean" in main_results.columns
    header = (
        f"{'Model':<28} {'nDCG@' + str(top_k):<14} {'ROI@' + str(top_k):<14}"
        f" {'PC@' + str(top_k):<10} {'Balance':<10}"
    )
    if has_lift:
        header += f" {'PC-lift@' + str(top_k):<10}"
    width = len(header)
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'-' * width}")
    for _, row in main_results.iterrows():
        line = (
            f"{row['display_name']:<28}"
            f" {row['ndcg_at_k_mean']:<14.4f}"
            f" {row['roi_at_k_mean']:<14.6f}"
            f" {row['profile_coherence_at_k_mean']:<10.4f}"
            f" {row['balance']:<10.4f}"
        )
        if has_lift:
            line += f" {row['profile_coherence_lift_at_k_mean']:<10.4f}"
        print(line)
    print(f"{'=' * width}")


def run_decomposition(
    evaluation_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build the best-trial main_results.csv table consumed by the findings renderers."""
    del data_paths
    print(f"Discovering evaluation runs in {evaluation_directory} ...")
    runs = discover_run_directories(evaluation_directory, run_timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No evaluation runs found under {evaluation_directory}. "
            "Run `uv run poe tune` first."
        )

    main_rows: list[dict[str, Any]] = []
    chosen_run_timestamp = run_timestamp

    for model_name, run_directory in runs.items():
        if model_name not in GRID_SPECS:
            print(f"  Skipping unknown model directory '{model_name}'")
            continue

        if chosen_run_timestamp is None:
            chosen_run_timestamp = run_directory.name

        primary_metric = GRID_SPECS[model_name].primary_metric
        primary_metric_key = PRIMARY_METRIC_TO_KEY[primary_metric]
        print(
            f"\n[{model_name}] reading {run_directory}"
            f" (best trial by {primary_metric_key})"
        )

        per_split_metrics = load_per_split_metrics(run_directory)
        if per_split_metrics.empty:
            print("  No per_split_metrics.csv found; skipping.")
            continue
        aggregated = average_per_trial(per_split_metrics)
        best_trial_id = select_best_trial_id(aggregated, primary_metric_key)
        if best_trial_id is None:
            print("  Could not determine best trial; skipping.")
            continue
        print(f"  Best trial: {best_trial_id}")

        main_rows.append(
            _build_main_results_row(
                model_name, aggregated, best_trial_id, primary_metric_key
            )
        )

    main_results = pd.DataFrame(main_rows)

    output_directory = output_root / (chosen_run_timestamp or "latest")
    output_directory.mkdir(parents=True, exist_ok=True)
    main_results.to_csv(output_directory / "main_results.csv", index=False)

    _print_main_results_table(main_results, top_k=top_k)
    print(f"\nDecomposition outputs saved to {output_directory}")
    return {
        "run_timestamp": chosen_run_timestamp,
        "evaluation_directory": str(evaluation_directory),
        "output_directory": str(output_directory),
        "models": [row["model"] for row in main_rows],
        "main_results": main_rows,
    }
