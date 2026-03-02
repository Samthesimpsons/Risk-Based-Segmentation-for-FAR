"""Experiment runner for training, evaluating, and comparing recommendation models.

The runner replicates the FAR-Trans benchmark: it iterates over the 61
preprocessed evaluation splits, trains each requested model from scratch on
each split, and aggregates nDCG@10 / ROI@10 across splits.

Adding a new model is one entry in `src/models/protocol.MODEL_REGISTRY` plus
the model implementation. The runner then picks it up automatically as long
as the user supplies its config in the tuning JSON or selects it via
`--models`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.config.schemas import EvaluationResult, SequenceData, TemporalSplitData
from src.config.settings import ExperimentConfig, ModelConfig
from src.evaluation.metrics import evaluate_model_on_split
from src.features.technical_indicators import build_indicator_dataframe
from src.models.protocol import (
    ALL_MODEL_NAMES,
    MODEL_REGISTRY,
    Recommender,
    build_recommender,
)
from src.models.train import set_random_seeds

from src.pipeline.preprocessing import (
    load_evaluation_sequences,
    load_evaluation_splits,
    load_preprocessed_close_prices,
)


def generate_recommendations(
    model: Recommender,
    split: TemporalSplitData,
    k: int = 10,
) -> dict[str, list[str]]:
    """Generate top-k recommendations for all eligible users in a split."""
    recommendations: dict[str, list[str]] = {}

    for customer_id in split.eligible_customer_ids:
        excluded_assets = split.training_interactions.get(customer_id, set())
        recommendations[customer_id] = model.recommend_for_user(
            customer_id, excluded_assets, k
        )

    return recommendations


def run_experiment(
    model: Recommender,
    splits: list[TemporalSplitData],
    close_prices: pd.DataFrame,
    k: int = 10,
    device: str = "cpu",
    sequences: list[SequenceData] | None = None,
) -> list[EvaluationResult]:
    """Run a model across all temporal splits and collect evaluation results."""
    results: list[EvaluationResult] = []
    total_splits = len(splits)

    for index, split in enumerate(splits):
        print(
            f"[{model.name}] Split {split.split_index + 1}/{total_splits}"
            f" at {split.time_point}"
        )

        train_kwargs: dict[str, object] = {"device": device}
        if sequences is not None:
            train_kwargs["user_sequences"] = sequences[index].user_sequences

        model.train_on_split(split, **train_kwargs)
        recommendations = generate_recommendations(model, split, k)
        result = evaluate_model_on_split(recommendations, split, close_prices, k)

        result_with_name = result.model_copy(update={"model_name": model.name})

        print(
            f"  nDCG@{k}: {result_with_name.ndcg_at_k:.4f}"
            f"  ROI@{k}: {result_with_name.roi_at_k:.6f}"
        )

        results.append(result_with_name)

    return results


def print_summary(results: list[EvaluationResult]) -> None:
    """Print aggregate results across all splits for a model."""
    if not results:
        return

    model_name = results[0].model_name
    average_ndcg = sum(r.ndcg_at_k for r in results) / len(results)
    average_roi = sum(r.roi_at_k for r in results) / len(results)

    print(f"\n{model_name} Summary:")
    print(f"  Average nDCG@k: {average_ndcg:.4f}")
    print(f"  Average ROI@k:  {average_roi:.6f}")


def save_evaluation_results(
    results: list[EvaluationResult],
    results_directory: Path,
    stage: str = "evaluation",
) -> Path:
    """Save per-split evaluation results for a model as a timestamped CSV."""
    if not results:
        return results_directory

    model_name = results[0].model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_directory = results_directory / stage / model_name
    model_directory.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "split_index": r.split_index,
            "time_point": r.time_point.isoformat(),
            "ndcg_at_k": r.ndcg_at_k,
            "roi_at_k": r.roi_at_k,
        }
        for r in results
    ]

    average_ndcg = sum(r.ndcg_at_k for r in results) / len(results)
    average_roi = sum(r.roi_at_k for r in results) / len(results)
    rows.append(
        {
            "split_index": -1,
            "time_point": "average",
            "ndcg_at_k": average_ndcg,
            "roi_at_k": average_roi,
        }
    )

    output_path = model_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    return output_path


def _resolve_device(preferred_device: str) -> str:
    """Determine the compute device, falling back to CPU when CUDA is unavailable."""
    if preferred_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return preferred_device


def _resolve_model_set(
    selected_models: list[str] | None,
    provided_configs: Mapping[str, ModelConfig],
) -> list[str]:
    """Determine which models to run, validated against the registry."""
    if selected_models:
        unknown = set(selected_models) - set(MODEL_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}")
        return list(selected_models)

    if provided_configs:
        return list(provided_configs.keys())

    raise ValueError(
        "No models selected. Pass --models or provide configs via --config."
    )


def run_all_experiments(
    splits_directory: Path,
    provided_configs: Mapping[str, ModelConfig] | None = None,
    experiment_config: ExperimentConfig | None = None,
    results_directory: Path | None = None,
    selected_models: list[str] | None = None,
) -> None:
    """Load preprocessed splits, train and evaluate the requested models.

    Each model is constructed via `build_recommender`, which uses paper
    defaults when no config is supplied for that model. This makes the
    benchmark replication path one command:
    `poe run --models random_forest light_gcn` reproduces the FAR-Trans
    Random Forest and LightGCN entries in Table 2 with their published
    hyperparameters.
    """
    provided_configs = dict(provided_configs or {})
    models_to_run = _resolve_model_set(selected_models, provided_configs)

    experiment_config = experiment_config or ExperimentConfig()
    set_random_seeds(experiment_config.seed)

    print("Loading evaluation splits...")
    splits = load_evaluation_splits(splits_directory)
    print(f"  Loaded {len(splits)} evaluation splits")

    print("Loading close prices...")
    close_prices = load_preprocessed_close_prices(splits_directory)

    needs_indicators = any(
        MODEL_REGISTRY[name].needs_indicators for name in models_to_run
    )
    needs_sequences = any(
        MODEL_REGISTRY[name].needs_sequences for name in models_to_run
    )

    indicator_dataframe: pd.DataFrame | None = None
    if needs_indicators:
        print("Building indicator DataFrame...")
        indicator_dataframe = build_indicator_dataframe(close_prices)
        print(f"  Built indicators: {len(indicator_dataframe)} rows")

    sequences: list[SequenceData] = []
    if needs_sequences:
        print("Loading evaluation sequences...")
        sequences = load_evaluation_sequences(splits_directory)
        print(f"  Loaded {len(sequences)} sequence files")

    unique_customers: set[str] = set()
    unique_assets: set[str] = set()
    for split in splits:
        unique_customers.update(split.eligible_customer_ids)
        unique_assets.update(split.eligible_asset_ids)

    print(f"  Unique eligible customers: {len(unique_customers)}")
    print(f"  Unique eligible assets: {len(unique_assets)}")

    device = _resolve_device(experiment_config.device)

    all_results: dict[str, list[EvaluationResult]] = {}

    for model_name in models_to_run:
        entry = MODEL_REGISTRY[model_name]
        config = provided_configs.get(model_name)
        recommender = build_recommender(
            model_name,
            config,
            indicator_dataframe=indicator_dataframe,
            close_prices=close_prices,
        )

        per_model_sequences = sequences if entry.needs_sequences else None
        results = run_experiment(
            recommender,
            splits,
            close_prices,
            experiment_config.top_k,
            device,
            per_model_sequences,
        )
        print_summary(results)
        if results_directory is not None:
            save_evaluation_results(results, results_directory, stage="evaluation")
        all_results[recommender.name] = results

    _print_comparison_table(all_results, experiment_config.top_k)


def _print_comparison_table(
    all_results: dict[str, list[EvaluationResult]],
    k: int,
) -> None:
    """Print a comparison table of average metrics across all models."""
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'nDCG@' + str(k):<15} {'ROI@' + str(k):<15}")
    print(f"{'-' * 60}")

    for model_name, results in all_results.items():
        if not results:
            continue
        average_ndcg = sum(r.ndcg_at_k for r in results) / len(results)
        average_roi = sum(r.roi_at_k for r in results) / len(results)
        print(f"{model_name:<25} {average_ndcg:<15.4f} {average_roi:<15.6f}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    from src.pipeline.tuning import load_configs_from_json

    parser = argparse.ArgumentParser(description="Run FAR-Trans baseline experiments")
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing preprocessed splits (default: data/splits)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to best_hyperparameters.json from tuning (optional; "
        "omit to use paper defaults from MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=None,
        help="Models to run (default: every model in --config). Options: "
        + ", ".join(ALL_MODEL_NAMES),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/results",
        help="Directory to save evaluation result CSVs (default: outputs/results)",
    )
    args = parser.parse_args()

    provided_configs: dict[str, ModelConfig] = {}
    if args.config is not None:
        print(f"Loading tuned configs from {args.config}")
        provided_configs = load_configs_from_json(Path(args.config))

    run_all_experiments(
        splits_directory=Path(args.splits_dir),
        provided_configs=provided_configs,
        experiment_config=ExperimentConfig(device=args.device),
        results_directory=Path(args.results_dir),
        selected_models=args.models,
    )
