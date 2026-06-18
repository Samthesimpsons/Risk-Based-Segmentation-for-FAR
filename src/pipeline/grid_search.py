"""RF + LightGCN grid sweep via Ray Tune."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune

from src.config.registry import (
    GRID_SPECS,
    PRIMARY_METRIC_TO_KEY,
    GridSpec,
)
from src.config.schemas import EvaluationResult, TemporalSplitData
from src.config.settings import (
    DataPaths,
    ExperimentConfig,
    LightGCNConfig,
    ModelConfig,
    RandomForestConfig,
)
from src.data.loading import load_assets, load_customers
from src.features.technical_indicators import build_indicator_dataframe
from src.models.light_gcn import LightGCNBaseline
from src.models.protocol import Recommender
from src.models.random_forest import RandomForestBaseline
from src.pipeline.preprocessing import (
    load_evaluation_splits,
    load_preprocessed_close_prices,
)
from src.utils.metrics import (
    build_price_lookup,
    compute_balance,
    compute_monthly_return,
    evaluate_model_on_split,
)
from src.utils.profile_coherence import (
    build_asset_risk_classes,
    build_customer_profile_lookup,
)


@dataclass(frozen=True)
class EvaluationContext:
    """Inputs every trial needs to train and score on the evaluation splits."""

    splits: list[TemporalSplitData]
    close_prices: pd.DataFrame
    indicator_dataframe: pd.DataFrame | None
    customer_profiles: dict[str, Any]
    asset_risk_classes: dict[str, int]
    top_k: int
    device: str
    project_root: Path
    run_timestamp: str


@dataclass(frozen=True)
class TrialEvaluation:
    """Per-split metrics and the recommendations generated for one trial."""

    per_split_results: list[EvaluationResult]
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]]


def _set_random_seeds(seed: int) -> None:
    """Seed numpy, torch, and Python's random module for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(preferred_device: str) -> str:
    """Return the requested device, falling back to CPU when CUDA is unavailable."""
    if preferred_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred_device


def _ray_runtime_env(project_root: Path) -> dict[str, object]:
    """Ship only the source tree to Ray workers."""
    return {"py_modules": [str(project_root / "src")]}


def _build_evaluation_context(
    splits_directory: Path,
    data_paths: DataPaths,
    needs_indicators: bool,
    splits_limit: int | None,
    top_k: int,
    device: str,
    project_root: Path,
    run_timestamp: str,
) -> EvaluationContext:
    """Load and bundle every input that the trainables need."""
    print("Loading evaluation splits...")
    splits = load_evaluation_splits(splits_directory)
    if splits_limit is not None:
        splits = splits[:splits_limit]
    print(f"  Loaded {len(splits)} evaluation splits")

    print("Loading close prices...")
    close_prices = load_preprocessed_close_prices(splits_directory)

    print("Loading customer profiles and asset risk classes...")
    customer_information = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    customer_profiles = build_customer_profile_lookup(customer_information)
    asset_information = load_assets(
        data_paths.data_directory / data_paths.asset_information_file
    )
    asset_risk_classes = build_asset_risk_classes(asset_information, close_prices)

    indicator_dataframe: pd.DataFrame | None = None
    if needs_indicators:
        print("Building indicator DataFrame for Random Forest...")
        indicator_dataframe = build_indicator_dataframe(close_prices)
        print(f"  Built indicators: {len(indicator_dataframe)} rows")

    return EvaluationContext(
        splits=splits,
        close_prices=close_prices,
        indicator_dataframe=indicator_dataframe,
        customer_profiles=customer_profiles,
        asset_risk_classes=asset_risk_classes,
        top_k=top_k,
        device=device,
        project_root=project_root,
        run_timestamp=run_timestamp,
    )


def _generate_recommendations(
    model: Recommender, split: TemporalSplitData, k: int
) -> dict[str, list[str]]:
    """Top-k recommendations for every eligible customer in the split."""
    return {
        customer_id: model.recommend_for_user(
            customer_id,
            split.training_interactions.get(customer_id, set()),
            k,
        )
        for customer_id in split.eligible_customer_ids
    }


def _instantiate_baseline_model(
    spec: GridSpec, config: ModelConfig, context: EvaluationContext
) -> Recommender:
    """Map a spec and config to a concrete RF or LightGCN recommender instance."""
    if spec.model_name == "random_forest":
        assert isinstance(config, RandomForestConfig)
        assert context.indicator_dataframe is not None
        return RandomForestBaseline(
            random_forest_config=config,
            indicator_dataframe=context.indicator_dataframe,
        )
    if spec.model_name == "light_gcn":
        assert isinstance(config, LightGCNConfig)
        return LightGCNBaseline(config=config)
    raise ValueError(f"Unknown baseline model: {spec.model_name}")


def _evaluate_over_splits(
    model: Recommender, context: EvaluationContext
) -> TrialEvaluation:
    """Train, evaluate, and retain per-split recommendations for one trial."""
    results: list[EvaluationResult] = []
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]] = []
    for split in context.splits:
        model.train_on_split(split, device=context.device)
        recommendations = _generate_recommendations(model, split, context.top_k)
        result = evaluate_model_on_split(
            recommendations,
            split,
            context.close_prices,
            context.customer_profiles,
            context.asset_risk_classes,
            context.top_k,
        )
        results.append(result.model_copy(update={"model_name": model.name}))
        recommendations_per_split.append((split, recommendations))
    return TrialEvaluation(
        per_split_results=results,
        recommendations_per_split=recommendations_per_split,
    )


def _summarise(results: list[EvaluationResult]) -> dict[str, float]:
    """Reduce per-split results to scalar averages."""
    if not results:
        return {
            "average_ndcg": 0.0,
            "average_roi": 0.0,
            "average_profile_coherence": 0.0,
            "average_profile_coherence_lift": 0.0,
            "average_balance": 0.0,
        }
    number_of_splits = len(results)
    average_ndcg = sum(r.ndcg_at_k for r in results) / number_of_splits
    average_roi = sum(r.roi_at_k for r in results) / number_of_splits
    average_profile_coherence = (
        sum(r.profile_coherence_at_k for r in results) / number_of_splits
    )
    return {
        "average_ndcg": average_ndcg,
        "average_roi": average_roi,
        "average_profile_coherence": average_profile_coherence,
        "average_profile_coherence_lift": sum(
            r.profile_coherence_lift_at_k for r in results
        )
        / number_of_splits,
        "average_balance": compute_balance(
            average_roi, average_ndcg, average_profile_coherence
        ),
    }


def _save_per_trial_metrics_csv(
    hyperparameters: dict[str, Any],
    trial_id: str,
    evaluation: TrialEvaluation,
    trial_directory: Path,
) -> None:
    """Save per-split scalar metrics for one trial to its own directory."""
    trial_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "trial_id": trial_id,
            **hyperparameters,
            "split_index": result.split_index,
            "time_point": result.time_point.isoformat(),
            "ndcg_at_k": result.ndcg_at_k,
            "roi_at_k": result.roi_at_k,
            "profile_coherence_at_k": result.profile_coherence_at_k,
            "profile_coherence_lift_at_k": result.profile_coherence_lift_at_k,
            "balance": compute_balance(
                result.roi_at_k,
                result.ndcg_at_k,
                result.profile_coherence_at_k,
            ),
        }
        for result in evaluation.per_split_results
    ]
    pd.DataFrame(rows).to_csv(trial_directory / "per_split_metrics.csv", index=False)


def _save_recommendations_parquet(
    hyperparameters: dict[str, Any],
    trial_id: str,
    evaluation: TrialEvaluation,
    close_prices: pd.DataFrame,
    trial_directory: Path,
) -> None:
    """Save flat per-recommendation rows for one trial as parquet."""
    trial_directory.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[Any]] = {"trial_id": []}
    for key in hyperparameters:
        columns[key] = []
    for column_name in (
        "split_index",
        "time_point",
        "test_end",
        "customer_id",
        "rank",
        "asset_id",
        "monthly_return",
        "is_relevant",
    ):
        columns[column_name] = []

    for split, recommendations in evaluation.recommendations_per_split:
        price_lookup = build_price_lookup(
            close_prices,
            split.time_point,
            split.test_end,
            split.eligible_asset_ids,
        )
        days_in_period = (split.test_end - split.time_point).days
        eligible_assets = set(split.eligible_asset_ids)
        time_point_iso = split.time_point.isoformat()
        test_end_iso = split.test_end.isoformat()

        for customer_id in split.eligible_customer_ids:
            relevant_assets = (
                split.test_interactions.get(customer_id, set()) & eligible_assets
            )
            customer_recommendations = recommendations.get(customer_id, [])
            for rank_index, asset_id in enumerate(customer_recommendations, start=1):
                start_price, end_price = price_lookup.get(asset_id, (0.0, 0.0))
                monthly_return = compute_monthly_return(
                    start_price, end_price, days_in_period
                )
                columns["trial_id"].append(trial_id)
                for key, value in hyperparameters.items():
                    columns[key].append(value)
                columns["split_index"].append(split.split_index)
                columns["time_point"].append(time_point_iso)
                columns["test_end"].append(test_end_iso)
                columns["customer_id"].append(customer_id)
                columns["rank"].append(rank_index)
                columns["asset_id"].append(asset_id)
                columns["monthly_return"].append(monthly_return)
                columns["is_relevant"].append(asset_id in relevant_assets)

    pd.DataFrame(columns).to_parquet(
        trial_directory / "recommendations.parquet", index=False
    )


def _build_grid_search_space(spec: GridSpec) -> dict[str, Any]:
    """Wrap each grid axis in tune.grid_search."""
    return {name: tune.grid_search(values) for name, values in spec.grid.items()}


def _trial_resources(
    spec: GridSpec, use_gpu: bool, max_concurrent_trials: int
) -> dict[str, float]:
    """Allocate fractional GPU when requested, CPU only otherwise."""
    if spec.use_gpu_per_trial and use_gpu:
        return {"gpu": 1.0 / max(1, max_concurrent_trials), "cpu": 1.0}
    return {"cpu": 1.0}


def _make_baseline_trainable(
    spec: GridSpec,
    context_reference: ray.ObjectRef,
    evaluation_directory: Path,
) -> Callable[[dict[str, Any]], None]:
    """Return a Ray-tunable function for the RF / LightGCN Cartesian-grid trials."""

    def trainable(hyperparameters: dict[str, Any]) -> None:
        context: EvaluationContext = ray.get(context_reference)
        config = spec.config_class(**hyperparameters)
        _set_random_seeds(0)
        model = _instantiate_baseline_model(spec, config, context)
        evaluation = _evaluate_over_splits(model, context)
        averages = _summarise(evaluation.per_split_results)

        trial_id = tune.get_context().get_trial_id()
        trial_directory = evaluation_directory / trial_id
        _save_per_trial_metrics_csv(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            trial_directory=trial_directory,
        )
        _save_recommendations_parquet(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            close_prices=context.close_prices,
            trial_directory=trial_directory,
        )

        tune.report(averages)

    return trainable


def _run_baseline_grid_for_spec(
    spec: GridSpec,
    context: EvaluationContext,
    use_gpu: bool,
    results_directory: Path,
) -> tuple[ModelConfig, tune.ResultGrid]:
    """Run one model's grid via Ray Tune and return the best config and the result grid."""
    evaluation_directory = (
        results_directory / "evaluation" / spec.model_name / context.run_timestamp
    ).resolve()
    context_reference = ray.put(context)
    trainable = _make_baseline_trainable(spec, context_reference, evaluation_directory)

    tuner = tune.Tuner(
        tune.with_resources(
            trainable, _trial_resources(spec, use_gpu, spec.max_concurrent_trials)
        ),
        param_space=_build_grid_search_space(spec),
        tune_config=tune.TuneConfig(
            metric=PRIMARY_METRIC_TO_KEY[spec.primary_metric],
            mode="max",
            num_samples=1,
            max_concurrent_trials=spec.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name=f"{spec.model_name}_grid_search",
            verbose=1,
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(
        metric=PRIMARY_METRIC_TO_KEY[spec.primary_metric], mode="max"
    )
    assert best_result.config is not None and best_result.metrics is not None
    best_config = spec.config_class(**dict(best_result.config))
    return best_config, result_grid


def _save_baseline_trial_summary(
    spec: GridSpec,
    result_grid: tune.ResultGrid,
    results_directory: Path,
    timestamp: str,
) -> Path:
    """Save one row per Cartesian-grid trial with hyperparameters and averaged metrics."""
    rows: list[dict[str, Any]] = []
    for result in result_grid:
        if result.metrics is None or result.config is None:
            continue
        row: dict[str, Any] = dict(result.config)
        row["average_ndcg"] = result.metrics.get("average_ndcg")
        row["average_roi"] = result.metrics.get("average_roi")
        row["average_profile_coherence"] = result.metrics.get(
            "average_profile_coherence"
        )
        row["average_profile_coherence_lift"] = result.metrics.get(
            "average_profile_coherence_lift"
        )
        row["average_balance"] = result.metrics.get("average_balance")
        rows.append(row)

    output_directory = results_directory / "tuning" / spec.model_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _save_best_configs(best_configs: dict[str, ModelConfig], output_path: Path) -> None:
    """Persist best-by-primary-metric configs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: config.model_dump() for name, config in best_configs.items()}
    output_path.write_text(json.dumps(payload, indent=2))


def _grid_size(spec: GridSpec) -> int:
    """Return the cardinality of the Cartesian product of the grid."""
    size = 1
    for values in spec.grid.values():
        size *= len(values)
    return size


def run_baseline_grid_search(
    splits_directory: Path,
    results_directory: Path,
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    selected_models: list[str] | None = None,
    splits_limit: int | None = None,
) -> tuple[dict[str, ModelConfig], str]:
    """Run RF + LightGCN grid sweeps and return best configs plus the run timestamp."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()
    requested = selected_models or list(GRID_SPECS.keys())
    unknown = set(requested) - set(GRID_SPECS.keys())
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    project_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    needs_indicators = any(GRID_SPECS[name].needs_indicators for name in requested)
    device = _resolve_device(experiment_config.device)

    context = _build_evaluation_context(
        splits_directory=splits_directory,
        data_paths=data_paths,
        needs_indicators=needs_indicators,
        splits_limit=splits_limit,
        top_k=experiment_config.top_k,
        device=device,
        project_root=project_root,
        run_timestamp=run_timestamp,
    )

    os.environ["RAY_RUNTIME_ENV_LOCAL_DEV_MODE"] = "1"
    os.environ["RAY_ENABLE_LOG_MONITOR"] = "0"
    ray.init(
        log_to_driver=False,
        logging_level="error",
        include_dashboard=False,
        _enable_object_reconstruction=False,
        object_store_memory=1_073_741_824,
        runtime_env=_ray_runtime_env(project_root),
    )

    best_configs: dict[str, ModelConfig] = {}
    use_gpu = device != "cpu"
    try:
        for model_name in requested:
            spec = GRID_SPECS[model_name]
            print(
                f"\n=== Running {spec.model_name} grid ({_grid_size(spec)} trials) ==="
            )
            best_config, result_grid = _run_baseline_grid_for_spec(
                spec,
                context,
                use_gpu,
                results_directory,
            )
            summary_path = _save_baseline_trial_summary(
                spec, result_grid, results_directory, run_timestamp
            )
            best_configs[model_name] = best_config
            print(f"  Trial summary saved to {summary_path}")
    finally:
        ray.shutdown()

    best_config_path = (
        results_directory.parent
        / "configs"
        / run_timestamp
        / "best_hyperparameters.json"
    )
    _save_best_configs(best_configs, best_config_path)
    print(f"\nBest configs saved to {best_config_path}")
    print("\nBest configs by primary metric:")
    for name, config in best_configs.items():
        print(f"  {name}: {config}")
    return best_configs, run_timestamp
