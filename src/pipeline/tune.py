"""End-to-end thesis pipeline: grid search, PC-LGCN sweep, decomposition, panel regressions."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import statsmodels.formula.api as smf
import torch
from ray import tune

from src.config.schemas import EvaluationResult, TemporalSplitData
from src.config.settings import (
    DataPaths,
    ExperimentConfig,
    LightGCNConfig,
    ModelConfig,
    ProfileCoherentLightGCNConfig,
    RandomForestConfig,
)
from src.data.loading import load_assets, load_close_prices, load_customers
from src.features.technical_indicators import build_indicator_dataframe
from src.models.light_gcn import LightGCNBaseline
from src.models.pc_light_gcn import ProfileCoherentLightGCNBaseline
from src.models.protocol import Recommender
from src.models.random_forest import RandomForestBaseline
from src.pipeline.preprocessing import (
    load_evaluation_splits,
    load_preprocessed_close_prices,
)
from src.utils.metrics import (
    build_price_lookup,
    evaluate_model_on_split,
)
from src.utils.profile_coherence import (
    CALENDAR_DAYS_PER_MONTH,
    RISK_BAND_NAMES,
    build_asset_risk_classes,
    build_customer_profile_lookup,
)

PrimaryMetric = Literal["ndcg", "roi", "recall", "profile_coherence"]

_PRIMARY_METRIC_TO_KEY: dict[PrimaryMetric, str] = {
    "ndcg": "average_ndcg",
    "roi": "average_roi",
    "recall": "average_recall",
    "profile_coherence": "average_profile_coherence",
}


RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "number_of_estimators": [20, 30, 40, 50],
    "max_depth": [None, 15, 25],
    "random_state": [42],
    "prediction_horizon_months": [6],
}


LIGHT_GCN_GRID: dict[str, list[Any]] = {
    "embedding_dimension": [64, 128],
    "number_of_layers": [2, 3],
    "learning_rate": [1e-2, 1e-3],
    "weight_decay": [1e-5],
    "keep_probability": [0.6],
    "number_of_epochs": [50],
    "batch_size": [1024],
}


@dataclass(frozen=True)
class GridSpec:
    """Static description of one model's tuning sweep."""

    model_name: str
    config_class: type[ModelConfig]
    grid: dict[str, list[Any]]
    primary_metric: PrimaryMetric
    needs_indicators: bool
    use_gpu_per_trial: bool
    max_concurrent_trials: int


GRID_SPECS: dict[str, GridSpec] = {
    "random_forest": GridSpec(
        model_name="random_forest",
        config_class=RandomForestConfig,
        grid=RANDOM_FOREST_GRID,
        primary_metric="roi",
        needs_indicators=True,
        use_gpu_per_trial=False,
        max_concurrent_trials=4,
    ),
    "light_gcn": GridSpec(
        model_name="light_gcn",
        config_class=LightGCNConfig,
        grid=LIGHT_GCN_GRID,
        primary_metric="ndcg",
        needs_indicators=False,
        use_gpu_per_trial=True,
        max_concurrent_trials=4,
    ),
}


PC_LGCN_MODEL_NAME = "profile_coherent_light_gcn"


@dataclass(frozen=True)
class ProfileCoherentSweepConfig:
    """Static knobs that drive the PC-LGCN profile-coherence sweep."""

    primary_metric: PrimaryMetric
    default_lambda: float
    lambda_sensitivity_values: tuple[float, ...]
    backbone_keys: tuple[str, ...]


_PC_LGCN_SWEEP = ProfileCoherentSweepConfig(
    primary_metric="ndcg",
    default_lambda=0.5,
    lambda_sensitivity_values=(0.1, 1.0, 2.0),
    backbone_keys=(
        "embedding_dimension",
        "number_of_layers",
        "learning_rate",
        "weight_decay",
        "keep_probability",
        "number_of_epochs",
        "batch_size",
    ),
)


PROFILE_COHERENT_GRID_SPEC = GridSpec(
    model_name=PC_LGCN_MODEL_NAME,
    config_class=ProfileCoherentLightGCNConfig,
    grid={},
    primary_metric=_PC_LGCN_SWEEP.primary_metric,
    needs_indicators=False,
    use_gpu_per_trial=True,
    max_concurrent_trials=4,
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


def _set_random_seeds(seed: int) -> None:
    """Seed numpy, torch, and Python random for reproducibility."""
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
    """Top-k recommendations for every eligible customer in `split`."""
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
    """Map a spec + config to a concrete RF or LightGCN recommender instance."""
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


@dataclass(frozen=True)
class TrialEvaluation:
    """Outputs of a single trial: per-split metrics + the recommendations themselves."""

    per_split_results: list[EvaluationResult]
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]]


def _evaluate_over_splits(
    model: Recommender, context: EvaluationContext
) -> TrialEvaluation:
    """Train, evaluate, AND retain per-split recommendations for one trial."""
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
            "average_recall": 0.0,
            "average_profile_coherence": 0.0,
            "average_profile_coherence_lift": 0.0,
        }
    n = len(results)
    return {
        "average_ndcg": sum(r.ndcg_at_k for r in results) / n,
        "average_roi": sum(r.roi_at_k for r in results) / n,
        "average_recall": sum(r.recall_at_k for r in results) / n,
        "average_profile_coherence": sum(r.profile_coherence_at_k for r in results) / n,
        "average_profile_coherence_lift": sum(
            r.profile_coherence_lift_at_k for r in results
        )
        / n,
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
            "recall_at_k": result.recall_at_k,
            "profile_coherence_at_k": result.profile_coherence_at_k,
            "profile_coherence_lift_at_k": result.profile_coherence_lift_at_k,
        }
        for result in evaluation.per_split_results
    ]
    pd.DataFrame(rows).to_csv(trial_directory / "per_split_metrics.csv", index=False)


def _compute_monthly_return(
    start_price: float, end_price: float, days_in_period: int
) -> float:
    """Compute the geometric monthly return for the ROI@k convention."""
    if start_price <= 0.0 or days_in_period <= 0:
        return 0.0
    total_return = (end_price - start_price) / start_price
    return pow(1.0 + total_return, CALENDAR_DAYS_PER_MONTH / days_in_period) - 1.0


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
            customer_recs = recommendations.get(customer_id, [])
            for rank_index, asset_id in enumerate(customer_recs, start=1):
                start_price, end_price = price_lookup.get(asset_id, (0.0, 0.0))
                monthly_return = _compute_monthly_return(
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
    """Wrap each grid axis in `tune.grid_search`."""
    return {name: tune.grid_search(values) for name, values in spec.grid.items()}


def _trial_resources(
    spec: GridSpec, use_gpu: bool, max_concurrent_trials: int
) -> dict[str, float]:
    """Allocate fractional GPU when requested; CPU only otherwise."""
    if spec.use_gpu_per_trial and use_gpu:
        return {"gpu": 1.0 / max(1, max_concurrent_trials), "cpu": 1.0}
    return {"cpu": 1.0}


def _make_baseline_trainable(
    spec: GridSpec, context_ref: ray.ObjectRef, evaluation_dir: Path
) -> Callable[[dict[str, Any]], None]:
    """Return a Ray-tunable function for the RF / LightGCN Cartesian-grid trials."""

    def trainable(hyperparameters: dict[str, Any]) -> None:
        context: EvaluationContext = ray.get(context_ref)
        config = spec.config_class(**hyperparameters)
        _set_random_seeds(0)
        model = _instantiate_baseline_model(spec, config, context)
        evaluation = _evaluate_over_splits(model, context)
        averages = _summarise(evaluation.per_split_results)

        trial_id = tune.get_context().get_trial_id()
        trial_directory = evaluation_dir / trial_id
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
    max_concurrent_trials_override: int | None,
) -> tuple[ModelConfig, tune.ResultGrid]:
    """Run one model's grid via Ray Tune and return the best config + the result grid."""
    evaluation_dir = (
        results_directory / "evaluation" / spec.model_name / context.run_timestamp
    ).resolve()
    context_ref = ray.put(context)
    trainable = _make_baseline_trainable(spec, context_ref, evaluation_dir)
    effective_concurrency = (
        max_concurrent_trials_override
        if max_concurrent_trials_override is not None
        else spec.max_concurrent_trials
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable, _trial_resources(spec, use_gpu, effective_concurrency)
        ),
        param_space=_build_grid_search_space(spec),
        tune_config=tune.TuneConfig(
            metric=_PRIMARY_METRIC_TO_KEY[spec.primary_metric],
            mode="max",
            num_samples=1,
            max_concurrent_trials=effective_concurrency,
        ),
        run_config=tune.RunConfig(
            name=f"{spec.model_name}_grid_search",
            verbose=1,
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(
        metric=_PRIMARY_METRIC_TO_KEY[spec.primary_metric], mode="max"
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
        row["average_recall"] = result.metrics.get("average_recall")
        row["average_profile_coherence"] = result.metrics.get(
            "average_profile_coherence"
        )
        row["average_profile_coherence_lift"] = result.metrics.get(
            "average_profile_coherence_lift"
        )
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
    """Return the cardinality of the cartesian product of the grid."""
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
    max_concurrent_trials_override: int | None = None,
) -> tuple[dict[str, ModelConfig], str]:
    """Run RF + LightGCN grid sweeps; return best configs and the run timestamp."""
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
                max_concurrent_trials_override,
            )
            summary_path = _save_baseline_trial_summary(
                spec, result_grid, results_directory, run_timestamp
            )
            best_configs[model_name] = best_config
            print(f"  Trial summary saved to {summary_path}")
    finally:
        ray.shutdown()

    best_config_path = (
        Path("outputs/configs") / run_timestamp / "best_hyperparameters.json"
    )
    _save_best_configs(best_configs, best_config_path)
    print(f"\nBest configs saved to {best_config_path}")
    print("\nBest configs by primary metric:")
    for name, config in best_configs.items():
        print(f"  {name}: {config}")
    return best_configs, run_timestamp


def _resolve_baseline_best_config_path(
    explicit_path: Path | None,
    configs_root: Path = Path("outputs/configs"),
) -> Path:
    """Pick the latest baseline best-config JSON if no explicit path was given."""
    if explicit_path is not None:
        return explicit_path
    if not configs_root.exists():
        raise FileNotFoundError(
            f"No {configs_root} directory; pass an explicit baseline_best_config_path."
        )
    timestamped_dirs = sorted(
        [path for path in configs_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    if not timestamped_dirs:
        raise FileNotFoundError(
            f"No timestamped subdirectories under {configs_root}; "
            "pass an explicit baseline_best_config_path."
        )
    return timestamped_dirs[-1] / "best_hyperparameters.json"


def _load_winning_lightgcn_backbone(best_config_path: Path) -> dict[str, Any]:
    """Read the LightGCN backbone hyperparameters from the baseline best-config JSON."""
    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Baseline best-config JSON not found at {best_config_path}. "
            "Run the baseline grid first or pass baseline_best_config_path."
        )
    payload = json.loads(best_config_path.read_text())
    if "light_gcn" not in payload:
        raise KeyError(
            f"'light_gcn' missing from {best_config_path}. "
            "Re-run baselines so the JSON includes the LightGCN winner."
        )
    backbone = payload["light_gcn"]
    missing = [key for key in _PC_LGCN_SWEEP.backbone_keys if key not in backbone]
    if missing:
        raise KeyError(f"LightGCN backbone is missing keys: {missing}")
    return {key: backbone[key] for key in _PC_LGCN_SWEEP.backbone_keys}


def _build_pc_lgcn_trial_configs(
    backbone: dict[str, Any], include_lambda_sweep: bool
) -> list[dict[str, Any]]:
    """Materialise the 2x2 ablation cells (and optional lambda sensitivity)."""
    cells: list[dict[str, Any]] = []
    for embedding_enabled in (False, True):
        for coherence_enabled in (False, True):
            cells.append(
                {
                    **backbone,
                    "profile_embedding_enabled": embedding_enabled,
                    "profile_coherence_enabled": coherence_enabled,
                    "profile_coherence_lambda": _PC_LGCN_SWEEP.default_lambda,
                }
            )
    if include_lambda_sweep:
        for value in _PC_LGCN_SWEEP.lambda_sensitivity_values:
            cells.append(
                {
                    **backbone,
                    "profile_embedding_enabled": True,
                    "profile_coherence_enabled": True,
                    "profile_coherence_lambda": value,
                }
            )
    return cells


def _make_pc_lgcn_trainable(
    context_ref: ray.ObjectRef, evaluation_dir: Path
) -> Callable[[dict[str, Any]], None]:
    """Return a Ray-tunable function that instantiates PC-LGCN, evaluates, and persists outputs."""

    def trainable(passed: dict[str, Any]) -> None:
        hyperparameters: dict[str, Any] = passed["hyperparams"]
        context: EvaluationContext = ray.get(context_ref)
        config = ProfileCoherentLightGCNConfig(**hyperparameters)
        _set_random_seeds(0)
        model = ProfileCoherentLightGCNBaseline(
            config=config,
            customer_profiles=context.customer_profiles,
            asset_risk_classes=context.asset_risk_classes,
        )
        evaluation = _evaluate_over_splits(model, context)
        averages = _summarise(evaluation.per_split_results)

        trial_id = tune.get_context().get_trial_id()
        trial_directory = evaluation_dir / trial_id
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


def _save_pc_lgcn_trial_summary(
    result_grid: tune.ResultGrid,
    results_directory: Path,
    timestamp: str,
) -> Path:
    """Save one row per PC-LGCN cell with full hyperparameters and averaged metrics."""
    rows: list[dict[str, Any]] = []
    for result in result_grid:
        if result.metrics is None or result.config is None:
            continue
        cell = dict(result.config).get("hyperparams", {})
        row: dict[str, Any] = dict(cell)
        row["average_ndcg"] = result.metrics.get("average_ndcg")
        row["average_roi"] = result.metrics.get("average_roi")
        row["average_recall"] = result.metrics.get("average_recall")
        row["average_profile_coherence"] = result.metrics.get(
            "average_profile_coherence"
        )
        rows.append(row)

    output_directory = results_directory / "tuning" / PC_LGCN_MODEL_NAME
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def run_profile_coherent_grid(
    splits_directory: Path,
    results_directory: Path,
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    baseline_best_config_path: Path | None = None,
    include_lambda_sweep: bool = True,
    splits_limit: int | None = None,
    max_concurrent_trials: int = 4,
) -> tuple[ProfileCoherentLightGCNConfig, str]:
    """Run the PC-LGCN ablation + sensitivity grid; return best cell and run timestamp."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()

    project_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = _resolve_device(experiment_config.device)

    backbone_path = _resolve_baseline_best_config_path(baseline_best_config_path)
    print(f"Loading winning LightGCN backbone from {backbone_path}")
    backbone = _load_winning_lightgcn_backbone(backbone_path)
    print(f"  Backbone: {backbone}")

    trial_configs = _build_pc_lgcn_trial_configs(backbone, include_lambda_sweep)
    sensitivity_count = len(trial_configs) - 4
    print(
        f"Configured {len(trial_configs)} trials"
        f" (4 ablation + {sensitivity_count} lambda sensitivity)"
    )

    context = _build_evaluation_context(
        splits_directory=splits_directory,
        data_paths=data_paths,
        needs_indicators=False,
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

    evaluation_dir = (
        results_directory / "evaluation" / PC_LGCN_MODEL_NAME / run_timestamp
    ).resolve()
    use_gpu = device != "cpu"
    resources: dict[str, float] = (
        {"gpu": 1.0 / max(1, max_concurrent_trials), "cpu": 1.0}
        if use_gpu
        else {"cpu": 1.0}
    )

    try:
        context_ref = ray.put(context)
        trainable = _make_pc_lgcn_trainable(context_ref, evaluation_dir)
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources),
            param_space={"hyperparams": tune.grid_search(trial_configs)},
            tune_config=tune.TuneConfig(
                metric=_PRIMARY_METRIC_TO_KEY[_PC_LGCN_SWEEP.primary_metric],
                mode="max",
                num_samples=1,
                max_concurrent_trials=max_concurrent_trials,
            ),
            run_config=tune.RunConfig(
                name=f"{PC_LGCN_MODEL_NAME}_grid_search",
                verbose=1,
            ),
        )
        result_grid = tuner.fit()
    finally:
        ray.shutdown()

    summary_path = _save_pc_lgcn_trial_summary(
        result_grid, results_directory, run_timestamp
    )
    print(f"  Trial summary saved to {summary_path}")

    best_result = result_grid.get_best_result(
        metric=_PRIMARY_METRIC_TO_KEY[_PC_LGCN_SWEEP.primary_metric], mode="max"
    )
    assert best_result.config is not None
    best_cell = dict(best_result.config)["hyperparams"]
    best_config = ProfileCoherentLightGCNConfig(**best_cell)
    print(f"\nBest PC-LGCN cell by nDCG@10: {best_config}")
    return best_config, run_timestamp


@dataclass(frozen=True)
class OutputDirectories:
    """Default output paths for evaluation, decomposition analyses, and panel regressions."""

    evaluation: Path
    baseline_decomposition: Path
    profile_coherent_decomposition: Path
    panel_regression: Path


_DEFAULT_OUTPUT_DIRECTORIES = OutputDirectories(
    evaluation=Path("outputs/results/evaluation"),
    baseline_decomposition=Path("outputs/analysis/baseline_decomposition"),
    profile_coherent_decomposition=Path(
        "outputs/analysis/profile_coherent_decomposition"
    ),
    panel_regression=Path("outputs/analysis/panel_regression"),
)

_PRIMARY_METRIC_TO_PER_SPLIT_COLUMN: dict[str, str] = {
    "average_ndcg": "ndcg_at_k",
    "average_roi": "roi_at_k",
    "average_recall": "recall_at_k",
    "average_profile_coherence": "profile_coherence_at_k",
}

_DISPLAY_MODEL_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "light_gcn": "LightGCN",
    PC_LGCN_MODEL_NAME: "Profile-Coherent LightGCN",
}


def _discover_run_directories(
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
            "profile_coherence_lift_at_k",
        )
        if column in metrics_dataframe.columns
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
    row: dict[str, Any] = {
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
    if "profile_coherence_lift_at_k_mean" in best.index:
        row["profile_coherence_lift_at_k_mean"] = float(
            best["profile_coherence_lift_at_k_mean"]
        )
        row["profile_coherence_lift_at_k_std"] = float(
            best["profile_coherence_lift_at_k_std"]
        )
    return row


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
    """Pretty-print the headline N-row table to stdout."""
    if main_results.empty:
        print("(no main results to display)")
        return
    has_lift = "profile_coherence_lift_at_k_mean" in main_results.columns
    header = (
        f"{'Model':<28} {'nDCG@' + str(top_k):<14} {'ROI@' + str(top_k):<14}"
        f" {'Recall@' + str(top_k):<14} {'PC@' + str(top_k):<10}"
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
            f" {row['recall_at_k_mean']:<14.4f}"
            f" {row['profile_coherence_at_k_mean']:<10.4f}"
        )
        if has_lift:
            line += f" {row['profile_coherence_lift_at_k_mean']:<10.4f}"
        print(line)
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
    evaluation_directory: Path = _DEFAULT_OUTPUT_DIRECTORIES.evaluation,
    output_root: Path = _DEFAULT_OUTPUT_DIRECTORIES.baseline_decomposition,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
    top_k: int = 10,
    extra_grid_specs: dict[str, GridSpec] | None = None,
) -> dict[str, Any]:
    """Run the 3-model decomposition; returns the headline summary dict."""
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
            "Run `uv run poe tune` first."
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
                f"  Missing recommendations parquet at {parquet_path}; "
                "skipping decomposition."
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


_PC_LGCN_METRIC_COLUMNS: tuple[str, ...] = (
    "ndcg_at_k_mean",
    "ndcg_at_k_std",
    "roi_at_k_mean",
    "roi_at_k_std",
    "recall_at_k_mean",
    "recall_at_k_std",
    "profile_coherence_at_k_mean",
    "profile_coherence_at_k_std",
    "profile_coherence_lift_at_k_mean",
    "profile_coherence_lift_at_k_std",
)


def _resolve_pc_lgcn_run_directory(
    evaluation_directory: Path, run_timestamp: str | None
) -> Path:
    """Return the PC-LGCN run directory to analyse (latest if not specified)."""
    model_directory = evaluation_directory / PC_LGCN_MODEL_NAME
    if not model_directory.exists():
        raise FileNotFoundError(
            f"No PC-LGCN evaluation directory at {model_directory}. "
            "Run `uv run poe tune` first."
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


def _load_pc_lgcn_per_trial_dataframe(run_directory: Path) -> pd.DataFrame:
    """Average each PC-LGCN trial's per-split metrics and merge in the cell hyperparameters."""
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

    is_default_lambda = (
        per_trial["profile_coherence_lambda"] == _PC_LGCN_SWEEP.default_lambda
    )
    sensitivity_mask = (
        per_trial["profile_embedding_enabled"] & per_trial["profile_coherence_enabled"]
    )

    ablation = (
        per_trial[is_default_lambda]
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


def _attach_pc_lgcn_decomposition(
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


def _print_pc_lgcn_table(title: str, table: pd.DataFrame, columns: list[str]) -> None:
    """Pretty-print a small PC-LGCN ablation/sensitivity table to stdout."""
    if table.empty:
        print(f"({title}: empty)")
        return
    print(f"\n=== {title} ===")
    display = table[columns].copy()
    print(display.to_string(index=False))


def run_profile_coherent_decomposition(
    evaluation_directory: Path = _DEFAULT_OUTPUT_DIRECTORIES.evaluation,
    output_root: Path = _DEFAULT_OUTPUT_DIRECTORIES.profile_coherent_decomposition,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
) -> dict[str, Any]:
    """Produce the ablation and lambda-sensitivity tables for one PC-LGCN run."""
    data_paths = data_paths or DataPaths()
    run_directory = _resolve_pc_lgcn_run_directory(evaluation_directory, run_timestamp)
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

    per_trial = _load_pc_lgcn_per_trial_dataframe(run_directory)
    if per_trial.empty:
        raise FileNotFoundError(
            f"No per_split_metrics.csv files under {run_directory}."
        )

    ablation, sensitivity = _split_ablation_and_sensitivity(per_trial)
    ablation = _attach_pc_lgcn_decomposition(
        ablation, run_directory, customer_profiles, asset_risk_classes
    )
    sensitivity = _attach_pc_lgcn_decomposition(
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
        "profile_coherence_lift_at_k_mean",
        "coherent_share",
    ]
    _print_pc_lgcn_table(
        "PC-LGCN ablation (lambda fixed)",
        ablation,
        [c for c in summary_columns if c in ablation.columns],
    )
    _print_pc_lgcn_table(
        "PC-LGCN lambda sensitivity",
        sensitivity,
        [c for c in summary_columns if c in sensitivity.columns],
    )

    summary = {
        "run_timestamp": chosen_run_timestamp,
        "run_directory": str(run_directory),
        "output_directory": str(output_directory),
        "metric_columns": list(_PC_LGCN_METRIC_COLUMNS),
        "ablation_rows": ablation.to_dict(orient="records"),
        "sensitivity_rows": sensitivity.to_dict(orient="records"),
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPC-LGCN decomposition outputs saved to {output_directory}")
    return summary


def _best_trial_recommendations(
    run_directory: Path, primary_metric_key: str
) -> tuple[str, pd.DataFrame] | None:
    """Return the best trial's id and recommendations parquet under one run dir."""
    per_split = _load_per_split_metrics(run_directory)
    if per_split.empty:
        return None
    aggregated = _average_per_trial(per_split)
    best_trial_id = _select_best_trial_id(aggregated, primary_metric_key)
    if best_trial_id is None:
        return None
    parquet_path = run_directory / best_trial_id / "recommendations.parquet"
    if not parquet_path.exists():
        return None
    return best_trial_id, pd.read_parquet(parquet_path)


def _build_panel_dataframe(
    evaluation_directory: Path,
    run_timestamp: str | None,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
    best_trial_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble one row per (model, split, customer) with coherent_share + declared_band."""
    runs = _discover_run_directories(evaluation_directory, run_timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No evaluation runs found under {evaluation_directory}."
        )

    overrides = best_trial_overrides or {}
    all_specs = {**GRID_SPECS, PC_LGCN_MODEL_NAME: PROFILE_COHERENT_GRID_SPEC}
    frames: list[pd.DataFrame] = []
    for model_name, run_directory in runs.items():
        if model_name not in all_specs:
            print(f"  Skipping unknown model directory '{model_name}'")
            continue
        if model_name in overrides:
            forced_trial_id = overrides[model_name]
            parquet_path = run_directory / forced_trial_id / "recommendations.parquet"
            if not parquet_path.exists():
                print(f"  Override trial {forced_trial_id} not found for {model_name}.")
                continue
            best_trial_id, recommendations = (
                forced_trial_id,
                pd.read_parquet(parquet_path),
            )
        else:
            primary_metric_key = _PRIMARY_METRIC_TO_KEY[
                all_specs[model_name].primary_metric
            ]
            result = _best_trial_recommendations(run_directory, primary_metric_key)
            if result is None:
                print(f"  No usable best trial for {model_name}; skipping.")
                continue
            best_trial_id, recommendations = result
        print(
            f"  {model_name}: best trial {best_trial_id} ({len(recommendations)} rows)"
        )

        annotated = _annotate_recommendations(
            recommendations, customer_profiles, asset_risk_classes
        )
        per_customer_split = (
            annotated.groupby(["customer_id", "split_index"], dropna=False)
            .agg(
                coherent_share=(
                    "is_coherent",
                    lambda series: float(series.fillna(False).mean()),
                ),
                declared_band=("customer_band", "first"),
            )
            .reset_index()
        )
        per_customer_split["model"] = model_name
        frames.append(per_customer_split)

    if not frames:
        raise RuntimeError("No model produced a usable best-trial parquet.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["declared_band"])
    panel["declared_band"] = panel["declared_band"].astype(int)
    panel["band_label"] = panel["declared_band"].map(RISK_BAND_NAMES)
    return panel


def _fit_panel_ols(panel: pd.DataFrame) -> Any:
    """Fit `coherent_share ~ C(declared_band) * C(model) + C(split_index)` with cluster SEs."""
    customer_codes = panel["customer_id"].astype("category").cat.codes
    return smf.ols(
        "coherent_share ~ C(declared_band) * C(model) + C(split_index)",
        data=panel,
    ).fit(cov_type="cluster", cov_kwds={"groups": customer_codes.values})


def _coefficients_dataframe(model_fit: Any) -> pd.DataFrame:
    """Return a tidy coefficients table with estimate, SE, t, p, and 95% CI."""
    confidence_intervals = model_fit.conf_int()
    confidence_intervals.columns = ["ci_lower", "ci_upper"]
    table = pd.DataFrame(
        {
            "term": model_fit.params.index,
            "estimate": model_fit.params.values,
            "std_error": model_fit.bse.values,
            "t_value": model_fit.tvalues.values,
            "p_value": model_fit.pvalues.values,
        }
    )
    return table.merge(
        confidence_intervals.reset_index().rename(columns={"index": "term"}),
        on="term",
        how="left",
    )


def _predicted_pc_grid(panel: pd.DataFrame, model_fit: Any) -> pd.DataFrame:
    """Predict PC@10 for every observed (band, model) at the median split."""
    median_split = int(panel["split_index"].median())
    bands = sorted(panel["declared_band"].unique())
    models = sorted(panel["model"].unique())
    grid = pd.DataFrame(
        [
            {
                "declared_band": band,
                "model": model_name,
                "split_index": median_split,
            }
            for band in bands
            for model_name in models
        ]
    )

    predictions = model_fit.get_prediction(grid)
    summary_frame = predictions.summary_frame(alpha=0.05)
    grid["predicted_pc"] = summary_frame["mean"].to_numpy()
    grid["ci_lower"] = summary_frame["mean_ci_lower"].to_numpy()
    grid["ci_upper"] = summary_frame["mean_ci_upper"].to_numpy()
    grid["band_label"] = grid["declared_band"].map(RISK_BAND_NAMES)
    grid["model_display"] = grid["model"].map(
        lambda key: _DISPLAY_MODEL_NAMES.get(key, key)
    )
    return grid


def _save_forest_figure(prediction_grid: pd.DataFrame, output_path: Path) -> None:
    """Render predicted PC@10 with 95% CIs, grouped by band and coloured by model."""
    bands = sorted(prediction_grid["declared_band"].unique())
    models = sorted(prediction_grid["model"].unique())
    band_positions = np.arange(len(bands))
    bar_offset = 0.18
    figure, axis = plt.subplots(figsize=(8, 4.5))

    for model_index, model_name in enumerate(models):
        subset = prediction_grid[prediction_grid["model"] == model_name].sort_values(
            "declared_band"
        )
        x_offsets = band_positions + (model_index - (len(models) - 1) / 2) * bar_offset
        lower_error = subset["predicted_pc"].to_numpy() - subset["ci_lower"].to_numpy()
        upper_error = subset["ci_upper"].to_numpy() - subset["predicted_pc"].to_numpy()
        axis.errorbar(
            x_offsets,
            subset["predicted_pc"].to_numpy(),
            yerr=[lower_error, upper_error],
            fmt="o",
            capsize=4,
            label=_DISPLAY_MODEL_NAMES.get(model_name, model_name),
        )

    axis.set_xticks(band_positions)
    axis.set_xticklabels([RISK_BAND_NAMES[band] for band in bands])
    axis.set_xlabel("Declared MiFID II risk band")
    axis.set_ylabel("Predicted PC@10 (95% CI)")
    axis.set_title("Profile coherence by declared band and model")
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def run_panel_regression(
    evaluation_directory: Path = _DEFAULT_OUTPUT_DIRECTORIES.evaluation,
    output_root: Path = _DEFAULT_OUTPUT_DIRECTORIES.panel_regression,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
    best_trial_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fit the panel OLS, save coefficients + predictions + forest figure."""
    data_paths = data_paths or DataPaths()
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

    print(f"Building panel from {evaluation_directory} ...")
    panel = _build_panel_dataframe(
        evaluation_directory=evaluation_directory,
        run_timestamp=run_timestamp,
        customer_profiles=customer_profiles,
        asset_risk_classes=asset_risk_classes,
        best_trial_overrides=best_trial_overrides,
    )
    print(
        f"Panel: {len(panel)} rows across "
        f"{panel['model'].nunique()} models, "
        f"{panel['declared_band'].nunique()} bands, "
        f"{panel['customer_id'].nunique()} customers, "
        f"{panel['split_index'].nunique()} splits."
    )

    print("Fitting OLS with cluster-robust SEs (clustered on customer_id) ...")
    model_fit = _fit_panel_ols(panel)
    coefficients = _coefficients_dataframe(model_fit)
    prediction_grid = _predicted_pc_grid(panel, model_fit)

    output_directory = output_root / (
        run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_directory / "coefficients.csv", index=False)
    prediction_grid.to_csv(
        output_directory / "predicted_pc_by_band_model.csv", index=False
    )
    _save_forest_figure(prediction_grid, output_directory / "forest_predicted_pc.png")
    (output_directory / "panel.csv").write_text(panel.to_csv(index=False))
    (output_directory / "regression_summary.txt").write_text(str(model_fit.summary()))

    print(f"\nPanel regression artefacts saved to {output_directory}")
    print("\nKey coefficients (interaction terms):")
    interaction_mask = coefficients["term"].str.contains(":", regex=False)
    print(
        coefficients[interaction_mask]
        .loc[:, ["term", "estimate", "std_error", "p_value", "ci_lower", "ci_upper"]]
        .to_string(index=False)
    )

    return {
        "output_directory": str(output_directory),
        "panel_rows": int(len(panel)),
        "model_summary": str(model_fit.summary()),
    }


def _latest_run_directory(model_root: Path) -> Path:
    """Return the lexicographically-latest timestamped subdirectory."""
    if not model_root.exists():
        raise FileNotFoundError(f"No model directory at {model_root}")
    candidates = [path for path in model_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped runs under {model_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _find_pc_lgcn_trial_by_hyperparameters(
    run_directory: Path,
    *,
    profile_embedding_enabled: bool,
    profile_coherence_enabled: bool,
    profile_coherence_lambda: float,
) -> str | None:
    """Return the trial_id matching the given PC-LGCN hyperparameter cell."""
    if not run_directory.exists():
        return None
    for trial_directory in sorted(run_directory.iterdir()):
        metrics_path = trial_directory / "per_split_metrics.csv"
        if not metrics_path.exists():
            continue
        first_row = pd.read_csv(metrics_path, nrows=1).iloc[0]
        if (
            bool(first_row["profile_embedding_enabled"]) == profile_embedding_enabled
            and bool(first_row["profile_coherence_enabled"])
            == profile_coherence_enabled
            and float(first_row["profile_coherence_lambda"]) == profile_coherence_lambda
        ):
            return str(first_row["trial_id"])
    return None


def run_tune(
    splits_directory: Path = Path("data/splits"),
    results_directory: Path = Path("outputs/results"),
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    splits_limit: int | None = None,
    max_concurrent_trials: int | None = None,
) -> None:
    """Orchestrate the full thesis pipeline in one invocation."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()

    print("\n=== Stage 1: baseline grid sweep (RF + LightGCN) ===")
    _, baseline_run_timestamp = run_baseline_grid_search(
        splits_directory=splits_directory,
        results_directory=results_directory,
        experiment_config=experiment_config,
        data_paths=data_paths,
        splits_limit=splits_limit,
        max_concurrent_trials_override=max_concurrent_trials,
    )

    print("\n=== Stage 2: Profile-Coherent LightGCN sweep ===")
    _, pc_lgcn_run_timestamp = run_profile_coherent_grid(
        splits_directory=splits_directory,
        results_directory=results_directory,
        experiment_config=experiment_config,
        data_paths=data_paths,
        splits_limit=splits_limit,
        max_concurrent_trials=max_concurrent_trials
        if max_concurrent_trials is not None
        else 4,
    )

    print("\n=== Stage 3: 3-model decomposition + PC-LGCN ablation tables ===")
    run_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=None,
        data_paths=data_paths,
        top_k=experiment_config.top_k,
        extra_grid_specs={PC_LGCN_MODEL_NAME: PROFILE_COHERENT_GRID_SPEC},
    )

    run_profile_coherent_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=pc_lgcn_run_timestamp,
        data_paths=data_paths,
    )

    print("\n=== Stage 4: panel regressions with canonical PC-LGCN trials ===")
    pc_lgcn_run = _latest_run_directory(
        results_directory / "evaluation" / PC_LGCN_MODEL_NAME
    )
    l_pc_alone_trial = _find_pc_lgcn_trial_by_hyperparameters(
        pc_lgcn_run,
        profile_embedding_enabled=False,
        profile_coherence_enabled=True,
        profile_coherence_lambda=0.5,
    )
    full_method_trial = _find_pc_lgcn_trial_by_hyperparameters(
        pc_lgcn_run,
        profile_embedding_enabled=True,
        profile_coherence_enabled=True,
        profile_coherence_lambda=1.0,
    )

    if l_pc_alone_trial is not None:
        print(f"\nPanel regression with L_PC-only cell (trial {l_pc_alone_trial}):")
        run_panel_regression(
            evaluation_directory=results_directory / "evaluation",
            output_root=Path("outputs/analysis/panel_regression_three_models"),
            best_trial_overrides={PC_LGCN_MODEL_NAME: l_pc_alone_trial},
            data_paths=data_paths,
        )
    else:
        print(
            "L_PC-only canonical cell not found; skipping the L_PC-only panel regression."
        )

    if full_method_trial is not None:
        print(f"\nPanel regression with full-method cell (trial {full_method_trial}):")
        run_panel_regression(
            evaluation_directory=results_directory / "evaluation",
            output_root=Path(
                "outputs/analysis/panel_regression_three_models_full_method"
            ),
            best_trial_overrides={PC_LGCN_MODEL_NAME: full_method_trial},
            data_paths=data_paths,
        )
    else:
        print(
            "Full-method canonical cell not found; skipping the full-method panel regression."
        )

    print(
        f"\nFull pipeline complete. Baseline run: {baseline_run_timestamp}; "
        f"PC-LGCN run: {pc_lgcn_run_timestamp}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end thesis pipeline: RF + LightGCN baseline grid, PC-LGCN "
            "ablation + lambda sweep, 3-model decomposition, PC-LGCN ablation "
            "tables, and both canonical panel regressions."
        )
    )
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits-limit", type=int, default=None)
    parser.add_argument("--max-concurrent-trials", type=int, default=None)
    arguments = parser.parse_args()

    run_tune(
        splits_directory=Path(arguments.splits_dir),
        results_directory=Path(arguments.results_dir),
        experiment_config=ExperimentConfig(device=arguments.device),
        splits_limit=arguments.splits_limit,
        max_concurrent_trials=arguments.max_concurrent_trials,
    )
