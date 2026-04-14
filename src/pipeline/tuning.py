"""Hyperparameter grid search pipeline using Ray Tune.

Each model declares a grid centered on its FAR-Trans paper configuration.
The grid is evaluated exhaustively via `tune.grid_search` so that benchmark
replication can be cleanly compared against tuned variants.

Adding a new model is a three-step extension:
  1. Create a `Recommender` implementation under `src/models/`.
  2. Add a `ModelTuningSpec` entry below describing its grid, factory, and
     primary metric.
  3. Add an entry in `_CONFIG_LOADERS` so saved JSON configs can be reloaded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import ray
from ray import tune

from src.config.schemas import SequenceData, TemporalSplitData
from src.config.settings import (
    HybridDualHeadConfig,
    LightGCNConfig,
    ModelConfig,
    RandomForestConfig,
    SASRecConfig,
    TiSASRecConfig,
)
from src.evaluation.metrics import evaluate_model_on_split
from src.features.technical_indicators import build_indicator_dataframe
from src.models.hybrid import HybridDualHeadRecommender
from src.models.light_gcn import LightGCNBaseline
from src.models.protocol import Recommender
from src.models.random_forest import RandomForestBaseline
from src.models.sasrec import SASRecRecommender
from src.models.tisasrec import TiSASRecRecommender
from src.pipeline.preprocessing import (
    load_preprocessed_close_prices,
    load_validation_sequences,
    load_validation_splits,
)
from src.pipeline.runner import generate_recommendations

PrimaryMetric = Literal["ndcg", "roi"]


@dataclass(frozen=True)
class ValidationContext:
    """Shared inputs for validation runs across all model tuners."""

    validation_splits: list[TemporalSplitData]
    validation_sequences: list[SequenceData]
    close_prices: pd.DataFrame
    indicator_dataframe: pd.DataFrame | None
    top_k: int
    device: str


@dataclass(frozen=True)
class ModelTuningSpec:
    """Declarative spec for tuning a single model.

    The grid maps each hyperparameter name to a list of values that the Ray
    Tune grid expander iterates over. `max_concurrent_trials` controls how
    many trials within this model's grid run in parallel.
    """

    model_name: str
    config_class: type[ModelConfig]
    grid: dict[str, list[int | float | str | None]]
    needs_indicators: bool
    needs_sequences: bool
    primary_metric: PrimaryMetric
    max_concurrent_trials: int = 1


def _evaluate_on_validation_splits(
    model: Recommender,
    context: ValidationContext,
) -> tuple[float, float]:
    """Train and evaluate a model across validation splits.

    Returns (average_ndcg, average_roi) across all validation splits.
    """
    needs_sequences = bool(context.validation_sequences)

    ndcg_scores: list[float] = []
    roi_scores: list[float] = []

    for index, split in enumerate(context.validation_splits):
        train_kwargs: dict[str, object] = {"device": context.device}
        if needs_sequences:
            train_kwargs["user_sequences"] = context.validation_sequences[
                index
            ].user_sequences

        model.train_on_split(split, **train_kwargs)
        recommendations = generate_recommendations(model, split, context.top_k)
        result = evaluate_model_on_split(
            recommendations, split, context.close_prices, context.top_k
        )
        ndcg_scores.append(result.ndcg_at_k)
        roi_scores.append(result.roi_at_k)

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    average_roi = sum(roi_scores) / len(roi_scores) if roi_scores else 0.0
    return average_ndcg, average_roi


def _build_model(
    spec: ModelTuningSpec,
    config: ModelConfig,
    context: ValidationContext,
) -> Recommender:
    """Construct a recommender from a tuning spec and a hyperparameter config."""
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
    if spec.model_name == "sasrec":
        assert isinstance(config, SASRecConfig)
        return SASRecRecommender(config=config)
    if spec.model_name == "tisasrec":
        assert isinstance(config, TiSASRecConfig)
        return TiSASRecRecommender(config=config)
    if spec.model_name == "hybrid_dual_head":
        assert isinstance(config, HybridDualHeadConfig)
        return HybridDualHeadRecommender(
            config=config,
            close_prices=context.close_prices,
            indicator_dataframe=context.indicator_dataframe,
        )
    raise ValueError(f"Unknown model: {spec.model_name}")


# Grids are intentionally small and centered on the FAR-Trans paper defaults
# so that the benchmark configuration is always one of the trial points.
# Edit these dictionaries to broaden / narrow the search.

RANDOM_FOREST_TUNING = ModelTuningSpec(
    model_name="random_forest",
    config_class=RandomForestConfig,
    grid={
        "number_of_estimators": [20, 50, 75],
        "max_depth": [None],
    },
    needs_indicators=True,
    needs_sequences=False,
    primary_metric="roi",
    max_concurrent_trials=3,
)

LIGHT_GCN_TUNING = ModelTuningSpec(
    model_name="light_gcn",
    config_class=LightGCNConfig,
    grid={
        "embedding_dimension": [64, 128],
        "number_of_layers": [3, 4],
        "learning_rate": [1e-3, 1e-2],
        "weight_decay": [1e-5, 1e-4],
        "keep_probability": [0.6, 1.0],
        "number_of_epochs": [50],
    },
    needs_indicators=False,
    needs_sequences=False,
    primary_metric="ndcg",
    max_concurrent_trials=1,
)

SASREC_TUNING = ModelTuningSpec(
    model_name="sasrec",
    config_class=SASRecConfig,
    grid={
        "embedding_dimension": [64],
        "number_of_attention_heads": [1, 2],
        "max_sequence_length": [50],
        "number_of_blocks": [1, 2],
        "dropout_rate": [0.2, 0.5],
        "learning_rate": [1e-3],
        "number_of_epochs": [200],
        "batch_size": [128],
    },
    needs_indicators=False,
    needs_sequences=True,
    primary_metric="ndcg",
    max_concurrent_trials=1,
)

TISASREC_TUNING = ModelTuningSpec(
    model_name="tisasrec",
    config_class=TiSASRecConfig,
    grid={
        "embedding_dimension": [64],
        "number_of_attention_heads": [1, 2],
        "max_sequence_length": [50],
        "number_of_blocks": [1, 2],
        "dropout_rate": [0.2],
        "learning_rate": [1e-3],
        "number_of_epochs": [200],
        "batch_size": [128],
        "time_bucket_count": [64, 256],
    },
    needs_indicators=False,
    needs_sequences=True,
    primary_metric="ndcg",
)

HYBRID_DUAL_HEAD_TUNING = ModelTuningSpec(
    model_name="hybrid_dual_head",
    config_class=HybridDualHeadConfig,
    grid={
        "embedding_dimension": [64],
        "max_sequence_length": [50],
        "number_of_attention_heads": [2],
        "number_of_blocks": [2],
        "dropout_rate": [0.2],
        "learning_rate": [1e-3],
        "number_of_epochs": [200],
        "batch_size": [128],
        "time_bucket_count": [256],
        "profitability_hidden_dimension": [32, 64],
        "loss_lambda": [0.1, 0.5, 1.0],
        "inference_alpha": [0.3, 0.5, 0.7],
    },
    needs_indicators=True,
    needs_sequences=True,
    primary_metric="roi",
)

ALL_TUNING_SPECS: dict[str, ModelTuningSpec] = {
    spec.model_name: spec
    for spec in (
        RANDOM_FOREST_TUNING,
        LIGHT_GCN_TUNING,
        SASREC_TUNING,
        TISASREC_TUNING,
        HYBRID_DUAL_HEAD_TUNING,
    )
}

ALL_MODEL_NAMES: list[str] = list(ALL_TUNING_SPECS.keys())


_CONFIG_LOADERS: dict[str, Callable[[dict[str, Any]], ModelConfig]] = {
    "random_forest": lambda payload: RandomForestConfig(**payload),
    "light_gcn": lambda payload: LightGCNConfig(**payload),
    "sasrec": lambda payload: SASRecConfig(**payload),
    "tisasrec": lambda payload: TiSASRecConfig(**payload),
    "hybrid_dual_head": lambda payload: HybridDualHeadConfig(**payload),
}


def _build_grid_search_space(spec: ModelTuningSpec) -> dict[str, Any]:
    """Wrap each grid axis in `tune.grid_search` so Ray expands the cartesian product."""
    return {name: tune.grid_search(values) for name, values in spec.grid.items()}


def _materialize_config(
    spec: ModelTuningSpec, hyperparameters: dict[str, Any]
) -> ModelConfig:
    """Build a Pydantic config object from a Ray Tune trial dictionary."""
    return spec.config_class(**hyperparameters)


def _metric_value(
    primary_metric: PrimaryMetric, average_ndcg: float, average_roi: float
) -> float:
    """Pick the aggregate metric used to rank trials for a given spec."""
    return average_roi if primary_metric == "roi" else average_ndcg


def tune_model(
    spec: ModelTuningSpec,
    context: ValidationContext,
    use_gpu: bool = False,
) -> tuple[ModelConfig, tune.ResultGrid]:
    """Run Ray Tune grid search for a single model and return the best config."""

    context_ref = ray.put(context)

    def trainable(hyperparameters: dict[str, Any]) -> None:
        resolved_context: ValidationContext = ray.get(context_ref)
        config = _materialize_config(spec, hyperparameters)
        model = _build_model(spec, config, resolved_context)
        average_ndcg, average_roi = _evaluate_on_validation_splits(
            model, resolved_context
        )
        tune.report(
            {
                "average_ndcg": average_ndcg,
                "average_roi": average_roi,
            }
        )

    metric_key = "average_roi" if spec.primary_metric == "roi" else "average_ndcg"

    # Random Forest is sklearn (CPU-only), so never claim a GPU slot for it even
    # when --device cuda is passed; otherwise it blocks the single GPU unnecessarily
    # and serializes RF trials that could otherwise run in parallel on CPU.
    trial_resources: dict[str, float] = (
        {"gpu": 1.0, "cpu": 1.0}
        if use_gpu and spec.model_name != "random_forest"
        else {"cpu": 1.0}
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, trial_resources),
        param_space=_build_grid_search_space(spec),
        tune_config=tune.TuneConfig(
            metric=metric_key,
            mode="max",
            num_samples=1,
            max_concurrent_trials=spec.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name=f"{spec.model_name}_grid_search",
            verbose=1,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric_key, mode="max")
    best_metrics = best_result.metrics
    best_hyperparameters = best_result.config
    assert best_metrics is not None, f"{spec.model_name} tuning produced no metrics"
    assert best_hyperparameters is not None, (
        f"{spec.model_name} tuning produced no config"
    )

    print(f"\n{spec.model_name} grid search complete:")
    print(f"  Best ROI: {best_metrics['average_roi']:.6f}")
    print(f"  Best nDCG: {best_metrics['average_ndcg']:.4f}")
    print(f"  Best config: {best_hyperparameters}")

    return _materialize_config(spec, dict(best_hyperparameters)), results


def _save_grid_search_results(
    result_grid: tune.ResultGrid,
    spec: ModelTuningSpec,
    results_directory: Path,
) -> None:
    """Save all grid search trial results to a timestamped CSV."""
    rows = []
    for result in result_grid:
        if result.metrics is None or result.config is None:
            continue
        row: dict[str, Any] = dict(result.config)
        row["average_ndcg"] = result.metrics.get("average_ndcg")
        row["average_roi"] = result.metrics.get("average_roi")
        rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_directory = results_directory / "tuning" / spec.model_name
    model_directory.mkdir(parents=True, exist_ok=True)
    output_path = model_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Grid search results saved to {output_path}")


def _save_tuned_configs_to_json(
    configs: Mapping[str, ModelConfig],
    output_path: Path,
) -> None:
    """Save best hyperparameter configs to JSON, merging with any existing file."""
    existing_payload: dict[str, dict[str, object]] = {}
    if output_path.exists():
        existing_payload = json.loads(output_path.read_text())

    for model_name, config in configs.items():
        existing_payload[model_name] = config.model_dump()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(existing_payload, indent=2))


def load_configs_from_json(config_path: Path) -> dict[str, ModelConfig]:
    """Load best hyperparameter configs from a JSON file saved by the tuning pipeline."""
    payload = json.loads(config_path.read_text())
    configs: dict[str, ModelConfig] = {}
    for model_name, loader in _CONFIG_LOADERS.items():
        if model_name in payload:
            configs[model_name] = loader(payload[model_name])
    return configs


def _build_validation_context(
    splits_directory: Path,
    needs_indicators: bool,
    needs_sequences: bool,
    top_k: int,
    device: str,
) -> ValidationContext:
    """Load all validation inputs needed by the requested model set."""
    print("Loading validation splits...")
    validation_splits = load_validation_splits(splits_directory)
    print(f"  Loaded {len(validation_splits)} validation splits")

    print("Loading close prices...")
    close_prices = load_preprocessed_close_prices(splits_directory)

    indicator_dataframe: pd.DataFrame | None = None
    if needs_indicators:
        print("Building indicator DataFrame...")
        indicator_dataframe = build_indicator_dataframe(close_prices)
        print(f"  Built indicators: {len(indicator_dataframe)} rows")

    validation_sequences: list[SequenceData] = []
    if needs_sequences:
        print("Loading validation sequences...")
        validation_sequences = load_validation_sequences(splits_directory)
        print(f"  Loaded {len(validation_sequences)} sequence files")

    return ValidationContext(
        validation_splits=validation_splits,
        validation_sequences=validation_sequences,
        close_prices=close_prices,
        indicator_dataframe=indicator_dataframe,
        top_k=top_k,
        device=device,
    )


def run_tuning_pipeline(
    splits_directory: Path,
    output_path: Path,
    top_k: int = 10,
    device: str = "cpu",
    selected_models: list[str] | None = None,
    results_directory: Path | None = None,
) -> dict[str, ModelConfig]:
    """Run the grid search pipeline and save the best configs to a JSON file.

    Models are tuned sequentially (one model fully completes before the next
    starts). Within each model, `spec.max_concurrent_trials` controls how many
    trials run in parallel; set this directly on each `ModelTuningSpec`.
    """
    requested = (
        list(selected_models) if selected_models else list(ALL_TUNING_SPECS.keys())
    )

    unknown = set(requested) - set(ALL_TUNING_SPECS.keys())
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    specs = [ALL_TUNING_SPECS[name] for name in requested]
    needs_indicators = any(spec.needs_indicators for spec in specs)
    needs_sequences = any(spec.needs_sequences for spec in specs)

    context = _build_validation_context(
        splits_directory=splits_directory,
        needs_indicators=needs_indicators,
        needs_sequences=needs_sequences,
        top_k=top_k,
        device=device,
    )

    project_root = Path(__file__).resolve().parents[2]
    os.environ["RAY_RUNTIME_ENV_LOCAL_DEV_MODE"] = "1"
    ray.init(
        log_to_driver=False,
        logging_level="error",
        include_dashboard=False,
        _enable_object_reconstruction=False,
        object_store_memory=1_073_741_824,
        runtime_env=_build_ray_runtime_env(project_root),
    )
    os.environ["RAY_ENABLE_LOG_MONITOR"] = "0"

    use_gpu = device != "cpu"
    best_configs: dict[str, ModelConfig] = {}

    try:
        for spec in specs:
            print(f"\nTuning {spec.model_name}...")
            spec_context = _filter_context_for_spec(context, spec)
            best_config, result_grid = tune_model(spec, spec_context, use_gpu=use_gpu)
            best_configs[spec.model_name] = best_config
            if results_directory is not None:
                _save_grid_search_results(result_grid, spec, results_directory)
    finally:
        ray.shutdown()

    _save_tuned_configs_to_json(best_configs, output_path)
    print(f"\nBest configs saved to {output_path}")

    return best_configs


def _filter_context_for_spec(
    context: ValidationContext, spec: ModelTuningSpec
) -> ValidationContext:
    """Return a context with sequence inputs blanked when the model does not need them.

    The shared loader populates everything; per-model contexts hide what is
    irrelevant so the trainable function can branch on `validation_sequences`
    being empty.
    """
    return ValidationContext(
        validation_splits=context.validation_splits,
        validation_sequences=context.validation_sequences
        if spec.needs_sequences
        else [],
        close_prices=context.close_prices,
        indicator_dataframe=context.indicator_dataframe
        if spec.needs_indicators
        else None,
        top_k=context.top_k,
        device=context.device,
    )


def _build_ray_runtime_env(project_root: Path) -> dict[str, object]:
    """Package only the source tree needed by Tune workers.

    Using ``py_modules`` avoids root-level ignore rules for large datasets from
    accidentally excluding the in-repo ``src.data`` package.
    """
    return {"py_modules": [str(project_root / "src")]}


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Grid search baseline model hyperparameters"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=None,
        help="Models to tune (default: all). Options: " + ", ".join(ALL_MODEL_NAMES),
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs/configs") / run_timestamp / "best_hyperparameters.json"

    best_configs = run_tuning_pipeline(
        splits_directory=Path("data/splits"),
        output_path=output_path,
        device=args.device,
        selected_models=args.models,
        results_directory=Path("outputs/results"),
    )

    print("\nBest configs found:")
    for model_name, config in best_configs.items():
        print(f"  {model_name}: {config}")
