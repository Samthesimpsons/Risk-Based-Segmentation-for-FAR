"""Catalog of pipeline-level model specs, grid axes, and metric keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.config.settings import LightGCNConfig, ModelConfig, RandomForestConfig

PrimaryMetric = Literal["ndcg", "roi", "profile_coherence"]

PRIMARY_METRIC_TO_KEY: dict[PrimaryMetric, str] = {
    "ndcg": "average_ndcg",
    "roi": "average_roi",
    "profile_coherence": "average_profile_coherence",
}

PRIMARY_METRIC_TO_PER_SPLIT_COLUMN: dict[str, str] = {
    "average_ndcg": "ndcg_at_k",
    "average_roi": "roi_at_k",
    "average_profile_coherence": "profile_coherence_at_k",
}

DISPLAY_MODEL_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "light_gcn": "LightGCN",
}

RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "number_of_estimators": [20, 30, 40, 50],
    "max_depth": [15, 25, 50],
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
        max_concurrent_trials=1,
    ),
    "light_gcn": GridSpec(
        model_name="light_gcn",
        config_class=LightGCNConfig,
        grid=LIGHT_GCN_GRID,
        primary_metric="ndcg",
        needs_indicators=False,
        use_gpu_per_trial=True,
        max_concurrent_trials=1,
    ),
}
