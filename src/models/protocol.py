"""Protocol and registry for the Recommender interface.

`Recommender` is the structural type that all models implement.

`MODEL_REGISTRY` is the single source of truth listing every model the
pipeline knows how to instantiate. Each entry holds the FAR-Trans paper
default config plus a factory and metadata used by `runner.py` and
`tuning.py`. Adding a new model is one entry plus the `Recommender`
implementation in `src/models/`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.config.schemas import TemporalSplitData
from src.config.settings import (
    HybridDualHeadConfig,
    LightGCNConfig,
    ModelConfig,
    RandomForestConfig,
    SASRecConfig,
    TiSASRecConfig,
)


class Recommender(Protocol):
    """Interface that all recommendation models must implement."""

    @property
    def name(self) -> str:
        """Return the display name of this recommender."""
        ...

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train the model on a single temporal split."""
        ...

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k recommended asset IDs for the given user."""
        ...


@dataclass(frozen=True)
class ModelEntry:
    """Static metadata describing one model in the recommendation pipeline."""

    model_name: str
    config_class: type[ModelConfig]
    needs_indicators: bool
    needs_sequences: bool
    needs_close_prices: bool


def _build_random_forest(
    config: ModelConfig,
    *,
    indicator_dataframe: pd.DataFrame | None,
    close_prices: pd.DataFrame | None,
) -> Recommender:
    from src.models.random_forest import RandomForestBaseline

    assert isinstance(config, RandomForestConfig)
    assert indicator_dataframe is not None
    return RandomForestBaseline(
        random_forest_config=config,
        indicator_dataframe=indicator_dataframe,
    )


def _build_light_gcn(
    config: ModelConfig,
    *,
    indicator_dataframe: pd.DataFrame | None,
    close_prices: pd.DataFrame | None,
) -> Recommender:
    from src.models.light_gcn import LightGCNBaseline

    assert isinstance(config, LightGCNConfig)
    return LightGCNBaseline(config=config)


def _build_sasrec(
    config: ModelConfig,
    *,
    indicator_dataframe: pd.DataFrame | None,
    close_prices: pd.DataFrame | None,
) -> Recommender:
    from src.models.sasrec import SASRecRecommender

    assert isinstance(config, SASRecConfig)
    return SASRecRecommender(config=config)


def _build_tisasrec(
    config: ModelConfig,
    *,
    indicator_dataframe: pd.DataFrame | None,
    close_prices: pd.DataFrame | None,
) -> Recommender:
    from src.models.tisasrec import TiSASRecRecommender

    assert isinstance(config, TiSASRecConfig)
    return TiSASRecRecommender(config=config)


def _build_hybrid_dual_head(
    config: ModelConfig,
    *,
    indicator_dataframe: pd.DataFrame | None,
    close_prices: pd.DataFrame | None,
) -> Recommender:
    from src.models.hybrid import HybridDualHeadRecommender

    assert isinstance(config, HybridDualHeadConfig)
    assert close_prices is not None
    return HybridDualHeadRecommender(
        config=config,
        close_prices=close_prices,
        indicator_dataframe=indicator_dataframe,
    )


RecommenderBuilder = Callable[..., Recommender]


_BUILDERS: dict[str, RecommenderBuilder] = {
    "random_forest": _build_random_forest,
    "light_gcn": _build_light_gcn,
    "sasrec": _build_sasrec,
    "tisasrec": _build_tisasrec,
    "hybrid_dual_head": _build_hybrid_dual_head,
}


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "random_forest": ModelEntry(
        model_name="random_forest",
        config_class=RandomForestConfig,
        needs_indicators=True,
        needs_sequences=False,
        needs_close_prices=False,
    ),
    "light_gcn": ModelEntry(
        model_name="light_gcn",
        config_class=LightGCNConfig,
        needs_indicators=False,
        needs_sequences=False,
        needs_close_prices=False,
    ),
    "sasrec": ModelEntry(
        model_name="sasrec",
        config_class=SASRecConfig,
        needs_indicators=False,
        needs_sequences=True,
        needs_close_prices=False,
    ),
    "tisasrec": ModelEntry(
        model_name="tisasrec",
        config_class=TiSASRecConfig,
        needs_indicators=False,
        needs_sequences=True,
        needs_close_prices=False,
    ),
    "hybrid_dual_head": ModelEntry(
        model_name="hybrid_dual_head",
        config_class=HybridDualHeadConfig,
        needs_indicators=True,
        needs_sequences=True,
        needs_close_prices=True,
    ),
}


def build_recommender(
    model_name: str,
    config: ModelConfig | None = None,
    *,
    indicator_dataframe: pd.DataFrame | None = None,
    close_prices: pd.DataFrame | None = None,
) -> Recommender:
    """Construct a recommender from the registry, using paper defaults when no config is given."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    entry = MODEL_REGISTRY[model_name]
    resolved_config = config if config is not None else entry.config_class()
    builder = _BUILDERS[model_name]
    return builder(
        resolved_config,
        indicator_dataframe=indicator_dataframe,
        close_prices=close_prices,
    )


ALL_MODEL_NAMES: list[str] = list(MODEL_REGISTRY.keys())
