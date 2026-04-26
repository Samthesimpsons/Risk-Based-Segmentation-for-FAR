"""Recommender protocol that all models implement."""

from __future__ import annotations

from typing import Protocol

from src.config.schemas import TemporalSplitData


class Recommender(Protocol):
    """Structural protocol implemented by every recommender model."""

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
        """Return top-k recommended asset ids for the given user."""
        ...
