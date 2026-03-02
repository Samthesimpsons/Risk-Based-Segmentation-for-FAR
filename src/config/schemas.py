"""Pydantic schemas for temporal splits, sequences, and evaluation results."""

from datetime import date

from pydantic import BaseModel


class SequenceData(BaseModel):
    """Per-user chronologically ordered purchase sequences for a split."""

    split_index: int
    time_point: date
    user_sequences: dict[str, list[tuple[str, date]]]


class TemporalSplitData(BaseModel):
    """A single temporal train/test split with filtered user and asset sets."""

    model_config = {"arbitrary_types_allowed": True}

    split_index: int
    time_point: date
    test_end: date
    training_interactions: dict[str, set[str]]
    test_interactions: dict[str, set[str]]
    eligible_customer_ids: list[str]
    eligible_asset_ids: list[str]
    customer_id_to_index: dict[str, int]
    asset_id_to_index: dict[str, int]


class EvaluationResult(BaseModel):
    """Evaluation metrics for a model on a single temporal split."""

    split_index: int
    time_point: date
    model_name: str
    ndcg_at_k: float
    roi_at_k: float
