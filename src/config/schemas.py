"""Pydantic schemas for temporal splits, customer profiles, and evaluation results."""

from datetime import date

from pydantic import BaseModel, field_serializer


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

    @field_serializer("training_interactions", "test_interactions")
    def _serialise_sorted(self, value: dict[str, set[str]]) -> dict[str, list[str]]:
        return {key: sorted(assets) for key, assets in value.items()}


class CustomerProfile(BaseModel):
    """Regulatory profile signals available for a single customer."""

    customer_id: str
    risk_band: int | None
    risk_band_is_predicted: bool
    customer_type: str | None


class EvaluationResult(BaseModel):
    """Evaluation metrics for a model on a single temporal split."""

    split_index: int
    time_point: date
    model_name: str
    ndcg_at_k: float
    roi_at_k: float
    profile_coherence_at_k: float
    profile_coherence_lift_at_k: float
