"""Evaluation metrics for recommendation quality: nDCG@k and ROI@k.

ROI@k uses the geometric monthly return formula from the FAR-Trans reference:
  monthly_return = pow(1 + total_return, 30 / days) - 1
where days is the calendar days between time_point and test_end.
"""

import math
from datetime import date

import pandas as pd

from src.config.schemas import EvaluationResult, TemporalSplitData

CALENDAR_DAYS_PER_MONTH = 30


def compute_ndcg_at_k(
    ranked_recommendations: list[str],
    relevant_assets: set[str],
    k: int = 10,
) -> float:
    """Compute nDCG@k for a single user's recommendation list."""
    if not relevant_assets:
        return 0.0

    top_k = ranked_recommendations[:k]

    dcg = 0.0
    for i, asset_id in enumerate(top_k):
        if asset_id in relevant_assets:
            dcg += 1.0 / math.log2(i + 2)

    ideal_relevant_count = min(k, len(relevant_assets))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant_count))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_roi_at_k(
    ranked_recommendations: list[str],
    price_lookup: dict[str, tuple[float, float]],
    days_in_period: int,
    k: int = 10,
) -> float:
    """Compute ROI@k: average geometric monthly return of top-k recommended assets.

    price_lookup maps ISIN to (price_at_time_point, price_at_test_end).
    Monthly return per asset = pow(1 + total_return, 30 / days) - 1.

    Matches FAR-Trans semantics (`metrics/kpi_evaluation_metric.py:30-32`,
    `metrics/kpi_monthly_evaluation_metric.py`): recommended assets without a
    valid price are imputed as a 0% return rather than skipped, so the per-user
    denominator is the actual number of items in the top-k slice.
    """
    top_k = ranked_recommendations[:k]
    if not top_k:
        return 0.0

    monthly_returns: list[float] = []

    for asset_id in top_k:
        if asset_id in price_lookup:
            start_price, end_price = price_lookup[asset_id]
            if start_price > 0.0:
                total_return = (end_price - start_price) / start_price
                monthly_return = (
                    math.pow(
                        1.0 + total_return,
                        CALENDAR_DAYS_PER_MONTH / days_in_period,
                    )
                    - 1.0
                )
                monthly_returns.append(monthly_return)
                continue
        monthly_returns.append(0.0)

    return sum(monthly_returns) / len(monthly_returns)


def build_price_lookup(
    close_prices: pd.DataFrame,
    time_point: date,
    test_end: date,
    asset_ids: list[str],
) -> dict[str, tuple[float, float]]:
    """Build a mapping from ISIN to (price_at_start, price_at_end) for ROI computation.

    For each asset, find the closest available price on or before time_point and test_end.
    """
    time_point_timestamp = pd.Timestamp(time_point)
    test_end_timestamp = pd.Timestamp(test_end)

    grouped = close_prices.groupby("ISIN")
    lookup: dict[str, tuple[float, float]] = {}

    for asset_id in asset_ids:
        if asset_id not in grouped.groups:
            continue

        asset_data = grouped.get_group(asset_id)

        start_candidates = asset_data[asset_data["timestamp"] <= time_point_timestamp]
        end_candidates = asset_data[asset_data["timestamp"] <= test_end_timestamp]

        if start_candidates.empty or end_candidates.empty:
            continue

        start_price = float(start_candidates.iloc[-1]["closePrice"])
        end_price = float(end_candidates.iloc[-1]["closePrice"])
        lookup[asset_id] = (start_price, end_price)

    return lookup


def evaluate_model_on_split(
    recommendations: dict[str, list[str]],
    split: TemporalSplitData,
    close_prices: pd.DataFrame,
    k: int = 10,
) -> EvaluationResult:
    """Evaluate a model's recommendations on one temporal split.

    Averages nDCG@k and ROI@k across all eligible users.
    """
    price_lookup = build_price_lookup(
        close_prices, split.time_point, split.test_end, split.eligible_asset_ids
    )

    days_in_period = (split.test_end - split.time_point).days
    if days_in_period <= 0:
        raise ValueError(
            f"Split {split.split_index}: test_end ({split.test_end}) is not after"
            f" time_point ({split.time_point})"
        )

    eligible_assets = set(split.eligible_asset_ids)
    ndcg_scores: list[float] = []
    roi_scores: list[float] = []

    for customer_id in split.eligible_customer_ids:
        customer_recommendations = recommendations.get(customer_id, [])
        relevant_assets = (
            split.test_interactions.get(customer_id, set()) & eligible_assets
        )

        ndcg_scores.append(
            compute_ndcg_at_k(customer_recommendations, relevant_assets, k)
        )
        roi_scores.append(
            compute_roi_at_k(customer_recommendations, price_lookup, days_in_period, k)
        )

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    average_roi = sum(roi_scores) / len(roi_scores) if roi_scores else 0.0

    return EvaluationResult(
        split_index=split.split_index,
        time_point=split.time_point,
        model_name="",
        ndcg_at_k=average_ndcg,
        roi_at_k=average_roi,
    )
