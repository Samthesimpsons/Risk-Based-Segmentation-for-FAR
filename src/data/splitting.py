"""Temporal train/test split generation with eligibility filtering."""

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from typing import cast

import pandas as pd

from src.config.schemas import TemporalSplitData

EVALUATION_DATE_RANGE: tuple[date, date, int, int] = (
    date(2019, 8, 1),
    date(2022, 5, 23),
    68,
    13,
)


def _add_delta_transactions(
    interactions: dict[str, set[str]],
    buy_transactions: pd.DataFrame,
    previous_time_point: date | None,
    current_time_point: date,
) -> None:
    """Mutate interactions in place by adding transactions from the delta window."""
    if previous_time_point is None:
        mask = buy_transactions["timestamp"] < pd.Timestamp(current_time_point)
    else:
        mask = (buy_transactions["timestamp"] >= pd.Timestamp(previous_time_point)) & (
            buy_transactions["timestamp"] < pd.Timestamp(current_time_point)
        )

    delta = buy_transactions[mask]
    for customer_id, asset_id in zip(delta["customerID"], delta["ISIN"], strict=True):
        interactions[customer_id].add(asset_id)


def _build_test_interactions(
    buy_transactions: pd.DataFrame,
    window_start: date,
    window_end: date,
) -> dict[str, set[str]]:
    """Build test interaction sets for a given time window."""
    mask = (buy_transactions["timestamp"] >= pd.Timestamp(window_start)) & (
        buy_transactions["timestamp"] <= pd.Timestamp(window_end)
    )
    test_transactions = buy_transactions[mask]
    interactions: dict[str, set[str]] = defaultdict(set)
    for customer_id, asset_id in zip(
        test_transactions["customerID"], test_transactions["ISIN"], strict=True
    ):
        interactions[customer_id].add(asset_id)
    return dict(interactions)


def _deduplicate_test_interactions(
    training_interactions: dict[str, set[str]],
    test_interactions: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Remove assets from test that already appear in the user's training set."""
    deduplicated: dict[str, set[str]] = {}
    for customer_id, test_assets in test_interactions.items():
        training_assets = training_interactions.get(customer_id, set())
        novel_assets = test_assets - training_assets
        if novel_assets:
            deduplicated[customer_id] = novel_assets
    return deduplicated


def _filter_test_to_training_assets(
    training_interactions: dict[str, set[str]],
    test_interactions: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Keep only test assets that have appeared somewhere in the training set."""
    training_assets = {
        asset_id
        for user_assets in training_interactions.values()
        for asset_id in user_assets
    }

    filtered: dict[str, set[str]] = {}
    for customer_id, test_assets in test_interactions.items():
        overlapping_assets = test_assets & training_assets
        if overlapping_assets:
            filtered[customer_id] = overlapping_assets
    return filtered


def _filter_eligible_assets(
    close_prices_asset_sets: dict[date, set[str]],
    time_point: date,
    test_end: date,
) -> set[str]:
    """Return asset ISINs that have a close price on both the split date and test-end date."""
    return close_prices_asset_sets.get(time_point, set()) & close_prices_asset_sets.get(
        test_end, set()
    )


def _filter_eligible_customers(
    training_interactions: dict[str, set[str]],
    test_interactions: dict[str, set[str]],
) -> list[str]:
    """Return customers who have at least one interaction in both training and test."""
    return sorted(
        customer_id
        for customer_id in test_interactions
        if customer_id in training_interactions
    )


def _build_id_to_index(ids: list[str]) -> dict[str, int]:
    """Map a sorted list of IDs to contiguous integer indices starting at 0."""
    return {identifier: index for index, identifier in enumerate(ids)}


def _build_trading_grid_schedule(
    close_prices: pd.DataFrame,
    min_date: date,
    max_date: date,
    number_of_splits: int,
    number_of_future_steps: int,
) -> list[tuple[date, date]]:
    """Match the upstream FAR-Trans trading-date schedule generation."""
    mask = close_prices["timestamp"].between(
        pd.Timestamp(min_date), pd.Timestamp(max_date)
    )
    available_dates = list(pd.to_datetime(close_prices.loc[mask, "timestamp"].unique()))
    available_dates.sort()

    if not available_dates:
        raise ValueError(
            f"No trading dates available between {min_date} and {max_date}"
        )

    total_splits = number_of_splits + math.ceil(number_of_future_steps)
    division = max(1, math.floor(len(available_dates) / total_splits))

    partial_dates = [available_dates[0]]
    for index in range(1, total_splits):
        partial_dates.append(available_dates[index * division])
    partial_dates.append(available_dates[-1])

    schedule: list[tuple[date, date]] = []
    for index in range(0, len(partial_dates) - math.ceil(number_of_future_steps)):
        schedule.append(
            (
                partial_dates[index].date(),
                partial_dates[index + number_of_future_steps].date(),
            )
        )
    return schedule


def _build_schedule(
    close_prices: pd.DataFrame,
    explicit_schedule: Sequence[tuple[date, date]] | None,
) -> list[tuple[date, date]]:
    """Resolve the evaluation schedule from the trading-day grid or an explicit override."""
    if explicit_schedule is not None:
        return list(explicit_schedule)

    min_date, max_date, number_of_splits, future_steps = EVALUATION_DATE_RANGE
    return _build_trading_grid_schedule(
        close_prices, min_date, max_date, number_of_splits, future_steps
    )


def generate_all_splits(
    transactions: pd.DataFrame,
    close_prices: pd.DataFrame,
    explicit_schedule: Sequence[tuple[date, date]] | None = None,
) -> list[TemporalSplitData]:
    """Generate all temporal train/test splits with filtering."""
    buy_transactions = transactions[transactions["transactionType"] == "Buy"].copy()
    buy_transactions = buy_transactions.sort_values("timestamp").reset_index(drop=True)

    close_prices_asset_sets: dict[date, set[str]] = {}
    for timestamp, asset_rows in close_prices.groupby("timestamp"):
        timestamp_date = cast("pd.Timestamp", timestamp).date()
        close_prices_asset_sets[timestamp_date] = set(asset_rows["ISIN"].unique())

    schedule = _build_schedule(close_prices, explicit_schedule)

    cumulative_training: dict[str, set[str]] = defaultdict(set)
    previous_time_point: date | None = None
    splits: list[TemporalSplitData] = []

    for split_index, (time_point, test_end) in enumerate(schedule):
        _add_delta_transactions(
            cumulative_training, buy_transactions, previous_time_point, time_point
        )
        previous_time_point = time_point

        training_snapshot = copy.deepcopy(dict(cumulative_training))

        test_interactions = _build_test_interactions(
            buy_transactions, time_point, test_end
        )
        test_interactions = _deduplicate_test_interactions(
            training_snapshot, test_interactions
        )
        test_interactions = _filter_test_to_training_assets(
            training_snapshot, test_interactions
        )

        eligible_asset_set = _filter_eligible_assets(
            close_prices_asset_sets,
            time_point,
            test_end,
        )
        eligible_asset_ids = sorted(eligible_asset_set)

        eligible_customer_ids = _filter_eligible_customers(
            training_snapshot, test_interactions
        )

        customer_id_to_index = _build_id_to_index(eligible_customer_ids)
        asset_id_to_index = _build_id_to_index(eligible_asset_ids)

        splits.append(
            TemporalSplitData(
                split_index=split_index,
                time_point=time_point,
                test_end=test_end,
                training_interactions=training_snapshot,
                test_interactions=test_interactions,
                eligible_customer_ids=eligible_customer_ids,
                eligible_asset_ids=eligible_asset_ids,
                customer_id_to_index=customer_id_to_index,
                asset_id_to_index=asset_id_to_index,
            )
        )

    return splits
