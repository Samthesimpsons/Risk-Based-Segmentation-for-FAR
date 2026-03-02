"""Chronological purchase sequence construction and time-encoding utilities."""

import math
from collections import defaultdict
from datetime import date

import pandas as pd


def build_user_sequences(
    buy_transactions: pd.DataFrame,
    cutoff_date: date,
) -> dict[str, list[tuple[str, date]]]:
    """Build chronologically ordered purchase sequences per user from buy transactions.

    Includes repeat purchases of the same asset (unlike interaction sets which deduplicate).
    """
    mask = buy_transactions["timestamp"] < pd.Timestamp(cutoff_date)
    filtered = buy_transactions[mask].sort_values("timestamp")

    sequences: dict[str, list[tuple[str, date]]] = defaultdict(list)
    for customer_id, asset_id, timestamp in zip(
        filtered["customerID"], filtered["ISIN"], filtered["timestamp"], strict=True
    ):
        sequences[customer_id].append((asset_id, timestamp.date()))

    return dict(sequences)


def truncate_sequences(
    sequences: dict[str, list[tuple[str, date]]],
    max_length: int,
) -> dict[str, list[tuple[str, date]]]:
    """Keep only the last max_length items per user sequence."""
    return {user_id: sequence[-max_length:] for user_id, sequence in sequences.items()}


def compute_relative_time_intervals(timestamps: list[date]) -> list[int]:
    """Compute days between consecutive timestamps. First element is 0."""
    if len(timestamps) <= 1:
        return [0] * len(timestamps)

    intervals = [0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).days
        intervals.append(max(0, delta))
    return intervals


def compute_absolute_positions(
    timestamps: list[date],
    reference_date: date,
) -> list[int]:
    """Compute days since reference_date for each timestamp."""
    return [max(0, (timestamp - reference_date).days) for timestamp in timestamps]


def bucket_time_values(values: list[int], bucket_count: int) -> list[int]:
    """Log-scale bucketing of time values into [0, bucket_count)."""
    if bucket_count <= 1:
        return [0] * len(values)

    max_bucket = bucket_count - 1
    result: list[int] = []
    for value in values:
        if value <= 0:
            result.append(0)
        else:
            bucketed = int(math.log1p(value) / math.log1p(max_bucket) * max_bucket)
            result.append(min(bucketed, max_bucket))
    return result
