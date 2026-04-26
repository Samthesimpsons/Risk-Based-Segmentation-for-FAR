"""Preprocessing pipeline: data loading, split generation, and serialization."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.config.schemas import TemporalSplitData
from src.config.settings import DataPaths
from src.data.loading import load_all, load_close_prices
from src.data.splitting import generate_all_splits


@dataclass(frozen=True)
class ValidationSchedule:
    """Schedule of validation recommendation dates."""

    dates: tuple[date, ...]


VALIDATION_SCHEDULE = ValidationSchedule(
    dates=(
        date(2019, 4, 1),
        date(2019, 10, 1),
        date(2020, 1, 31),
    ),
)


def _save_split(split: TemporalSplitData, path: Path) -> None:
    """Serialize a single TemporalSplitData to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(split.model_dump_json(indent=2))


def _load_split(path: Path) -> TemporalSplitData:
    """Deserialize a single TemporalSplitData from a JSON file."""
    return TemporalSplitData.model_validate_json(path.read_text())


def save_splits(
    splits: list[TemporalSplitData],
    directory: Path,
) -> None:
    """Save a list of splits to numbered JSON files in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        filename = f"split_{split.split_index:03d}.json"
        _save_split(split, directory / filename)


def load_splits(directory: Path) -> list[TemporalSplitData]:
    """Load all split JSON files from a directory, sorted by split index."""
    split_files = sorted(directory.glob("split_*.json"))
    return [_load_split(path) for path in split_files]


def load_evaluation_splits(splits_root: Path) -> list[TemporalSplitData]:
    """Load evaluation splits from the preprocessing output directory."""
    return load_splits(splits_root / "evaluation")


def load_preprocessed_close_prices(splits_root: Path) -> pd.DataFrame:
    """Load close prices using the path stored in preprocessing metadata."""
    metadata_path = splits_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    close_prices_path = Path(metadata["close_prices_path"])
    return load_close_prices(close_prices_path)


def _snap_to_trading_day(
    target: date, trading_days: list[date], mode: Literal["nearest", "next"]
) -> date:
    """Snap to a trading day: nearest absolute distance, or first on/after target."""
    if not trading_days:
        raise ValueError("trading_days must not be empty")

    if mode == "next":
        for trading_day in trading_days:
            if trading_day >= target:
                return trading_day
        return trading_days[-1]

    return min(trading_days, key=lambda d: abs((d - target).days))


def _generate_validation_splits(
    transactions: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> list[TemporalSplitData]:
    """Generate splits at the validation dates, snapped to trading days."""
    trading_days = sorted(
        {timestamp.date() for timestamp in close_prices["timestamp"].unique()}
    )

    validation_splits: list[TemporalSplitData] = []

    for index, validation_date in enumerate(VALIDATION_SCHEDULE.dates):
        snapped_time_point = _snap_to_trading_day(
            validation_date, trading_days, mode="nearest"
        )
        target_test_end = snapped_time_point + relativedelta(months=6)
        snapped_test_end = _snap_to_trading_day(
            target_test_end, trading_days, mode="next"
        )

        splits = generate_all_splits(
            transactions,
            close_prices,
            explicit_schedule=[(snapped_time_point, snapped_test_end)],
        )

        if not splits:
            continue

        split_with_correct_index = splits[0].model_copy(update={"split_index": index})
        validation_splits.append(split_with_correct_index)

    return validation_splits


def _save_metadata(
    output_directory: Path,
    close_prices_path: Path,
    number_of_evaluation_splits: int,
    number_of_validation_splits: int,
) -> None:
    """Save preprocessing metadata to a JSON file."""
    metadata = {
        "validation_dates": [d.isoformat() for d in VALIDATION_SCHEDULE.dates],
        "number_of_evaluation_splits": number_of_evaluation_splits,
        "number_of_validation_splits": number_of_validation_splits,
        "close_prices_path": str(close_prices_path),
    }
    (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run_preprocessing(
    output_directory: Path,
    data_paths: DataPaths | None = None,
) -> None:
    """Load raw data, generate all splits, and save to disk."""
    data_paths = data_paths or DataPaths()

    print("Loading raw data...")
    datasets = load_all(data_paths)
    transactions = datasets["transactions"]
    close_prices = datasets["close_prices"]

    close_prices_path = data_paths.data_directory / data_paths.close_prices_file

    print("Generating evaluation splits...")
    evaluation_splits = generate_all_splits(transactions, close_prices)
    save_splits(evaluation_splits, output_directory / "evaluation")
    print(f"  Saved {len(evaluation_splits)} evaluation splits")

    print("Generating 3 validation splits...")
    validation_splits = _generate_validation_splits(transactions, close_prices)
    save_splits(validation_splits, output_directory / "validation")
    print(f"  Saved {len(validation_splits)} validation splits")

    _save_metadata(
        output_directory,
        close_prices_path,
        len(evaluation_splits),
        len(validation_splits),
    )

    unique_customers: set[str] = set()
    unique_assets: set[str] = set()
    for split in evaluation_splits:
        unique_customers.update(split.eligible_customer_ids)
        unique_assets.update(split.eligible_asset_ids)

    print(f"\nPreprocessing complete: {output_directory}/")
    print(f"  Evaluation splits: {len(evaluation_splits)}")
    print(f"  Validation splits: {len(validation_splits)}")
    print(f"  Unique eligible customers: {len(unique_customers)}")
    print(f"  Unique eligible assets: {len(unique_assets)}")


if __name__ == "__main__":
    run_preprocessing(output_directory=Path("data/splits"))
