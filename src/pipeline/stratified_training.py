"""Stratified profile-coherent LightGCN training and evaluation."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.schemas import EvaluationResult, TemporalSplitData
from src.config.settings import (
    DataPaths,
    ExperimentConfig,
    LightGCNConfig,
    ProfileCoherentLightGCNConfig,
)
from src.data.loading import load_assets, load_customers
from src.models.profile_coherent_light_gcn import ProfileCoherentLightGCN
from src.pipeline.preprocessing import (
    load_evaluation_splits,
    load_preprocessed_close_prices,
)
from src.utils.metrics import (
    build_price_lookup,
    compute_monthly_return,
    evaluate_model_on_split,
)
from src.utils.profile_coherence import (
    build_asset_risk_classes,
    build_customer_profile_lookup,
)

MODEL_NAME = "pc_lgcn"
TRIAL_LAMBDA_VALUES: tuple[float, ...] = (0.0, 1.0)
DEFAULT_CONFIGS_DIRECTORY = Path("outputs/configs")


def _set_random_seeds(seed: int) -> None:
    """Seed numpy, torch, and Python's random module for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(preferred_device: str) -> str:
    """Return the requested device, falling back to CPU when CUDA is unavailable."""
    if preferred_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred_device


def _load_best_light_gcn_config(
    configs_directory: Path,
) -> LightGCNConfig:
    """Load the best-LightGCN hyperparameters from the most recent baseline run."""
    if not configs_directory.exists():
        print(f"  No configs directory found at {configs_directory}; using defaults.")
        return LightGCNConfig()

    timestamped_directories = sorted(
        (path for path in configs_directory.iterdir() if path.is_dir()),
        reverse=True,
    )
    for directory in timestamped_directories:
        best_hyperparameters_path = directory / "best_hyperparameters.json"
        if not best_hyperparameters_path.exists():
            continue
        payload = json.loads(best_hyperparameters_path.read_text())
        light_gcn_payload = payload.get("light_gcn")
        if light_gcn_payload is None:
            continue
        print(
            f"  Loaded best LightGCN hyperparameters from {best_hyperparameters_path}"
        )
        return LightGCNConfig(**light_gcn_payload)

    print(
        f"  No best_hyperparameters.json with a light_gcn block found in "
        f"{configs_directory}; using defaults."
    )
    return LightGCNConfig()


def _build_pc_config(
    light_gcn_config: LightGCNConfig, coherence_loss_weight: float
) -> ProfileCoherentLightGCNConfig:
    """Layer the coherence loss weight on top of the inherited LightGCN backbone config."""
    return ProfileCoherentLightGCNConfig(
        embedding_dimension=light_gcn_config.embedding_dimension,
        number_of_layers=light_gcn_config.number_of_layers,
        learning_rate=light_gcn_config.learning_rate,
        weight_decay=light_gcn_config.weight_decay,
        keep_probability=light_gcn_config.keep_probability,
        number_of_epochs=light_gcn_config.number_of_epochs,
        batch_size=light_gcn_config.batch_size,
        coherence_loss_weight=coherence_loss_weight,
    )


def _generate_recommendations(
    model: ProfileCoherentLightGCN, split: TemporalSplitData, k: int
) -> dict[str, list[str]]:
    """Top-k recommendations for every eligible customer in the split."""
    return {
        customer_id: model.recommend_for_user(
            customer_id,
            split.training_interactions.get(customer_id, set()),
            k,
        )
        for customer_id in split.eligible_customer_ids
    }


def _summarise(per_split_results: list[EvaluationResult]) -> dict[str, float]:
    """Reduce per-split results to scalar averages."""
    if not per_split_results:
        return {
            "average_ndcg": 0.0,
            "average_roi": 0.0,
            "average_recall": 0.0,
            "average_profile_coherence": 0.0,
            "average_profile_coherence_lift": 0.0,
        }
    number_of_splits = len(per_split_results)
    return {
        "average_ndcg": sum(r.ndcg_at_k for r in per_split_results) / number_of_splits,
        "average_roi": sum(r.roi_at_k for r in per_split_results) / number_of_splits,
        "average_recall": sum(r.recall_at_k for r in per_split_results)
        / number_of_splits,
        "average_profile_coherence": sum(
            r.profile_coherence_at_k for r in per_split_results
        )
        / number_of_splits,
        "average_profile_coherence_lift": sum(
            r.profile_coherence_lift_at_k for r in per_split_results
        )
        / number_of_splits,
    }


def _save_per_split_metrics(
    config: ProfileCoherentLightGCNConfig,
    trial_id: str,
    per_split_results: list[EvaluationResult],
    trial_directory: Path,
) -> None:
    """Save per-split scalar metrics for one trial to its own directory."""
    trial_directory.mkdir(parents=True, exist_ok=True)
    config_payload = config.model_dump()
    rows = [
        {
            "trial_id": trial_id,
            **config_payload,
            "split_index": result.split_index,
            "time_point": result.time_point.isoformat(),
            "ndcg_at_k": result.ndcg_at_k,
            "roi_at_k": result.roi_at_k,
            "recall_at_k": result.recall_at_k,
            "profile_coherence_at_k": result.profile_coherence_at_k,
            "profile_coherence_lift_at_k": result.profile_coherence_lift_at_k,
        }
        for result in per_split_results
    ]
    pd.DataFrame(rows).to_csv(trial_directory / "per_split_metrics.csv", index=False)


def _save_recommendations(
    config: ProfileCoherentLightGCNConfig,
    trial_id: str,
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]],
    close_prices: pd.DataFrame,
    trial_directory: Path,
) -> None:
    """Save flat per-recommendation rows for one trial as parquet."""
    trial_directory.mkdir(parents=True, exist_ok=True)
    config_payload = config.model_dump()

    columns: dict[str, list[Any]] = {"trial_id": []}
    for key in config_payload:
        columns[key] = []
    for column_name in (
        "split_index",
        "time_point",
        "test_end",
        "customer_id",
        "rank",
        "asset_id",
        "monthly_return",
        "is_relevant",
    ):
        columns[column_name] = []

    for split, recommendations in recommendations_per_split:
        price_lookup = build_price_lookup(
            close_prices,
            split.time_point,
            split.test_end,
            split.eligible_asset_ids,
        )
        days_in_period = (split.test_end - split.time_point).days
        eligible_assets = set(split.eligible_asset_ids)
        time_point_iso = split.time_point.isoformat()
        test_end_iso = split.test_end.isoformat()

        for customer_id in split.eligible_customer_ids:
            relevant_assets = (
                split.test_interactions.get(customer_id, set()) & eligible_assets
            )
            customer_recommendations = recommendations.get(customer_id, [])
            for rank_index, asset_id in enumerate(customer_recommendations, start=1):
                start_price, end_price = price_lookup.get(asset_id, (0.0, 0.0))
                columns["trial_id"].append(trial_id)
                for key, value in config_payload.items():
                    columns[key].append(value)
                columns["split_index"].append(split.split_index)
                columns["time_point"].append(time_point_iso)
                columns["test_end"].append(test_end_iso)
                columns["customer_id"].append(customer_id)
                columns["rank"].append(rank_index)
                columns["asset_id"].append(asset_id)
                columns["monthly_return"].append(
                    compute_monthly_return(start_price, end_price, days_in_period)
                )
                columns["is_relevant"].append(asset_id in relevant_assets)

    pd.DataFrame(columns).to_parquet(
        trial_directory / "recommendations.parquet", index=False
    )


def _save_trial_summary(
    timestamp: str,
    rows: list[dict[str, Any]],
    results_directory: Path,
) -> Path:
    """Save one row per stratified configuration with averaged metrics."""
    output_directory = results_directory / "tuning" / MODEL_NAME
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def run_stratified_training(
    splits_directory: Path = Path("data/splits"),
    results_directory: Path = Path("outputs/results"),
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    splits_limit: int | None = None,
    lambda_values: tuple[float, ...] = TRIAL_LAMBDA_VALUES,
    configs_directory: Path = DEFAULT_CONFIGS_DIRECTORY,
) -> str:
    """Train and evaluate the stratified PC-LightGCN configurations and persist artefacts."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()

    print("Loading evaluation splits...")
    splits = load_evaluation_splits(splits_directory)
    if splits_limit is not None:
        splits = splits[:splits_limit]
    print(f"  Loaded {len(splits)} evaluation splits")

    print("Loading close prices...")
    close_prices = load_preprocessed_close_prices(splits_directory)

    print("Loading customer profiles and asset risk classes...")
    customers = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    customer_profiles = build_customer_profile_lookup(customers)
    assets = load_assets(data_paths.data_directory / data_paths.asset_information_file)
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print("Loading best LightGCN backbone hyperparameters...")
    backbone_config = _load_best_light_gcn_config(configs_directory)
    print(f"  Backbone: {backbone_config}")

    device = _resolve_device(experiment_config.device)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        config = _build_pc_config(backbone_config, lambda_value)
        trial_id = f"stratified_lambda_{lambda_value:.1f}"
        print(f"\n=== Trial {trial_id} (coherence_loss_weight={lambda_value}) ===")
        _set_random_seeds(experiment_config.seed)

        per_split_results: list[EvaluationResult] = []
        recommendations_per_split: list[
            tuple[TemporalSplitData, dict[str, list[str]]]
        ] = []
        for split in splits:
            print(
                f"  split {split.split_index} ({split.time_point} -> {split.test_end})"
            )
            model = ProfileCoherentLightGCN(
                config=config,
                customer_profiles=customer_profiles,
                asset_risk_classes=asset_risk_classes,
            )
            model.train_on_split(split, device=device)
            recommendations = _generate_recommendations(
                model, split, experiment_config.top_k
            )
            result = evaluate_model_on_split(
                recommendations,
                split,
                close_prices,
                customer_profiles,
                asset_risk_classes,
                experiment_config.top_k,
            )
            per_split_results.append(
                result.model_copy(update={"model_name": model.name})
            )
            recommendations_per_split.append((split, recommendations))

        trial_directory = (
            results_directory / "evaluation" / MODEL_NAME / run_timestamp / trial_id
        )
        _save_per_split_metrics(config, trial_id, per_split_results, trial_directory)
        _save_recommendations(
            config, trial_id, recommendations_per_split, close_prices, trial_directory
        )

        averages = _summarise(per_split_results)
        summary_rows.append(
            {
                "trial_id": trial_id,
                **config.model_dump(),
                **averages,
            }
        )
        print(
            f"  averages: ndcg={averages['average_ndcg']:.4f} "
            f"roi={averages['average_roi']:.4f} "
            f"recall={averages['average_recall']:.4f} "
            f"pc={averages['average_profile_coherence']:.4f} "
            f"pc_lift={averages['average_profile_coherence_lift']:.4f}"
        )

    summary_path = _save_trial_summary(run_timestamp, summary_rows, results_directory)
    print(f"\nTrial summary saved to {summary_path}")
    print(
        f"Per-trial artefacts under "
        f"{results_directory / 'evaluation' / MODEL_NAME / run_timestamp}"
    )
    return run_timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified profile-coherent LightGCN training and evaluation."
    )
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits-limit", type=int, default=None)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DataPaths().data_directory),
        help="Directory of customer and asset CSVs (use data/regrouped for the regrouping extension).",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=str(DEFAULT_CONFIGS_DIRECTORY),
        help="Directory of timestamped baseline configs to inherit hyperparameters from.",
    )
    arguments = parser.parse_args()

    run_stratified_training(
        splits_directory=Path(arguments.splits_dir),
        results_directory=Path(arguments.results_dir),
        experiment_config=ExperimentConfig(device=arguments.device),
        data_paths=DataPaths(data_directory=Path(arguments.data_dir)),
        splits_limit=arguments.splits_limit,
        configs_directory=Path(arguments.configs_dir),
    )
