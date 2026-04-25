"""Ray-driven Profile-Coherent LightGCN ablation + lambda-sensitivity sweep."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import ray
from ray import tune

from src.config.settings import (
    DataPaths,
    ExperimentConfig,
    ProfileCoherentLightGCNConfig,
)
from src.models.profile_coherent_lgcn import ProfileCoherentLightGCNBaseline
from src.pipeline.baseline_evaluation import (
    EvaluationContext,
    GridSpec,
    _PRIMARY_METRIC_TO_KEY,
    _build_evaluation_context,
    _evaluate_over_splits,
    _ray_runtime_env,
    _resolve_device,
    _save_per_trial_metrics_csv,
    _save_recommendations_parquet,
    _set_random_seeds,
    _summarise,
)

MODEL_NAME = "profile_coherent_light_gcn"
_PRIMARY_METRIC = "ndcg"
_PRIMARY_METRIC_KEY = _PRIMARY_METRIC_TO_KEY[_PRIMARY_METRIC]
_DEFAULT_LAMBDA = 0.5
_LAMBDA_SENSITIVITY_VALUES: list[float] = [0.1, 1.0, 2.0]
_BACKBONE_KEYS: tuple[str, ...] = (
    "embedding_dimension",
    "number_of_layers",
    "learning_rate",
    "weight_decay",
    "keep_probability",
    "number_of_epochs",
    "batch_size",
)

PROFILE_COHERENT_GRID_SPEC = GridSpec(
    model_name=MODEL_NAME,
    config_class=ProfileCoherentLightGCNConfig,
    grid={},
    primary_metric=_PRIMARY_METRIC,
    needs_indicators=False,
    use_gpu_per_trial=True,
    max_concurrent_trials=4,
)


def _resolve_baseline_best_config_path(
    explicit_path: Path | None,
    configs_root: Path = Path("outputs/configs"),
) -> Path:
    """Pick the latest baseline best-config JSON if no explicit path was given."""
    if explicit_path is not None:
        return explicit_path
    if not configs_root.exists():
        raise FileNotFoundError(
            f"No {configs_root} directory; pass --baseline-best-config-path explicitly."
        )
    timestamped_dirs = sorted(
        [path for path in configs_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    if not timestamped_dirs:
        raise FileNotFoundError(
            f"No timestamped subdirectories under {configs_root}; "
            "pass --baseline-best-config-path explicitly."
        )
    return timestamped_dirs[-1] / "best_hyperparameters.json"


def _load_winning_lightgcn_backbone(best_config_path: Path) -> dict[str, Any]:
    """Read the LightGCN backbone hyperparameters from the baseline best-config JSON."""
    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Baseline best-config JSON not found at {best_config_path}. "
            "Run `uv run poe evaluate-baselines` first or pass "
            "--baseline-best-config-path."
        )
    payload = json.loads(best_config_path.read_text())
    if "light_gcn" not in payload:
        raise KeyError(
            f"'light_gcn' missing from {best_config_path}. "
            "Re-run baselines so the JSON includes the LightGCN winner."
        )
    backbone = payload["light_gcn"]
    missing = [key for key in _BACKBONE_KEYS if key not in backbone]
    if missing:
        raise KeyError(f"LightGCN backbone is missing keys: {missing}")
    return {key: backbone[key] for key in _BACKBONE_KEYS}


def _build_trial_configs(
    backbone: dict[str, Any], include_lambda_sweep: bool
) -> list[dict[str, Any]]:
    """Materialise the 2x2 ablation cells (and optional lambda sensitivity) as full configs."""
    cells: list[dict[str, Any]] = []
    for embedding_enabled in (False, True):
        for coherence_enabled in (False, True):
            cells.append(
                {
                    **backbone,
                    "profile_embedding_enabled": embedding_enabled,
                    "profile_coherence_enabled": coherence_enabled,
                    "profile_coherence_lambda": _DEFAULT_LAMBDA,
                }
            )
    if include_lambda_sweep:
        for value in _LAMBDA_SENSITIVITY_VALUES:
            cells.append(
                {
                    **backbone,
                    "profile_embedding_enabled": True,
                    "profile_coherence_enabled": True,
                    "profile_coherence_lambda": value,
                }
            )
    return cells


def _make_trainable(context_ref: ray.ObjectRef, evaluation_dir: Path):
    """Trainable that instantiates PC-LGCN, evaluates over 69 splits, and persists outputs."""

    def trainable(passed: dict[str, Any]) -> None:
        hyperparameters: dict[str, Any] = passed["hyperparams"]
        context: EvaluationContext = ray.get(context_ref)
        config = ProfileCoherentLightGCNConfig(**hyperparameters)
        _set_random_seeds(0)
        model = ProfileCoherentLightGCNBaseline(
            config=config,
            customer_profiles=context.customer_profiles,
            asset_risk_classes=context.asset_risk_classes,
        )
        evaluation = _evaluate_over_splits(model, context)
        averages = _summarise(evaluation.per_split_results)

        trial_id = tune.get_context().get_trial_id()
        trial_directory = evaluation_dir / trial_id
        _save_per_trial_metrics_csv(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            trial_directory=trial_directory,
        )
        _save_recommendations_parquet(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            close_prices=context.close_prices,
            trial_directory=trial_directory,
        )
        tune.report(averages)

    return trainable


def _save_trial_summary(
    result_grid: tune.ResultGrid,
    results_directory: Path,
    timestamp: str,
) -> Path:
    """Save one row per trial with full hyperparameters and the four averaged metrics."""
    rows: list[dict[str, Any]] = []
    for result in result_grid:
        if result.metrics is None or result.config is None:
            continue
        cell = dict(result.config).get("hyperparams", {})
        row: dict[str, Any] = dict(cell)
        row["average_ndcg"] = result.metrics.get("average_ndcg")
        row["average_roi"] = result.metrics.get("average_roi")
        row["average_recall"] = result.metrics.get("average_recall")
        row["average_profile_coherence"] = result.metrics.get(
            "average_profile_coherence"
        )
        rows.append(row)

    output_directory = results_directory / "tuning" / MODEL_NAME
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def run_profile_coherent_grid(
    splits_directory: Path,
    results_directory: Path,
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    baseline_best_config_path: Path | None = None,
    include_lambda_sweep: bool = True,
    splits_limit: int | None = None,
    max_concurrent_trials: int = 4,
) -> ProfileCoherentLightGCNConfig:
    """Run the PC-LGCN ablation + sensitivity grid, persist outputs, return best cell."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()

    project_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = _resolve_device(experiment_config.device)

    backbone_path = _resolve_baseline_best_config_path(baseline_best_config_path)
    print(f"Loading winning LightGCN backbone from {backbone_path}")
    backbone = _load_winning_lightgcn_backbone(backbone_path)
    print(f"  Backbone: {backbone}")

    trial_configs = _build_trial_configs(backbone, include_lambda_sweep)
    sensitivity_count = len(trial_configs) - 4
    print(
        f"Configured {len(trial_configs)} trials"
        f" (4 ablation + {sensitivity_count} lambda sensitivity)"
    )

    context = _build_evaluation_context(
        splits_directory=splits_directory,
        data_paths=data_paths,
        needs_indicators=False,
        splits_limit=splits_limit,
        top_k=experiment_config.top_k,
        device=device,
        project_root=project_root,
        run_timestamp=run_timestamp,
    )

    os.environ["RAY_RUNTIME_ENV_LOCAL_DEV_MODE"] = "1"
    os.environ["RAY_ENABLE_LOG_MONITOR"] = "0"
    ray.init(
        log_to_driver=False,
        logging_level="error",
        include_dashboard=False,
        _enable_object_reconstruction=False,
        object_store_memory=1_073_741_824,
        runtime_env=_ray_runtime_env(project_root),
    )

    evaluation_dir = (
        results_directory / "evaluation" / MODEL_NAME / run_timestamp
    ).resolve()
    use_gpu = device != "cpu"
    resources: dict[str, float] = (
        {"gpu": 1.0 / max(1, max_concurrent_trials), "cpu": 1.0}
        if use_gpu
        else {"cpu": 1.0}
    )

    try:
        context_ref = ray.put(context)
        trainable = _make_trainable(context_ref, evaluation_dir)
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources),
            param_space={"hyperparams": tune.grid_search(trial_configs)},
            tune_config=tune.TuneConfig(
                metric=_PRIMARY_METRIC_KEY,
                mode="max",
                num_samples=1,
                max_concurrent_trials=max_concurrent_trials,
            ),
            run_config=tune.RunConfig(
                name=f"{MODEL_NAME}_grid_search",
                verbose=1,
            ),
        )
        result_grid = tuner.fit()
    finally:
        ray.shutdown()

    summary_path = _save_trial_summary(result_grid, results_directory, run_timestamp)
    print(f"  Trial summary saved to {summary_path}")

    best_result = result_grid.get_best_result(metric=_PRIMARY_METRIC_KEY, mode="max")
    assert best_result.config is not None
    best_cell = dict(best_result.config)["hyperparams"]
    best_config = ProfileCoherentLightGCNConfig(**best_cell)
    print(f"\nBest PC-LGCN cell by {_PRIMARY_METRIC_KEY}: {best_config}")

    from src.analysis.baseline_decomposition import run_decomposition

    print("\nRunning post-evaluation decomposition (with PC-LGCN included)...")
    run_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=run_timestamp,
        data_paths=data_paths,
        top_k=experiment_config.top_k,
        extra_grid_specs={MODEL_NAME: PROFILE_COHERENT_GRID_SPEC},
    )

    from src.analysis.profile_coherent_decomposition import (
        run_profile_coherent_decomposition,
    )

    print("\nRunning Profile-Coherent ablation + sensitivity decomposition...")
    run_profile_coherent_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=run_timestamp,
        data_paths=data_paths,
        top_k=experiment_config.top_k,
    )
    return best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Profile-Coherent LightGCN ablation + lambda sensitivity sweep. "
            "Reads the winning LightGCN backbone from the baseline best-config "
            "JSON and runs a 4-cell ablation plus a small lambda grid."
        )
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing preprocessed splits (default: data/splits)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/results",
        help="Directory to write per-trial CSVs (default: outputs/results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--baseline-best-config-path",
        type=Path,
        default=None,
        help=(
            "Path to baseline best_hyperparameters.json. "
            "Default: latest under outputs/configs/."
        ),
    )
    parser.add_argument(
        "--no-lambda-sweep",
        action="store_true",
        help="Skip the lambda sensitivity sweep; run only the 4-cell ablation.",
    )
    parser.add_argument(
        "--splits-limit",
        type=int,
        default=None,
        help="Run only the first N splits (smoke test). Default: all 69 splits.",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=4,
        help=(
            "Cap concurrent Ray trials. Default 4 matches the L40S fractional "
            "GPU split used by the baseline LightGCN sweep."
        ),
    )
    args = parser.parse_args()

    run_profile_coherent_grid(
        splits_directory=Path(args.splits_dir),
        results_directory=Path(args.results_dir),
        experiment_config=ExperimentConfig(device=args.device),
        baseline_best_config_path=args.baseline_best_config_path,
        include_lambda_sweep=not args.no_lambda_sweep,
        splits_limit=args.splits_limit,
        max_concurrent_trials=args.max_concurrent_trials,
    )
