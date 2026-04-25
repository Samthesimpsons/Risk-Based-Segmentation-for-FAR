"""Panel OLS of profile coherence on declared band x model with cluster-robust SEs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.baseline_decomposition import (
    _annotate_recommendations,
    _average_per_trial,
    _discover_run_directories,
    _load_per_split_metrics,
    _select_best_trial_id,
)
from src.config.settings import DataPaths
from src.data.loading import load_assets, load_close_prices, load_customers
from src.pipeline.baseline_evaluation import GRID_SPECS, _PRIMARY_METRIC_TO_KEY
from src.pipeline.profile_coherent_evaluation import (
    MODEL_NAME as PROFILE_COHERENT_MODEL_NAME,
    PROFILE_COHERENT_GRID_SPEC,
)
from src.profile_coherence.customer_profile import build_customer_profile_lookup
from src.profile_coherence.risk_classification import build_asset_risk_classes

DEFAULT_EVALUATION_DIRECTORY = Path("outputs/results/evaluation")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/panel_regression")

_BAND_LABELS: dict[int, str] = {
    0: "Conservative",
    1: "Income",
    2: "Balanced",
    3: "Aggressive",
}

_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "light_gcn": "LightGCN",
    PROFILE_COHERENT_MODEL_NAME: "Profile-Coherent LightGCN",
}


def _best_trial_recommendations(
    run_directory: Path, primary_metric_key: str
) -> tuple[str, pd.DataFrame] | None:
    """Return the best trial's id and recommendations parquet under one run dir."""
    per_split = _load_per_split_metrics(run_directory)
    if per_split.empty:
        return None
    aggregated = _average_per_trial(per_split)
    best_trial_id = _select_best_trial_id(aggregated, primary_metric_key)
    if best_trial_id is None:
        return None
    parquet_path = run_directory / best_trial_id / "recommendations.parquet"
    if not parquet_path.exists():
        return None
    return best_trial_id, pd.read_parquet(parquet_path)


def _build_panel_dataframe(
    evaluation_directory: Path,
    run_timestamp: str | None,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Assemble one row per (model, split, customer) with coherent_share + declared_band."""
    runs = _discover_run_directories(evaluation_directory, run_timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No evaluation runs found under {evaluation_directory}."
        )

    all_specs = {**GRID_SPECS, PROFILE_COHERENT_MODEL_NAME: PROFILE_COHERENT_GRID_SPEC}
    frames: list[pd.DataFrame] = []
    for model_name, run_directory in runs.items():
        if model_name not in all_specs:
            print(f"  Skipping unknown model directory '{model_name}'")
            continue
        primary_metric_key = _PRIMARY_METRIC_TO_KEY[
            all_specs[model_name].primary_metric
        ]
        result = _best_trial_recommendations(run_directory, primary_metric_key)
        if result is None:
            print(f"  No usable best trial for {model_name}; skipping.")
            continue
        best_trial_id, recommendations = result
        print(
            f"  {model_name}: best trial {best_trial_id} ({len(recommendations)} rows)"
        )

        annotated = _annotate_recommendations(
            recommendations, customer_profiles, asset_risk_classes
        )
        per_customer_split = (
            annotated.groupby(["customer_id", "split_index"], dropna=False)
            .agg(
                coherent_share=(
                    "is_coherent",
                    lambda series: float(series.fillna(False).mean()),
                ),
                declared_band=("customer_band", "first"),
            )
            .reset_index()
        )
        per_customer_split["model"] = model_name
        frames.append(per_customer_split)

    if not frames:
        raise RuntimeError("No model produced a usable best-trial parquet.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["declared_band"])
    panel["declared_band"] = panel["declared_band"].astype(int)
    panel["band_label"] = panel["declared_band"].map(_BAND_LABELS)
    return panel


def _fit_panel_ols(panel: pd.DataFrame) -> Any:
    """Fit `coherent_share ~ C(declared_band) * C(model) + C(split_index)` with cluster SEs."""
    customer_codes = panel["customer_id"].astype("category").cat.codes
    return smf.ols(
        "coherent_share ~ C(declared_band) * C(model) + C(split_index)",
        data=panel,
    ).fit(cov_type="cluster", cov_kwds={"groups": customer_codes.values})


def _coefficients_dataframe(model_fit: Any) -> pd.DataFrame:
    """Return a tidy coefficients table with estimate, SE, t, p, and 95% CI."""
    confidence_intervals = model_fit.conf_int()
    confidence_intervals.columns = ["ci_lower", "ci_upper"]
    table = pd.DataFrame(
        {
            "term": model_fit.params.index,
            "estimate": model_fit.params.values,
            "std_error": model_fit.bse.values,
            "t_value": model_fit.tvalues.values,
            "p_value": model_fit.pvalues.values,
        }
    )
    return table.merge(
        confidence_intervals.reset_index().rename(columns={"index": "term"}),
        on="term",
        how="left",
    )


def _predicted_pc_grid(panel: pd.DataFrame, model_fit: Any) -> pd.DataFrame:
    """Predict PC@10 for every observed (band, model) at the median split."""
    median_split = int(panel["split_index"].median())
    bands = sorted(panel["declared_band"].unique())
    models = sorted(panel["model"].unique())
    grid = pd.DataFrame(
        [
            {
                "declared_band": band,
                "model": model_name,
                "split_index": median_split,
            }
            for band in bands
            for model_name in models
        ]
    )

    predictions = model_fit.get_prediction(grid)
    summary_frame = predictions.summary_frame(alpha=0.05)
    grid["predicted_pc"] = summary_frame["mean"].to_numpy()
    grid["ci_lower"] = summary_frame["mean_ci_lower"].to_numpy()
    grid["ci_upper"] = summary_frame["mean_ci_upper"].to_numpy()
    grid["band_label"] = grid["declared_band"].map(_BAND_LABELS)
    grid["model_display"] = grid["model"].map(
        lambda key: _MODEL_DISPLAY_NAMES.get(key, key)
    )
    return grid


def _save_forest_figure(prediction_grid: pd.DataFrame, output_path: Path) -> None:
    """Render predicted PC@10 with 95% CIs, grouped by band and coloured by model."""
    bands = sorted(prediction_grid["declared_band"].unique())
    models = sorted(prediction_grid["model"].unique())
    band_positions = np.arange(len(bands))
    bar_offset = 0.18
    figure, axis = plt.subplots(figsize=(8, 4.5))

    for model_index, model_name in enumerate(models):
        subset = prediction_grid[prediction_grid["model"] == model_name].sort_values(
            "declared_band"
        )
        x_offsets = band_positions + (model_index - (len(models) - 1) / 2) * bar_offset
        lower_error = subset["predicted_pc"].to_numpy() - subset["ci_lower"].to_numpy()
        upper_error = subset["ci_upper"].to_numpy() - subset["predicted_pc"].to_numpy()
        axis.errorbar(
            x_offsets,
            subset["predicted_pc"].to_numpy(),
            yerr=[lower_error, upper_error],
            fmt="o",
            capsize=4,
            label=_MODEL_DISPLAY_NAMES.get(model_name, model_name),
        )

    axis.set_xticks(band_positions)
    axis.set_xticklabels([_BAND_LABELS[band] for band in bands])
    axis.set_xlabel("Declared MiFID II risk band")
    axis.set_ylabel("Predicted PC@10 (95% CI)")
    axis.set_title("Profile coherence by declared band and model")
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def run_panel_regression(
    evaluation_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
) -> dict[str, Any]:
    """Fit the panel OLS, save coefficients + predictions + forest figure."""
    data_paths = data_paths or DataPaths()
    print("Loading customer profiles and asset risk classes...")
    customers = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    customer_profiles = build_customer_profile_lookup(customers)
    assets = load_assets(data_paths.data_directory / data_paths.asset_information_file)
    close_prices = load_close_prices(
        data_paths.data_directory / data_paths.close_prices_file
    )
    asset_risk_classes = build_asset_risk_classes(assets, close_prices)

    print(f"Building panel from {evaluation_directory} ...")
    panel = _build_panel_dataframe(
        evaluation_directory=evaluation_directory,
        run_timestamp=run_timestamp,
        customer_profiles=customer_profiles,
        asset_risk_classes=asset_risk_classes,
    )
    print(
        f"Panel: {len(panel)} rows across "
        f"{panel['model'].nunique()} models, "
        f"{panel['declared_band'].nunique()} bands, "
        f"{panel['customer_id'].nunique()} customers, "
        f"{panel['split_index'].nunique()} splits."
    )

    print("Fitting OLS with cluster-robust SEs (clustered on customer_id) ...")
    model_fit = _fit_panel_ols(panel)
    coefficients = _coefficients_dataframe(model_fit)
    prediction_grid = _predicted_pc_grid(panel, model_fit)

    output_directory = output_root / (
        run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_directory / "coefficients.csv", index=False)
    prediction_grid.to_csv(
        output_directory / "predicted_pc_by_band_model.csv", index=False
    )
    _save_forest_figure(prediction_grid, output_directory / "forest_predicted_pc.png")
    (output_directory / "panel.csv").write_text(panel.to_csv(index=False))
    (output_directory / "regression_summary.txt").write_text(str(model_fit.summary()))

    print(f"\nPanel regression artefacts saved to {output_directory}")
    print("\nKey coefficients (interaction terms):")
    interaction_mask = coefficients["term"].str.contains(":", regex=False)
    print(
        coefficients[interaction_mask]
        .loc[:, ["term", "estimate", "std_error", "p_value", "ci_lower", "ci_upper"]]
        .to_string(index=False)
    )

    return {
        "output_directory": str(output_directory),
        "panel_rows": int(len(panel)),
        "model_summary": str(model_fit.summary()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panel OLS of profile coherence on declared band x model."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIRECTORY,
        help="Directory containing per-model per-run evaluation outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write panel-regression artefacts.",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help=(
            "Specific run timestamp (matches directory name under each model). "
            "Default: pick each model's latest."
        ),
    )
    arguments = parser.parse_args()

    run_panel_regression(
        evaluation_directory=arguments.evaluation_dir,
        output_root=arguments.output_root,
        run_timestamp=arguments.run_timestamp,
    )
