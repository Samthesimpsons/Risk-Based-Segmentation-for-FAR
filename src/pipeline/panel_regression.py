"""Per-(customer, split, model) profile-coherence panel for the band-decomposition plot."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.annotate import annotate_recommendations
from src.analysis.run_artefacts import (
    best_trial_recommendations,
    discover_run_directories,
)
from src.config.registry import (
    GRID_SPECS,
    PRIMARY_METRIC_TO_KEY,
)
from src.config.settings import DataPaths
from src.data.loading import load_assets, load_close_prices, load_customers
from src.utils.profile_coherence import (
    RISK_BAND_NAMES,
    build_asset_risk_classes,
    build_customer_profile_lookup,
)

DEFAULT_EVALUATION_DIRECTORY = Path("outputs/results/evaluation")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/panel_regression")


def _build_panel(
    evaluation_directory: Path,
    run_timestamp: str | None,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Assemble one row per (model, split, customer) with coherent_share and declared_band."""
    runs = discover_run_directories(evaluation_directory, run_timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No evaluation runs found under {evaluation_directory}."
        )

    frames: list[pd.DataFrame] = []
    for model_name, run_directory in runs.items():
        if model_name not in GRID_SPECS:
            print(f"  Skipping unknown model directory '{model_name}'")
            continue
        primary_metric_key = PRIMARY_METRIC_TO_KEY[
            GRID_SPECS[model_name].primary_metric
        ]
        result = best_trial_recommendations(run_directory, primary_metric_key)
        if result is None:
            print(f"  No usable best trial for {model_name}; skipping.")
            continue
        best_trial_id, recommendations = result
        print(
            f"  {model_name}: best trial {best_trial_id} ({len(recommendations)} rows)"
        )

        annotated = annotate_recommendations(
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
    panel["band_label"] = panel["declared_band"].map(RISK_BAND_NAMES)
    return panel


def run_panel_regression(
    evaluation_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_timestamp: str | None = None,
    data_paths: DataPaths | None = None,
) -> dict[str, Any]:
    """Build the (customer, split, model) panel.csv consumed by findings.ipynb."""
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
    panel = _build_panel(
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

    output_directory = output_root / (
        run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "panel.csv").write_text(panel.to_csv(index=False))

    print(f"\nPanel artefacts saved to {output_directory}")
    return {
        "output_directory": str(output_directory),
        "panel_rows": int(len(panel)),
    }
