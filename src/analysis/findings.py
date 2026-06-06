"""Plot, table, and figure-export renderers for the thesis, driven by the export_figures pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RISK_BAND_NAMES: dict[int, str] = {
    0: "Conservative",
    1: "Income",
    2: "Balanced",
    3: "Aggressive",
}
BANDS_ORDER: list[str] = ["Conservative", "Income", "Balanced", "Aggressive"]
BAND_COLOURS: dict[str, str] = {
    "Conservative": "#1f77b4",
    "Income": "#2ca02c",
    "Balanced": "#ff7f0e",
    "Aggressive": "#d62728",
}
MODEL_DISPLAY_MAP: dict[str, str] = {
    "light_gcn": "LightGCN",
    "random_forest": "Random Forest",
}
METRIC_COLUMNS: list[str] = [
    "ndcg_at_k",
    "roi_at_k",
    "recall_at_k",
    "profile_coherence_at_k",
    "profile_coherence_lift_at_k",
]
PRETTY_METRIC: dict[str, str] = {
    "ndcg_at_k": "nDCG@10",
    "roi_at_k": "ROI@10 (mo.)",
    "recall_at_k": "Recall@10",
    "profile_coherence_at_k": "PC@10",
    "profile_coherence_lift_at_k": "PC-lift@10",
}
# PC-lift@10 is omitted from the paired figures: regrouping changes each customer's
# band and hence the denominator pi(b_u), so PC-lift is not comparable across the
# declared and regrouped configurations shown side by side. The figures compare on
# raw PC@10 instead.
PC_METRIC_ORDER: list[str] = [
    "PC@10",
    "ROI@10 (mo.)",
    "Recall@10",
    "nDCG@10",
]
CONTRAST_DESCRIPTIONS: dict[str, str] = {
    "Stratify": "λ=0, declared bands",
    "Stratify+PC-loss": "λ=1, declared bands",
    "Stratify+Regroup": "λ=0, regrouped bands",
}
CONTRAST_COLOURS: dict[str, str] = {
    "Stratify": "#4C72B0",
    "Stratify+PC-loss": "#DD8452",
    "Stratify+Regroup": "#55A868",
}
CONTRAST_ORDER: list[str] = list(CONTRAST_DESCRIPTIONS.keys())
SIGNIFICANCE_ALPHA: float = 0.05

SINGLE_COLUMN_WIDTH_INCHES: float = 3.3
DEFAULT_FIGURE_HEIGHT_INCHES: float = 2.2

PLOT_RC: dict[str, Any] = {
    "figure.figsize": (SINGLE_COLUMN_WIDTH_INCHES, DEFAULT_FIGURE_HEIGHT_INCHES),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman",
        "Times",
        "Times New Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
}


@dataclass(frozen=True)
class NotebookPaths:
    """Locations of every precomputed artefact rendered into the thesis figures."""

    eda_directory: Path
    return_regression_directory: Path
    baseline_decomposition_directory: Path
    panel_regression_directory: Path
    stratified_results_directory: Path
    stratified_baseline_directory: Path
    regrouped_results_directory: Path

    @classmethod
    def from_root(cls, root: Path) -> NotebookPaths:
        """Resolve the default thesis-run paths under the project root."""
        return cls(
            eda_directory=root / "outputs/eda",
            return_regression_directory=root
            / "outputs/analysis/transaction_return_regression/20260504_004849",
            baseline_decomposition_directory=root
            / "outputs/analysis/baseline_decomposition/20260427_122215",
            panel_regression_directory=root
            / "outputs/analysis/panel_regression/20260504_003145",
            stratified_results_directory=root
            / "outputs/results/evaluation/pc_lgcn/20260505_125105",
            stratified_baseline_directory=root
            / "outputs/results/evaluation/light_gcn/20260427_122215/eb788_00006",
            regrouped_results_directory=root
            / "outputs/results_regrouped/evaluation/pc_lgcn/20260605_223513",
        )


def configure_matplotlib() -> None:
    """Apply the thesis plot defaults to matplotlib's global rcParams."""
    plt.rcParams.update(PLOT_RC)


def load_eda_summary(eda_directory: Path) -> dict[str, Any]:
    """Read outputs/eda/summary.json into a plain dict."""
    with open(eda_directory / "summary.json") as fh:
        return json.load(fh)


def _format_p_value(p: float) -> str:
    """Render a p-value with scientific notation below 1e-3, fixed otherwise."""
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _finalise(fig: plt.Figure, save_path: Path | None) -> None:
    """Either render inline or write `fig` to `save_path` as PDF and close it."""
    if save_path is None:
        plt.show()
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_discordance_distribution(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Bar chart of pairwise discordance d in {0,1,2,3} across all scoreable Buys."""
    discordance = eda_summary["transaction_discordance_summary"]
    counts = discordance["discordance_counts"]
    total = discordance["transactions_with_both_bands"]

    distribution = pd.DataFrame(
        [{"d": int(k), "count": v, "share": v / total} for k, v in counts.items()]
    ).sort_values("d")
    distribution["label"] = distribution["d"].map(
        {0: "$d{=}0$", 1: "$d{=}1$", 2: "$d{=}2$", 3: "$d{=}3$"}
    )

    print(
        f"Total scoreable Buys:      {total:,}\n"
        f"Coherent (d<=1) share:     {discordance['fraction_coherent_default']:.1%}\n"
        f"Strict-coherent (d=0):     {discordance['fraction_coherent_strict']:.1%}\n"
        f"Mean profile discordance:  {discordance['mean_discordance']:.3f} bands"
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 2.4))
    colours = ["#2ca02c", "#9bcd9b", "#ff9900", "#d62728"]
    ax.bar(distribution["label"], distribution["share"], color=colours)
    for x, value in enumerate(distribution["share"]):
        ax.text(x, value + 0.01, f"{value:.1%}", ha="center", va="bottom", fontsize=6)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Share of scoreable Buys")
    ax.grid(axis="x", visible=False)

    coherent_share = discordance["fraction_coherent_default"]
    discoherent_share = 1.0 - coherent_share
    _draw_coherence_bracket(
        ax, 0, 1, 0.62, f"Coherent ($d{{\\leq}}1$): {coherent_share:.1%}", "#1f4e1f"
    )
    _draw_coherence_bracket(
        ax,
        2,
        3,
        0.62,
        f"Discordant ($d{{\\geq}}2$): {discoherent_share:.1%}",
        "#7a1f1f",
    )

    ax.text(
        0.98,
        0.97,
        f"Mean $d$ = {discordance['mean_discordance']:.3f} bands",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#888",
            linewidth=0.4,
            alpha=0.9,
        ),
    )

    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def _draw_coherence_bracket(
    ax: plt.Axes, x_left: float, x_right: float, y: float, label: str, colour: str
) -> None:
    ax.plot([x_left, x_right], [y, y], color=colour, linewidth=0.9)
    ax.plot([x_left, x_left], [y - 0.015, y], color=colour, linewidth=0.9)
    ax.plot([x_right, x_right], [y - 0.015, y], color=colour, linewidth=0.9)
    ax.text(
        (x_left + x_right) / 2,
        y + 0.012,
        label,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=colour,
        fontweight="bold",
    )


def plot_self_discordance(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Histogram of per-customer discordant share; demonstrates the bimodal customer trait."""
    self_disc = eda_summary["customer_self_discordance_summary"]
    populations = eda_summary["populations"]
    transaction_summary = eda_summary["transaction_discordance_summary"]

    population_discordance_rate = 1 - transaction_summary["fraction_coherent_default"]
    mean_buys_per_scoreable_customer = (
        populations["transactions_with_both_bands"] / populations["customers_with_band"]
    )

    print(
        f"Population discordance rate (per Buy):                 {population_discordance_rate:.1%}\n"
        f"Mean Buys per customer      (with a known band):       {mean_buys_per_scoreable_customer:.2f}\n"
        f"Fully coherent customers    (every Buy in tolerance):  {self_disc['fraction_fully_coherent']:.1%}\n"
        f"Fully discordant customers  (every Buy out of toler.): {self_disc['fraction_fully_discordant']:.1%}"
    )

    edges = self_disc["discordant_share_histogram_edges"]
    counts = np.array(self_disc["discordant_share_histogram_counts"], dtype=float)
    total_customers = counts.sum()
    shares = counts / total_customers
    bin_labels = [
        f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}" for i in range(len(counts))
    ]

    bar_colours = ["#cccccc"] * len(counts)
    bar_colours[0] = "#2ca02c"
    bar_colours[-1] = "#d62728"

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 2.6))
    positions = np.arange(len(counts))
    ax.bar(positions, shares, color=bar_colours, edgecolor="white", linewidth=0.3)

    ax.annotate(
        f"Fully coherent: {self_disc['fraction_fully_coherent']:.1%}",
        xy=(positions[0], shares[0]),
        xytext=(positions[0] + 1.5, shares[0] + 0.05),
        arrowprops=dict(arrowstyle="-", linestyle=":", color="#1f4e1f", linewidth=0.7),
        fontsize=6.5,
        color="#1f4e1f",
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.annotate(
        f"Fully discordant: {self_disc['fraction_fully_discordant']:.1%}",
        xy=(positions[-1], shares[-1]),
        xytext=(positions[-1] - 1.5, shares[-1] + 0.18),
        arrowprops=dict(arrowstyle="-", linestyle=":", color="#7a1f1f", linewidth=0.7),
        fontsize=6.5,
        color="#7a1f1f",
        fontweight="bold",
        ha="right",
        va="center",
    )

    mean_x_position = population_discordance_rate * len(counts) - 0.5
    ax.axvline(mean_x_position, color="black", linestyle="--", linewidth=0.7)
    ax.text(
        mean_x_position + 0.15,
        max(shares) * 0.28,
        f"Pop. mean\n{population_discordance_rate:.1%}",
        fontsize=6,
        va="top",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, max(shares) * 1.3)
    ax.set_ylabel("Share of customers")
    ax.set_xlabel("Per-customer discordant share (%)")
    ax.grid(axis="x", visible=False)

    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def plot_per_band_coherence(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Per-band coherent share showing the U-shape on Conservative and Aggressive extremes."""
    risk_level_counts = eda_summary["transaction_discordance_by_risk_level"]
    rows = []
    for band_label, by_d in risk_level_counts.items():
        total_band = sum(by_d.values())
        coherent_band = by_d.get("0", 0) + by_d.get("1", 0)
        rows.append(
            {
                "band_label": band_label,
                "transactions": total_band,
                "coherent_share": coherent_band / total_band if total_band else 0,
            }
        )
    band_table = pd.DataFrame(rows)
    band_table["band_label"] = pd.Categorical(
        band_table["band_label"], categories=BANDS_ORDER, ordered=True
    )
    band_table = band_table.sort_values("band_label")

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 2.3))
    colours = [BAND_COLOURS[label] for label in band_table["band_label"]]
    ax.bar(
        band_table["band_label"].astype(str),
        band_table["coherent_share"],
        color=colours,
    )
    for x, (count, share) in enumerate(
        zip(band_table["transactions"], band_table["coherent_share"])
    ):
        ax.text(
            x,
            share + 0.015,
            f"{share:.1%}\nn={count:,}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Coherent share ($d{\\leq}1$)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def plot_yearly_discordance(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Per-year mean discordance line chart with the 2018 MiFID II elevation flagged."""
    year_disc = eda_summary["mean_discordance_by_year"]
    buy_coverage = eda_summary["buy_coverage_by_year"]
    years = sorted(year_disc)
    year_values = [year_disc[y] for y in years]
    mean_across_years = float(np.mean(year_values))

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 2.4))
    ax.plot(
        years,
        year_values,
        marker="o",
        linewidth=1.2,
        markersize=4,
        color="#1f77b4",
        label="Per-year mean",
        zorder=3,
    )
    ax.axhline(
        mean_across_years,
        linestyle="--",
        color="black",
        linewidth=0.8,
        label=f"Avg. = {mean_across_years:.2f}",
    )

    for x, year in enumerate(years):
        coverage = buy_coverage[year]
        partial_flag = "*" if _is_partial_year(coverage) else ""
        ax.annotate(
            f"{year_values[x]:.2f}{partial_flag}",
            xy=(year, year_values[x]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    partial_years = [y for y in years if _is_partial_year(buy_coverage[y])]
    if partial_years:
        caption = (
            "* partial year ("
            + ", ".join(f"to {buy_coverage[y]['last_date']}" for y in partial_years)
            + ")"
        )
        ax.text(
            0.01, -0.28, caption, transform=ax.transAxes, fontsize=6, color="#666666"
        )

    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Mean discordance")
    ax.set_xlabel("Calendar year")
    ax.legend(loc="lower left", fontsize=6.5)
    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def _is_partial_year(coverage: dict[str, Any]) -> bool:
    last = pd.Timestamp(coverage["last_date"])
    return (last.month, last.day) != (12, 31)


def load_return_regression_coefficients(
    return_regression_directory: Path,
) -> pd.DataFrame:
    """Load the transaction-return regression coefficients dataframe."""
    return pd.read_csv(return_regression_directory / "coefficients.csv")


_COEFFICIENT_DISPLAY_ORDER: list[str] = [
    "Intercept",
    "is_coherent",
    "asset_volatility",
    "C(customer_type)[T.Legal Entity]",
    "C(customer_type)[T.Mass]",
    "C(customer_type)[T.Premium]",
    "C(customer_type)[T.Professional]",
    "C(year)[T.2019]",
    "C(year)[T.2020]",
    "C(year)[T.2021]",
    "C(year)[T.2022]",
]
_COEFFICIENT_TERM_EXPLANATIONS: dict[str, str] = {
    "Intercept": "Baseline = Inactive customer, 2018, all numeric regressors = 0",
    "is_coherent": "Coherence premium after controlling for vol, segment, year",
    "asset_volatility": "Return penalty per 1.0 unit of annualised vol (negative = low-vol won)",
    "C(customer_type)[T.Legal Entity]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Mass]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Premium]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Professional]": "Segment effect vs Inactive reference",
    "C(year)[T.2019]": "Calendar effect vs 2018 reference (pre-COVID equity recovery)",
    "C(year)[T.2020]": "Calendar effect vs 2018 reference (COVID rebound)",
    "C(year)[T.2021]": "Calendar effect vs 2018 reference (continued post-COVID recovery)",
    "C(year)[T.2022]": "Calendar effect vs 2018 reference (bond / rate-shock drawdown)",
}


def _filtered_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    filtered = coefficients[
        coefficients["term"].isin(_COEFFICIENT_DISPLAY_ORDER)
    ].copy()
    return filtered.set_index("term").reindex(_COEFFICIENT_DISPLAY_ORDER).reset_index()


def style_coefficient_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Return the human-readable display table for the transaction-return regression coefficients."""
    filtered = _filtered_coefficients(coefficients)
    display_df = filtered.assign(
        estimate=lambda d: d["estimate"].map("{:+.4f}".format),
        std_error=lambda d: d["std_error"].map("{:.4f}".format),
        ci=lambda d: d.apply(
            lambda r: f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]", axis=1
        ),
        p_value=lambda d: d["p_value"].map(_format_p_value),
        meaning=lambda d: d["term"].map(_COEFFICIENT_TERM_EXPLANATIONS),
    )
    return display_df[["term", "estimate", "std_error", "ci", "p_value", "meaning"]]


def plot_coefficient_forest(
    coefficients: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Forest plot of every transaction-return regression coefficient with 95% CIs in pp."""
    filtered = _filtered_coefficients(coefficients)
    plot_df = filtered.copy()
    plot_df["significant"] = plot_df["p_value"] < SIGNIFICANCE_ALPHA
    plot_df["estimate_pp"] = plot_df["estimate"] * 100
    plot_df["ci_lower_pp"] = plot_df["ci_lower"] * 100
    plot_df["ci_upper_pp"] = plot_df["ci_upper"] * 100
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    y_positions = np.arange(len(plot_df))
    sig_colour = "#1f4e1f"
    ns_colour = "#888888"
    point_colours = [sig_colour if s else ns_colour for s in plot_df["significant"]]

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 3.6))
    for y, row, colour in zip(y_positions, plot_df.itertuples(), point_colours):
        ax.errorbar(
            row.estimate_pp,
            y,
            xerr=[
                [row.estimate_pp - row.ci_lower_pp],
                [row.ci_upper_pp - row.estimate_pp],
            ],
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2,
            markersize=4,
        )
        label = f"{row.estimate_pp:+.2f} pp  (p={_format_p_value(row.p_value)})"
        ax.text(
            row.ci_upper_pp + 0.4,
            y,
            label,
            va="center",
            ha="left",
            fontsize=6,
            color=colour,
        )

    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["term"], fontsize=6.5)
    ax.set_xlabel("Coefficient estimate (pp on 6-mo. realised return)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(
        min(plot_df["ci_lower_pp"].min(), 0) - 2,
        plot_df["ci_upper_pp"].max() + 26,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=sig_colour,
            markersize=5,
            label=f"$p < {SIGNIFICANCE_ALPHA}$",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=ns_colour,
            markersize=5,
            label=f"$p \\geq {SIGNIFICANCE_ALPHA}$",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.5)
    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def style_baseline_main_results(baseline_decomposition_directory: Path) -> Any:
    """Headline metrics table for the two FAR-Trans baselines at their primary-metric optima."""
    main_results = pd.read_csv(baseline_decomposition_directory / "main_results.csv")
    display_columns = [
        "display_name",
        "best_trial_id",
        "primary_metric",
        "ndcg_at_k_mean",
        "roi_at_k_mean",
        "recall_at_k_mean",
        "profile_coherence_at_k_mean",
        "profile_coherence_lift_at_k_mean",
    ]
    renamed = main_results[display_columns].rename(
        columns={
            "display_name": "Model",
            "best_trial_id": "Best trial",
            "primary_metric": "Primary metric",
            "ndcg_at_k_mean": "nDCG@10",
            "roi_at_k_mean": "ROI@10 (mo.)",
            "recall_at_k_mean": "Recall@10",
            "profile_coherence_at_k_mean": "PC@10",
            "profile_coherence_lift_at_k_mean": "PC-lift@10",
        }
    )
    return renamed.style.format(
        {
            "nDCG@10": "{:.3f}",
            "ROI@10 (mo.)": "{:+.4f}",
            "Recall@10": "{:.3f}",
            "PC@10": "{:.3f}",
            "PC-lift@10": "{:.3f}",
        }
    )


def build_per_band_summary(
    panel_regression_directory: Path, eda_summary: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-(band, model) mean PC@10 plus the band-conditional random baseline pi(b)."""
    panel = pd.read_csv(panel_regression_directory / "panel.csv")
    pi_series = _compute_pi_series(eda_summary)

    summary = panel.groupby(["model", "band_label"], as_index=False)[
        "coherent_share"
    ].agg(mean_pc="mean", n_observations="count")
    summary["model_display"] = summary["model"].map(MODEL_DISPLAY_MAP)
    summary["band_label"] = pd.Categorical(
        summary["band_label"], categories=BANDS_ORDER, ordered=True
    )
    summary["pi_b"] = summary["band_label"].astype(str).map(pi_series.to_dict())
    summary["lift"] = (summary["mean_pc"] / summary["pi_b"]).round(3)
    summary = summary.sort_values(["model_display", "band_label"]).reset_index(
        drop=True
    )
    return summary, pi_series


def _compute_pi_series(eda_summary: dict[str, Any]) -> pd.Series:
    asset_band_dist = eda_summary["asset_band_distribution_by_category"]
    total_per_band: dict[str, int] = {band: 0 for band in RISK_BAND_NAMES.values()}
    for band_counts in asset_band_dist.values():
        for band_label, count in band_counts.items():
            total_per_band[band_label] += count
    total_assets = sum(total_per_band.values())
    pi: dict[str, float] = {}
    for band_idx, label in RISK_BAND_NAMES.items():
        coherent_count = sum(
            total_per_band[other_label]
            for other_idx, other_label in RISK_BAND_NAMES.items()
            if abs(other_idx - band_idx) <= 1
        )
        pi[label] = coherent_count / total_assets
    return pd.Series(pi).reindex(BANDS_ORDER)


def style_per_band_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the band/model PC@10 summary in the column order shown in the notebook."""
    return summary[
        ["band_label", "model_display", "n_observations", "mean_pc", "pi_b", "lift"]
    ]


def plot_per_band_pc(
    summary: pd.DataFrame,
    pi_series: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart of mean PC@10 by band/model, with pi(b) reference lines."""
    models = list(summary["model_display"].drop_duplicates())
    tab10 = plt.get_cmap("tab10")
    model_colours = {model: tab10(i) for i, model in enumerate(models)}

    x_positions = np.arange(len(BANDS_ORDER))
    group_width = 0.78
    bar_width = group_width / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_INCHES, 2.6))
    bar_handles = []
    for i, model in enumerate(models):
        rows = (
            summary[summary["model_display"] == model]
            .set_index("band_label")
            .reindex(BANDS_ORDER)
            .reset_index()
        )
        offsets = -group_width / 2 + bar_width / 2 + i * bar_width
        xs = x_positions + offsets
        bars = ax.bar(
            xs,
            rows["mean_pc"],
            width=bar_width * 0.95,
            color=model_colours[model],
            label=model,
            edgecolor="white",
            linewidth=0.3,
        )
        bar_handles.append(bars)
        for x, pc, lift in zip(xs, rows["mean_pc"], rows["lift"]):
            ax.text(
                x,
                pc + 0.012,
                f"{pc:.0%}\n{lift:.2f}x",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=model_colours[model],
                fontweight="bold",
            )

    for j, band in enumerate(BANDS_ORDER):
        pi_val = pi_series[band]
        left = x_positions[j] - group_width / 2
        right = x_positions[j] + group_width / 2
        ax.hlines(pi_val, left, right, colors="black", linestyles="--", linewidth=0.7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(BANDS_ORDER, fontsize=6.5)
    ax.set_ylabel("Mean PC@10")
    ax.set_ylim(0, max(summary["mean_pc"].max(), pi_series.max()) * 1.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    baseline_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=0.7,
        label="Random baseline $\\pi(b)$",
    )
    ax.legend(
        handles=[*bar_handles, baseline_handle],
        loc="upper left",
        fontsize=6,
    )
    plt.tight_layout(pad=0.3)
    _finalise(fig, save_path)


def load_stratified_per_split(
    stratified_baseline_directory: Path,
    stratified_results_directory: Path,
    regrouped_results_directory: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-split metric tables for the baseline, the declared-band trials, and (optionally)
    the regrouped-band trials."""
    per_split = {
        "baseline": _read_per_split(
            stratified_baseline_directory / "per_split_metrics.csv", "baseline"
        ),
        "lambda_0": _read_per_split(
            stratified_results_directory
            / "stratified_lambda_0.0/per_split_metrics.csv",
            "lambda_0",
        ),
        "lambda_1": _read_per_split(
            stratified_results_directory
            / "stratified_lambda_1.0/per_split_metrics.csv",
            "lambda_1",
        ),
    }
    if regrouped_results_directory is not None:
        per_split["regroup_lambda_0"] = _read_per_split(
            regrouped_results_directory / "stratified_lambda_0.0/per_split_metrics.csv",
            "regroup_lambda_0",
        )
        per_split["regroup_lambda_1"] = _read_per_split(
            regrouped_results_directory / "stratified_lambda_1.0/per_split_metrics.csv",
            "regroup_lambda_1",
        )
    return per_split


def _read_per_split(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trial"] = label
    return df


def style_stratified_aggregate(per_split: dict[str, pd.DataFrame]) -> Any:
    """Aggregate-mean metrics across 69 splits for every available trial."""
    trial_labels = {
        "baseline": "LightGCN baseline",
        "lambda_0": "Stratified, λ=0",
        "lambda_1": "Stratified, λ=1",
        "regroup_lambda_0": "Regrouped, λ=0",
        "regroup_lambda_1": "Regrouped, λ=1",
    }
    available = [key for key in trial_labels if key in per_split]

    aggregate = pd.DataFrame(
        {
            "Configuration": [trial_labels[key] for key in available],
            "nDCG@10": [per_split[key]["ndcg_at_k"].mean() for key in available],
            "ROI@10 (mo.)": [per_split[key]["roi_at_k"].mean() for key in available],
            "Recall@10": [per_split[key]["recall_at_k"].mean() for key in available],
            "PC@10": [
                per_split[key]["profile_coherence_at_k"].mean() for key in available
            ],
            "PC-lift@10": [
                per_split[key]["profile_coherence_lift_at_k"].mean()
                for key in available
            ],
        }
    )

    print("Aggregate metrics: mean across 69 evaluation splits, per configuration")
    return aggregate.style.format(
        {
            "nDCG@10": "{:.4f}",
            "ROI@10 (mo.)": "{:+.4f}",
            "Recall@10": "{:.4f}",
            "PC@10": "{:.4f}",
            "PC-lift@10": "{:.3f}",
        }
    )


def compute_paired_table(per_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format mean Δ and win rate against the baseline for each stratified trial."""
    paired = pd.concat(
        [
            _paired_summary(per_split["lambda_0"], per_split["baseline"], "Stratify"),
            _paired_summary(
                per_split["lambda_1"], per_split["baseline"], "Stratify+PC-loss"
            ),
            _paired_summary(
                per_split["regroup_lambda_0"], per_split["baseline"], "Stratify+Regroup"
            ),
        ],
        ignore_index=True,
    )
    paired["metric"] = paired["metric"].map(PRETTY_METRIC)
    return paired


def _paired_summary(
    treatment: pd.DataFrame, control: pd.DataFrame, label: str
) -> pd.DataFrame:
    rows = []
    for column in METRIC_COLUMNS:
        delta = treatment[column].to_numpy() - control[column].to_numpy()
        rows.append(
            {
                "contrast": label,
                "metric": column,
                "mean_delta": float(delta.mean()),
                "win_rate": float((delta > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def style_paired_table(paired_table: pd.DataFrame) -> pd.DataFrame:
    """Pivoted display of mean_delta and win_rate for the three stratified contrasts."""
    formatted = paired_table.assign(
        mean_delta=lambda d: d["mean_delta"].map("{:+.5f}".format),
        win_rate=lambda d: d["win_rate"].map("{:.0%}".format),
    )
    return formatted.pivot(
        index="contrast",
        columns="metric",
        values=["mean_delta", "win_rate"],
    )


def build_stratified_per_band_summary(
    stratified_baseline_directory: Path,
    stratified_results_directory: Path,
    eda_summary: dict[str, Any],
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Per-band PC-lift@10 for all three stratified trials (baseline, lambda=0, lambda=1)."""
    trial_paths = {
        "Baseline": stratified_baseline_directory / "recommendations.parquet",
        r"$\lambda{=}0$": stratified_results_directory
        / "stratified_lambda_0.0/recommendations.parquet",
        r"$\lambda{=}1$": stratified_results_directory
        / "stratified_lambda_1.0/recommendations.parquet",
    }
    pi_series = _compute_pi_series(eda_summary)

    customer_band_lookup: dict[str, str] = {
        customer_id: RISK_BAND_NAMES[profile.risk_band]
        for customer_id, profile in customer_profiles.items()
        if profile.risk_band is not None
    }
    rows: list[dict[str, Any]] = []
    for trial_label, parquet_path in trial_paths.items():
        recommendations = pd.read_parquet(parquet_path)
        recommendations["customer_band"] = recommendations["customer_id"].map(
            customer_band_lookup
        )
        recommendations["asset_band"] = recommendations["asset_id"].map(
            asset_risk_classes
        )
        recommendations["customer_band_num"] = recommendations["customer_id"].map(
            lambda cid, profiles=customer_profiles: (
                profiles[cid].risk_band
                if cid in profiles and profiles[cid].risk_band is not None
                else None
            )
        )
        recommendations = recommendations.dropna(
            subset=["customer_band", "asset_band", "customer_band_num"]
        )
        recommendations["is_coherent"] = (
            recommendations["customer_band_num"] - recommendations["asset_band"]
        ).abs() <= 1

        per_customer_split = (
            recommendations.groupby(["split_index", "customer_id", "customer_band"])
            .agg(pc_at_10=("is_coherent", "mean"))
            .reset_index()
        )
        per_band = (
            per_customer_split.groupby("customer_band")
            .agg(
                mean_pc=("pc_at_10", "mean"),
                n_observations=("pc_at_10", "count"),
            )
            .reindex(BANDS_ORDER)
        )
        per_band["pi_b"] = per_band.index.map(pi_series.to_dict())
        per_band["lift"] = per_band["mean_pc"] / per_band["pi_b"]

        for band_label in BANDS_ORDER:
            band_row = per_band.loc[band_label]
            rows.append(
                {
                    "trial": trial_label,
                    "band": band_label,
                    "pc_lift_at_10": round(float(band_row["lift"]), 3),
                    "n_observations": int(band_row["n_observations"]),
                }
            )

    return pd.DataFrame(rows)


def style_stratified_per_band_summary(summary: pd.DataFrame) -> Any:
    """Pivoted display of per-band PC-lift@10 across the three stratified trials."""
    pivoted = summary.pivot(index="band", columns="trial", values="pc_lift_at_10")
    pivoted = pivoted.reindex(index=BANDS_ORDER)
    trial_order = ["Baseline", r"$\lambda{=}0$", r"$\lambda{=}1$"]
    pivoted = pivoted[[col for col in trial_order if col in pivoted.columns]]
    return pivoted.style.format("{:.2f}\\times")


def plot_paired_deltas(
    paired_table: pd.DataFrame, save_path: Path | None = None
) -> None:
    """Per-metric horizontal bars of mean Delta across the configurations, stacked vertically."""
    _plot_paired_grid(
        paired_table=paired_table,
        value_column="mean_delta",
        formatter=lambda v: f"{v:+.4f}",
        save_path=save_path,
        x_limits=None,
        reference_line=0.0,
        reference_style="solid",
    )


def plot_paired_win_rates(
    paired_table: pd.DataFrame, save_path: Path | None = None
) -> None:
    """Per-metric horizontal bars of win rate across the configurations, stacked vertically."""
    _plot_paired_grid(
        paired_table=paired_table,
        value_column="win_rate",
        formatter=lambda v: f"{v:.0%}",
        save_path=save_path,
        x_limits=(0.0, 1.0),
        reference_line=0.5,
        reference_style="dashed",
        x_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_tick_labels=["0%", "25%", "50%", "75%", "100%"],
    )


def _plot_paired_grid(
    paired_table: pd.DataFrame,
    value_column: str,
    formatter: Any,
    save_path: Path | None,
    x_limits: tuple[float, float] | None,
    reference_line: float,
    reference_style: str,
    x_ticks: list[float] | None = None,
    x_tick_labels: list[str] | None = None,
) -> None:
    metrics_top_to_bottom = list(PC_METRIC_ORDER)
    fig, axes = plt.subplots(
        len(metrics_top_to_bottom),
        1,
        figsize=(SINGLE_COLUMN_WIDTH_INCHES, 1.2 * len(metrics_top_to_bottom)),
        sharex=(x_limits is not None),
    )

    bar_colors = [CONTRAST_COLOURS[c] for c in CONTRAST_ORDER]
    y_positions = np.arange(len(CONTRAST_ORDER))

    for axis, metric in zip(axes, metrics_top_to_bottom):
        subset = paired_table[paired_table["metric"] == metric].set_index("contrast")
        values = np.array([subset.loc[c, value_column] for c in CONTRAST_ORDER])

        axis.barh(
            y_positions, values, color=bar_colors, edgecolor="black", linewidth=0.3
        )
        axis.axvline(
            reference_line,
            color="black",
            linewidth=0.6,
            linestyle="--" if reference_style == "dashed" else "-",
            alpha=0.7,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(CONTRAST_ORDER, fontsize=6)
        axis.set_ylabel(
            metric, fontsize=7, rotation=0, ha="right", va="center", labelpad=2
        )
        axis.tick_params(axis="x", labelsize=6)
        axis.grid(True, axis="x", linewidth=0.3, alpha=0.5)
        axis.invert_yaxis()

        if x_limits is not None:
            axis.set_xlim(*x_limits)
        else:
            max_abs = max(np.abs(values).max(), 1e-6)
            axis.set_xlim(-max_abs * 2.0, max_abs * 2.0)
        if x_ticks is not None:
            axis.set_xticks(x_ticks)
        if x_tick_labels is not None:
            axis.set_xticklabels(x_tick_labels)

        span = axis.get_xlim()[1] - axis.get_xlim()[0]
        for y_pos, value in zip(y_positions, values):
            goes_right = value >= 0
            offset = 0.025 * span if goes_right else -0.025 * span
            axis.text(
                value + offset,
                y_pos,
                formatter(value),
                va="center",
                ha="left" if goes_right else "right",
                fontsize=5.5,
            )

    fig.tight_layout(rect=(0, 0.16, 1, 1.0), pad=0.4, h_pad=0.6)
    fig.legend(
        handles=_contrast_legend_handles(),
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=6,
    )
    _finalise(fig, save_path)


def _contrast_legend_handles() -> list[Any]:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=CONTRAST_COLOURS[c],
            label=f"{c}: {CONTRAST_DESCRIPTIONS[c]}",
        )
        for c in CONTRAST_ORDER
    ]


def export_figures_as_pdf(
    output_directory: Path,
    eda_summary: dict[str, Any],
    paths: NotebookPaths,
) -> list[Path]:
    """Render every findings figure to PDF under `output_directory` and return the written paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    coefficients = load_return_regression_coefficients(
        paths.return_regression_directory
    )
    per_band_summary, pi_series = build_per_band_summary(
        paths.panel_regression_directory, eda_summary
    )
    stratified_per_split = load_stratified_per_split(
        paths.stratified_baseline_directory,
        paths.stratified_results_directory,
        paths.regrouped_results_directory,
    )
    paired_table = compute_paired_table(stratified_per_split)

    figure_plan: list[tuple[str, Any]] = [
        (
            "discordance_distribution.pdf",
            lambda p: plot_discordance_distribution(eda_summary, save_path=p),
        ),
        (
            "self_discordance.pdf",
            lambda p: plot_self_discordance(eda_summary, save_path=p),
        ),
        (
            "per_band_coherence.pdf",
            lambda p: plot_per_band_coherence(eda_summary, save_path=p),
        ),
        (
            "yearly_discordance.pdf",
            lambda p: plot_yearly_discordance(eda_summary, save_path=p),
        ),
        (
            "coefficient_forest.pdf",
            lambda p: plot_coefficient_forest(coefficients, save_path=p),
        ),
        (
            "per_band_pc.pdf",
            lambda p: plot_per_band_pc(per_band_summary, pi_series, save_path=p),
        ),
        (
            "paired_deltas.pdf",
            lambda p: plot_paired_deltas(paired_table, save_path=p),
        ),
        (
            "paired_win_rates.pdf",
            lambda p: plot_paired_win_rates(paired_table, save_path=p),
        ),
    ]
    written: list[Path] = []
    for filename, render in figure_plan:
        target = output_directory / filename
        render(target)
        written.append(target)
    return written
