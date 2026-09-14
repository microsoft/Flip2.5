"""Build compact figures for the 2026-08-31 collaborator presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
OUTPUT = HERE / "assets"

BOX_RULE = "#2C6EAB"
GRAY_TEXT = "#5B6169"
RULE_GRAY = "#BFC5CB"
GRID = "#E4E8EC"
ORANGE = "#D95F02"
FAMILY_ORDER = ("pdz", "trpb", "nuc")
FAMILY_LABELS = {"pdz": "PDZ", "trpb": "TrpB", "nuc": "NucB"}
FAMILY_COLORS = {"pdz": BOX_RULE, "trpb": "#3F7A4F", "nuc": ORANGE}
METHOD_ORDER = ("HMC posterior", "MLE predictive")
METHOD_STYLES = {
    "HMC posterior": {"marker": "o", "linestyle": "-", "color": BOX_RULE},
    "MLE predictive": {"marker": "s", "linestyle": "--", "color": ORANGE},
}
TABLE_ROOTS = {
    "HMC posterior": PAPER_ROOT / "output/hmc/tables",
    "MLE predictive": PAPER_ROOT / "output/mle_predictive/tables",
}
CONTRAST_ANALYSES = (
    "trpb__trpb-libi__mle-predictive__exponential-exponential__mle-49",
    "trpb__trpb-libf__mle-predictive__exponential-exponential__mle-31",
)
CONTRAST_LABELS = {
    CONTRAST_ANALYSES[0]: "TrpB libI",
    CONTRAST_ANALYSES[1]: "TrpB libF",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 12.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": RULE_GRAY,
            "axes.labelcolor": GRAY_TEXT,
            "text.color": GRAY_TEXT,
            "xtick.color": GRAY_TEXT,
            "ytick.color": GRAY_TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _load_table(name: str) -> pd.DataFrame:
    frames = []
    for method, root in TABLE_ROOTS.items():
        frame = pd.read_csv(root / f"{name}.csv", low_memory=False)
        frame.insert(0, "uncertainty_procedure", method)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _shared_landscape_scope(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"uncertainty_procedure", "protein_family", "landscape"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Shared-scope rows are missing columns: {sorted(missing)}")
    available = rows[list(required)].drop_duplicates()
    shared: dict[str, set[str]] = {}
    for family in FAMILY_ORDER:
        method_landscapes = [
            set(
                available.loc[
                    available["protein_family"].eq(family)
                    & available["uncertainty_procedure"].eq(method),
                    "landscape",
                ]
            )
            for method in METHOD_ORDER
        ]
        shared[family] = set.intersection(*method_landscapes)
        if not shared[family]:
            raise ValueError(f"No shared landscapes for {family}.")
    include = np.array(
        [
            landscape in shared[family]
            for family, landscape in zip(
                rows["protein_family"],
                rows["landscape"],
                strict=True,
            )
        ]
    )
    return rows.loc[include].copy()


def _positive_plot_floor(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Plot-floor values must be finite and non-negative.")
    positive = array[array > 0]
    if not len(positive):
        raise ValueError("At least one positive value is required for a log plot.")
    return float(positive.min() / 2)


def _style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    axis.spines[["left", "bottom"]].set_color(RULE_GRAY)


def _save(figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{stem}.pdf")
    figure.savefig(OUTPUT / f"{stem}.png", dpi=220)
    plt.close(figure)


def _strip_points(
    axis,
    rows: pd.DataFrame,
    *,
    value: str,
    log_scale: bool = False,
) -> None:
    method_offsets = {"HMC posterior": -0.17, "MLE predictive": 0.17}
    for family_index, family in enumerate(FAMILY_ORDER):
        for method in METHOD_ORDER:
            selected = rows.loc[
                rows["protein_family"].eq(family)
                & rows["uncertainty_procedure"].eq(method)
            ]
            values = selected[value].dropna().sort_values().to_numpy(dtype=float)
            landscape_values = (
                selected.groupby("landscape", as_index=False)[value]
                .median()[value]
                .dropna()
                .to_numpy(dtype=float)
            )
            if not len(values) or not len(landscape_values):
                continue
            jitter = np.linspace(-0.055, 0.055, len(values))
            center = family_index + method_offsets[method]
            style = METHOD_STYLES[method]
            axis.scatter(
                center + jitter,
                values,
                s=25,
                marker=style["marker"],
                facecolor=FAMILY_COLORS[family],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.6,
                zorder=3,
            )
            median = float(np.median(landscape_values))
            axis.plot(
                [center - 0.095, center + 0.095],
                [median, median],
                color=FAMILY_COLORS[family],
                linewidth=3.0,
                solid_capstyle="round",
                zorder=4,
            )
    axis.set_xticks(range(len(FAMILY_ORDER)), [FAMILY_LABELS[item] for item in FAMILY_ORDER])
    if log_scale:
        axis.set_yscale("log")
    _style_axis(axis)


def build_reported_concordance() -> None:
    rows = _load_table("rank_percentiles")
    rows = rows[rows["cohort_top_n"].eq(100)].drop_duplicates(
        ["uncertainty_procedure", "analysis_id"]
    )
    rows = _shared_landscape_scope(rows)
    figure, axis = plt.subplots(figsize=(7.4, 3.25))
    _strip_points(
        axis,
        rows,
        value="reported_vs_posterior_median_rank_spearman",
    )
    axis.axhline(0, color=RULE_GRAY, linestyle="--", linewidth=1.1)
    axis.set(
        ylabel="Spearman correlation\npublished rank vs sampled-median rank",
        ylim=(-0.25, 1.03),
    )
    axis.text(
        0.99,
        0.97,
        "1 = identical ordering",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=METHOD_STYLES[method]["marker"],
                color="none",
                markerfacecolor=GRAY_TEXT,
                markeredgecolor="white",
                markersize=7,
                label=method,
            )
            for method in METHOD_ORDER
        ],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
    )
    figure.tight_layout()
    _save(figure, "reported_concordance")


def build_rank_example() -> None:
    rows = pd.read_csv(
        TABLE_ROOTS["HMC posterior"] / "model_rank_intervals.csv",
        low_memory=False,
    )
    analysis_id = "pdz__pdz-cis__hmc__lomax-exponential__small-clean-3"
    rows = (
        rows[rows["analysis_id"].eq(analysis_id)]
        .nsmallest(100, "model_reference_rank")
        .sort_values("model_reference_rank")
    )
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    axis.axvspan(0.5, 100.5, color="#E7F0E5", zorder=0)
    axis.hlines(
        y,
        rows["posterior_rank_lower"],
        rows["posterior_rank_upper"],
        color="#9CBAD0",
        linewidth=0.8,
        zorder=2,
    )
    axis.scatter(
        rows["posterior_median_rank"],
        y,
        color=BOX_RULE,
        s=10,
        zorder=3,
    )
    axis.scatter(
        rows["reported_rank"],
        y,
        color="#3F7A4F",
        marker="D",
        edgecolor="white",
        linewidth=0.25,
        s=12,
        zorder=4,
    )
    axis.axvline(100.5, color="#66835F", linewidth=1.0)
    displayed_ranks = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axis.set(
        xlabel="Full-landscape sampled rank",
        ylabel="Median-score reference rank",
        yticks=displayed_ranks - 1,
        yticklabels=displayed_ranks,
        xlim=(0, 210),
        xticks=[0, 50, 100, 150, 200],
    )
    axis.invert_yaxis()
    axis.legend(
        handles=[
            Line2D([0], [0], color="#9CBAD0", linewidth=2, label="90% sampled-rank interval"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BOX_RULE,
                label="Median sampled rank",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#3F7A4F",
                label="Published rank",
            ),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
    )
    _style_axis(axis, grid_axis="x")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    _save(figure, "rank_example_pdz")


def build_rank_resolution() -> None:
    intervals = _load_table("model_rank_intervals")
    runs = (
        intervals.groupby(
            [
                "uncertainty_procedure",
                "analysis_id",
                "protein_family",
                "landscape",
            ],
            as_index=False,
        )
        .agg(
            median_rank_width=("rank_interval_width", "median"),
            exit_fraction=("rank_interval_leaves_top_k", "mean"),
        )
    )
    runs = _shared_landscape_scope(runs)
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 3.55))
    _strip_points(axes[0], runs, value="median_rank_width", log_scale=True)
    axes[0].axhline(100, color=RULE_GRAY, linestyle="--", linewidth=1.1)
    axes[0].text(
        0.03,
        100,
        " one top-100 span",
        va="bottom",
        fontsize=9.5,
        color=GRAY_TEXT,
    )
    axes[0].set(
        title="Width of rank intervals",
        ylabel="Run median 90% rank-interval width",
    )

    _strip_points(axes[1], runs, value="exit_fraction")
    axes[1].set(
        title="Uncertain top-100 membership",
        ylabel="Fraction of model top 100 with\n95th-percentile rank > 100",
        ylim=(-0.04, 1.04),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=METHOD_STYLES[method]["marker"],
                color="none",
                markerfacecolor=GRAY_TEXT,
                markeredgecolor="white",
                markersize=7,
                label=method,
            )
            for method in METHOD_ORDER
        ],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.2)
    _save(figure, "rank_resolution")


def build_top_k_stability() -> None:
    curves = _load_table("top_k_curves")
    curves = _shared_landscape_scope(curves)
    curves = curves[
        [
            "uncertainty_procedure",
            "protein_family",
            "landscape",
            "model_spec",
            "k",
            "sample_self_consistency",
        ]
    ]
    landscape = (
        curves.groupby(
            ["uncertainty_procedure", "protein_family", "landscape", "k"],
            as_index=False,
        )["sample_self_consistency"]
        .median()
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for axis, family in zip(axes, FAMILY_ORDER, strict=True):
        for method in METHOD_ORDER:
            selected = landscape[
                landscape["protein_family"].eq(family)
                & landscape["uncertainty_procedure"].eq(method)
            ]
            summary = (
                selected.groupby("k")["sample_self_consistency"]
                .agg(
                    median="median",
                    lower=lambda values: values.quantile(0.25),
                    upper=lambda values: values.quantile(0.75),
                )
                .reset_index()
            )
            style = METHOD_STYLES[method]
            axis.fill_between(
                summary["k"],
                summary["lower"],
                summary["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                summary["k"],
                summary["median"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.5,
                linewidth=2.0,
                label=method,
            )
        axis.axvline(100, color=RULE_GRAY, linestyle=":", linewidth=1.0)
        axis.set(
            title=FAMILY_LABELS[family],
            xlabel="Top-set size K",
            xscale="log",
            ylim=(-0.03, 1.03),
        )
        axis.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        _style_axis(axis)
    axes[0].set_ylabel("Expected overlap of two sampled top-K sets")
    axes[1].text(
        100,
        0.04,
        "K=100",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9.5,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=METHOD_STYLES[method]["color"],
                linestyle=METHOD_STYLES[method]["linestyle"],
                marker=METHOD_STYLES[method]["marker"],
                label=method,
            )
            for method in METHOD_ORDER
        ],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    _save(figure, "top_k_stability")


def _decile(values: pd.Series) -> pd.Series:
    return np.ceil(values.rank(method="first", pct=True) * 10).clip(1, 10).astype(int)


def _mechanism_summary(
    rows: pd.DataFrame,
    *,
    predictor: str,
    response: str,
) -> pd.DataFrame:
    selected = rows[
        rows["is_primary_cohort"].astype(bool)
        & np.isfinite(rows[predictor])
        & np.isfinite(rows[response])
    ].copy()
    if (selected[response] < 0).any():
        raise ValueError(f"Mechanism response {response!r} must be non-negative.")
    selected["decile"] = selected.groupby(
        ["uncertainty_procedure", "analysis_id"],
        sort=False,
    )[predictor].transform(_decile)
    run = (
        selected.groupby(
            [
                "uncertainty_procedure",
                "analysis_id",
                "protein_family",
                "landscape",
                "decile",
            ],
            as_index=False,
        )[response]
        .median()
    )
    landscape = (
        run.groupby(
            ["uncertainty_procedure", "protein_family", "landscape", "decile"],
            as_index=False,
        )[response]
        .median()
    )
    return (
        landscape.groupby(
            ["uncertainty_procedure", "protein_family", "decile"],
            as_index=False,
        )[response]
        .median()
    )


def build_mechanism_chain() -> None:
    rows = _load_table("mechanism_variants")
    rows = _shared_landscape_scope(rows)
    support = _mechanism_summary(
        rows,
        predictor="mechanism_log_input_support",
        response="normalized_score_interval_width",
    )
    propagation = _mechanism_summary(
        rows,
        predictor="normalized_score_interval_width",
        response="rank_interval_width",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.65))
    for axis, summary, response in (
        (axes[0], support, "normalized_score_interval_width"),
        (axes[1], propagation, "rank_interval_width"),
    ):
        plot_floor = _positive_plot_floor(summary[response])
        for family in FAMILY_ORDER:
            for method in METHOD_ORDER:
                selected = summary[
                    summary["protein_family"].eq(family)
                    & summary["uncertainty_procedure"].eq(method)
                ]
                axis.plot(
                    selected["decile"],
                    np.maximum(selected[response], plot_floor),
                    color=FAMILY_COLORS[family],
                    linestyle=METHOD_STYLES[method]["linestyle"],
                    marker=METHOD_STYLES[method]["marker"],
                    markersize=3.5,
                    linewidth=2.0,
                )
        axis.set_yscale("log")
        axis.set_xticks([1, 3, 5, 7, 10])
        _style_axis(axis)
    axes[0].set(
        title="Experimental support → score precision",
        xlabel="Within-run input-support decile",
        ylabel="Median 90% score width / run score IQR",
    )
    axes[1].set(
        title="Score uncertainty → rank uncertainty",
        xlabel="Within-run score-uncertainty decile",
        ylabel="Median 90% rank-interval width",
    )
    family_handles = [
        Line2D([0], [0], color=FAMILY_COLORS[family], linewidth=2.5, label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=GRAY_TEXT,
            linestyle=METHOD_STYLES[method]["linestyle"],
            marker=METHOD_STYLES[method]["marker"],
            label=method,
        )
        for method in METHOD_ORDER
    ]
    figure.legend(
        handles=[*family_handles, *method_handles],
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.0)
    _save(figure, "mechanism_chain")


def _count_ticks(maximum: float) -> tuple[np.ndarray, list[str]]:
    values = np.array([0, 10, 1_000, 100_000, 1_000_000])
    values = values[np.log10(values + 1) <= maximum + 1e-9]
    labels = [
        "0"
        if value == 0
        else f"{value:g}"
        if value < 1_000
        else f"{value / 1_000:g}k"
        if value < 1_000_000
        else f"{value / 1_000_000:g}M"
        for value in values
    ]
    return np.log10(values + 1), labels


def _plot_contrast_points(axis, rows: pd.DataFrame, *, x: str, y: str) -> None:
    incumbent = rows["is_model_top_100"].astype(bool)
    for include, color, filled in (
        (incumbent, BOX_RULE, True),
        (~incumbent, ORANGE, False),
    ):
        selected = rows.loc[include]
        axis.scatter(
            selected[x],
            selected[y],
            s=11,
            color=color if filled else "none",
            edgecolor=color,
            linewidth=0.5,
            alpha=0.42,
            rasterized=True,
        )


def _plot_linear_trend(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
    y_range: tuple[float, float],
) -> None:
    x_values = rows[x].to_numpy(dtype=float)
    y_values = rows[y].to_numpy(dtype=float)
    if len(x_values) < 2 or np.ptp(x_values) <= 0:
        return
    slope, intercept = np.polyfit(x_values, y_values, 1)
    line_x = np.linspace(x_values.min(), x_values.max(), 200)
    line_y = intercept + slope * line_x
    visible = (line_y >= y_range[0]) & (line_y <= y_range[1])
    axis.plot(line_x[visible], line_y[visible], color=GRAY_TEXT, linewidth=1.5)


def build_figure10_contrast() -> None:
    root = TABLE_ROOTS["MLE predictive"]
    all_rows = pd.read_csv(root / "mechanism_variants.csv", low_memory=False)
    runs = pd.read_csv(root / "mechanism_runs.csv", low_memory=False).set_index(
        "analysis_id"
    )
    all_rows = all_rows[all_rows["is_primary_cohort"].astype(bool)].copy()
    floor = _positive_plot_floor(all_rows["normalized_score_interval_width"])
    family_rows = all_rows[all_rows["protein_family"].eq("trpb")].copy()
    family_rows["input_log_count"] = np.log10(
        family_rows["mechanism_input_count"] + 1
    )
    family_rows["output_log_count"] = np.log10(
        family_rows["mechanism_output_count"] + 1
    )
    family_rows["decision_margin"] = (
        family_rows["score_margin_to_rank_100"] / family_rows["reference_score_iqr"]
    )
    family_rows["score_uncertainty_plot"] = np.log10(
        np.maximum(family_rows["normalized_score_interval_width"], floor)
    )
    rows = family_rows[family_rows["analysis_id"].isin(CONTRAST_ANALYSES)].copy()
    if set(rows["analysis_id"]) != set(CONTRAST_ANALYSES):
        raise ValueError("The Figure 10 contrast analyses are incomplete.")
    if not rows.groupby("analysis_id")["is_model_top_100"].sum().eq(100).all():
        raise ValueError("Each contrast analysis must contain 100 fixed top members.")

    count_ranges = {
        "input_log_count": (
            -0.08,
            float(family_rows["input_log_count"].max()) * 1.02,
        ),
        "output_log_count": (
            -0.08,
            float(family_rows["output_log_count"].max()) * 1.02,
        ),
    }
    uncertainty = family_rows["score_uncertainty_plot"]
    uncertainty_padding = max(float(np.ptp(uncertainty)) * 0.04, 0.05)
    uncertainty_range = (
        float(uncertainty.min()) - uncertainty_padding,
        float(uncertainty.max()) + uncertainty_padding,
    )

    figure, axes = plt.subplots(2, 4, figsize=(12.2, 5.7), squeeze=False)
    column_titles = (
        "Input support",
        "Output abundance",
        "Margin vs score width",
        "Margin vs inclusion",
    )
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=7)

    for row_index, analysis_id in enumerate(CONTRAST_ANALYSES):
        analysis_rows = rows[rows["analysis_id"].eq(analysis_id)]
        decision_min = float(analysis_rows["decision_margin"].min())
        decision_max = float(analysis_rows["decision_margin"].max())
        decision_padding = max((decision_max - decision_min) * 0.04, 0.05)
        decision_range = (
            min(decision_min - decision_padding, -0.01),
            max(decision_max + decision_padding, 0.01),
        )

        for column_index, x_column in enumerate(
            ("input_log_count", "output_log_count")
        ):
            axis = axes[row_index, column_index]
            _plot_contrast_points(
                axis,
                analysis_rows,
                x=x_column,
                y="score_uncertainty_plot",
            )
            _plot_linear_trend(
                axis,
                analysis_rows,
                x=x_column,
                y="score_uncertainty_plot",
                y_range=uncertainty_range,
            )
            correlation = spearmanr(
                analysis_rows[x_column],
                analysis_rows["score_uncertainty_plot"],
            ).statistic
            axis.text(
                0.97,
                0.92,
                rf"$\rho = {correlation:.2f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=10.5,
            )
            axis.set(xlim=count_ranges[x_column], ylim=uncertainty_range)
            ticks, labels = _count_ticks(count_ranges[x_column][1])
            axis.set_xticks(ticks, labels)
            _style_axis(axis)

        uncertainty_axis = axes[row_index, 2]
        _plot_contrast_points(
            uncertainty_axis,
            analysis_rows,
            x="decision_margin",
            y="score_uncertainty_plot",
        )
        uncertainty_axis.axvline(0, color=RULE_GRAY, linestyle=":", linewidth=1)
        margin_correlation = spearmanr(
            analysis_rows["decision_margin"],
            analysis_rows["score_uncertainty_plot"],
        ).statistic
        uncertainty_axis.text(
            0.97,
            0.92,
            rf"$\rho = {margin_correlation:.2f}$",
            transform=uncertainty_axis.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
        )
        uncertainty_axis.set(xlim=decision_range, ylim=uncertainty_range)
        _style_axis(uncertainty_axis)

        inclusion_axis = axes[row_index, 3]
        _plot_contrast_points(
            inclusion_axis,
            analysis_rows,
            x="decision_margin",
            y="top_100_inclusion_probability",
        )
        inclusion_axis.axvline(0, color=RULE_GRAY, linestyle=":", linewidth=1)
        inclusion_axis.axhline(
            0.5,
            color=RULE_GRAY,
            linestyle="--",
            linewidth=0.8,
        )
        inclusion_axis.set(xlim=decision_range, ylim=(-0.03, 1.03))
        inclusion_axis.yaxis.set_major_formatter(PercentFormatter(1))
        _style_axis(inclusion_axis)

        run = runs.loc[analysis_id]
        axes[row_index, 0].text(
            -0.52,
            0.5,
            (
                f"{CONTRAST_LABELS[analysis_id]}\n"
                f"{run['top_k_instability']:.0%} instability\n"
                f"{int(run['challenger_count_q_ge_001'])} challengers"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        for column_index in (1, 2):
            axes[row_index, column_index].tick_params(labelleft=False)
        if row_index == 0:
            for axis in axes[row_index]:
                axis.tick_params(labelbottom=False)

    axes[-1, 0].set_xlabel("Input count")
    axes[-1, 1].set_xlabel("Output count")
    for axis in axes[-1, 2:]:
        axis.set_xlabel("Decision margin")
    axes[0, 0].set_ylabel("Log relative\n90% score width")
    axes[1, 0].set_ylabel("Log relative\n90% score width")

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BOX_RULE,
                markeredgecolor=BOX_RULE,
                label="Fixed median-score top 100",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=ORANGE,
                label="Outside challenger (q >= 1%)",
            ),
            Line2D([0], [0], color=GRAY_TEXT, linewidth=1.5, label="OLS trend"),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.99,
        top=0.91,
        bottom=0.18,
        hspace=0.27,
        wspace=0.32,
    )
    _save(figure, "figure10_contrast")


def main() -> None:
    _set_style()
    build_reported_concordance()
    build_rank_example()
    build_rank_resolution()
    build_top_k_stability()
    build_mechanism_chain()
    build_figure10_contrast()
    print(f"Saved talk assets to {OUTPUT}")


if __name__ == "__main__":
    main()
