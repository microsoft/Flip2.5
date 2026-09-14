"""Publication figure composition from paper summary tables."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde, spearmanr

from dms_fitness_stability.plotting import (
    DEFAULT_COLOR,
    INTERVAL_COLOR,
    REFERENCE_COLOR,
)

TABLE_NAMES = (
    "inventory",
    "rank_percentiles",
    "rank_agreement",
    "model_rank_intervals",
    "top_k_curves",
    "top_k_candidates",
)
MECHANISM_TABLE_NAMES = (
    "mechanism_inventory",
    "mechanism_variants",
    "mechanism_runs",
    "mechanism_effects",
)
FAMILY_ORDER = ("pdz", "trpb", "nuc", "k50")
FAMILY_COLORS = {
    "pdz": DEFAULT_COLOR,
    "trpb": REFERENCE_COLOR,
    "nuc": "#D55E00",
    "k50": "#009E73",
}
FAMILY_LABELS = {"pdz": "PDZ", "trpb": "TrpB", "nuc": "NucB", "k50": "K50"}
FAMILY_FILE_STEMS = {"pdz": "pdz", "trpb": "trpb", "nuc": "nucb", "k50": "k50"}
FAMILY_SCORE_LABELS = {
    "pdz": "Posterior-median log mean growth",
    "trpb": "Posterior-median log mean growth",
    "nuc": "Posterior-median modeled log fluorescence",
    "k50": "Posterior-median protease resistance",
}
SCORE_LABELS_BY_INFERENCE = {
    "hmc": FAMILY_SCORE_LABELS,
    "mle_predictive": {
        "pdz": "Median DiMSum-style proxy",
        "trpb": "Median Johnston fitness",
        "nuc": "Median highest-gate enrichment",
        "k50": "Median protease-specific log enrichment",
    },
    "hmc_count_based": {
        "pdz": "Median DiMSum-style proxy",
        "trpb": "Median Johnston fitness",
        "nuc": "Median highest-gate enrichment",
        "k50": "Median protease-specific log enrichment",
    },
}
LANDSCAPE_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#393B79",
    "#637939",
    "#8C6D31",
    "#7B4173",
)
MODEL_MARKERS = ("o", "s", "^", "D", "P", "X")
MODEL_LINESTYLES = ("-", "--", "-.", ":")
METRIC_STYLES = {
    "sample_self_consistency": (DEFAULT_COLOR, "Two sampled top-K sets"),
    "sample_vs_model_reference": (
        "#FF7F0E",
        "Sampled top-K vs median-score top-K",
    ),
    "sample_vs_reported": (REFERENCE_COLOR, "Sampled top-K vs published top-K"),
}
GRID_COLOR = "#E2E5E9"
AXIS_COLOR = "#8A8F98"
TEXT_COLOR = "#4E535B"
TOP_K_FILL = "#DCEAD8"
CONTEXT_FILL = "#ECEFF2"


def load_tables(table_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the six paper tables from one directory."""
    root = Path(table_dir)
    missing = [name for name in TABLE_NAMES if not (root / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing paper tables in {root}: {missing}")
    return {name: pd.read_csv(root / f"{name}.csv", low_memory=False) for name in TABLE_NAMES}


def load_mechanism_tables(table_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the four Section 4 mechanism tables from one directory."""
    root = Path(table_dir)
    missing = [
        name
        for name in MECHANISM_TABLE_NAMES
        if not (root / f"{name}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing mechanism tables in {root}: {missing}")
    return {
        name: pd.read_csv(root / f"{name}.csv", low_memory=False)
        for name in MECHANISM_TABLE_NAMES
    }


def load_functionality_scores(table_dir: str | Path) -> pd.DataFrame:
    """Load all-candidate posterior-median scores for Figure 8."""
    root = Path(table_dir)
    paths = (
        root / "functionality_scores.csv.gz",
        root / "functionality_scores.csv",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing functionality_scores.csv.gz or functionality_scores.csv in {root}."
        )
    return pd.read_csv(path, low_memory=False)


def set_paper_style() -> None:
    """Apply the established HMC paper visual language."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 0.8,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def _short_landscape(value: str) -> str:
    return (
        value.removeprefix("pdz-")
        .removeprefix("trpb-")
        .removeprefix("nuc-")
        .removeprefix("k50-")
        .replace("-", " ")
    )


def _legend_landscape(value: str) -> str:
    if value.startswith("pdz-"):
        name = value.removeprefix("pdz-")
        if name.startswith("cript-"):
            name = name.upper()
        return f"PDZ {name}"
    if value.startswith("trpb-"):
        return f"TrpB {value.removeprefix('trpb-')}"
    if value.startswith("nuc-"):
        return f"NucB {value.removeprefix('nuc-')}"
    if value.startswith("k50-"):
        return f"K50 {value.removeprefix('k50-').title()}"
    return value


def _family_order(inventory: pd.DataFrame) -> tuple[str, ...]:
    observed = set(inventory["protein_family"].astype(str))
    unknown = observed - set(FAMILY_ORDER)
    if unknown:
        raise ValueError(f"Unknown protein families: {sorted(unknown)}")
    return tuple(family for family in FAMILY_ORDER if family in observed)


def _short_model(value: str) -> str:
    if value == "(MASSIVE, only MLE)":
        return "MLE"
    replacements = {
        "exponential": "Exp",
        "gamma": "Gamma",
        "lomax": "Lomax",
        "sigmoid": "Sigmoid",
    }
    return "-".join(replacements.get(part, part.title()) for part in value.split("-"))


def _analysis_label(row: pd.Series) -> str:
    inference = "" if row["inference_method"] == "hmc" else f" | {row['inference_method']}"
    return f"{_short_landscape(row['landscape'])} | {_short_model(row['model_spec'])}{inference}"


def _inference_method(rows: pd.DataFrame) -> str:
    methods = rows["inference_method"].drop_duplicates().astype(str).tolist()
    if len(methods) != 1 or methods[0] not in SCORE_LABELS_BY_INFERENCE:
        raise ValueError(f"Expected one supported inference method, found {methods}.")
    return methods[0]


def _uncertainty_terms(inference_method: str) -> tuple[str, str]:
    if inference_method == "hmc":
        return "posterior median", "posterior score width"
    if inference_method == "mle_predictive":
        return "predictive proxy median", "predictive proxy width"
    if inference_method == "hmc_count_based":
        return "HMC count-proxy median", "HMC count-proxy width"
    raise ValueError(f"Unsupported inference method: {inference_method!r}.")


def _positive_plot_floor(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Uncertainty widths must be finite and non-negative.")
    positive = array[array > 0]
    if not len(positive):
        raise ValueError("At least one positive uncertainty width is required.")
    return float(positive.min() / 2)


def _variant_name(row: pd.Series, *, max_length: int) -> str:
    value = row.get("variant_label")
    if pd.isna(value) or not str(value).strip():
        value = row["variant_id"]
    value = str(value)
    return value if len(value) <= max_length else f"{value[: max_length - 3]}..."


def _ranked_variant_label(row: pd.Series, *, max_length: int) -> str:
    rank = row["model_reference_rank"]
    rank_label = f"{rank:g}" if np.isfinite(rank) else "unretained"
    return f"{rank_label}  ({_variant_name(row, max_length=max_length)})"


def _encodings(inventory: pd.DataFrame):
    landscapes = inventory.sort_values("analysis_order")["landscape"].drop_duplicates().tolist()
    models = inventory.sort_values("analysis_order")["model_spec"].drop_duplicates().tolist()
    landscape_colors = {
        landscape: LANDSCAPE_COLORS[index % len(LANDSCAPE_COLORS)]
        for index, landscape in enumerate(landscapes)
    }
    model_markers = {
        model: MODEL_MARKERS[index % len(MODEL_MARKERS)]
        for index, model in enumerate(models)
    }
    model_lines = {
        model: MODEL_LINESTYLES[index % len(MODEL_LINESTYLES)]
        for index, model in enumerate(models)
    }
    return landscape_colors, model_markers, model_lines


def _style_axis(axis, *, grid_axis: str = "both") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(AXIS_COLOR)
    axis.tick_params(colors=TEXT_COLOR)


def figure_reported_order(
    tables: Mapping[str, pd.DataFrame],
    *,
    cohort_top_n: int = 100,
    example_id: str | None = None,
) -> Figure:
    """Compose the descriptive reported-rank percentile figure."""
    percentiles = tables["rank_percentiles"]
    inventory = tables["inventory"].sort_values("analysis_order")
    selected = percentiles[percentiles["cohort_top_n"] == cohort_top_n]
    if selected.empty:
        raise ValueError(f"No rank-percentile rows for cohort top {cohort_top_n}.")
    landscape_colors, _, model_lines = _encodings(inventory)

    identifiable_families = set(selected["protein_family"])
    families = tuple(
        family
        for family in _family_order(inventory)
        if family in identifiable_families
    )
    n_panels = len(families) + 3
    n_columns = max(2, (n_panels + 1) // 2)
    figure, axes = plt.subplots(
        2,
        n_columns,
        figsize=(6.6 * n_columns, 9.2),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    example_axis = flat_axes[0]
    family_axes = flat_axes[1 : 1 + len(families)]
    diagnostic_axis = flat_axes[1 + len(families)]
    concordance_axis = flat_axes[2 + len(families)]
    example_id = example_id or inventory.iloc[0]["analysis_id"]
    if example_id not in set(inventory["analysis_id"]):
        raise ValueError(f"Unknown example analysis ID: {example_id}")
    example = selected[selected["analysis_id"] == example_id].sort_values(
        "reported_rank_sample_percentile"
    )
    example_axis.step(
        np.r_[0, example["reported_rank_sample_percentile"], 1],
        np.r_[0, example["ecdf"], 1],
        where="post",
        color=FAMILY_COLORS[example.iloc[0]["protein_family"]],
        linewidth=2,
    )
    example_axis.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle="--", linewidth=1)
    area = example["ecdf_to_uniform_area"].iloc[0]
    example_axis.text(0.04, 0.94, f"ECDF-to-uniform area = {area:.3f}", va="top")
    example_axis.set_title(f"Example analysis\n{_analysis_label(example.iloc[0])}")
    example_axis.set(
        xlabel=(
            "Fraction of samples ranked above the published rank\n"
            "(half credit for exact ties)"
        ),
        ylabel="Cumulative fraction of reported variants",
    )
    example_axis.legend(
        handles=[
            Line2D([0], [0], color=FAMILY_COLORS[example.iloc[0]["protein_family"]],
                   linewidth=2, label="Observed ECDF"),
            Line2D([0], [0], color=AXIS_COLOR, linestyle="--", linewidth=1,
                   label="Uniform reference"),
        ],
        frameon=False,
        loc="lower right",
    )

    for axis, family in zip(family_axes, families, strict=True):
        title = f"{FAMILY_LABELS[family]} analysis runs"
        family_rows = inventory[inventory["protein_family"] == family]
        for _, run in family_rows.iterrows():
            rows = selected[selected["analysis_id"] == run["analysis_id"]].sort_values(
                "reported_rank_sample_percentile"
            )
            axis.step(
                np.r_[0, rows["reported_rank_sample_percentile"], 1],
                np.r_[0, rows["ecdf"], 1],
                where="post",
                color=landscape_colors[run["landscape"]],
                linestyle=model_lines[run["model_spec"]],
                linewidth=1.15,
                alpha=0.8,
            )
        axis.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle="--", linewidth=1)
        axis.set_title(f"{title}: reported top {cohort_top_n}")
        axis.set(
            xlabel="Fraction of samples ranked above the published rank",
            ylabel="Cumulative fraction of reported variants",
        )
        family_landscapes = family_rows["landscape"].drop_duplicates().tolist()
        family_models = family_rows["model_spec"].drop_duplicates().tolist()
        handles = [
            Line2D(
                [0],
                [0],
                color=landscape_colors[landscape],
                linewidth=2,
                label=_short_landscape(landscape),
            )
            for landscape in family_landscapes
        ]
        handles.extend(
            Line2D(
                [0],
                [0],
                color=AXIS_COLOR,
                linestyle=model_lines[model],
                linewidth=1.5,
                label=_short_model(model),
            )
            for model in family_models
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=AXIS_COLOR,
                linestyle="--",
                linewidth=1,
                label="Uniform reference",
            )
        )
        axis.legend(
            handles=handles,
            frameon=False,
            ncol=2,
            fontsize=6,
            loc="lower right",
            title="Color = landscape; line = model",
            title_fontsize=6,
        )

    diagnostics = (
        selected.groupby(
            [
                "analysis_order",
                "analysis_id",
                "protein_family",
                "landscape",
                "inference_method",
                "model_spec",
            ],
            as_index=False,
        ).agg(
            ecdf_to_uniform_area=("ecdf_to_uniform_area", "first"),
            reported_vs_posterior_median_rank_spearman=(
                "reported_vs_posterior_median_rank_spearman",
                "first",
            ),
        )
        .sort_values("analysis_order")
    )
    _, model_markers, _ = _encodings(inventory)
    for _, row in diagnostics.iterrows():
        diagnostic_axis.scatter(
            row["analysis_order"],
            row["ecdf_to_uniform_area"],
            color=landscape_colors[row["landscape"]],
            marker=model_markers[row["model_spec"]],
            s=35,
            edgecolor="white",
            linewidth=0.5,
        )
    grouped = diagnostics.groupby("landscape", sort=False)["analysis_order"].mean()
    diagnostic_axis.set_xticks(
        grouped.to_numpy(),
        [_short_landscape(value) for value in grouped.index],
    )
    diagnostic_axis.tick_params(axis="x", rotation=45)
    diagnostic_axis.set(
        title="ECDF-to-uniform area",
        xlabel="Landscape (points are model/inference runs)",
        ylabel="Absolute ECDF area",
        ylim=(0, None),
    )
    model_specs = diagnostics["model_spec"].drop_duplicates().tolist()
    diagnostic_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=model_markers[model],
                linestyle="none",
                markerfacecolor=AXIS_COLOR,
                markeredgecolor="white",
                label=_short_model(model),
            )
            for model in model_specs
        ],
        frameon=False,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=len(model_specs),
        borderaxespad=0,
    )

    for _, row in diagnostics.iterrows():
        concordance_axis.scatter(
            row["analysis_order"],
            row["reported_vs_posterior_median_rank_spearman"],
            color=landscape_colors[row["landscape"]],
            marker=model_markers[row["model_spec"]],
            s=35,
            edgecolor="white",
            linewidth=0.5,
        )
    concordance_axis.axhline(0, color=AXIS_COLOR, linestyle="--", linewidth=1)
    concordance_axis.set_xticks(
        grouped.to_numpy(),
        [_short_landscape(value) for value in grouped.index],
    )
    concordance_axis.tick_params(axis="x", rotation=45)
    concordance_axis.set(
        title="Direct reported-model rank concordance",
        xlabel="Landscape (points are model/inference runs)",
        ylabel="Spearman rank correlation",
        ylim=(-1.05, 1.05),
    )

    for axis in flat_axes[n_panels:]:
        axis.set_axis_off()
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", flat_axes[:n_panels]):
        _panel_label(axis, label)
        is_curve_axis = axis not in {diagnostic_axis, concordance_axis}
        axis.set_xlim(0, 1) if is_curve_axis else None
        if is_curve_axis:
            axis.set_ylim(0, 1)
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        _style_axis(axis)
    figure.suptitle(
        f"Published ranks within sampled rank distributions | reported top {cohort_top_n}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _representative_ids(inventory: pd.DataFrame) -> tuple[str, ...]:
    ordered = inventory.sort_values("analysis_order")
    return tuple(
        ordered[ordered["protein_family"] == family].iloc[0]["analysis_id"]
        for family in _family_order(inventory)
    )


def _rank_interval_panel(
    axis,
    rows: pd.DataFrame,
    *,
    main_k: int,
    detail_top_n: int,
) -> None:
    displayed = rows.nsmallest(detail_top_n, "model_reference_rank").sort_values(
        "model_reference_rank"
    )
    y = np.arange(len(displayed))
    axis.axvspan(0.5, main_k + 0.5, color=TOP_K_FILL, alpha=0.65, zorder=0)
    axis.axvspan(
        main_k + 0.5,
        2 * main_k + 0.5,
        color=CONTEXT_FILL,
        alpha=0.8,
        zorder=0,
    )
    axis.hlines(
        y,
        displayed["posterior_rank_lower"],
        displayed["posterior_rank_upper"],
        color=INTERVAL_COLOR,
        linewidth=1.5,
    )
    axis.scatter(
        displayed["posterior_median_rank"],
        y,
        color=DEFAULT_COLOR,
        s=24,
        zorder=3,
    )
    reported = np.isfinite(displayed["reported_rank"])
    axis.scatter(
        displayed.loc[reported, "reported_rank"],
        y[reported],
        marker="D",
        color=REFERENCE_COLOR,
        edgecolor="white",
        linewidth=0.5,
        s=32,
        label="Published rank",
        zorder=4,
    )
    labels = [
        _ranked_variant_label(row, max_length=28)
        for _, row in displayed.iterrows()
    ]
    last_rank = int(displayed["model_reference_rank"].max())
    axis.set(
        title=(
            f"Median-score reference ranks 1-{last_rank}\n"
            f"{_analysis_label(displayed.iloc[0])}"
        ),
        xlabel="Full-landscape sample rank",
        ylabel="Median-score reference rank (variant)",
        yticks=y,
        yticklabels=labels,
    )
    axis.invert_yaxis()
    axis.tick_params(axis="y", labelsize=6.5)
    axis.axvline(main_k + 0.5, color="#66835F", linewidth=1)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INTERVAL_COLOR,
                linewidth=1.5,
                label="5th-95th percentile sampled rank",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DEFAULT_COLOR,
                markeredgecolor=DEFAULT_COLOR,
                label="Median sampled rank",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor=REFERENCE_COLOR,
                markeredgecolor="white",
                label="Published rank",
            ),
            Patch(facecolor=TOP_K_FILL, edgecolor="none", label=f"Sample rank in top {main_k}"),
            Patch(
                facecolor=CONTEXT_FILL,
                edgecolor="none",
                label=f"Sample rank {main_k + 1}-{2 * main_k}",
            ),
        ],
        frameon=False,
        fontsize=6,
        loc="lower right",
    )
    _style_axis(axis, grid_axis="x")


def figure_rank_uncertainty(
    tables: Mapping[str, pd.DataFrame],
    *,
    representative_ids: tuple[str, ...] | None = None,
    main_k: int = 100,
    detail_top_n: int = 50,
) -> Figure:
    """Compose exact model-cohort rank intervals and cross-run summaries."""
    inventory = tables["inventory"].sort_values("analysis_order")
    intervals = tables["model_rank_intervals"]
    representatives = representative_ids or _representative_ids(inventory)
    if not representatives:
        raise ValueError("At least one representative analysis ID is required.")
    missing = set(representatives) - set(intervals["analysis_id"])
    if missing:
        raise ValueError(f"Unknown representative analysis IDs: {sorted(missing)}")
    landscape_colors, model_markers, _ = _encodings(inventory)

    if detail_top_n < 1:
        raise ValueError("detail_top_n must be positive.")
    detail_top_n = min(detail_top_n, main_k)
    detail_height = max(5.0, 2.5 + 0.27 * detail_top_n)
    summary_height = max(5.0, 2.5 + 0.25 * len(inventory))
    figure = plt.figure(
        figsize=(
            max(14.2, 6.5 * len(representatives)),
            detail_height + summary_height,
        ),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(detail_height, summary_height),
        hspace=0.2,
    )
    representative_grid = grid[0].subgridspec(
        1,
        len(representatives),
        wspace=0.28,
    )
    summary_grid = grid[1].subgridspec(
        1,
        2,
        width_ratios=(1.15, 0.85),
        wspace=0.16,
    )
    axes = np.array(
        [
            *(
                figure.add_subplot(representative_grid[0, index])
                for index in range(len(representatives))
            ),
            figure.add_subplot(summary_grid[0, 0]),
            figure.add_subplot(summary_grid[0, 1]),
        ]
    )
    summary_axes = axes[-2:]
    for axis, analysis_id in zip(
        axes[: len(representatives)],
        representatives,
        strict=True,
    ):
        _rank_interval_panel(
            axis,
            intervals[intervals["analysis_id"] == analysis_id],
            main_k=main_k,
            detail_top_n=detail_top_n,
        )

    summaries = (
        intervals.groupby(
            [
                "analysis_order",
                "analysis_id",
                "protein_family",
                "landscape",
                "inference_method",
                "model_spec",
            ],
            as_index=False,
        )
        .agg(
            interval_width_q05=(
                "rank_interval_width",
                lambda values: values.quantile(0.05),
            ),
            interval_width_q25=(
                "rank_interval_width",
                lambda values: values.quantile(0.25),
            ),
            median_interval_width=("rank_interval_width", "median"),
            interval_width_q75=(
                "rank_interval_width",
                lambda values: values.quantile(0.75),
            ),
            interval_width_q95=(
                "rank_interval_width",
                lambda values: values.quantile(0.95),
            ),
            fraction_leaving_top_k=("rank_interval_leaves_top_k", "mean"),
        )
        .sort_values("analysis_order")
    )
    y = np.arange(len(summaries))
    labels = [_analysis_label(row) for _, row in summaries.iterrows()]
    boxplot = summary_axes[0].bxp(
        [
            {
                "whislo": row["interval_width_q05"],
                "q1": row["interval_width_q25"],
                "med": row["median_interval_width"],
                "q3": row["interval_width_q75"],
                "whishi": row["interval_width_q95"],
                "fliers": [],
            }
            for _, row in summaries.iterrows()
        ],
        positions=y,
        orientation="horizontal",
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for position, (_, row) in enumerate(summaries.iterrows()):
        color = landscape_colors[row["landscape"]]
        boxplot["boxes"][position].set(facecolor=to_rgba(color, 0.18), edgecolor=color)
        boxplot["medians"][position].set(color=color, linewidth=1.0)
        for artist in (
            *boxplot["whiskers"][2 * position : 2 * position + 2],
            *boxplot["caps"][2 * position : 2 * position + 2],
        ):
            artist.set(color=color, linewidth=0.9)
        summary_axes[0].scatter(
            row["median_interval_width"],
            position,
            color=color,
            marker=model_markers[row["model_spec"]],
            s=30,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    summary_axes[0].set(
        title=(
            "Distribution of 90% sampled-rank uncertainty\n"
            "(each box = one analysis run)"
        ),
        yticks=y,
        yticklabels=labels,
        xlabel=(
            "90% sampled-rank interval width (rank positions)\n"
            f"box = 25th-75th; whiskers = 5th-95th across reference top {main_k}"
        ),
        ylabel="",
    )
    summary_axes[0].invert_yaxis()
    _style_axis(summary_axes[0], grid_axis="x")
    summary_axes[0].axvline(
        main_k,
        color=AXIS_COLOR,
        linestyle=":",
        linewidth=1.2,
        zorder=0,
        label=f"K={main_k} reference",
    )
    summary_axes[0].text(
        main_k,
        0.02,
        f"K={main_k}",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=6,
        color=TEXT_COLOR,
        transform=axes[2].get_xaxis_transform(),
    )

    for position, (_, row) in enumerate(summaries.iterrows()):
        summary_axes[1].scatter(
            row["fraction_leaving_top_k"],
            position,
            color=landscape_colors[row["landscape"]],
            marker=model_markers[row["model_spec"]],
            s=36,
            edgecolor="white",
            linewidth=0.45,
        )
    summary_axes[1].set(
        title=f"Uncertain membership in the top {main_k}\n(each point = one analysis run)",
        yticks=y,
        xlabel=(
            f"Fraction of median-score reference top {main_k} with\n"
            f"95th-percentile sampled rank > {main_k}"
        ),
        ylabel="",
    )
    summary_axes[1].tick_params(labelleft=False)
    summary_axes[1].invert_yaxis()
    _style_axis(summary_axes[1], grid_axis="x")
    summary_axes[0].set_xscale("log")
    summary_axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    summary_axes[1].set_xlim(-0.02, 1.02)

    for label, axis in zip("abcd", axes):
        _panel_label(axis, label)
    figure.suptitle(
        f"Sampled rank uncertainty for the median-score reference top {main_k}",
        fontsize=13,
        fontweight="bold",
    )
    return figure


def _candidate_panel(axis, rows: pd.DataFrame, *, main_k: int) -> None:
    incumbents = rows[
        rows["is_model_reference_top_k"]
        & (rows["model_reference_rank"] > main_k - 10)
    ]
    challengers = rows[~rows["is_model_reference_top_k"]].nlargest(
        10, "top_k_probability"
    )
    displayed = (
        pd.concat([incumbents, challengers], ignore_index=True)
        .drop_duplicates("variant_index")
        .sort_values("model_reference_rank")
    )
    y = np.arange(len(displayed))
    axis.hlines(y, 0, displayed["top_k_probability"], color="#D3D8DE", linewidth=1)
    model_member = displayed["is_model_reference_top_k"].astype(bool)
    reported_member = displayed["is_reported_top_k"].fillna(False).astype(bool)
    categories = (
        (
            model_member & reported_member,
            "D",
            REFERENCE_COLOR,
            DEFAULT_COLOR,
            "In median-score and published top K",
        ),
        (
            model_member & ~reported_member,
            "o",
            DEFAULT_COLOR,
            "white",
            "Median-score top K only",
        ),
        (
            ~model_member & reported_member,
            "D",
            REFERENCE_COLOR,
            "white",
            "Published top K only",
        ),
        (
            ~model_member & ~reported_member,
            "o",
            INTERVAL_COLOR,
            "white",
            "Outside both: sampled challenger",
        ),
    )
    for selected, marker, color, edgecolor, label in categories:
        if not selected.any():
            continue
        axis.scatter(
            displayed.loc[selected, "top_k_probability"],
            y[selected],
            marker=marker,
            color=color,
            s=34,
            edgecolor=edgecolor,
            linewidth=1.0 if edgecolor == DEFAULT_COLOR else 0.5,
            label=label,
            zorder=3,
        )
    labels = [
        _ranked_variant_label(row, max_length=27)
        for _, row in displayed.iterrows()
    ]
    axis.set(
        title=(
            f"Median-score ranks {main_k - 9}-{main_k} + 10 outside challengers\n"
            f"{_analysis_label(displayed.iloc[0])}"
        ),
        xlabel=f"Top-{main_k} inclusion probability across samples",
        ylabel="Median-score reference rank (variant)",
        yticks=y,
        yticklabels=labels,
        xlim=(-0.02, 1.02),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.invert_yaxis()
    axis.legend(frameon=False, fontsize=6, loc="lower right", title="Candidate status")
    _style_axis(axis, grid_axis="x")


def _stability_curve_panel(
    axis,
    rows: pd.DataFrame,
    *,
    family: str,
    main_k: int,
) -> None:
    for metric, (color, label) in METRIC_STYLES.items():
        if not np.isfinite(rows[metric]).any():
            continue
        for _, run in rows.groupby("analysis_id", sort=False):
            axis.plot(
                run["k"],
                run[metric],
                color=color,
                linewidth=0.75,
                alpha=0.18,
            )
        landscape_curves = (
            rows.groupby(["landscape", "k"], as_index=False)[metric].median()
        )
        center = landscape_curves.groupby("k", as_index=False)[metric].median()
        axis.plot(
            center["k"],
            center[metric],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=label,
        )
    k_values = sorted(rows["k"].unique())
    axis.axvline(
        main_k,
        color=AXIS_COLOR,
        linestyle=":",
        linewidth=1.2,
        label=f"Main detailed view: K={main_k}",
    )
    if 50 in k_values and main_k != 50:
        axis.axvline(50, color=AXIS_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
        axis.text(
            50,
            0.02,
            "supporting K=50",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=6,
            color=TEXT_COLOR,
            transform=axis.get_xaxis_transform(),
        )
    axis.set(
        title=(
            f"{FAMILY_LABELS[family]} stability across K\n"
            "thin = analysis run; thick with points = landscape-weighted median"
        ),
        xlabel="Top-set size K (log scale)",
        ylabel="Expected fraction of top-K variants shared",
        xscale="log",
        ylim=(-0.02, 1.02),
    )
    axis.set_xticks(k_values, [str(value) for value in k_values])
    axis.set_xlim(min(k_values) * 0.9, max(k_values) * 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(
        frameon=False,
        loc="lower right",
        fontsize=6,
        title="Expected shared fraction",
        title_fontsize=6,
    )
    _style_axis(axis)


def figure_top_k_stability(
    tables: Mapping[str, pd.DataFrame],
    *,
    representative_ids: tuple[str, ...] | None = None,
    main_k: int = 100,
) -> Figure:
    """Compose candidate-boundary examples and dense top-K stability curves."""
    inventory = tables["inventory"].sort_values("analysis_order")
    candidates = tables["top_k_candidates"]
    curves = tables["top_k_curves"]
    families = _family_order(inventory)
    representatives = representative_ids or _representative_ids(inventory)
    if len(representatives) != len(families):
        raise ValueError(
            "representative_ids must contain one analysis for each protein family."
        )
    figure, axes = plt.subplots(
        2,
        len(families),
        figsize=(7.1 * len(families), 11.2),
        squeeze=False,
    )
    for axis, analysis_id in zip(axes[0], representatives, strict=True):
        _candidate_panel(
            axis,
            candidates[candidates["analysis_id"] == analysis_id],
            main_k=main_k,
        )
    for axis, family in zip(axes[1], families, strict=True):
        _stability_curve_panel(
            axis,
            curves[curves["protein_family"] == family],
            family=family,
            main_k=main_k,
        )
    for label, axis in zip("abcd", axes.ravel()):
        _panel_label(axis, label)
    figure.suptitle(
        "Top-K decision stability",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def figure_k50_top_k_stability(
    tables: Mapping[str, pd.DataFrame],
    *,
    main_k: int = 100,
) -> Figure:
    """Compare K50 candidate and top-K stability by protease."""
    inventory = tables["inventory"].sort_values("analysis_order")
    inventory = inventory[inventory["protein_family"] == "k50"]
    if len(inventory) != 2:
        raise ValueError("K50 top-K figure requires exactly two protease views.")
    candidates = tables["top_k_candidates"]
    curves = tables["top_k_curves"]
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 11.2), squeeze=False)
    for column, (_, analysis) in enumerate(inventory.iterrows()):
        analysis_id = analysis["analysis_id"]
        _candidate_panel(
            axes[0, column],
            candidates[candidates["analysis_id"] == analysis_id],
            main_k=main_k,
        )
        _stability_curve_panel(
            axes[1, column],
            curves[curves["analysis_id"] == analysis_id],
            family="k50",
            main_k=main_k,
        )
        axes[1, column].set_title(
            f"{_legend_landscape(analysis['landscape'])} stability across K"
        )
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _panel_label(axis, label)
    figure.suptitle(
        "K50 top-K decision stability by protease",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _quantile_bin(
    frame: pd.DataFrame,
    *,
    column: str,
    n_bins: int,
) -> pd.Series:
    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")
    return frame.groupby("analysis_id", sort=False)[column].transform(
        lambda values: np.ceil(
            values.rank(method="first", pct=True) * n_bins
        ).clip(1, n_bins)
    ).astype(np.int16)


def _run_decile_summary(
    variants: pd.DataFrame,
    *,
    predictor: str,
    response: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    required = {
        "analysis_id",
        "protein_family",
        "landscape",
        predictor,
        response,
    }
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Mechanism variants are missing columns: {sorted(missing)}")
    selected = variants[
        np.isfinite(variants[predictor]) & np.isfinite(variants[response])
    ].copy()
    if selected.empty:
        raise ValueError(f"No finite rows for {predictor!r} and {response!r}.")
    selected["support_decile"] = _quantile_bin(
        selected,
        column=predictor,
        n_bins=n_bins,
    )
    return (
        selected.groupby(
            [
                "analysis_id",
                "protein_family",
                "landscape",
                "support_decile",
            ],
            as_index=False,
        )
        .agg(
            predictor_median=(predictor, "median"),
            response_median=(response, "median"),
            n_variants=(response, "size"),
        )
    )


def _plot_run_deciles(
    axis,
    summary: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    for _, run in summary.groupby("analysis_id", sort=False):
        family = run["protein_family"].iloc[0]
        axis.plot(
            run["support_decile"],
            run["response_median"],
            color=FAMILY_COLORS[family],
            linewidth=0.7,
            alpha=0.16,
        )
    landscape = (
        summary.groupby(
            ["protein_family", "landscape", "support_decile"],
            as_index=False,
        )["response_median"]
        .median()
    )
    family_summary = (
        landscape.groupby(["protein_family", "support_decile"], as_index=False)[
            "response_median"
        ]
        .median()
    )
    for family, rows in family_summary.groupby("protein_family", sort=False):
        axis.plot(
            rows["support_decile"],
            rows["response_median"],
            color=FAMILY_COLORS[family],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=f"{FAMILY_LABELS[family]} landscape-weighted median",
        )
    axis.set(
        title=f"{title}\nthin = analysis run; thick = landscape-weighted median",
        xlabel=xlabel,
        ylabel=ylabel,
        yscale="log",
        xticks=np.arange(1, 11),
    )
    axis.legend(frameon=False, fontsize=6, loc="best")
    _style_axis(axis)


def _plot_crowding_interaction(axis, variants: pd.DataFrame) -> None:
    required = {
        "analysis_id",
        "landscape",
        "normalized_score_interval_width",
        "score_crowding",
        "rank_interval_width",
    }
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Mechanism variants are missing columns: {sorted(missing)}")
    selected = variants[
        np.isfinite(variants["normalized_score_interval_width"])
        & np.isfinite(variants["score_crowding"])
        & np.isfinite(variants["rank_interval_width"])
        & (variants["normalized_score_interval_width"] > 0)
        & (variants["rank_interval_width"] > 0)
    ].copy()
    selected["uncertainty_decile"] = _quantile_bin(
        selected,
        column="normalized_score_interval_width",
        n_bins=10,
    )
    selected["crowding_tertile"] = _quantile_bin(
        selected,
        column="score_crowding",
        n_bins=3,
    )
    run_summary = (
        selected.groupby(
            ["analysis_id", "landscape", "crowding_tertile", "uncertainty_decile"],
            as_index=False,
        )
        .agg(
            score_uncertainty=("normalized_score_interval_width", "median"),
            rank_uncertainty=("rank_interval_width", "median"),
        )
    )
    colors = {1: REFERENCE_COLOR, 2: "#FF7F0E", 3: "#B24C63"}
    labels = {1: "Low crowding", 2: "Medium crowding", 3: "High crowding"}
    for (_, tertile), rows in run_summary.groupby(
        ["analysis_id", "crowding_tertile"],
        sort=False,
    ):
        axis.plot(
            rows["score_uncertainty"],
            rows["rank_uncertainty"],
            color=colors[int(tertile)],
            linewidth=0.6,
            alpha=0.1,
        )
    landscape = (
        run_summary.groupby(
            ["landscape", "crowding_tertile", "uncertainty_decile"],
            as_index=False,
        )[["score_uncertainty", "rank_uncertainty"]]
        .median()
    )
    center = (
        landscape.groupby(
            ["crowding_tertile", "uncertainty_decile"],
            as_index=False,
        )[["score_uncertainty", "rank_uncertainty"]]
        .median()
    )
    for tertile, rows in center.groupby("crowding_tertile", sort=True):
        axis.plot(
            rows["score_uncertainty"],
            rows["rank_uncertainty"],
            color=colors[int(tertile)],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=labels[int(tertile)],
        )
    axis.set(
        title=(
            "Score and rank uncertainty by local crowding\n"
            "thin = analysis run; thick = landscape-weighted median"
        ),
        xlabel="Normalized 90% sampled score-interval width (log scale)",
        ylabel="90% sampled rank-interval width (log scale)",
        xscale="log",
        yscale="log",
    )
    axis.legend(
        frameon=False,
        fontsize=6,
        title="Within-run crowding tertile",
        title_fontsize=6,
    )
    _style_axis(axis)


def _select_score_geometry_pair(
    variants: pd.DataFrame,
    *,
    neighborhood: int = 5,
    min_rank: int = 10,
    max_rank: int = 95,
    max_rank_distance: int = 15,
    max_uncertainty_ratio: float = 1.2,
) -> tuple[pd.Series, pd.Series]:
    """Select a same-run pair matched on uncertainty but contrasting in spacing."""
    required = {
        "analysis_id",
        "is_model_top_100",
        "model_reference_rank",
        "normalized_score_interval_width",
        "normalized_local_score_spacing",
        "score_lower",
        "score_median",
        "score_upper",
        "reference_score_iqr",
        "rank_interval_width",
    }
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Mechanism variants are missing columns: {sorted(missing)}")
    if neighborhood < 1:
        raise ValueError("neighborhood must be positive.")
    if min_rank <= neighborhood or max_rank < min_rank:
        raise ValueError("rank limits must contain complete score neighborhoods.")
    if max_rank_distance < 1 or max_uncertainty_ratio < 1:
        raise ValueError("matching tolerances must be positive.")

    numeric = (
        "model_reference_rank",
        "normalized_score_interval_width",
        "normalized_local_score_spacing",
        "score_lower",
        "score_median",
        "score_upper",
        "reference_score_iqr",
        "rank_interval_width",
    )
    selected = variants[
        variants["is_model_top_100"].astype(bool)
        & np.isfinite(variants[list(numeric)]).all(axis=1)
        & (variants["normalized_score_interval_width"] > 0)
        & (variants["normalized_local_score_spacing"] > 0)
        & (variants["reference_score_iqr"] > 0)
    ].copy()
    candidates: list[tuple[tuple[object, ...], int, int]] = []
    for analysis_id, run in selected.groupby("analysis_id", sort=True):
        run = run.sort_values("model_reference_rank")
        ranks = run["model_reference_rank"].round().astype(int)
        if not np.allclose(run["model_reference_rank"], ranks):
            continue
        rank_set = set(ranks)
        complete = run[
            ranks.between(min_rank, max_rank)
            & ranks.map(
                lambda rank: all(
                    neighbor in rank_set
                    for neighbor in range(rank - neighborhood, rank + neighborhood + 1)
                )
            )
        ]
        records = list(complete.iterrows())
        for left_position, (left_index, left) in enumerate(records):
            for right_index, right in records[left_position + 1 :]:
                rank_distance = abs(
                    int(left["model_reference_rank"])
                    - int(right["model_reference_rank"])
                )
                if rank_distance > max_rank_distance:
                    continue
                left_uncertainty = float(left["normalized_score_interval_width"])
                right_uncertainty = float(right["normalized_score_interval_width"])
                uncertainty_ratio = max(left_uncertainty, right_uncertainty) / min(
                    left_uncertainty,
                    right_uncertainty,
                )
                if uncertainty_ratio > max_uncertainty_ratio:
                    continue
                if (
                    left["normalized_local_score_spacing"]
                    > right["normalized_local_score_spacing"]
                ):
                    sparse_index, sparse = left_index, left
                    crowded_index, crowded = right_index, right
                else:
                    sparse_index, sparse = right_index, right
                    crowded_index, crowded = left_index, left
                spacing_ratio = float(sparse["normalized_local_score_spacing"]) / float(
                    crowded["normalized_local_score_spacing"]
                )
                if spacing_ratio <= 1:
                    continue
                key = (
                    -spacing_ratio,
                    abs(np.log(left_uncertainty / right_uncertainty)),
                    rank_distance,
                    str(analysis_id),
                    int(sparse["model_reference_rank"]),
                    int(crowded["model_reference_rank"]),
                )
                candidates.append((key, sparse_index, crowded_index))
    if not candidates:
        raise ValueError("No matched sparse/crowded score-neighborhood pair was found.")
    _, sparse_index, crowded_index = min(candidates, key=lambda candidate: candidate[0])
    return variants.loc[sparse_index], variants.loc[crowded_index]


def _plot_score_geometry(
    axis,
    run: pd.DataFrame,
    focal: pd.Series,
    *,
    neighborhood: int,
    title: str,
) -> tuple[float, float]:
    focal_rank = int(focal["model_reference_rank"])
    displayed = run[
        run["model_reference_rank"].between(
            focal_rank - neighborhood,
            focal_rank + neighborhood,
        )
    ].sort_values("model_reference_rank")
    expected_ranks = np.arange(
        focal_rank - neighborhood,
        focal_rank + neighborhood + 1,
    )
    if not np.array_equal(
        displayed["model_reference_rank"].to_numpy(dtype=int),
        expected_ranks,
    ):
        raise ValueError(f"Incomplete score neighborhood around reference rank {focal_rank}.")
    scale = float(focal["reference_score_iqr"])
    center = float(focal["score_median"])
    lower = (displayed["score_lower"].to_numpy() - center) / scale
    median = (displayed["score_median"].to_numpy() - center) / scale
    upper = (displayed["score_upper"].to_numpy() - center) / scale
    ranks = displayed["model_reference_rank"].to_numpy(dtype=int)
    focal_position = np.flatnonzero(ranks == focal_rank)[0]
    median_term, _ = _uncertainty_terms(str(focal["inference_method"]))

    axis.hlines(ranks, lower, upper, color=INTERVAL_COLOR, linewidth=1.5)
    axis.scatter(median, ranks, color=DEFAULT_COLOR, s=24, zorder=3)
    axis.hlines(
        focal_rank,
        lower[focal_position],
        upper[focal_position],
        color=REFERENCE_COLOR,
        linewidth=2.4,
        zorder=4,
    )
    axis.scatter(
        median[focal_position],
        focal_rank,
        marker="D",
        color=REFERENCE_COLOR,
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=5,
    )
    axis.axvline(0, color=AXIS_COLOR, linestyle=":", linewidth=1)
    axis.set(
        title=f"{title}\n{_analysis_label(focal)} | focal rank {focal_rank}",
        xlabel=(
            f"Score relative to focal {median_term}\n"
            "(run top-1,000 median-score IQR units)"
        ),
        ylabel="Median-score reference rank",
        yticks=ranks,
    )
    axis.invert_yaxis()
    axis.text(
        0.98,
        0.03,
        (
            "Focal (score Q95 - Q05) / run IQR: "
            f"{focal['normalized_score_interval_width']:.2f}\n"
            "Median-score drop per rank / run IQR: "
            f"{focal['normalized_local_score_spacing']:.3f}\n"
            "Focal rank Q95 - Q05: "
            f"{focal['rank_interval_width']:.0f} rank positions"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": GRID_COLOR, "pad": 3},
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INTERVAL_COLOR,
                marker="o",
                markerfacecolor=DEFAULT_COLOR,
                markeredgecolor=DEFAULT_COLOR,
                label=f"Neighbor score Q05-Q95 and {median_term}",
            ),
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                marker="D",
                markerfacecolor=REFERENCE_COLOR,
                markeredgecolor="white",
                label="Matched focal variant's score summary",
            ),
        ],
        frameon=False,
        fontsize=6,
        loc="upper right",
    )
    _style_axis(axis, grid_axis="x")
    return float(lower.min()), float(upper.max())


def _rank_interval_overlap_fraction(rows: pd.DataFrame) -> float:
    """Return the fraction of variant pairs with intersecting rank intervals."""
    required = {"rank_lower", "rank_upper"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Rank intervals are missing columns: {sorted(missing)}")
    intervals = rows[["rank_lower", "rank_upper"]].to_numpy(dtype=float)
    if len(intervals) < 2:
        raise ValueError("At least two rank intervals are required.")
    if not np.isfinite(intervals).all() or np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError("Rank intervals must be finite with lower <= upper.")
    overlap_count = sum(
        max(intervals[left, 0], intervals[right, 0])
        <= min(intervals[left, 1], intervals[right, 1])
        for left, right in combinations(range(len(intervals)), 2)
    )
    pair_count = len(intervals) * (len(intervals) - 1) // 2
    return overlap_count / pair_count


def _select_rank_interval_windows(
    variants: pd.DataFrame,
    *,
    window_size: int = 10,
    min_rank: int = 10,
    max_rank: int = 90,
    max_center_distance: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select same-run rank windows with the largest interval-overlap contrast."""
    required = {
        "analysis_id",
        "model_reference_rank",
        "rank_lower",
        "rank_median",
        "rank_upper",
        "rank_interval_width",
    }
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Mechanism variants are missing columns: {sorted(missing)}")
    if window_size < 2:
        raise ValueError("window_size must be at least two.")
    if min_rank < 1 or max_rank - min_rank + 1 < 2 * window_size:
        raise ValueError("rank limits must accommodate two non-overlapping windows.")
    if max_center_distance < window_size:
        raise ValueError("max_center_distance must accommodate separate windows.")

    windows: list[dict[str, object]] = []
    numeric = (
        "model_reference_rank",
        "rank_lower",
        "rank_median",
        "rank_upper",
        "rank_interval_width",
    )
    selected = variants[np.isfinite(variants[list(numeric)]).all(axis=1)].copy()
    for analysis_id, run in selected.groupby("analysis_id", sort=True):
        ranks = run["model_reference_rank"].round().astype(int)
        if not np.allclose(run["model_reference_rank"], ranks):
            continue
        if ranks.duplicated().any():
            raise ValueError(f"{analysis_id}: model-reference ranks must be unique.")
        run = run.assign(_integer_rank=ranks).set_index("_integer_rank", drop=True)
        for start_rank in range(min_rank, max_rank - window_size + 2):
            rank_range = np.arange(start_rank, start_rank + window_size)
            if not set(rank_range).issubset(run.index):
                continue
            window = run.loc[rank_range].sort_values("model_reference_rank").copy()
            windows.append(
                {
                    "analysis_id": analysis_id,
                    "start_rank": start_rank,
                    "stop_rank": start_rank + window_size - 1,
                    "overlap_fraction": _rank_interval_overlap_fraction(window),
                    "median_width": float(window["rank_interval_width"].median()),
                    "rows": window,
                }
            )

    candidates: list[tuple[tuple[object, ...], pd.DataFrame, pd.DataFrame]] = []
    window_table = pd.DataFrame(windows)
    if window_table.empty:
        raise ValueError("No complete consecutive rank windows were found.")
    for analysis_id, run_windows in window_table.groupby("analysis_id", sort=True):
        records = run_windows.to_dict("records")
        for left, right in combinations(records, 2):
            if not (
                int(left["stop_rank"]) < int(right["start_rank"])
                or int(right["stop_rank"]) < int(left["start_rank"])
            ):
                continue
            left_center = (int(left["start_rank"]) + int(left["stop_rank"])) / 2
            right_center = (int(right["start_rank"]) + int(right["stop_rank"])) / 2
            center_distance = abs(left_center - right_center)
            if center_distance > max_center_distance:
                continue
            low, high = sorted(
                (left, right),
                key=lambda window: (
                    float(window["overlap_fraction"]),
                    int(window["start_rank"]),
                ),
            )
            difference = float(high["overlap_fraction"]) - float(
                low["overlap_fraction"]
            )
            if difference <= 0:
                continue
            key = (
                -difference,
                abs(float(high["median_width"]) - float(low["median_width"])),
                center_distance,
                str(analysis_id),
                int(low["start_rank"]),
                int(high["start_rank"]),
            )
            candidates.append((key, low["rows"], high["rows"]))
    if not candidates:
        raise ValueError("No same-run rank windows with different overlap were found.")
    _, separated, mixed = min(candidates, key=lambda candidate: candidate[0])
    return separated.copy(), mixed.copy()


def _plot_rank_interval_window(
    axis,
    rows: pd.DataFrame,
    *,
    title: str,
) -> tuple[float, float]:
    displayed = rows.sort_values("model_reference_rank")
    y = np.arange(len(displayed))
    axis.hlines(
        y,
        displayed["rank_lower"],
        displayed["rank_upper"],
        color=INTERVAL_COLOR,
        linewidth=1.7,
    )
    axis.scatter(
        displayed["rank_median"],
        y,
        color=DEFAULT_COLOR,
        s=28,
        zorder=3,
    )
    axis.scatter(
        displayed["model_reference_rank"],
        y,
        marker="D",
        color=REFERENCE_COLOR,
        edgecolor="white",
        linewidth=0.5,
        s=34,
        zorder=4,
    )
    start_rank = int(displayed["model_reference_rank"].min())
    stop_rank = int(displayed["model_reference_rank"].max())
    overlap = _rank_interval_overlap_fraction(displayed)
    median_width = float(displayed["rank_interval_width"].median())
    axis.set(
        title=(
            f"{title}\n{_analysis_label(displayed.iloc[0])} | "
            f"reference ranks {start_rank}-{stop_rank}"
        ),
        xlabel="Sampled full-landscape rank",
        ylabel="Fixed median-score reference rank",
        yticks=y,
        yticklabels=displayed["model_reference_rank"].astype(int),
    )
    axis.invert_yaxis()
    axis.text(
        0.98,
        0.04,
        (
            f"Overlapping interval pairs: {overlap:.0%}\n"
            f"Median Q95 - Q05 width: {median_width:g} ranks"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": GRID_COLOR, "pad": 3},
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INTERVAL_COLOR,
                marker="o",
                markerfacecolor=DEFAULT_COLOR,
                markeredgecolor=DEFAULT_COLOR,
                label="Sampled-rank Q05-Q95 and median",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor=REFERENCE_COLOR,
                markeredgecolor="white",
                label="Fixed median-score reference rank",
            ),
        ],
        frameon=False,
        fontsize=6,
        loc="upper right",
    )
    _style_axis(axis, grid_axis="x")
    return float(displayed["rank_lower"].min()), float(
        displayed["rank_upper"].max()
    )


def _plot_boundary_difficulty(axis, runs: pd.DataFrame) -> None:
    required = {
        "analysis_id",
        "protein_family",
        "landscape",
        "model_spec",
        "uncertainty_to_spacing_ratio",
        "top_k_instability",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Mechanism runs are missing columns: {sorted(missing)}")
    selected = runs[
        np.isfinite(runs["uncertainty_to_spacing_ratio"])
        & np.isfinite(runs["top_k_instability"])
        & (runs["uncertainty_to_spacing_ratio"] > 0)
    ].copy()
    if selected.empty:
        raise ValueError("No finite run-level boundary mechanism rows.")
    inventory = (
        selected[
            ["analysis_id", "protein_family", "landscape", "model_spec"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    inventory["analysis_order"] = np.arange(len(inventory))
    landscape_colors, model_markers, _ = _encodings(inventory)
    for _, row in selected.iterrows():
        axis.scatter(
            row["uncertainty_to_spacing_ratio"],
            row["top_k_instability"],
            color=landscape_colors[row["landscape"]],
            marker=model_markers[row["model_spec"]],
            s=40,
            edgecolor="white",
            linewidth=0.5,
        )
    landscape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=landscape_colors[landscape],
            markeredgecolor="white",
            label=_short_landscape(landscape),
        )
        for landscape in inventory["landscape"].drop_duplicates()
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[model],
            linestyle="none",
            markerfacecolor=AXIS_COLOR,
            markeredgecolor="white",
            label=_short_model(model),
        )
        for model in inventory["model_spec"].drop_duplicates()
    ]
    axis.set(
        title=(
            "Rank-100 boundary difficulty and decision instability\n"
            "each point = one landscape/model analysis run"
        ),
        xlabel="Score uncertainty / local median-score spacing near rank 100 (log scale)",
        ylabel="Top-100 instability = 1 - sample self-consistency",
        xscale="log",
        ylim=(-0.02, 1.02),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(
        handles=[*landscape_handles, *model_handles],
        frameon=False,
        fontsize=5.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        title="Color = landscape; marker = model",
        title_fontsize=6,
    )
    _style_axis(axis)


def figure_ranking_mechanisms(
    tables: Mapping[str, pd.DataFrame],
) -> Figure:
    """Compose Section 4 mechanisms from observational feature tables."""
    missing = set(MECHANISM_TABLE_NAMES) - set(tables)
    if missing:
        raise ValueError(f"Missing mechanism tables: {sorted(missing)}")
    variants = tables["mechanism_variants"]
    primary = variants[variants["is_primary_cohort"].astype(bool)].copy()
    if primary.empty:
        raise ValueError("Mechanism variants contain no primary-cohort rows.")
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.8))
    count_summary = _run_decile_summary(
        primary,
        predictor="mechanism_log_input_support",
        response="normalized_score_interval_width",
    )
    _plot_run_deciles(
        axes[0, 0],
        count_summary,
        title="Input-count support to score uncertainty",
        xlabel="Within-run input-support decile (1 = lowest)",
        ylabel=(
            "Median (score Q95 - Q05) / run score IQR\n"
            "(log scale)"
        ),
    )
    propagation_summary = _run_decile_summary(
        primary,
        predictor="normalized_score_interval_width",
        response="rank_interval_width",
    )
    _plot_run_deciles(
        axes[0, 1],
        propagation_summary,
        title="Score uncertainty to rank uncertainty",
        xlabel="Within-run score-uncertainty decile (1 = lowest)",
        ylabel=(
            "Median sampled-rank Q95 - Q05\n"
            "(rank positions; log scale)"
        ),
    )
    sparse, crowded = _select_score_geometry_pair(primary)
    if sparse["analysis_id"] != crowded["analysis_id"]:
        raise ValueError("Matched score-neighborhood examples must share one analysis run.")
    example_run = primary[primary["analysis_id"] == sparse["analysis_id"]]
    score_limits = [
        _plot_score_geometry(
            axes[1, 0],
            example_run,
            sparse,
            neighborhood=5,
            title="More-separated score neighborhood",
        ),
        _plot_score_geometry(
            axes[1, 1],
            example_run,
            crowded,
            neighborhood=5,
            title="Crowded score neighborhood",
        ),
    ]
    lower = min(limit[0] for limit in score_limits)
    upper = max(limit[1] for limit in score_limits)
    padding = max((upper - lower) * 0.04, 0.02)
    for axis in axes[1]:
        axis.set_xlim(lower - padding, upper + padding)
    for label, axis in zip("abcd", axes.ravel()):
        _panel_label(axis, label)
    figure.suptitle(
        "From experimental support to ranking resolution",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def figure_rank_space_sensitivity(
    tables: Mapping[str, pd.DataFrame],
) -> Figure:
    """Compose a temporary rank-space sensitivity view of ranking resolution."""
    missing = set(MECHANISM_TABLE_NAMES) - set(tables)
    if missing:
        raise ValueError(f"Missing mechanism tables: {sorted(missing)}")
    variants = tables["mechanism_variants"]
    primary = variants[variants["is_primary_cohort"].astype(bool)].copy()
    if primary.empty:
        raise ValueError("Mechanism variants contain no primary-cohort rows.")

    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.8))
    support_summary = _run_decile_summary(
        primary,
        predictor="mechanism_log_input_support",
        response="rank_interval_width",
    )
    _plot_run_deciles(
        axes[0, 0],
        support_summary,
        title="Input-count support to rank uncertainty",
        xlabel="Within-run input-support decile (1 = lowest)",
        ylabel=(
            "Median sampled-rank Q95 - Q05\n"
            "(rank positions; log scale)"
        ),
    )
    propagation_summary = _run_decile_summary(
        primary,
        predictor="score_interval_width",
        response="rank_interval_width",
    )
    _plot_run_deciles(
        axes[0, 1],
        propagation_summary,
        title="Score uncertainty to rank uncertainty",
        xlabel="Within-run raw score-width decile (1 = lowest)",
        ylabel=(
            "Median sampled-rank Q95 - Q05\n"
            "(rank positions; log scale)"
        ),
    )

    separated, mixed = _select_rank_interval_windows(primary)
    if separated["analysis_id"].iloc[0] != mixed["analysis_id"].iloc[0]:
        raise ValueError("Selected rank windows must share one analysis run.")
    rank_limits = [
        _plot_rank_interval_window(
            axes[1, 0],
            separated,
            title="More-separated sampled-rank intervals",
        ),
        _plot_rank_interval_window(
            axes[1, 1],
            mixed,
            title="Overlapping sampled-rank intervals",
        ),
    ]
    lower = min(limit[0] for limit in rank_limits)
    upper = max(limit[1] for limit in rank_limits)
    padding = max((upper - lower) * 0.04, 1)
    for axis in axes[1]:
        axis.set_xlim(max(0.5, lower - padding), upper + padding)
    for label, axis in zip("abcd", axes.ravel()):
        _panel_label(axis, label)
    figure.suptitle(
        "Temporary rank-space sensitivity view",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def _count_uncertainty_frame(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    missing = set(MECHANISM_TABLE_NAMES) - set(tables)
    if missing:
        raise ValueError(f"Missing mechanism tables: {sorted(missing)}")
    variants = tables["mechanism_variants"].copy()
    runs = tables["mechanism_runs"]
    required_variants = {
        "analysis_id",
        "protein_family",
        "landscape",
        "model_spec",
        "mechanism_input_count",
        "mechanism_output_count",
        "score_interval_width",
        "reference_score_iqr",
        "normalized_score_interval_width",
        "rank_lower",
        "rank_upper",
        "rank_interval_width",
    }
    missing_variants = required_variants - set(variants)
    if missing_variants:
        raise ValueError(
            f"Mechanism variants are missing columns: {sorted(missing_variants)}"
        )
    if "n_variants" not in runs:
        raise ValueError("Mechanism runs are missing column: n_variants")
    frame = variants.merge(
        runs[["analysis_id", "n_variants"]],
        on="analysis_id",
        how="left",
        validate="many_to_one",
    )
    numeric = (
        "mechanism_input_count",
        "mechanism_output_count",
        "score_interval_width",
        "reference_score_iqr",
        "normalized_score_interval_width",
        "rank_lower",
        "rank_upper",
        "rank_interval_width",
        "n_variants",
    )
    if not np.isfinite(frame[list(numeric)]).all().all():
        raise ValueError("Count-uncertainty figure inputs must be finite.")
    if (
        (frame["mechanism_input_count"] < 0).any()
        or (frame["mechanism_output_count"] < 0).any()
    ):
        raise ValueError("Observed counts must be non-negative.")
    if (frame["normalized_score_interval_width"] < 0).any():
        raise ValueError("Normalized score-interval widths must be non-negative.")
    if (frame["reference_score_iqr"] <= 0).any() or not np.allclose(
        frame["normalized_score_interval_width"],
        frame["score_interval_width"] / frame["reference_score_iqr"],
    ):
        raise ValueError("Normalized score widths must equal raw width / run score IQR.")
    if (frame["rank_interval_width"] < 0).any() or (frame["n_variants"] <= 0).any():
        raise ValueError("Rank widths must be non-negative and candidate counts positive.")
    if (
        (frame["rank_lower"] < 1).any()
        or (frame["rank_upper"] > frame["n_variants"]).any()
        or not np.allclose(
            frame["rank_interval_width"],
            frame["rank_upper"] - frame["rank_lower"],
        )
    ):
        raise ValueError("Rank intervals must be exact and lie in the fitted rank universe.")

    frame["input_log_count"] = np.log10(frame["mechanism_input_count"] + 1)
    frame["output_log_count"] = np.log10(frame["mechanism_output_count"] + 1)
    score_floor = _positive_plot_floor(frame["normalized_score_interval_width"])
    frame["score_uncertainty_floor"] = score_floor
    frame["score_uncertainty_plot"] = np.log10(
        np.maximum(frame["normalized_score_interval_width"], score_floor)
    )
    frame["relative_rank_uncertainty"] = (
        frame["rank_interval_width"] / frame["n_variants"]
    )
    if (frame["relative_rank_uncertainty"] > 1).any():
        raise ValueError("Rank-interval width cannot exceed the fitted candidate count.")
    return frame


def _count_uncertainty_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "x": "input_log_count",
            "y": "score_uncertainty_plot",
            "title": "Input count and relative score uncertainty",
            "xlabel": "Observed input count (log scale; zero retained)",
            "ylabel": "(score Q95 - Q05) / run score IQR (log scale)",
            "response": "score",
        },
        {
            "x": "input_log_count",
            "y": "relative_rank_uncertainty",
            "title": "Input count and relative rank uncertainty",
            "xlabel": "Observed input count (log scale; zero retained)",
            "ylabel": "(rank Q95 - Q05) / fitted candidate count",
            "response": "rank",
        },
        {
            "x": "output_log_count",
            "y": "score_uncertainty_plot",
            "title": "Output count and relative score uncertainty",
            "xlabel": "Observed output count (log scale; zero retained)",
            "ylabel": "(score Q95 - Q05) / run score IQR (log scale)",
            "response": "score",
        },
        {
            "x": "output_log_count",
            "y": "relative_rank_uncertainty",
            "title": "Output count and relative rank uncertainty",
            "xlabel": "Observed output count (log scale; zero retained)",
            "ylabel": "(rank Q95 - Q05) / fitted candidate count",
            "response": "rank",
        },
    )


def _count_ticks(maximum: float) -> tuple[np.ndarray, list[str]]:
    values = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    values = values[np.log10(values + 1) <= maximum + 1e-9]
    labels = [
        "0" if value == 0 else f"{value:g}" if value < 1_000 else f"{value / 1_000:g}k"
        if value < 1_000_000
        else f"{value / 1_000_000:g}M"
        for value in values
    ]
    return np.log10(values + 1), labels


def _score_ticks(lower: float, upper: float) -> tuple[np.ndarray, list[str]]:
    values = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    positions = np.log10(values)
    selected = (positions >= lower - 1e-9) & (positions <= upper + 1e-9)
    return positions[selected], [f"{value:g}" for value in values[selected]]


def _descriptive_trend(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
    y_range: tuple[float, float],
    linewidth: float = 2.2,
) -> None:
    x_values = rows[x].to_numpy(dtype=float)
    y_values = rows[y].to_numpy(dtype=float)
    if len(x_values) < 2 or np.ptp(x_values) <= 0:
        return
    slope, intercept = np.polyfit(x_values, y_values, 1)
    line_x = np.linspace(x_values.min(), x_values.max(), 200)
    line_y = intercept + slope * line_x
    visible = (line_y >= y_range[0]) & (line_y <= y_range[1])
    axis.plot(
        line_x[visible],
        line_y[visible],
        color=TEXT_COLOR,
        linewidth=linewidth,
        zorder=6,
    )


def _landscape_density_cmap(
    landscape: str,
    landscape_colors: Mapping[str, str],
) -> LinearSegmentedColormap:
    red, green, blue, _ = to_rgba(landscape_colors[landscape])
    return LinearSegmentedColormap.from_list(
        f"{landscape}_density",
        [(red, green, blue, 0.0), (red, green, blue, 0.95)],
    )


def _count_uncertainty_encodings(
    rows: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    inventory = rows[
        ["analysis_id", "landscape", "model_spec"]
    ].drop_duplicates("analysis_id").reset_index(drop=True)
    inventory["analysis_order"] = np.arange(len(inventory))
    landscape_colors, model_markers, _ = _encodings(inventory)
    return landscape_colors, model_markers


def _plot_analysis_medians(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
    landscape_colors: Mapping[str, str],
    model_markers: Mapping[str, str],
) -> None:
    centers = (
        rows.groupby(
            ["analysis_id", "landscape", "model_spec"],
            as_index=False,
        )[[x, y]]
        .median()
    )
    for _, center in centers.iterrows():
        axis.scatter(
            center[x],
            center[y],
            color=landscape_colors[center["landscape"]],
            marker=model_markers[center["model_spec"]],
            s=30,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.95,
            zorder=5,
        )


def _draw_count_uncertainty_panel(
    axis,
    rows: pd.DataFrame,
    spec: Mapping[str, object],
    *,
    renderer: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    landscape_colors: Mapping[str, str],
    model_markers: Mapping[str, str],
) -> None:
    x = str(spec["x"])
    y = str(spec["y"])
    if renderer == "scatter":
        for _, analysis_rows in rows.groupby("analysis_id", sort=False):
            landscape = analysis_rows["landscape"].iloc[0]
            model = analysis_rows["model_spec"].iloc[0]
            axis.scatter(
                analysis_rows[x],
                analysis_rows[y],
                color=landscape_colors[landscape],
                marker=model_markers[model],
                s=7,
                alpha=0.1,
                linewidth=0,
                rasterized=True,
            )
    elif renderer == "datashader":
        import datashader as ds
        import datashader.transfer_functions as tf

        shaded_rows = rows[[x, y, "landscape"]].copy()
        shaded_rows["landscape"] = pd.Categorical(
            shaded_rows["landscape"],
            categories=list(landscape_colors),
        )
        canvas = ds.Canvas(
            plot_width=720,
            plot_height=440,
            x_range=x_range,
            y_range=y_range,
        )
        aggregate = canvas.points(
            shaded_rows,
            x,
            y,
            agg=ds.count_cat("landscape"),
        )
        image = tf.shade(
            aggregate,
            color_key=landscape_colors,
            how="log",
            min_alpha=20,
        )
        axis.imshow(
            np.asarray(image.to_pil()),
            extent=(*x_range, *y_range),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            zorder=1,
        )
        _plot_analysis_medians(
            axis,
            rows,
            x=x,
            y=y,
            landscape_colors=landscape_colors,
            model_markers=model_markers,
        )
    elif renderer == "hexbin":
        for landscape, landscape_rows in rows.groupby("landscape", sort=False):
            axis.hexbin(
                landscape_rows[x],
                landscape_rows[y],
                gridsize=55,
                extent=(*x_range, *y_range),
                mincnt=1,
                bins="log",
                cmap=_landscape_density_cmap(landscape, landscape_colors),
                linewidths=0,
                zorder=2,
            )
        _plot_analysis_medians(
            axis,
            rows,
            x=x,
            y=y,
            landscape_colors=landscape_colors,
            model_markers=model_markers,
        )
    else:
        raise ValueError("renderer must be 'scatter', 'datashader', or 'hexbin'.")

    _descriptive_trend(axis, rows, x=x, y=y, y_range=y_range)
    axis.set(
        title=str(spec["title"]),
        xlabel=str(spec["xlabel"]),
        ylabel=str(spec["ylabel"]),
        xlim=x_range,
        ylim=y_range,
    )
    x_ticks, x_labels = _count_ticks(x_range[1])
    axis.set_xticks(x_ticks, x_labels)
    if spec["response"] == "score":
        y_ticks, y_labels = _score_ticks(*y_range)
        axis.set_yticks(y_ticks, y_labels)
    else:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    _style_axis(axis)


def _figure_count_uncertainty(
    tables: Mapping[str, pd.DataFrame],
    *,
    renderer: str,
) -> Figure:
    rows = _count_uncertainty_frame(tables)
    specs = _count_uncertainty_specs()
    landscape_colors, model_markers = _count_uncertainty_encodings(rows)
    x_ranges = {
        "input_log_count": (-0.08, float(rows["input_log_count"].max()) * 1.02),
        "output_log_count": (-0.08, float(rows["output_log_count"].max()) * 1.02),
    }
    score_values = rows["score_uncertainty_plot"]
    score_padding = max(float(np.ptp(score_values)) * 0.03, 0.05)
    y_ranges = {
        "score": (
            float(score_values.min()) - score_padding,
            float(score_values.max()) + score_padding,
        ),
        "rank": (-0.015, max(1.0, float(rows["relative_rank_uncertainty"].max()) * 1.03)),
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.8))
    for axis, spec in zip(axes.ravel(), specs):
        _draw_count_uncertainty_panel(
            axis,
            rows,
            spec,
            renderer=renderer,
            x_range=x_ranges[str(spec["x"])],
            y_range=y_ranges[str(spec["response"])],
            landscape_colors=landscape_colors,
            model_markers=model_markers,
        )
    for label, axis in zip("abcd", axes.ravel()):
        _panel_label(axis, label)
    landscape_handles = [
        Patch(
            facecolor=landscape_colors[landscape],
            edgecolor="none",
            label=_legend_landscape(landscape),
        )
        for landscape in landscape_colors
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[model],
            linestyle="none",
            markerfacecolor=TEXT_COLOR,
            markeredgecolor="white",
            markersize=8.5,
            label=_short_model(model),
        )
        for model in model_markers
    ]
    trend_handle = Line2D(
        [0],
        [0],
        color=TEXT_COLOR,
        linewidth=2.2,
        label="Pooled descriptive linear trend",
    )
    handles = [
        *landscape_handles,
        *model_handles,
        trend_handle,
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        handleheight=1.2,
        handletextpad=0.7,
        columnspacing=1.3,
        labelspacing=0.8,
        title=(
            "Color = landscape; marker = model family; "
            "line = pooled least-squares fit in plotted coordinates"
        ),
        title_fontsize=9,
    )
    figure.suptitle(
        f"Count support and fitted uncertainty ({renderer})",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.22, 1, 0.97))
    return figure


def figure_count_uncertainty_scatter(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show all tracked variants as standard scatter points."""
    return _figure_count_uncertainty(tables, renderer="scatter")


def figure_count_uncertainty_datashader(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Rasterize all tracked variants with Datashader category mixing."""
    return _figure_count_uncertainty(tables, renderer="datashader")


def figure_count_uncertainty_hexbin(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Aggregate all tracked variants into family-colored hexagonal bins."""
    return _figure_count_uncertainty(tables, renderer="hexbin")


def _count_uncertainty_legend_handles(
    landscape_colors: Mapping[str, str],
    model_markers: Mapping[str, str],
    *,
    trend_label: str,
) -> list[Patch | Line2D]:
    landscape_handles = [
        Patch(
            facecolor=landscape_colors[landscape],
            edgecolor="none",
            label=_legend_landscape(landscape),
        )
        for landscape in landscape_colors
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[model],
            linestyle="none",
            markerfacecolor=TEXT_COLOR,
            markeredgecolor="white",
            markersize=8.5,
            label=_short_model(model),
        )
        for model in model_markers
    ]
    return [
        *landscape_handles,
        *model_handles,
        Line2D(
            [0],
            [0],
            color=TEXT_COLOR,
            linewidth=2.2,
            label=trend_label,
        ),
    ]


def _figure_count_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
    response: str,
) -> Figure:
    if protein_family not in FAMILY_COLORS:
        raise ValueError(f"Unknown protein family: {protein_family!r}")
    if response not in {"score", "rank"}:
        raise ValueError("response must be 'score' or 'rank'.")
    all_rows = _count_uncertainty_frame(tables)
    rows = all_rows[all_rows["protein_family"] == protein_family].copy()
    if rows.empty:
        raise ValueError(f"No tracked rows for protein family {protein_family!r}.")
    landscape_colors, model_markers = _count_uncertainty_encodings(all_rows)
    family_landscapes = rows["landscape"].drop_duplicates().tolist()
    family_models = rows["model_spec"].drop_duplicates().tolist()
    family_landscape_colors = {
        landscape: landscape_colors[landscape] for landscape in family_landscapes
    }
    family_model_markers = {model: model_markers[model] for model in family_models}
    analyses = rows[
        ["analysis_id", "landscape", "model_spec", "inference_method"]
    ].drop_duplicates("analysis_id")
    n_analyses = len(analyses)

    y_column = (
        "score_uncertainty_plot" if response == "score" else "relative_rank_uncertainty"
    )
    if response == "score":
        y_values = rows[y_column]
        y_padding = max(float(np.ptp(y_values)) * 0.03, 0.05)
        y_range = (
            float(y_values.min()) - y_padding,
            float(y_values.max()) + y_padding,
        )
        y_label = "(score Q95 - Q05) / run score IQR (log scale)"
        response_name = "relative score uncertainty"
    else:
        y_range = (
            -0.015,
            max(1.0, float(rows[y_column].max()) * 1.03),
        )
        y_label = "(rank Q95 - Q05) / fitted candidate count"
        response_name = "relative rank uncertainty"
    x_ranges = {
        "input_log_count": (-0.08, float(rows["input_log_count"].max()) * 1.02),
        "output_log_count": (-0.08, float(rows["output_log_count"].max()) * 1.02),
    }

    figure_height = 3.6 + 1.8 * n_analyses
    figure, axes = plt.subplots(
        n_analyses,
        2,
        figsize=(14.2, figure_height),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    for row_index, (_, analysis) in enumerate(analyses.iterrows()):
        analysis_rows = rows[rows["analysis_id"] == analysis["analysis_id"]]
        color = family_landscape_colors[analysis["landscape"]]
        marker = family_model_markers[analysis["model_spec"]]
        for column_index, x_column in enumerate(
            ("input_log_count", "output_log_count")
        ):
            axis = axes[row_index, column_index]
            axis.scatter(
                analysis_rows[x_column],
                analysis_rows[y_column],
                color=color,
                marker=marker,
                s=8,
                alpha=0.13,
                linewidth=0,
                rasterized=True,
            )
            _descriptive_trend(
                axis,
                analysis_rows,
                x=x_column,
                y=y_column,
                y_range=y_range,
            )
            axis.set(xlim=x_ranges[x_column], ylim=y_range)
            x_ticks, x_labels = _count_ticks(x_ranges[x_column][1])
            axis.set_xticks(x_ticks, x_labels)
            if response == "score":
                y_ticks, y_labels = _score_ticks(*y_range)
                axis.set_yticks(y_ticks, y_labels)
            else:
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            _style_axis(axis)
        axes[row_index, 0].set_title(
            _analysis_label(analysis),
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=7,
        )

    axes[-1, 0].set_xlabel("Observed input count (log scale; zero retained)")
    axes[-1, 1].set_xlabel("Observed output count (log scale; zero retained)")
    figure.supylabel(
        y_label,
        x=0.025 if protein_family == "k50" else 0.012,
        fontsize=9,
    )
    figure.text(
        0.29,
        0.93,
        "Input count",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.76,
        0.93,
        "Output count",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    handles = _count_uncertainty_legend_handles(
        family_landscape_colors,
        family_model_markers,
        trend_label="Within-analysis descriptive linear trend",
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        handleheight=1.2,
        handletextpad=0.7,
        columnspacing=1.3,
        labelspacing=0.8,
        title="Color = landscape; marker = model family",
        title_fontsize=9,
    )
    figure.suptitle(
        f"{FAMILY_LABELS[protein_family]} count support and {response_name} by analysis",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=(0.035, 0.075, 1, 0.9), h_pad=1.15, w_pad=1.2)
    return figure


def figure_count_score_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
) -> Figure:
    """Show input/output count versus relative score uncertainty for each run."""
    return _figure_count_uncertainty_by_analysis(
        tables,
        protein_family=protein_family,
        response="score",
    )


def figure_count_rank_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
) -> Figure:
    """Show input/output count versus relative rank uncertainty for each run."""
    return _figure_count_uncertainty_by_analysis(
        tables,
        protein_family=protein_family,
        response="rank",
    )


def _plot_count_score_corner_points(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> None:
    model_top = rows["is_model_top_100"]
    challenger = rows["is_challenger_q_ge_001"]
    context = ~(model_top | challenger)
    axis.scatter(
        rows.loc[context, x],
        rows.loc[context, y],
        color=AXIS_COLOR,
        marker=".",
        s=2,
        alpha=0.24,
        linewidth=0,
        rasterized=True,
        zorder=1,
        label="Reported top-1,000 context only",
    )
    axis.scatter(
        rows.loc[model_top, x],
        rows.loc[model_top, y],
        color=DEFAULT_COLOR,
        marker="o",
        s=4,
        alpha=0.65,
        linewidth=0,
        rasterized=True,
        zorder=2,
        label="Median-score top 100",
    )
    axis.scatter(
        rows.loc[challenger, x],
        rows.loc[challenger, y],
        facecolors="none",
        edgecolors="#FF7F0E",
        marker="o",
        s=8,
        alpha=0.85,
        linewidth=0.4,
        rasterized=True,
        zorder=3,
        label="Outside challenger (q >= 1%)",
    )


def _plot_count_score_corner_density(
    axis,
    values: pd.Series,
    *,
    x_range: tuple[float, float],
) -> None:
    array = values.to_numpy(dtype=float)
    grid = np.linspace(*x_range, 256)
    if np.ptp(array) > 0:
        try:
            density = gaussian_kde(array, bw_method="scott")(grid)
        except (np.linalg.LinAlgError, ValueError):
            density = None
        if density is not None and np.isfinite(density).all():
            axis.fill_between(
                grid,
                density,
                color="black",
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(grid, density, color="black", linewidth=1.05)
        else:
            axis.axvline(np.median(array), color="black", linewidth=1.05)
    else:
        axis.axvline(array[0], color="black", linewidth=1.05)
    axis.set_xlim(x_range)
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="y", left=False, labelleft=False)
    _style_axis(axis)


def figure_count_score_corner_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
    n_columns: int = 3,
) -> Figure:
    """Show count and score uncertainty with models stacked by landscape."""
    if protein_family not in FAMILY_COLORS:
        raise ValueError(f"Unknown protein family: {protein_family!r}")
    if n_columns < 1:
        raise ValueError("n_columns must be positive.")
    all_rows = _count_uncertainty_frame(tables)
    rows = all_rows[all_rows["protein_family"] == protein_family].copy()
    if rows.empty:
        raise ValueError(f"No tracked rows for protein family {protein_family!r}.")
    status_columns = {"is_model_top_100", "is_challenger_q_ge_001"}
    missing_status = status_columns - set(rows)
    if missing_status:
        raise ValueError(
            f"Corner-plot variants are missing columns: {sorted(missing_status)}"
        )
    if rows[list(status_columns)].isna().any().any():
        raise ValueError("Corner-plot status flags must not be missing.")
    rows[list(status_columns)] = rows[list(status_columns)].astype(bool)
    if (rows["is_model_top_100"] & rows["is_challenger_q_ge_001"]).any():
        raise ValueError("Model-top variants and outside challengers must be disjoint.")
    inference_method = _inference_method(rows)
    _, width_term = _uncertainty_terms(inference_method)
    analyses = rows[
        ["analysis_id", "landscape", "model_spec", "inference_method"]
    ].drop_duplicates("analysis_id")
    landscapes = analyses["landscape"].drop_duplicates().tolist()
    resolved_columns = min(n_columns, len(landscapes))
    landscape_bands = [
        landscapes[start : start + resolved_columns]
        for start in range(0, len(landscapes), resolved_columns)
    ]
    band_heights = [
        max(int((analyses["landscape"] == landscape).sum()) for landscape in band)
        for band in landscape_bands
    ]
    band_offsets = np.cumsum([0, *band_heights[:-1]])
    n_model_rows = sum(band_heights)

    score_values = rows["score_uncertainty_plot"]
    score_padding = max(float(np.ptp(score_values)) * 0.03, 0.05)
    variable_specs = (
        {
            "column": "input_log_count",
            "label": "Input count",
            "range": (
                -0.08,
                max(1.0, float(rows["input_log_count"].max()) * 1.02),
            ),
        },
        {
            "column": "output_log_count",
            "label": "Output count",
            "range": (
                -0.08,
                max(1.0, float(rows["output_log_count"].max()) * 1.02),
            ),
        },
        {
            "column": "score_uncertainty_plot",
            "label": "Relative score\nuncertainty",
            "range": (
                float(score_values.min()) - score_padding,
                float(score_values.max()) + score_padding,
            ),
        },
    )
    tick_specs: list[tuple[np.ndarray, list[str]]] = []
    for index, spec in enumerate(variable_specs):
        if index < 2:
            positions, labels = _count_ticks(float(spec["range"][1]))
        else:
            positions, labels = _score_ticks(*spec["range"])
        if len(positions) > 4:
            selected = np.unique(
                np.rint(np.linspace(0, len(positions) - 1, 4)).astype(int)
            )
            positions = positions[selected]
            labels = [labels[position] for position in selected]
        tick_specs.append((positions, labels))

    figure = plt.figure(
        figsize=(14.2, 1.2 + 4.2 * n_model_rows),
        constrained_layout=True,
    )
    outer_grid = figure.add_gridspec(n_model_rows, resolved_columns)
    for band, row_offset in zip(landscape_bands, band_offsets):
        for column_index, landscape in enumerate(band):
            landscape_analyses = analyses[analyses["landscape"] == landscape]
            for model_position, (_, analysis) in enumerate(
                landscape_analyses.iterrows()
            ):
                row_index = int(row_offset) + model_position
                subfigure = figure.add_subfigure(
                    outer_grid[row_index, column_index]
                )
                axes = subfigure.subplots(3, 3, squeeze=False)
                analysis_rows = rows[
                    rows["analysis_id"] == analysis["analysis_id"]
                ]

                for variable_index, spec in enumerate(variable_specs):
                    _plot_count_score_corner_density(
                        axes[variable_index, variable_index],
                        analysis_rows[str(spec["column"])],
                        x_range=spec["range"],
                    )

                for y_index in range(1, len(variable_specs)):
                    y_spec = variable_specs[y_index]
                    for x_index in range(y_index):
                        x_spec = variable_specs[x_index]
                        axis = axes[y_index, x_index]
                        _plot_count_score_corner_points(
                            axis,
                            analysis_rows,
                            x=str(x_spec["column"]),
                            y=str(y_spec["column"]),
                        )
                        axis.set(xlim=x_spec["range"], ylim=y_spec["range"])
                        _style_axis(axis)

                        correlation_axis = axes[x_index, y_index]
                        x_values = analysis_rows[
                            str(x_spec["column"])
                        ].to_numpy(dtype=float)
                        y_values = analysis_rows[
                            str(y_spec["column"])
                        ].to_numpy(dtype=float)
                        if np.ptp(x_values) > 0 and np.ptp(y_values) > 0:
                            correlation = float(
                                spearmanr(x_values, y_values).statistic
                            )
                            correlation_label = rf"$\rho$ = {correlation:.2f}"
                        else:
                            correlation_label = r"$\rho$ = n/a"
                        correlation_axis.set_axis_off()
                        correlation_axis.text(
                            0.5,
                            0.5,
                            correlation_label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=TEXT_COLOR,
                        )

                for y_index, row_axes in enumerate(axes):
                    for x_index, axis in enumerate(row_axes):
                        if x_index <= y_index:
                            x_positions, x_labels = tick_specs[x_index]
                            axis.set_xticks(x_positions)
                            if y_index == len(variable_specs) - 1:
                                axis.set_xticklabels(x_labels)
                                axis.set_xlabel(
                                    variable_specs[x_index]["label"],
                                    fontsize=7,
                                )
                            else:
                                axis.tick_params(axis="x", labelbottom=False)
                            if x_index == 0 and y_index > 0:
                                y_positions, y_labels = tick_specs[y_index]
                                axis.set_yticks(y_positions, y_labels)
                                axis.set_ylabel(
                                    variable_specs[y_index]["label"],
                                    fontsize=7,
                                )
                            elif x_index < y_index:
                                axis.tick_params(axis="y", labelleft=False)
                            axis.tick_params(labelsize=6, pad=1.5)

                model_title = (
                    f"{_short_model(analysis['model_spec'])} | "
                    f"n={len(analysis_rows):,}"
                )
                title = (
                    f"{_legend_landscape(landscape)}\n{model_title}"
                    if model_position == 0
                    else model_title
                )
                subfigure.suptitle(
                    title,
                    x=0.04,
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                )

    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color=AXIS_COLOR,
                marker=".",
                linestyle="none",
                markersize=3,
                alpha=0.55,
                label="Reported top-1,000 context only",
            ),
            Line2D(
                [0],
                [0],
                color=DEFAULT_COLOR,
                marker="o",
                linestyle="none",
                markersize=3.5,
                label="Median-score top 100",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#FF7F0E",
                linestyle="none",
                markersize=4.5,
                label="Outside challenger (q >= 1%)",
            ),
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    figure.suptitle(
        (
            f"{FAMILY_LABELS[protein_family]} input count, output count, and "
            f"{width_term} by landscape and model"
        ),
        fontsize=13,
        fontweight="bold",
    )
    return figure


def _functionality_frame(
    inventory: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    inventory_columns = (
        "analysis_order",
        "analysis_id",
        "protein_family",
        "landscape",
        "inference_method",
        "model_spec",
        "n_variants",
    )
    score_columns = (
        "analysis_id",
        "variant_index",
        "score_median",
        "model_reference_rank",
    )
    missing_inventory = set(inventory_columns) - set(inventory)
    missing_scores = set(score_columns) - set(scores)
    if missing_inventory or missing_scores:
        raise ValueError(
            "Functionality inputs have missing columns: "
            f"inventory={sorted(missing_inventory)}, scores={sorted(missing_scores)}."
        )
    run_metadata = inventory.loc[:, inventory_columns].drop_duplicates()
    if run_metadata["analysis_id"].duplicated().any():
        raise ValueError("Inventory contains inconsistent rows for an analysis ID.")
    if scores.duplicated(["analysis_id", "variant_index"]).any():
        raise ValueError("Functionality scores contain duplicate candidate rows.")
    frame = scores.loc[:, score_columns].merge(
        run_metadata,
        on="analysis_id",
        how="left",
        validate="many_to_one",
    )
    numeric = frame[
        ["variant_index", "score_median", "model_reference_rank", "n_variants"]
    ]
    if not np.isfinite(numeric).all().all():
        raise ValueError("Functionality scores and ranks must be finite.")

    for analysis_id, rows in frame.groupby("analysis_id", sort=False):
        expected_n = int(rows["n_variants"].iloc[0])
        if not 2 <= len(rows) <= expected_n:
            raise ValueError(
                f"{analysis_id}: expected 2..{expected_n:,} candidate scores, "
                f"found {len(rows):,}."
            )
        ordered = rows.sort_values("model_reference_rank")
        expected_ranks = np.arange(1, len(rows) + 1)
        if not np.array_equal(ordered["model_reference_rank"].to_numpy(), expected_ranks):
            raise ValueError(f"{analysis_id}: model-reference ranks are not 1..N.")
        ordered_scores = ordered["score_median"].to_numpy(dtype=float)
        if np.any(ordered_scores[:-1] < ordered_scores[1:]):
            raise ValueError(
                f"{analysis_id}: median scores are not descending in reference rank."
            )
    return frame


def _score_kde(values: np.ndarray, *, grid_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("KDE scores must contain at least two finite values.")
    if np.ptp(scores) <= 0:
        raise ValueError("KDE scores must not all be identical.")
    density = gaussian_kde(scores, bw_method="scott")
    bandwidth = float(np.sqrt(density.covariance.squeeze()))
    padding = max(3 * bandwidth, float(np.ptp(scores)) * 0.015)
    grid = np.linspace(scores.min() - padding, scores.max() + padding, grid_size)
    return grid, density(grid)


def figure_functionality_distributions(
    inventory: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    protein_family: str,
    top_k: int = 100,
    n_columns: int | None = None,
) -> Figure:
    """Show all-candidate median-score KDEs and top-K cutoffs by landscape."""
    if protein_family not in FAMILY_COLORS:
        raise ValueError(f"Unknown protein family: {protein_family!r}")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    frame = _functionality_frame(inventory, scores)
    family_rows = frame[frame["protein_family"] == protein_family]
    if family_rows.empty:
        raise ValueError(f"No functionality scores for family {protein_family!r}.")
    inference_method = _inference_method(family_rows)
    analyses = (
        family_rows[
            [
                "analysis_order",
                "analysis_id",
                "landscape",
                "inference_method",
                "model_spec",
                "n_variants",
            ]
        ]
        .drop_duplicates("analysis_id")
        .sort_values("analysis_order")
    )
    if (analyses["n_variants"] < top_k).any():
        raise ValueError(f"Every analysis must contain at least top_k={top_k} candidates.")
    landscapes = analyses["landscape"].drop_duplicates().tolist()
    resolved_columns = n_columns or (2 if protein_family == "nuc" else 3)
    if resolved_columns < 1:
        raise ValueError("n_columns must be positive.")
    n_rows = (len(landscapes) + resolved_columns - 1) // resolved_columns
    figure = plt.figure(figsize=(14.2, 1.8 + 2.8 * n_rows))
    if len(landscapes) == 5 and resolved_columns == 3:
        grid = figure.add_gridspec(2, 6)
        axes = np.array(
            [
                figure.add_subplot(grid[0, 0:2]),
                figure.add_subplot(grid[0, 2:4]),
                figure.add_subplot(grid[0, 4:6]),
                figure.add_subplot(grid[1, 0:3]),
                figure.add_subplot(grid[1, 3:6]),
            ]
        )
    else:
        axes = np.array(
            [
                figure.add_subplot(n_rows, resolved_columns, index + 1)
                for index in range(n_rows * resolved_columns)
            ]
        )
    landscape_colors, model_markers, model_lines = _encodings(inventory)
    x_lower = np.inf
    x_upper = -np.inf
    density_upper = 0.0

    for axis, landscape in zip(axes, landscapes):
        landscape_analyses = analyses[analyses["landscape"] == landscape]
        color = landscape_colors[landscape]
        panel_lower = np.inf
        panel_upper = -np.inf
        panel_density_upper = 0.0
        for _, analysis in landscape_analyses.iterrows():
            analysis_rows = family_rows[
                family_rows["analysis_id"] == analysis["analysis_id"]
            ]
            values = analysis_rows["score_median"].to_numpy(dtype=float)
            grid, density = _score_kde(values)
            x_lower = min(x_lower, float(grid[0]))
            x_upper = max(x_upper, float(grid[-1]))
            density_upper = max(density_upper, float(density.max()))
            panel_lower = min(panel_lower, float(grid[0]))
            panel_upper = max(panel_upper, float(grid[-1]))
            panel_density_upper = max(panel_density_upper, float(density.max()))
            cutoff_rows = analysis_rows[
                analysis_rows["model_reference_rank"] == top_k
            ]
            if len(cutoff_rows) != 1:
                raise ValueError(
                    f"{analysis['analysis_id']}: expected exactly one rank-{top_k} score."
                )
            cutoff = float(cutoff_rows["score_median"].iloc[0])
            model = analysis["model_spec"]
            axis.plot(
                grid,
                density,
                color=color,
                linestyle=model_lines[model],
                linewidth=1.8,
                marker=model_markers[model],
                markevery=64,
                markersize=3.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=f"KDE:{analysis['analysis_id']}",
            )
            axis.axvline(
                cutoff,
                color=color,
                linestyle=model_lines[model],
                linewidth=1.15,
                alpha=0.75,
                label=f"Rank-{top_k} cutoff:{analysis['analysis_id']}",
            )
            axis.scatter(
                [cutoff],
                [0],
                color=color,
                marker=model_markers[model],
                s=24,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        n_candidates = landscape_analyses["n_variants"].drop_duplicates()
        n_label = (
            f"N={int(n_candidates.iloc[0]):,}"
            if len(n_candidates) == 1
            else "N=" + ", ".join(f"{int(value):,}" for value in n_candidates)
        )
        axis.text(
            0.02,
            0.92,
            n_label,
            transform=axis.transAxes,
            va="top",
            fontsize=7,
        )
        axis.set_title(
            _legend_landscape(landscape),
            loc="left",
            fontsize=9,
            fontweight="bold",
            pad=6,
        )
        axis.set_yticks([])
        if protein_family == "nuc":
            axis.set(
                xlim=(panel_lower, panel_upper),
                ylim=(0, panel_density_upper * 1.05),
            )
        _style_axis(axis, grid_axis="x")

    if protein_family != "nuc":
        for axis in axes[: len(landscapes)]:
            axis.set(xlim=(x_lower, x_upper), ylim=(0, density_upper * 1.05))
    for axis in axes[len(landscapes) :]:
        axis.set_axis_off()
    family_models = analyses["model_spec"].drop_duplicates().tolist()
    model_handles = [
        Line2D(
            [0],
            [0],
            color=TEXT_COLOR,
            linestyle=model_lines[model],
            marker=model_markers[model],
            markerfacecolor=TEXT_COLOR,
            markeredgecolor="white",
            linewidth=1.8,
            label=_short_model(model),
        )
        for model in family_models
    ]
    model_handles.append(
        Line2D(
            [0],
            [0],
            color=TEXT_COLOR,
            marker="|",
            markersize=11,
            linestyle="none",
            label=f"Vertical line = rank {top_k} cutoff",
        )
    )
    figure.supxlabel(
        SCORE_LABELS_BY_INFERENCE[inference_method][protein_family],
        y=0.09,
        fontsize=9,
    )
    figure.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=len(model_handles),
        frameon=False,
        fontsize=8,
        title="Curve and cutoff style = model",
        title_fontsize=8,
    )
    score_kind = "fitness-score" if inference_method == "hmc" else "proxy-score"
    figure.suptitle(
        f"{FAMILY_LABELS[protein_family]} median {score_kind} distributions by landscape",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.025, 0.14, 1, 0.96), h_pad=1.3, w_pad=1.15)
    return figure


def _decision_relative_score_frame(
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if "mechanism_variants" not in tables:
        raise ValueError("Figure 9 requires the mechanism_variants table.")
    variants = tables["mechanism_variants"]
    required = {
        "analysis_id",
        "protein_family",
        "landscape",
        "inference_method",
        "model_spec",
        "score_median",
        "score_interval_width",
        "normalized_score_interval_width",
        "reference_score_iqr",
        "score_margin_to_rank_100",
        "is_model_top_100",
        "is_challenger_q_ge_001",
        "is_primary_cohort",
    }
    missing = required - set(variants)
    if missing:
        raise ValueError(f"Figure 9 inputs have missing columns: {sorted(missing)}.")
    rows = variants[variants["is_primary_cohort"].astype(bool)].copy()
    if rows.empty:
        raise ValueError("Figure 9 has no primary-cohort variants.")
    numeric_columns = (
        "score_median",
        "score_interval_width",
        "normalized_score_interval_width",
        "reference_score_iqr",
        "score_margin_to_rank_100",
    )
    if not np.isfinite(rows.loc[:, numeric_columns]).all().all():
        raise ValueError("Figure 9 score quantities must be finite.")
    if (
        (rows["score_interval_width"] < 0).any()
        or (rows["reference_score_iqr"] <= 0).any()
        or (rows["normalized_score_interval_width"] < 0).any()
    ):
        raise ValueError(
            "Figure 9 score widths must be non-negative and run score IQRs positive."
        )
    if not np.allclose(
        rows["normalized_score_interval_width"],
        rows["score_interval_width"] / rows["reference_score_iqr"],
    ):
        raise ValueError("Normalized score widths must equal raw width / run score IQR.")
    model_member = rows["is_model_top_100"].astype(bool)
    challenger = rows["is_challenger_q_ge_001"].astype(bool)
    if not np.all(model_member ^ challenger):
        raise ValueError(
            "Primary Figure 9 rows must be either model top 100 or outside challenger."
        )
    rows["decision_relative_fitness"] = (
        rows["score_margin_to_rank_100"] / rows["reference_score_iqr"]
    )
    rows["decision_status"] = np.where(
        model_member,
        "Median-score top 100",
        "Outside challenger (q >= 1%)",
    )
    score_floor = _positive_plot_floor(rows["normalized_score_interval_width"])
    rows["normalized_score_interval_width_plot"] = np.maximum(
        rows["normalized_score_interval_width"],
        score_floor,
    )
    return rows


def figure_fitness_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
    n_columns: int | None = None,
) -> Figure:
    """Relate decision-centered median fitness to score uncertainty by run."""
    if protein_family not in FAMILY_COLORS:
        raise ValueError(f"Unknown protein family: {protein_family!r}")
    frame = _decision_relative_score_frame(tables)
    rows = frame[frame["protein_family"] == protein_family]
    if rows.empty:
        raise ValueError(f"No Figure 9 rows for protein family {protein_family!r}.")
    inference_method = _inference_method(rows)
    median_term, width_term = _uncertainty_terms(inference_method)
    analyses = rows[
        ["analysis_id", "landscape", "inference_method", "model_spec"]
    ].drop_duplicates("analysis_id")
    resolved_columns = n_columns or (2 if protein_family == "pdz" else 3)
    if resolved_columns < 1:
        raise ValueError("n_columns must be positive.")
    n_rows = (len(analyses) + resolved_columns - 1) // resolved_columns
    figure, axes = plt.subplots(
        n_rows,
        resolved_columns,
        figsize=(14.2, 1.8 + 2.55 * n_rows),
        sharey=True,
        squeeze=False,
    )
    y_min = float(rows["normalized_score_interval_width_plot"].min())
    y_max = float(rows["normalized_score_interval_width_plot"].max())
    y_limits = (y_min / 1.35, y_max * 1.35)
    categories = (
        ("Median-score top 100", DEFAULT_COLOR, "o", True),
        ("Outside challenger (q >= 1%)", "#FF7F0E", "o", False),
    )

    for axis, (_, analysis) in zip(axes.ravel(), analyses.iterrows()):
        analysis_rows = rows[rows["analysis_id"] == analysis["analysis_id"]]
        for status, color, marker, filled in categories:
            selected = analysis_rows[analysis_rows["decision_status"] == status]
            axis.scatter(
                selected["decision_relative_fitness"],
                selected["normalized_score_interval_width_plot"],
                color=color if filled else "none",
                edgecolor=color,
                marker=marker,
                s=14,
                alpha=0.42,
                linewidth=0.55,
                rasterized=True,
                label=status,
            )
        axis.axvline(
            0,
            color=AXIS_COLOR,
            linestyle=":",
            linewidth=1.1,
            label="Rank-100 boundary",
        )
        correlation = spearmanr(
            analysis_rows["decision_relative_fitness"],
            analysis_rows["normalized_score_interval_width"],
        ).statistic
        axis.text(
            0.98,
            0.94,
            f"Spearman rho = {correlation:.2f}\nn = {len(analysis_rows):,}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        axis.set(
            title=_analysis_label(analysis),
            yscale="log",
            ylim=y_limits,
        )
        axis.title.set_fontsize(8.5)
        axis.title.set_fontweight("bold")
        axis.title.set_ha("left")
        axis.title.set_position((0, 1.0))
        _style_axis(axis)

    for axis in axes.ravel()[len(analyses) :]:
        axis.set_axis_off()
    figure.supxlabel(
        f"({median_term} score - rank-100 score) / top-1,000 score IQR",
        y=0.055,
        fontsize=9,
    )
    figure.supylabel(
        f"90% {width_term} / top-1,000 score IQR (log scale)",
        x=0.012,
        fontsize=9,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DEFAULT_COLOR,
                markeredgecolor=DEFAULT_COLOR,
                label="Median-score top 100",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#FF7F0E",
                label="Outside challenger (q >= 1%)",
            ),
            Line2D(
                [0],
                [0],
                color=AXIS_COLOR,
                linestyle=":",
                linewidth=1.1,
                label="Fixed rank-100 boundary",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    score_kind = "fitness" if inference_method == "hmc" else "proxy score"
    figure.suptitle(
        f"{FAMILY_LABELS[protein_family]} decision-relative {score_kind} and uncertainty",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.035, 0.1, 1, 0.965), h_pad=1.15, w_pad=1.15)
    return figure


def _summary_why_frame(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _decision_relative_score_frame(tables)
    required_variant_columns = {
        "mechanism_input_count",
        "mechanism_output_count",
        "top_100_inclusion_probability",
    }
    missing_variants = required_variant_columns - set(rows)
    if missing_variants:
        raise ValueError(
            f"Summary-why variants have missing columns: {sorted(missing_variants)}."
        )
    numeric = rows[
        [
            "mechanism_input_count",
            "mechanism_output_count",
            "top_100_inclusion_probability",
        ]
    ]
    if not np.isfinite(numeric).all().all():
        raise ValueError("Summary-why count and inclusion quantities must be finite.")
    if (
        (rows["mechanism_input_count"] < 0).any()
        or (rows["mechanism_output_count"] < 0).any()
    ):
        raise ValueError("Summary-why counts must be non-negative.")
    if not rows["top_100_inclusion_probability"].between(0, 1).all():
        raise ValueError("Top-100 inclusion probabilities must lie between zero and one.")
    rows["input_log_count"] = np.log10(rows["mechanism_input_count"] + 1)
    rows["output_log_count"] = np.log10(rows["mechanism_output_count"] + 1)
    rows["score_uncertainty_plot"] = np.log10(
        rows["normalized_score_interval_width_plot"]
    )

    if "mechanism_runs" not in tables:
        raise ValueError("Summary-why figure requires the mechanism_runs table.")
    runs = tables["mechanism_runs"]
    required_run_columns = {
        "analysis_id",
        "sample_self_consistency",
        "top_k_instability",
        "challenger_count_q_ge_001",
    }
    missing_runs = required_run_columns - set(runs)
    if missing_runs:
        raise ValueError(
            f"Summary-why runs have missing columns: {sorted(missing_runs)}."
        )
    run_rows = runs.loc[:, sorted(required_run_columns)].drop_duplicates()
    if run_rows["analysis_id"].duplicated().any():
        raise ValueError("Summary-why run annotations are not unique by analysis ID.")
    run_numeric = run_rows[
        [
            "sample_self_consistency",
            "top_k_instability",
            "challenger_count_q_ge_001",
        ]
    ]
    if not np.isfinite(run_numeric).all().all():
        raise ValueError("Summary-why run annotations must be finite.")
    if (
        not run_rows["sample_self_consistency"].between(0, 1).all()
        or not run_rows["top_k_instability"].between(0, 1).all()
        or (run_rows["challenger_count_q_ge_001"] < 0).any()
        or not np.allclose(
            run_rows["top_k_instability"],
            1 - run_rows["sample_self_consistency"],
        )
    ):
        raise ValueError("Summary-why run-level stability quantities are invalid.")
    missing_annotations = set(rows["analysis_id"]) - set(run_rows["analysis_id"])
    if missing_annotations:
        raise ValueError(
            f"Summary-why runs lack annotations for: {sorted(missing_annotations)}."
        )
    return rows, run_rows


def _plot_summary_status_points(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> None:
    categories = (
        ("Median-score top 100", DEFAULT_COLOR, True),
        ("Outside challenger (q >= 1%)", "#FF7F0E", False),
    )
    for status, color, filled in categories:
        selected = rows[rows["decision_status"] == status]
        axis.scatter(
            selected[x],
            selected[y],
            color=color if filled else "none",
            edgecolor=color,
            marker="o",
            s=11,
            alpha=0.38,
            linewidth=0.5,
            rasterized=True,
            label=status,
        )


def figure_summary_why_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    protein_family: str,
) -> Figure:
    """Summarize support, score precision, and top-100 consequence by run."""
    if protein_family not in FAMILY_COLORS:
        raise ValueError(f"Unknown protein family: {protein_family!r}")
    frame, run_annotations = _summary_why_frame(tables)
    rows = frame[frame["protein_family"] == protein_family]
    if rows.empty:
        raise ValueError(f"No summary-why rows for protein family {protein_family!r}.")
    inference_method = _inference_method(rows)
    _, width_term = _uncertainty_terms(inference_method)
    analyses = rows[
        ["analysis_id", "landscape", "inference_method", "model_spec"]
    ].drop_duplicates("analysis_id")
    n_analyses = len(analyses)
    count_upper = float(
        rows[["input_log_count", "output_log_count"]].to_numpy().max()
    )
    count_range = (-0.08, count_upper * 1.02)
    uncertainty_values = rows["score_uncertainty_plot"]
    uncertainty_padding = max(float(np.ptp(uncertainty_values)) * 0.03, 0.05)
    uncertainty_range = (
        float(uncertainty_values.min()) - uncertainty_padding,
        float(uncertainty_values.max()) + uncertainty_padding,
    )
    figure, axes = plt.subplots(
        n_analyses,
        4,
        figsize=(
            18.0,
            4.0 + 2.0 * n_analyses
            if protein_family == "k50"
            else 2.6 + 1.65 * n_analyses,
        ),
        squeeze=False,
    )
    count_ticks, count_labels = _count_ticks(count_range[1])
    uncertainty_ticks, uncertainty_labels = _score_ticks(*uncertainty_range)

    for row_index, (_, analysis) in enumerate(analyses.iterrows()):
        analysis_rows = rows[rows["analysis_id"] == analysis["analysis_id"]]
        for column_index, x_column in enumerate(
            ("input_log_count", "output_log_count")
        ):
            axis = axes[row_index, column_index]
            _plot_summary_status_points(
                axis,
                analysis_rows,
                x=x_column,
                y="score_uncertainty_plot",
            )
            _descriptive_trend(
                axis,
                analysis_rows,
                x=x_column,
                y="score_uncertainty_plot",
                y_range=uncertainty_range,
                linewidth=1.5,
            )
            correlation = spearmanr(
                analysis_rows[x_column],
                analysis_rows["score_uncertainty_plot"],
            ).statistic
            axis.text(
                0.98,
                0.92,
                f"rho = {correlation:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
            )
            axis.set(
                xlim=count_range,
                ylim=uncertainty_range,
                xticks=count_ticks,
                xticklabels=count_labels,
                yticks=uncertainty_ticks,
                yticklabels=uncertainty_labels,
            )
            _style_axis(axis)

        decision_min = float(analysis_rows["decision_relative_fitness"].min())
        decision_max = float(analysis_rows["decision_relative_fitness"].max())
        decision_span = max(decision_max - decision_min, 0.1)
        decision_padding = decision_span * 0.04
        decision_range = (
            min(decision_min - decision_padding, -0.01),
            max(decision_max + decision_padding, 0.01),
        )

        uncertainty_axis = axes[row_index, 2]
        _plot_summary_status_points(
            uncertainty_axis,
            analysis_rows,
            x="decision_relative_fitness",
            y="score_uncertainty_plot",
        )
        uncertainty_axis.axvline(
            0,
            color=AXIS_COLOR,
            linestyle=":",
            linewidth=1.0,
            label="Rank-100 boundary",
        )
        decision_correlation = spearmanr(
            analysis_rows["decision_relative_fitness"],
            analysis_rows["score_uncertainty_plot"],
        ).statistic
        uncertainty_axis.text(
            0.98,
            0.92,
            f"rho = {decision_correlation:.2f}",
            transform=uncertainty_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
        )
        uncertainty_axis.set(
            xlim=decision_range,
            ylim=uncertainty_range,
            yticks=uncertainty_ticks,
            yticklabels=uncertainty_labels,
        )
        _style_axis(uncertainty_axis)

        inclusion_axis = axes[row_index, 3]
        _plot_summary_status_points(
            inclusion_axis,
            analysis_rows,
            x="decision_relative_fitness",
            y="top_100_inclusion_probability",
        )
        inclusion_axis.axvline(
            0,
            color=AXIS_COLOR,
            linestyle=":",
            linewidth=1.0,
            label="Rank-100 boundary",
        )
        inclusion_axis.axhline(
            0.5,
            color=AXIS_COLOR,
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            label="50% inclusion",
        )
        annotation = run_annotations[
            run_annotations["analysis_id"] == analysis["analysis_id"]
        ].iloc[0]
        inclusion_axis.set(xlim=decision_range, ylim=(-0.02, 1.02))
        inclusion_axis.yaxis.set_major_formatter(PercentFormatter(1))
        _style_axis(inclusion_axis)

        axes[row_index, 0].text(
            -0.11,
            0.5,
            (
                f"{_analysis_label(analysis)}\n"
                f"instability {annotation['top_k_instability']:.0%}; "
                f"{int(annotation['challenger_count_q_ge_001']):,} challengers"
            ),
            transform=axes[row_index, 0].transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="right",
            va="center",
        )
        for column_index in (1, 2):
            axes[row_index, column_index].tick_params(labelleft=False)
        if row_index < n_analyses - 1:
            for axis in axes[row_index, :2]:
                axis.tick_params(labelbottom=False)

    column_titles = (
        "Input support -> score uncertainty",
        "Output abundance -> score uncertainty",
        "Decision margin -> score uncertainty",
        "Decision margin -> inclusion probability",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=9, fontweight="bold", pad=8)
    axes[-1, 0].set_xlabel("Input count (log10[count + 1])")
    axes[-1, 1].set_xlabel("Output count (log10[count + 1])")
    for axis in axes[-1, 2:]:
        axis.set_xlabel(
            "(median score - rank-100 score) / top-1,000 score IQR"
        )
    figure.text(
        0.025 if protein_family == "k50" else 0.012,
        0.5,
        f"Relative 90% {width_term} (log scale)",
        rotation=90,
        va="center",
        fontsize=9,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DEFAULT_COLOR,
                markeredgecolor=DEFAULT_COLOR,
                label="Median-score top 100",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#FF7F0E",
                label="Outside challenger (q >= 1%)",
            ),
            Line2D(
                [0],
                [0],
                color=TEXT_COLOR,
                linewidth=1.5,
                label="Within-analysis descriptive trend",
            ),
            Line2D(
                [0],
                [0],
                color=AXIS_COLOR,
                linestyle=":",
                linewidth=1.0,
                label="Fixed rank-100 boundary",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        f"{FAMILY_LABELS[protein_family]} per-analysis summary: why is the top 100 uncertain?",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.18 if protein_family == "k50" else 0.14,
        right=0.99,
        top=0.86 if protein_family == "k50" else 0.94,
        bottom=0.18 if protein_family == "k50" else 0.075,
        hspace=0.32,
        wspace=0.2,
    )
    return figure


def save_figure(figure: Figure, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Save one figure as vector PDF and 300-dpi PNG."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    return pdf, png


def build_all_figures(
    table_dir: str | Path,
    output_dir: str | Path,
    *,
    representative_ids: tuple[str, ...] | None = None,
    main_k: int = 100,
) -> dict[str, tuple[Path, Path]]:
    """Load available paper tables and save all implemented paper figures."""
    set_paper_style()
    tables = load_tables(table_dir)
    inventory = tables["inventory"].sort_values("analysis_order")
    families = _family_order(inventory)
    outputs: dict[str, tuple[Path, Path]] = {}

    def save(name: str, figure: Figure) -> None:
        outputs[name] = save_figure(figure, output_dir, name)
        plt.close(figure)

    save(
        "figure1_reported_order",
        figure_reported_order(
            tables,
            cohort_top_n=main_k,
            example_id=None if representative_ids is None else representative_ids[0],
        ),
    )
    save(
        "figure2_rank_uncertainty",
        figure_rank_uncertainty(
            tables,
            representative_ids=representative_ids,
            main_k=main_k,
        ),
    )
    shared_inventory = inventory[inventory["protein_family"] != "k50"]
    shared_ids = set(shared_inventory["analysis_id"])
    shared_tables = {
        name: table[
            ~table.get(
                "protein_family",
                pd.Series("", index=table.index),
            ).eq("k50")
        ].copy()
        for name, table in tables.items()
    }
    shared_representatives = (
        None
        if representative_ids is None
        else tuple(identifier for identifier in representative_ids if identifier in shared_ids)
    )
    save(
        "figure3_top_k_stability",
        figure_top_k_stability(
            shared_tables,
            representative_ids=shared_representatives,
            main_k=main_k,
        ),
    )
    if "k50" in families:
        save(
            "figure3_k50_top_k_stability",
            figure_k50_top_k_stability(tables, main_k=main_k),
        )
    mechanism_paths = [
        Path(table_dir) / f"{name}.csv"
        for name in MECHANISM_TABLE_NAMES
    ]
    if all(path.is_file() for path in mechanism_paths):
        mechanism_tables = load_mechanism_tables(table_dir)
        save(
            "figure4_ranking_mechanisms",
            figure_ranking_mechanisms(mechanism_tables),
        )
        save(
            "figure5_rank_space_sensitivity",
            figure_rank_space_sensitivity(mechanism_tables),
        )
        save(
            "figure6_1_count_uncertainty_scatter",
            figure_count_uncertainty_scatter(mechanism_tables),
        )
        save(
            "figure6_2_count_uncertainty_datashader",
            figure_count_uncertainty_datashader(mechanism_tables),
        )
        save(
            "figure6_3_count_uncertainty_hexbin",
            figure_count_uncertainty_hexbin(mechanism_tables),
        )
        mechanism_families = set(
            mechanism_tables["mechanism_variants"]["protein_family"]
        )
        for family in families:
            if family not in mechanism_families:
                continue
            family_stem = FAMILY_FILE_STEMS[family]
            save(
                f"figure7_1_{family_stem}_count_score_uncertainty",
                figure_count_score_uncertainty_by_analysis(
                    mechanism_tables,
                    protein_family=family,
                ),
            )
            save(
                f"figure7_2_{family_stem}_count_rank_uncertainty",
                figure_count_rank_uncertainty_by_analysis(
                    mechanism_tables,
                    protein_family=family,
                ),
            )
            save(
                f"figure9_{family_stem}_fitness_uncertainty",
                figure_fitness_uncertainty_by_analysis(
                    mechanism_tables,
                    protein_family=family,
                ),
            )
            save(
                f"figure10_{family_stem}_summary_why",
                figure_summary_why_by_analysis(
                    mechanism_tables,
                    protein_family=family,
                ),
            )
    functionality_paths = (
        Path(table_dir) / "functionality_scores.csv.gz",
        Path(table_dir) / "functionality_scores.csv",
    )
    if any(path.is_file() for path in functionality_paths):
        functionality_scores = load_functionality_scores(table_dir)
        score_families = set(inventory["protein_family"])
        for family in families:
            if family not in score_families:
                continue
            family_stem = FAMILY_FILE_STEMS[family]
            save(
                f"figure8_{family_stem}_functionality",
                figure_functionality_distributions(
                    inventory,
                    functionality_scores,
                    protein_family=family,
                    top_k=main_k,
                ),
            )
    return outputs
