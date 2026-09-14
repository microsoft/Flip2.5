"""Paper figure composition from generated numerical tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, NullFormatter, PercentFormatter
from scipy.stats import gaussian_kde, spearmanr

CORE_TABLE_NAMES = (
    "runs",
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
FAMILY_ORDER = ("pdz", "trpb", "nuc")
FAMILY_LABELS = {"pdz": "PDZ", "trpb": "TrpB", "nuc": "NucB"}
FAMILY_FILE_STEMS = {"pdz": "pdz", "trpb": "trpb", "nuc": "nucb"}
SAMPLING_LABELS = {
    "parameter_posterior": "parameter-posterior",
    "mle_predictive": "MLE-predictive",
    "posterior_predictive": "posterior-predictive",
}
PALETTE = (
    "#276FBF",
    "#D1495B",
    "#2A9D8F",
    "#E9A03B",
    "#7251B5",
    "#4D908E",
    "#C8553D",
    "#577590",
    "#7A9E48",
    "#B56576",
)
MODEL_MARKERS = ("o", "s", "^", "D", "P", "X")
MODEL_LINESTYLES = ("-", "--", "-.", ":")
TEXT_COLOR = "#293241"
AXIS_COLOR = "#7D8597"
GRID_COLOR = "#DDE2E8"
MODEL_COLOR = "#276FBF"
EXTERNAL_COLOR = "#2A9D8F"
CHALLENGER_COLOR = "#E07A5F"
FIGURE_FONT = "DejaVu Sans"
FIGURE_TEXT_SIZE = 8
FIGURE_HEADING_SIZE = 9
MANUSCRIPT_FIGURE_WIDTH = 6.5


def _load_named_tables(root: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    missing = [name for name in names if not (root / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tables in {root}: {missing}")
    return {name: pd.read_csv(root / f"{name}.csv", low_memory=False) for name in names}


def load_tables(table_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the six rank-stability tables from one sampling directory."""
    return _load_named_tables(Path(table_dir), CORE_TABLE_NAMES)


def load_mechanism_tables(table_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the four mechanism tables from one sampling directory."""
    return _load_named_tables(Path(table_dir), MECHANISM_TABLE_NAMES)


def load_functionality_scores(table_dir: str | Path) -> pd.DataFrame:
    """Load all-candidate median scores and reference ranks."""
    root = Path(table_dir)
    paths = (
        root / "functionality_scores.csv.gz",
        root / "functionality_scores.csv",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing functionality scores in {root}.")
    return pd.read_csv(path, low_memory=False)


def load_all_tables(table_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every table needed by the figure builders."""
    return {
        **load_tables(table_dir),
        **load_mechanism_tables(table_dir),
        "functionality_scores": load_functionality_scores(table_dir),
    }


def set_paper_style() -> None:
    """Apply the common restrained figure style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FIGURE_FONT],
            "font.size": FIGURE_TEXT_SIZE,
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": FIGURE_HEADING_SIZE,
            "axes.labelsize": FIGURE_TEXT_SIZE,
            "figure.titlesize": FIGURE_HEADING_SIZE,
            "legend.fontsize": FIGURE_TEXT_SIZE,
            "legend.title_fontsize": FIGURE_HEADING_SIZE,
            "xtick.labelsize": FIGURE_TEXT_SIZE,
            "ytick.labelsize": FIGURE_TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def apply_figure_typography(figure: Figure) -> None:
    """Apply one typeface and consistent text roles, including builder overrides."""
    for text in figure.findobj(match=Text):
        text.set_fontfamily(FIGURE_FONT)
        text.set_fontsize(FIGURE_TEXT_SIZE)
        if text.get_gid() == "panel-label":
            text.set_fontsize(FIGURE_HEADING_SIZE)
    for axis in figure.axes:
        axis.tick_params(axis="both", which="both", labelsize=FIGURE_TEXT_SIZE)
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_fontsize(FIGURE_HEADING_SIZE)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_title().set_fontsize(FIGURE_HEADING_SIZE)
    containers = [figure]
    while containers:
        container = containers.pop()
        if container._suptitle is not None:
            container._suptitle.set_fontsize(FIGURE_HEADING_SIZE)
        for legend in container.legends:
            legend.get_title().set_fontsize(FIGURE_HEADING_SIZE)
        containers.extend(container.subfigs)


def save_figure(figure: Figure, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Save one figure as vector PDF and 300-dpi PNG."""
    apply_figure_typography(figure)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    return pdf, png


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _families(runs: pd.DataFrame) -> tuple[str, ...]:
    observed = runs["family"].drop_duplicates().astype(str).tolist()
    known = [family for family in FAMILY_ORDER if family in observed]
    return tuple([*known, *sorted(set(observed) - set(known))])


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())


def _short_landscape(landscape: str, family: str | None = None) -> str:
    prefix = f"{family}-" if family else ""
    return landscape.removeprefix(prefix)


def _short_model(model: str) -> str:
    names = {"exponential": "Exp", "gamma": "Gamma", "lomax": "Lomax"}
    return "-".join(names.get(part, part.title()) for part in model.split("-"))


def _run_label(row: pd.Series) -> str:
    sampling = SAMPLING_LABELS.get(str(row["sampling"]), str(row["sampling"]))
    return (
        f"{_short_landscape(str(row['landscape']), str(row['family']))} | "
        f"{_short_model(str(row['model']))} | {sampling}"
    )


def _encodings(runs: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    landscapes = runs["landscape"].drop_duplicates().astype(str).tolist()
    models = runs["model"].drop_duplicates().astype(str).tolist()
    colors = {
        landscape: PALETTE[index % len(PALETTE)] for index, landscape in enumerate(landscapes)
    }
    markers = {
        model: MODEL_MARKERS[index % len(MODEL_MARKERS)] for index, model in enumerate(models)
    }
    lines = {
        model: MODEL_LINESTYLES[index % len(MODEL_LINESTYLES)] for index, model in enumerate(models)
    }
    return colors, markers, lines


def _style_axis(axis, *, grid_axis: str = "both") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=FIGURE_HEADING_SIZE,
        fontweight="bold",
        gid="panel-label",
        va="bottom",
    )


def _representative_ids(runs: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        runs.loc[runs["family"].eq(family), "run_id"].iloc[0] for family in _families(runs)
    )


def _positive_floor(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    positive = array[np.isfinite(array) & (array > 0)]
    if not len(positive):
        raise ValueError("At least one positive uncertainty value is required.")
    return float(positive.min() / 2)


def _safe_spearman(first: pd.Series, second: pd.Series) -> float:
    complete = np.isfinite(first.to_numpy(dtype=float)) & np.isfinite(second.to_numpy(dtype=float))
    left = first.to_numpy(dtype=float)[complete]
    right = second.to_numpy(dtype=float)[complete]
    if len(left) < 3 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def _trend(axis, rows: pd.DataFrame, x: str, y: str) -> None:
    finite = np.isfinite(rows[x]) & np.isfinite(rows[y])
    values = rows.loc[finite, [x, y]]
    if len(values) < 2 or np.ptp(values[x]) <= 0:
        return
    slope, intercept = np.polyfit(values[x], values[y], 1)
    line_x = np.linspace(values[x].min(), values[x].max(), 100)
    axis.plot(line_x, intercept + slope * line_x, color=TEXT_COLOR, linewidth=1.5)


def figure_reported_order(
    tables: Mapping[str, pd.DataFrame],
    *,
    cohort_top_n: int = 100,
    example_id: str | None = None,
) -> Figure:
    """Show sampled-rank percentiles for the external ranking."""
    runs = tables["runs"]
    percentiles = tables["rank_percentiles"]
    _require(runs, {"run_id", "family", "landscape", "model", "sampling"}, "runs")
    _require(
        percentiles,
        {
            "run_id",
            "cohort_top_n",
            "external_rank_sample_percentile",
            "ecdf",
            "ecdf_to_uniform_area",
            "external_vs_posterior_median_rank_spearman",
        },
        "rank_percentiles",
    )
    selected = percentiles[percentiles["cohort_top_n"].eq(cohort_top_n)]
    if selected.empty:
        raise ValueError(f"No rank-percentile rows for cohort top {cohort_top_n}.")
    families = tuple(family for family in _families(runs) if family in set(selected["family"]))
    colors, markers, lines = _encodings(runs)
    n_panels = len(families) + 3
    n_columns = max(2, (n_panels + 1) // 2)
    figure, axes = plt.subplots(2, n_columns, figsize=(6.0 * n_columns, 8.4), squeeze=False)
    flat = axes.ravel()
    example_id = example_id or str(
        runs.loc[runs["run_id"].isin(selected["run_id"]), "run_id"].iloc[0]
    )
    example = selected[selected["run_id"].eq(example_id)].sort_values(
        "external_rank_sample_percentile"
    )
    if example.empty:
        raise ValueError(f"Unknown example run ID: {example_id}")
    example_run = runs[runs["run_id"].eq(example_id)].iloc[0]
    axis = flat[0]
    axis.step(
        np.r_[0, example["external_rank_sample_percentile"], 1],
        np.r_[0, example["ecdf"], 1],
        where="post",
        color=colors[str(example_run["landscape"])],
        linewidth=2,
    )
    axis.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle="--")
    axis.set(
        title=f"Example run\n{_run_label(example_run)}",
        xlabel="Fraction of samples ranked above the external rank",
        ylabel="Cumulative fraction of variants",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.text(0.04, 0.94, f"Area = {example['ecdf_to_uniform_area'].iloc[0]:.3f}", va="top")
    _style_axis(axis)

    for axis, family in zip(flat[1:], families):
        family_runs = runs[runs["family"].eq(family)]
        for _, run in family_runs.iterrows():
            rows = selected[selected["run_id"].eq(run["run_id"])].sort_values(
                "external_rank_sample_percentile"
            )
            if rows.empty:
                continue
            axis.step(
                np.r_[0, rows["external_rank_sample_percentile"], 1],
                np.r_[0, rows["ecdf"], 1],
                where="post",
                color=colors[str(run["landscape"])],
                linestyle=lines[str(run["model"])],
                linewidth=1,
                alpha=0.8,
            )
        axis.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle="--")
        axis.set(
            title=f"{_family_label(family)} runs: external top {cohort_top_n}",
            xlabel="Sample percentile of external rank",
            ylabel="Cumulative fraction of variants",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        _style_axis(axis)

    summaries = selected.drop_duplicates("run_id").merge(
        runs[["run_id", "family", "landscape", "model"]],
        on=["run_id", "family", "landscape", "model"],
        how="left",
        validate="one_to_one",
    )
    diagnostic_axis = flat[1 + len(families)]
    concordance_axis = flat[2 + len(families)]
    for _, row in summaries.iterrows():
        style = {
            "color": colors[str(row["landscape"])],
            "marker": markers[str(row["model"])],
            "s": 35,
            "edgecolor": "white",
            "linewidth": 0.4,
        }
        diagnostic_axis.scatter(row["ecdf_to_uniform_area"], row["family"], **style)
        concordance_axis.scatter(
            row["external_vs_posterior_median_rank_spearman"],
            row["family"],
            **style,
        )
    diagnostic_axis.set(
        title="Departure from uniform sampled-rank percentiles",
        xlabel="Absolute ECDF area",
        ylabel="Family",
    )
    concordance_axis.set(
        title="External versus median sampled rank",
        xlabel="Spearman correlation",
        ylabel="Family",
        xlim=(-1.05, 1.05),
    )
    _style_axis(diagnostic_axis, grid_axis="x")
    _style_axis(concordance_axis, grid_axis="x")
    used = 3 + len(families)
    for axis in flat[used:]:
        axis.set_axis_off()
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", flat[:used]):
        _panel_label(axis, label)
    figure.suptitle(
        "External-order calibration across sampled landscapes", fontsize=13, fontweight="bold"
    )
    figure.tight_layout()
    return figure


def _rank_interval_panel(axis, rows: pd.DataFrame, main_k: int, detail_top_n: int) -> None:
    rows = rows.nsmallest(detail_top_n, "model_reference_rank").sort_values(
        "model_reference_rank",
        ascending=False,
    )
    y = np.arange(len(rows))
    median = rows["posterior_median_rank"].to_numpy(dtype=float)
    axis.errorbar(
        median,
        y,
        xerr=np.vstack(
            [
                median - rows["posterior_rank_lower"].to_numpy(dtype=float),
                rows["posterior_rank_upper"].to_numpy(dtype=float) - median,
            ]
        ),
        fmt="o",
        color=MODEL_COLOR,
        ecolor=to_rgba(MODEL_COLOR, 0.45),
        markersize=3,
    )
    axis.scatter(rows["model_reference_rank"], y, marker="|", color=TEXT_COLOR, s=45)
    axis.axvline(main_k, color=AXIS_COLOR, linestyle=":")
    labels = rows.get("variant_label", rows["variant_id"]).fillna(rows["variant_id"])
    axis.set(
        title=_run_label(rows.iloc[0]),
        xlabel="Sampled rank (point = median; line = central interval)",
        yticks=y,
        yticklabels=labels,
    )
    _style_axis(axis, grid_axis="x")


def figure_rank_uncertainty(
    tables: Mapping[str, pd.DataFrame],
    *,
    representative_ids: tuple[str, ...] | None = None,
    main_k: int = 100,
    detail_top_n: int = 50,
) -> Figure:
    """Show exact model-cohort rank intervals and cross-run summaries."""
    runs = tables["runs"]
    intervals = tables["model_rank_intervals"]
    representatives = representative_ids or _representative_ids(runs)
    if set(representatives) - set(intervals["run_id"]):
        raise ValueError("Representative IDs must be present in model_rank_intervals.")
    colors, markers, _ = _encodings(runs)
    figure = plt.figure(
        figsize=(max(12, 5.2 * len(representatives)), 10),
        layout="constrained",
    )
    grid = figure.add_gridspec(2, 1, height_ratios=(1.2, 1))
    detail_grid = grid[0].subgridspec(1, len(representatives))
    summary_grid = grid[1].subgridspec(1, 2, width_ratios=(1.25, 0.75))
    detail_axes = [
        figure.add_subplot(detail_grid[0, index]) for index in range(len(representatives))
    ]
    summary_axes = [figure.add_subplot(summary_grid[0, index]) for index in range(2)]
    for axis, run_id in zip(detail_axes, representatives, strict=True):
        _rank_interval_panel(
            axis,
            intervals[intervals["run_id"].eq(run_id)],
            main_k,
            min(detail_top_n, main_k),
        )
    summaries = intervals.groupby(
        ["run_id", "family", "landscape", "model", "sampling"], as_index=False
    ).agg(
        interval_width_q05=("rank_interval_width", lambda values: values.quantile(0.05)),
        interval_width_q25=("rank_interval_width", lambda values: values.quantile(0.25)),
        interval_width_median=("rank_interval_width", "median"),
        interval_width_q75=("rank_interval_width", lambda values: values.quantile(0.75)),
        interval_width_q95=("rank_interval_width", lambda values: values.quantile(0.95)),
        fraction_leaving_top_k=("rank_interval_leaves_top_k", "mean"),
    )
    summaries["run_order"] = summaries["run_id"].map(
        {run_id: index for index, run_id in enumerate(runs["run_id"])}
    )
    summaries = summaries.sort_values("run_order")
    y = np.arange(len(summaries))
    labels = [_run_label(row) for _, row in summaries.iterrows()]
    for position, (_, row) in enumerate(summaries.iterrows()):
        color = colors[str(row["landscape"])]
        summary_axes[0].hlines(
            position, row["interval_width_q05"], row["interval_width_q95"], color=color
        )
        summary_axes[0].hlines(
            position,
            row["interval_width_q25"],
            row["interval_width_q75"],
            color=color,
            linewidth=5,
            alpha=0.35,
        )
        summary_axes[0].scatter(
            row["interval_width_median"],
            position,
            color=color,
            marker=markers[str(row["model"])],
            s=28,
        )
        summary_axes[1].scatter(
            row["fraction_leaving_top_k"],
            position,
            color=color,
            marker=markers[str(row["model"])],
            s=32,
        )
    summary_axes[0].set(
        title="Central rank-interval width by run",
        xlabel="Rank positions (5th-95th range across variants)",
        yticks=y,
        yticklabels=labels,
        xscale="log",
    )
    summary_axes[1].set(
        title=f"Membership uncertainty at K={main_k}",
        xlabel="Fraction whose upper rank leaves top K",
        yticks=y,
        yticklabels=[],
        xlim=(-0.02, 1.02),
    )
    summary_axes[0].invert_yaxis()
    summary_axes[1].invert_yaxis()
    summary_axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    _style_axis(summary_axes[0], grid_axis="x")
    _style_axis(summary_axes[1], grid_axis="x")
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", [*detail_axes, *summary_axes]):
        _panel_label(axis, label)
    figure.suptitle(
        f"Sampled rank uncertainty for the model top {main_k}", fontsize=13, fontweight="bold"
    )
    return figure


def _candidate_panel(axis, rows: pd.DataFrame, main_k: int) -> None:
    incumbents = rows[rows["is_model_reference_top_k"]].nlargest(10, "model_reference_rank")
    challengers = rows[~rows["is_model_reference_top_k"]].nlargest(10, "top_k_probability")
    displayed = pd.concat([incumbents, challengers]).drop_duplicates("variant_index")
    displayed = displayed.sort_values(
        ["is_model_reference_top_k", "model_reference_rank", "top_k_probability"],
        ascending=[False, True, False],
        na_position="last",
    )
    y = np.arange(len(displayed))
    axis.hlines(y, 0, displayed["top_k_probability"], color=GRID_COLOR)
    model = displayed["is_model_reference_top_k"].astype(bool)
    external = displayed["is_external_reference_top_k"].astype(bool)
    for mask, color, marker, label in (
        (model & external, EXTERNAL_COLOR, "D", "In both reference top-K sets"),
        (model & ~external, MODEL_COLOR, "o", "Model top K only"),
        (~model & external, EXTERNAL_COLOR, "s", "External top K only"),
        (~model & ~external, CHALLENGER_COLOR, "o", "Outside challenger"),
    ):
        if mask.any():
            axis.scatter(
                displayed.loc[mask, "top_k_probability"],
                y[mask],
                color=color,
                marker=marker,
                s=30,
                label=label,
            )
    labels = displayed.get("variant_label", displayed["variant_id"]).fillna(displayed["variant_id"])
    axis.set(
        title=f"Decision boundary and outside challengers\n{_run_label(displayed.iloc[0])}",
        xlabel=f"Top-{main_k} inclusion probability",
        yticks=y,
        yticklabels=labels,
        xlim=(-0.02, 1.02),
    )
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False, fontsize=6)
    _style_axis(axis, grid_axis="x")


def figure_top_k_stability(
    tables: Mapping[str, pd.DataFrame],
    *,
    representative_ids: tuple[str, ...] | None = None,
    main_k: int = 100,
) -> Figure:
    """Show decision-boundary examples and top-K stability curves."""
    runs = tables["runs"]
    candidates = tables["top_k_candidates"]
    curves = tables["top_k_curves"]
    families = _families(runs)
    representatives = representative_ids or _representative_ids(runs)
    if len(representatives) != len(families):
        raise ValueError("Provide one representative run for each family.")
    figure, axes = plt.subplots(2, len(families), figsize=(6.2 * len(families), 9.5), squeeze=False)
    for axis, run_id in zip(axes[0], representatives, strict=True):
        rows = candidates[candidates["run_id"].eq(run_id)]
        if rows.empty:
            raise ValueError(f"No top-K candidate rows for {run_id}.")
        _candidate_panel(axis, rows, main_k)
    metrics = (
        ("sample_self_consistency", MODEL_COLOR, "Two sampled top-K sets"),
        ("sample_vs_model_reference", "#E9A03B", "Sample versus model reference"),
        ("sample_vs_external_reference", EXTERNAL_COLOR, "Sample versus external reference"),
    )
    for axis, family in zip(axes[1], families, strict=True):
        rows = curves[curves["family"].eq(family)]
        for metric, color, label in metrics:
            for _, run_rows in rows.groupby("run_id", sort=False):
                axis.plot(run_rows["k"], run_rows[metric], color=color, alpha=0.18, linewidth=0.8)
            center = rows.groupby("k", as_index=False)[metric].median()
            axis.plot(
                center["k"],
                center[metric],
                color=color,
                marker="o",
                markersize=3,
                linewidth=2,
                label=label,
            )
        axis.axvline(main_k, color=AXIS_COLOR, linestyle=":")
        axis.set(
            title=f"{_family_label(family)} stability across K",
            xlabel="Top-set size K",
            ylabel="Expected shared fraction",
            xscale="log",
            ylim=(-0.02, 1.02),
        )
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(frameon=False, fontsize=6)
        _style_axis(axis)
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", axes.ravel()):
        _panel_label(axis, label)
    figure.suptitle("Top-K decision stability", fontsize=13, fontweight="bold")
    figure.tight_layout()
    return figure


def _primary_rows(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    variants = tables["mechanism_variants"]
    rows = variants[variants["is_primary_cohort"].astype(bool)].copy()
    if rows.empty:
        raise ValueError("Mechanism tables contain no primary-cohort rows.")
    return rows


def _quantile_summary(rows: pd.DataFrame, predictor: str, response: str) -> pd.DataFrame:
    selected = rows[np.isfinite(rows[predictor]) & np.isfinite(rows[response])].copy()
    selected["quantile"] = selected.groupby("run_id", sort=False)[predictor].transform(
        lambda values: np.ceil(values.rank(method="first", pct=True) * 10).clip(1, 10)
    )
    return selected.groupby(["run_id", "family", "quantile"], as_index=False).agg(
        predictor_median=(predictor, "median"),
        response_median=(response, "median"),
    )


def _plot_quantile_summary(axis, summary: pd.DataFrame, title: str, ylabel: str) -> None:
    for _, rows in summary.groupby("run_id", sort=False):
        axis.plot(rows["quantile"], rows["response_median"], color=AXIS_COLOR, alpha=0.2)
    center = summary.groupby("quantile", as_index=False)["response_median"].median()
    axis.plot(
        center["quantile"], center["response_median"], color=MODEL_COLOR, marker="o", linewidth=2
    )
    if (summary["response_median"] > 0).all():
        axis.set_yscale("log")
    axis.set(title=title, xlabel="Within-run predictor decile", ylabel=ylabel, xticks=range(1, 11))
    _style_axis(axis)


def _score_geometry(axis, rows: pd.DataFrame, *, crowded: bool) -> None:
    target = rows.nlargest(1, "score_crowding") if crowded else rows.nsmallest(1, "score_crowding")
    rank = int(target["model_reference_rank"].iloc[0])
    neighborhood = rows[rows["model_reference_rank"].between(rank - 5, rank + 5)].sort_values(
        "model_reference_rank"
    )
    y = neighborhood["model_reference_rank"]
    axis.hlines(
        y,
        neighborhood["score_lower"],
        neighborhood["score_upper"],
        color=to_rgba(MODEL_COLOR, 0.45),
    )
    axis.scatter(neighborhood["score_median"], y, color=MODEL_COLOR, s=20)
    axis.scatter(
        target["score_median"],
        target["model_reference_rank"],
        color=CHALLENGER_COLOR,
        s=45,
        zorder=3,
    )
    axis.set(
        title="Crowded score neighborhood" if crowded else "Separated score neighborhood",
        xlabel="Sampled score interval",
        ylabel="Model reference rank",
    )
    axis.invert_yaxis()
    _style_axis(axis)


def figure_ranking_mechanisms(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show the pathway from count support to score and rank resolution."""
    rows = _primary_rows(tables)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    _plot_quantile_summary(
        axes[0, 0],
        _quantile_summary(rows, "mechanism_log_input_support", "normalized_score_interval_width"),
        "Input support to score uncertainty",
        "Median normalized score width",
    )
    _plot_quantile_summary(
        axes[0, 1],
        _quantile_summary(rows, "normalized_score_interval_width", "rank_interval_width"),
        "Score uncertainty to rank uncertainty",
        "Median rank-interval width",
    )
    example_id = rows.groupby("run_id").size().idxmax()
    example = rows[rows["run_id"].eq(example_id)]
    _score_geometry(axes[1, 0], example, crowded=False)
    _score_geometry(axes[1, 1], example, crowded=True)
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _panel_label(axis, label)
    figure.suptitle(
        "From experimental support to ranking resolution", fontsize=13, fontweight="bold"
    )
    figure.tight_layout()
    return figure


def _rank_window(axis, rows: pd.DataFrame, *, overlapping: bool) -> None:
    selected = (
        rows.nlargest(8, "rank_interval_width")
        if overlapping
        else rows.nsmallest(8, "rank_interval_width")
    )
    selected = selected.sort_values("model_reference_rank", ascending=False)
    y = np.arange(len(selected))
    median = selected["rank_median"].to_numpy(dtype=float)
    axis.errorbar(
        median,
        y,
        xerr=np.vstack([median - selected["rank_lower"], selected["rank_upper"] - median]),
        fmt="o",
        color=MODEL_COLOR,
        ecolor=to_rgba(MODEL_COLOR, 0.45),
    )
    axis.set(
        title="Overlapping rank intervals" if overlapping else "Separated rank intervals",
        xlabel="Sampled rank interval",
        yticks=y,
        yticklabels=selected["variant_id"],
    )
    _style_axis(axis, grid_axis="x")


def figure_rank_space_sensitivity(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show exploratory sensitivity directly in sampled-rank space."""
    rows = _primary_rows(tables)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    _plot_quantile_summary(
        axes[0, 0],
        _quantile_summary(rows, "mechanism_log_input_support", "rank_interval_width"),
        "Input support to rank uncertainty",
        "Median rank-interval width",
    )
    _plot_quantile_summary(
        axes[0, 1],
        _quantile_summary(rows, "score_interval_width", "rank_interval_width"),
        "Score uncertainty to rank uncertainty",
        "Median rank-interval width",
    )
    example_id = rows.groupby("run_id").size().idxmax()
    example = rows[rows["run_id"].eq(example_id)]
    _rank_window(axes[1, 0], example, overlapping=False)
    _rank_window(axes[1, 1], example, overlapping=True)
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _panel_label(axis, label)
    figure.suptitle("Rank-space sensitivity", fontsize=13, fontweight="bold")
    figure.tight_layout()
    return figure


def _count_uncertainty_frame(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = tables["mechanism_variants"].copy()
    runs = tables["mechanism_runs"]
    _require(
        rows,
        {
            "run_id",
            "family",
            "landscape",
            "model",
            "mechanism_input_count",
            "mechanism_output_count",
            "normalized_score_interval_width",
            "rank_interval_width",
            "is_model_top_k",
            "is_challenger",
        },
        "mechanism_variants",
    )
    frame = rows.merge(
        runs[["run_id", "n_candidates"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    numeric = (
        "mechanism_input_count",
        "mechanism_output_count",
        "normalized_score_interval_width",
        "rank_interval_width",
        "n_candidates",
    )
    if not np.isfinite(frame[list(numeric)]).all().all():
        raise ValueError("Count-uncertainty inputs must be finite.")
    if (frame[["mechanism_input_count", "mechanism_output_count"]] < 0).any().any():
        raise ValueError("Count features must be non-negative.")
    frame["input_log_count"] = np.log10(frame["mechanism_input_count"] + 1)
    frame["output_log_count"] = np.log10(frame["mechanism_output_count"] + 1)
    floor = _positive_floor(frame["normalized_score_interval_width"])
    frame["score_uncertainty_plot"] = np.log10(
        np.maximum(frame["normalized_score_interval_width"], floor)
    )
    frame["relative_rank_uncertainty"] = frame["rank_interval_width"] / frame["n_candidates"]
    return frame


def _draw_density_panel(
    axis,
    rows: pd.DataFrame,
    *,
    x: str,
    y: str,
    renderer: str,
    colors: Mapping[str, str],
) -> None:
    if renderer == "scatter":
        for landscape, group in rows.groupby("landscape", sort=False):
            axis.scatter(
                group[x],
                group[y],
                color=colors[str(landscape)],
                s=7,
                alpha=0.15,
                linewidth=0,
                rasterized=True,
            )
    elif renderer == "hexbin":
        axis.hexbin(
            rows[x], rows[y], gridsize=45, mincnt=1, bins="log", cmap="viridis", linewidths=0
        )
    elif renderer == "datashader":
        import datashader as ds
        import datashader.transfer_functions as tf

        frame = rows[[x, y, "landscape"]].copy()
        frame["landscape"] = pd.Categorical(frame["landscape"], categories=list(colors))
        x_range = (float(frame[x].min()), float(frame[x].max()))
        y_range = (float(frame[y].min()), float(frame[y].max()))
        if x_range[0] == x_range[1] or y_range[0] == y_range[1]:
            raise ValueError("Datashader panels require varying x and y values.")
        canvas = ds.Canvas(plot_width=600, plot_height=400, x_range=x_range, y_range=y_range)
        aggregate = canvas.points(frame, x, y, agg=ds.count_cat("landscape"))
        image = tf.shade(aggregate, color_key=dict(colors), how="log", min_alpha=20)
        axis.imshow(
            np.asarray(image.to_pil()),
            extent=(*x_range, *y_range),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
    else:
        raise ValueError("renderer must be 'scatter', 'datashader', or 'hexbin'.")
    _trend(axis, rows, x, y)
    _style_axis(axis)


def figure_count_uncertainty(
    tables: Mapping[str, pd.DataFrame],
    *,
    renderer: str = "scatter",
) -> Figure:
    """Show count support against score and rank uncertainty."""
    rows = _count_uncertainty_frame(tables)
    runs = tables["runs"]
    colors, _, _ = _encodings(runs)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    specs = (
        (
            axes[0, 0],
            "input_log_count",
            "score_uncertainty_plot",
            "Input count",
            "Normalized score width (log10)",
        ),
        (
            axes[0, 1],
            "output_log_count",
            "score_uncertainty_plot",
            "Output count",
            "Normalized score width (log10)",
        ),
        (
            axes[1, 0],
            "input_log_count",
            "relative_rank_uncertainty",
            "Input count",
            "Rank width / candidate count",
        ),
        (
            axes[1, 1],
            "output_log_count",
            "relative_rank_uncertainty",
            "Output count",
            "Rank width / candidate count",
        ),
    )
    for axis, x, y, x_label, y_label in specs:
        _draw_density_panel(axis, rows, x=x, y=y, renderer=renderer, colors=colors)
        axis.set(xlabel=f"{x_label} (log10[count + 1])", ylabel=y_label)
    for axis in axes[1]:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _panel_label(axis, label)
    figure.suptitle(
        f"Count support and fitted uncertainty ({renderer})", fontsize=13, fontweight="bold"
    )
    figure.tight_layout()
    return figure


def _figure_count_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str,
    response: str,
) -> Figure:
    rows = _count_uncertainty_frame(tables)
    rows = rows[rows["family"].eq(family)]
    if rows.empty:
        raise ValueError(f"No mechanism rows for family {family!r}.")
    run_order = tables["runs"].loc[tables["runs"]["family"].eq(family), "run_id"]
    run_ids = [run_id for run_id in run_order if run_id in set(rows["run_id"])]
    y = "score_uncertainty_plot" if response == "score" else "relative_rank_uncertainty"
    y_label = (
        "Normalized score width (log10)" if response == "score" else "Rank width / candidate count"
    )
    figure, axes = plt.subplots(
        len(run_ids), 2, figsize=(12.5, 2.1 * len(run_ids) + 1.8), squeeze=False, sharex="col"
    )
    for row_index, run_id in enumerate(run_ids):
        run_rows = rows[rows["run_id"].eq(run_id)]
        for column_index, x in enumerate(("input_log_count", "output_log_count")):
            axis = axes[row_index, column_index]
            axis.scatter(
                run_rows[x],
                run_rows[y],
                color=MODEL_COLOR,
                s=7,
                alpha=0.2,
                linewidth=0,
                rasterized=True,
            )
            _trend(axis, run_rows, x, y)
            correlation = _safe_spearman(run_rows[x], run_rows[y])
            label = "n/a" if not np.isfinite(correlation) else f"{correlation:.2f}"
            axis.text(
                0.98,
                0.9,
                f"rho = {label}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            _style_axis(axis)
        axes[row_index, 0].set_title(_run_label(run_rows.iloc[0]), loc="left", fontsize=8)
    axes[-1, 0].set_xlabel("Input count (log10[count + 1])")
    axes[-1, 1].set_xlabel("Output count (log10[count + 1])")
    figure.supylabel(y_label)
    figure.suptitle(
        f"{_family_label(family)} count support and {response} uncertainty by run",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def figure_count_score_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str,
) -> Figure:
    """Show input/output count against score uncertainty for each run."""
    return _figure_count_uncertainty_by_analysis(tables, family=family, response="score")


def figure_count_rank_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str,
) -> Figure:
    """Show input/output count against rank uncertainty for each run."""
    return _figure_count_uncertainty_by_analysis(tables, family=family, response="rank")


def _density(axis, values: np.ndarray, color: str = TEXT_COLOR) -> None:
    values = np.asarray(values, dtype=float)
    if len(values) >= 3 and np.ptp(values) > 0:
        try:
            estimator = gaussian_kde(values)
            padding = max(np.ptp(values) * 0.05, 1e-6)
            grid = np.linspace(values.min() - padding, values.max() + padding, 200)
            density = estimator(grid)
            axis.fill_between(grid, density, color=to_rgba(color, 0.15))
            axis.plot(grid, density, color=color)
            return
        except (np.linalg.LinAlgError, ValueError):
            pass
    axis.hist(values, bins=min(10, max(1, len(values))), color=to_rgba(color, 0.35))


def figure_count_score_corner_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str,
    n_columns: int = 3,
) -> Figure:
    """Show joint input count, output count, and score uncertainty by run."""
    if n_columns < 1:
        raise ValueError("n_columns must be positive.")
    rows = _count_uncertainty_frame(tables)
    rows = rows[rows["family"].eq(family)]
    if rows.empty:
        raise ValueError(f"No mechanism rows for family {family!r}.")
    run_ids = rows["run_id"].drop_duplicates().tolist()
    n_columns = min(n_columns, len(run_ids))
    n_rows = (len(run_ids) + n_columns - 1) // n_columns
    figure = plt.figure(figsize=(4.7 * n_columns, 4.7 * n_rows), layout="constrained")
    outer = figure.add_gridspec(n_rows, n_columns)
    variables = (
        ("input_log_count", "Input count"),
        ("output_log_count", "Output count"),
        ("score_uncertainty_plot", "Score uncertainty"),
    )
    for run_position, run_id in enumerate(run_ids):
        row_index, column_index = divmod(run_position, n_columns)
        grid = outer[row_index, column_index].subgridspec(3, 3, wspace=0.08, hspace=0.08)
        run_rows = rows[rows["run_id"].eq(run_id)]
        for y_index, (y, y_label) in enumerate(variables):
            for x_index, (x, x_label) in enumerate(variables):
                axis = figure.add_subplot(grid[y_index, x_index])
                if x_index == y_index:
                    _density(axis, run_rows[x].to_numpy())
                    axis.tick_params(axis="y", left=False, labelleft=False)
                elif y_index > x_index:
                    context = ~(run_rows["is_model_top_k"] | run_rows["is_challenger"])
                    axis.scatter(
                        run_rows.loc[context, x],
                        run_rows.loc[context, y],
                        color=AXIS_COLOR,
                        s=3,
                        alpha=0.25,
                    )
                    axis.scatter(
                        run_rows.loc[run_rows["is_model_top_k"], x],
                        run_rows.loc[run_rows["is_model_top_k"], y],
                        color=MODEL_COLOR,
                        s=5,
                        alpha=0.55,
                    )
                    axis.scatter(
                        run_rows.loc[run_rows["is_challenger"], x],
                        run_rows.loc[run_rows["is_challenger"], y],
                        facecolors="none",
                        edgecolors=CHALLENGER_COLOR,
                        s=9,
                        linewidth=0.5,
                    )
                else:
                    correlation = _safe_spearman(run_rows[x], run_rows[y])
                    label = "n/a" if not np.isfinite(correlation) else f"{correlation:.2f}"
                    axis.text(0.5, 0.5, f"rho = {label}", ha="center", va="center")
                    axis.set_axis_off()
                if y_index == 2:
                    axis.set_xlabel(x_label, fontsize=7)
                else:
                    axis.tick_params(labelbottom=False)
                if x_index == 0 and y_index > 0:
                    axis.set_ylabel(y_label, fontsize=7)
                elif x_index < y_index:
                    axis.tick_params(labelleft=False)
                axis.tick_params(labelsize=6)
                if axis.axison:
                    _style_axis(axis)
        first_axis = figure.axes[-9]
        first_axis.set_title(
            _run_label(run_rows.iloc[0]), loc="left", fontsize=8, fontweight="bold"
        )
    figure.suptitle(
        f"{_family_label(family)} count and score-uncertainty geometry",
        fontsize=13,
        fontweight="bold",
    )
    return figure


def figure_functionality_distributions(
    runs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    family: str,
    top_k: int = 100,
    n_columns: int | None = None,
) -> Figure:
    """Show all-candidate median-score densities and top-K cutoffs."""
    family_runs = runs[runs["family"].eq(family)]
    if family_runs.empty:
        raise ValueError(f"No runs for family {family!r}.")
    landscapes = family_runs["landscape"].drop_duplicates().tolist()
    columns = n_columns or (2 if family == "nuc" else 3)
    columns = min(columns, len(landscapes))
    rows_count = (len(landscapes) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(5 * columns, 3.2 * rows_count), squeeze=False
    )
    colors, markers, lines = _encodings(runs)
    for axis, landscape in zip(axes.ravel(), landscapes):
        landscape_runs = family_runs[family_runs["landscape"].eq(landscape)]
        for _, run in landscape_runs.iterrows():
            values = scores[scores["run_id"].eq(run["run_id"])].sort_values("model_reference_rank")
            if len(values) < top_k:
                raise ValueError(f"{run['run_id']}: fewer than top_k={top_k} candidate scores.")
            color = colors[str(landscape)]
            score_values = values["score_median"].to_numpy(dtype=float)
            if len(score_values) >= 3 and np.ptp(score_values) > 0:
                estimator = gaussian_kde(score_values)
                grid = np.linspace(score_values.min(), score_values.max(), 300)
                axis.plot(
                    grid,
                    estimator(grid),
                    color=color,
                    linestyle=lines[str(run["model"])],
                    label=_short_model(str(run["model"])),
                )
            else:
                axis.hist(
                    score_values, bins=min(10, len(score_values)), histtype="step", color=color
                )
            cutoff = float(
                values.loc[values["model_reference_rank"].eq(top_k), "score_median"].iloc[0]
            )
            axis.axvline(cutoff, color=color, linestyle=lines[str(run["model"])], alpha=0.7)
            axis.scatter(cutoff, 0, color=color, marker=markers[str(run["model"])], s=20)
        axis.set(
            title=_short_landscape(str(landscape), family),
            xlabel="Median sampled score",
            ylabel="Density",
        )
        axis.legend(frameon=False, fontsize=6)
        _style_axis(axis)
    for axis in axes.ravel()[len(landscapes) :]:
        axis.set_axis_off()
    figure.suptitle(
        f"{_family_label(family)} median-score distributions", fontsize=13, fontweight="bold"
    )
    figure.tight_layout()
    return figure


def _decision_rows(tables: Mapping[str, pd.DataFrame], family: str | None) -> pd.DataFrame:
    rows = _primary_rows(tables)
    if family is not None:
        rows = rows[rows["family"].eq(family)].copy()
    if rows.empty:
        raise ValueError(f"No primary mechanism rows for family {family!r}.")
    rows["decision_relative_score"] = (
        rows["score_margin_to_top_k_boundary"] / rows["reference_score_iqr"]
    )
    floor = _positive_floor(rows["normalized_score_interval_width"])
    rows["score_uncertainty_plot"] = np.maximum(rows["normalized_score_interval_width"], floor)
    return rows


def _status_scatter(axis, rows: pd.DataFrame, x: str, y: str) -> None:
    model = rows["is_model_top_k"].astype(bool)
    challenger = rows["is_challenger"].astype(bool)
    axis.scatter(
        rows.loc[model, x], rows.loc[model, y], color=MODEL_COLOR, s=11, alpha=0.45, linewidth=0
    )
    axis.scatter(
        rows.loc[challenger, x],
        rows.loc[challenger, y],
        facecolors="none",
        edgecolors=CHALLENGER_COLOR,
        s=15,
        alpha=0.75,
        linewidth=0.6,
    )


def figure_fitness_uncertainty_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str,
    n_columns: int | None = None,
) -> Figure:
    """Relate decision-centered median score to uncertainty by run."""
    rows = _decision_rows(tables, family)
    run_ids = rows["run_id"].drop_duplicates().tolist()
    columns = n_columns or (2 if family == "pdz" else 3)
    columns = min(columns, len(run_ids))
    n_rows = (len(run_ids) + columns - 1) // columns
    figure, axes = plt.subplots(
        n_rows, columns, figsize=(5 * columns, 3.1 * n_rows), squeeze=False, sharey=True
    )
    for axis, run_id in zip(axes.ravel(), run_ids):
        run_rows = rows[rows["run_id"].eq(run_id)]
        _status_scatter(axis, run_rows, "decision_relative_score", "score_uncertainty_plot")
        axis.axvline(0, color=AXIS_COLOR, linestyle=":")
        correlation = _safe_spearman(
            run_rows["decision_relative_score"], run_rows["normalized_score_interval_width"]
        )
        label = "n/a" if not np.isfinite(correlation) else f"{correlation:.2f}"
        axis.text(
            0.98, 0.94, f"rho = {label}", transform=axis.transAxes, ha="right", va="top", fontsize=7
        )
        axis.set(
            title=_run_label(run_rows.iloc[0]),
            xlabel="Decision-relative median score",
            ylabel="Normalized score width",
            yscale="log",
        )
        _style_axis(axis)
    for axis in axes.ravel()[len(run_ids) :]:
        axis.set_axis_off()
    figure.suptitle(
        f"{_family_label(family)} decision-relative score and uncertainty",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def figure_summary_why_by_analysis(
    tables: Mapping[str, pd.DataFrame],
    *,
    family: str | None = None,
    run_ids: tuple[str, ...] | None = None,
) -> Figure:
    """Show Figure 10's four relationships without fitted regression lines."""
    if family is None:
        if not run_ids:
            raise ValueError("Cross-family summaries require explicit run IDs.")
        variants = tables["mechanism_variants"]
        tables = {**tables, "mechanism_variants": variants[variants["run_id"].isin(run_ids)]}
    rows = _decision_rows(tables, family)
    annotations = tables["mechanism_runs"].set_index("run_id")
    rows["input_log_count"] = np.log10(rows["mechanism_input_count"] + 1)
    rows["output_log_count"] = np.log10(rows["mechanism_output_count"] + 1)
    available_ids = rows["run_id"].drop_duplicates().tolist()
    selected_ids = available_ids if run_ids is None else list(run_ids)
    if (not selected_ids or len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids).issubset(available_ids)):
        raise ValueError("Summary run IDs must be unique, nonempty, and present in the family.")
    sampling = rows["sampling"].drop_duplicates().tolist()
    if len(sampling) != 1:
        raise ValueError("Each summary figure must contain exactly one resampling scheme.")
    sampling_label = {
        "parameter_posterior": "Parameter resampling",
        "mle_predictive": "Optimized-fit count resampling",
        "posterior_predictive": "Posterior count resampling",
    }.get(str(sampling[0]), str(sampling[0]))
    count_max = max(float(rows["input_log_count"].max()), float(rows["output_log_count"].max()))
    score_limits = (
        float(rows["score_uncertainty_plot"].min()) * 0.8,
        float(rows["score_uncertainty_plot"].max()) * 1.25,
    )
    figure = plt.figure(
        figsize=(MANUSCRIPT_FIGURE_WIDTH, 2.0 * len(selected_ids) + 0.6), layout="constrained"
    )
    subfigures = figure.subfigures(len(selected_ids), 1, squeeze=False)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=MODEL_COLOR,
               label="Reference top-k member", markersize=4),
        Line2D([], [], marker="o", linestyle="none", color=CHALLENGER_COLOR,
               markerfacecolor="none", label="Outside challenger", markersize=4),
    ]
    figure.legend(
        handles=handles, loc="outside upper center", ncol=2, frameon=False,
        title=sampling_label if family is None else f"{_family_label(family)} | {sampling_label}",
    )
    for row_index, run_id in enumerate(selected_ids):
        run_rows = rows[rows["run_id"].eq(run_id)]
        annotation = annotations.loc[run_id]
        reference_count = int(run_rows["is_model_top_k"].sum())
        subfigure = subfigures[row_index, 0]
        axes = subfigure.subplots(1, 4)
        model_label = str(run_rows["model"].iloc[0]).replace("-", " / ").title()
        row_family = str(run_rows["family"].iloc[0])
        landscape_label = _short_landscape(str(run_rows["landscape"].iloc[0]), row_family)
        if family is None:
            landscape_label = f"{_family_label(row_family)} {landscape_label}"
        subfigure.suptitle(
            f"{landscape_label} | {model_label}\n"
            f"{int(annotation['n_samples']):,} samples; reference top {reference_count} + "
            f"{int(annotation['challenger_count']):,} challengers; "
            f"mean shared fraction {annotation['sample_self_consistency']:.1%}",
            x=0.01, ha="left",
        )
        specs = (
            ("input_log_count", "score_uncertainty_plot"),
            ("output_log_count", "score_uncertainty_plot"),
            ("decision_relative_score", "score_uncertainty_plot"),
            ("decision_relative_score", "top_k_inclusion_probability"),
        )
        for column, (axis, (x, y)) in enumerate(zip(axes, specs, strict=True)):
            _status_scatter(axis, run_rows, x, y)
            for collection in axis.collections:
                collection.set_rasterized(True)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.text(0.02, 0.98, chr(97 + 4 * row_index + column),
                      transform=axis.transAxes, va="top", fontweight="bold", gid="panel-label")
            if column < 3:
                axis.set(yscale="log", ylim=score_limits)
                axis.yaxis.set_minor_formatter(NullFormatter())
            if column in (1, 2):
                axis.tick_params(axis="y", labelleft=False)
            _style_axis(axis)
        for axis in axes[:2]:
            axis.set_xlim(-0.02, max(0.1, count_max * 1.04))
        margin = run_rows["decision_relative_score"]
        padding = max(float(margin.max() - margin.min()) * 0.04, 0.01)
        for axis in axes[2:]:
            axis.set_xlim(float(margin.min()) - padding, float(margin.max()) + padding)
            axis.axvline(0, color=AXIS_COLOR, linestyle=":", linewidth=0.8)
            axis.set_xlabel("Score margin / IQR")
        axes[0].set_xlabel("Input count\nlog10(n + 1)")
        axes[1].set_xlabel("Output count\nlog10(n + 1)")
        axes[0].set_ylabel("90% score width / IQR")
        axes[3].set(ylim=(-0.03, 1.03), yticks=[0, 0.5, 1])
        axes[3].set_ylabel(f"Top-{reference_count} inclusion")
        axes[3].yaxis.set_major_formatter(PercentFormatter(1))
    apply_figure_typography(figure)
    return figure
