"""Notebook-friendly plots for saved fitness-stability results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .results import StabilityResult

DEFAULT_COLOR = "#2F6F9F"
REFERENCE_COLOR = "#487A57"
INTERVAL_COLOR = "#9CB8CB"
SCORE_LABELS = {
    "log_mean_growth": "Posterior log mean growth",
    "count_ratio": "Posterior-predictive output/input ratio",
    "dimsum_fitness": "DiMSum-style count fitness",
}


@dataclass(frozen=True)
class _PlotData:
    metadata: dict[str, object]
    summary: pd.DataFrame
    overlaps: pd.DataFrame


ResultLike = StabilityResult | _PlotData | str | Path
ResultCollection = Mapping[str, ResultLike] | Sequence[tuple[str, ResultLike]]


def _load_plot_data(result: ResultLike) -> _PlotData:
    if isinstance(result, _PlotData):
        return result
    if isinstance(result, StabilityResult):
        return _PlotData(result.metadata, result.summary, result.overlaps)
    result_dir = Path(result)
    required = ("metadata.json", "summary.csv", "overlaps.csv")
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise ValueError(f"Result directory is missing plotting files: {missing}")
    return _PlotData(
        metadata=json.loads((result_dir / "metadata.json").read_text(encoding="utf-8")),
        summary=pd.read_csv(result_dir / "summary.csv", low_memory=False),
        overlaps=pd.read_csv(result_dir / "overlaps.csv"),
    )


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")


def _available_values(frame: pd.DataFrame, column: str) -> list[object]:
    return frame[column].drop_duplicates().tolist()


def _selected_summary(data: _PlotData, *, score: str, k: int) -> pd.DataFrame:
    _require_columns(data.summary, {"score", "k"})
    available_scores = _available_values(data.summary, "score")
    if score not in available_scores:
        raise ValueError(f"Unknown score {score!r}; choose from {available_scores}.")
    available_k = sorted(data.summary.loc[data.summary["score"].eq(score), "k"].unique())
    if k not in available_k:
        raise ValueError(f"K={k} is unavailable for {score!r}; choose from {available_k}.")
    return data.summary[data.summary["score"].eq(score) & data.summary["k"].eq(k)].copy()


def _score_label(score: str) -> str:
    return SCORE_LABELS.get(score, score.replace("_", " ").capitalize())


def _landscape_label(data: _PlotData) -> str:
    value = data.metadata.get("landscape")
    if value is None and "landscape" in data.summary:
        values = _available_values(data.summary, "landscape")
        value = values[0] if values else "Landscape"
    return str(value or "Landscape").replace("-", " ").upper()


def _variant_labels(frame: pd.DataFrame, label_column: str | None) -> pd.Series:
    resolved = label_column
    if resolved is None:
        resolved = "variant_label" if "variant_label" in frame else "variant_id"
    if resolved not in frame:
        raise ValueError(f"Unknown variant label column {resolved!r}.")
    labels = frame[resolved].fillna(frame.get("variant_id", "variant")).astype(str)
    if resolved == "variant_id":
        labels = labels.map(
            lambda value: value if len(value) <= 52 else f"{value[:24]}...{value[-24:]}"
        )
    return labels


def _style_axis(axis: Axes, *, grid_axis: str) -> None:
    axis.grid(axis=grid_axis, color="#E2E5E9", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8A8F98")
    axis.tick_params(colors="#4E535B")


def _result_items(results: ResultCollection) -> list[tuple[str, ResultLike]]:
    items = list(results.items()) if isinstance(results, Mapping) else list(results)
    if not items:
        raise ValueError("results must contain at least one labeled result.")
    labels = [str(label) for label, _ in items]
    if len(set(labels)) != len(labels):
        raise ValueError("Result labels must be unique.")
    return [(str(label), result) for (label, result) in items]


def _result_colors(
    labels: Sequence[str],
    colors: Mapping[str, str] | None,
) -> dict[str, object]:
    if colors is not None:
        missing = set(labels) - set(colors)
        if missing:
            raise ValueError(f"Missing colors for result labels: {sorted(missing)}")
        return {label: colors[label] for label in labels}
    color_map = plt.get_cmap("tab20")
    denominator = max(1, len(labels) - 1)
    return {
        label: color_map(index / denominator)
        for index, label in enumerate(labels)
    }


def _calibration_subset(
    data: _PlotData,
    *,
    score: str,
    k: int,
    cohort: str,
) -> pd.DataFrame:
    if cohort not in {"tracked", "reported_top_k"}:
        raise ValueError("cohort must be 'tracked' or 'reported_top_k'.")
    summary = _selected_summary(data, score=score, k=k)
    _require_columns(
        summary,
        {
            "reported_rank",
            "reported_rank_percentile",
            "is_tracked_reported_cohort",
        },
    )
    selected = summary[summary["is_tracked_reported_cohort"].astype(bool)]
    if cohort == "reported_top_k":
        selected = selected[selected["reported_rank"] <= k]
    return selected


def _absolute_ecdf_area(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration percentiles must be finite and non-empty.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Calibration percentiles must be between 0 and 1.")

    edges = np.r_[0.0, values, 1.0]
    levels = np.r_[0.0, np.arange(1, len(values) + 1) / len(values)]
    area = 0.0
    for left, right, level in zip(edges[:-1], edges[1:], levels):
        if level <= left:
            area += 0.5 * ((right - level) ** 2 - (left - level) ** 2)
        elif level >= right:
            area += 0.5 * ((level - left) ** 2 - (level - right) ** 2)
        else:
            area += 0.5 * ((level - left) ** 2 + (right - level) ** 2)
    return float(area)


def _calibration_curves(
    results: ResultCollection,
    *,
    score: str,
    k: int,
    cohort: str,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], pd.DataFrame]:
    curves = []
    diagnostics = []
    for label, result in _result_items(results):
        selected = _calibration_subset(
            _load_plot_data(result),
            score=score,
            k=k,
            cohort=cohort,
        )
        percentiles = np.sort(
            selected["reported_rank_percentile"].dropna().to_numpy(dtype=float)
        )
        if not len(percentiles):
            raise ValueError(f"No finite calibration percentiles for {label!r}.")
        ecdf = np.arange(1, len(percentiles) + 1) / len(percentiles)
        curves.append(
            (
                label,
                np.r_[0.0, percentiles, 1.0],
                np.r_[0.0, ecdf, 1.0],
            )
        )
        diagnostics.append(
            {
                "result": label,
                "cohort": cohort,
                "n_variants": len(percentiles),
                "calibration_l1_area": _absolute_ecdf_area(percentiles),
                "median_reported_rank_percentile": np.median(percentiles),
                "fraction_below_0.05": np.mean(percentiles < 0.05),
                "fraction_above_0.95": np.mean(percentiles > 0.95),
            }
        )
    return curves, pd.DataFrame(diagnostics)


def plot_calibration_overlay(
    results: ResultCollection,
    *,
    score: str,
    k: int,
    cohort: str = "tracked",
    colors: Mapping[str, str] | None = None,
    alpha: float = 0.6,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Overlay reported-rank calibration ECDFs for multiple saved results."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    curves, diagnostics = _calibration_curves(
        results,
        score=score,
        k=k,
        cohort=cohort,
    )
    labels = [label for label, _, _ in curves]
    resolved_colors = _result_colors(labels, colors)
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    for label, percentile, ecdf in curves:
        axis.step(
            percentile,
            ecdf,
            where="post",
            color=resolved_colors[label],
            linewidth=1.5,
            alpha=alpha,
            label=label,
        )
    axis.plot([0, 1], [0, 1], color="#343A40", linestyle="--", linewidth=1.2)
    cohort_label = "tracked reported cohort" if cohort == "tracked" else f"reported top {k}"
    axis.set(
        title=(
            "Reported-rank calibration across results\n"
            f"{_score_label(score)}; {cohort_label}"
        ),
        xlabel="Directional reported-rank percentile",
        ylabel="ECDF",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1 if len(labels) <= 10 else 2,
        fontsize=8.5,
    )
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return figure, axis, diagnostics


def plot_calibration_divergence(
    results: ResultCollection,
    *,
    score: str,
    k: int,
    cohort: str = "tracked",
    colors: Mapping[str, str] | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Rank results by area between their calibration ECDF and the diagonal."""
    _, diagnostics = _calibration_curves(
        results,
        score=score,
        k=k,
        cohort=cohort,
    )
    diagnostics = diagnostics.sort_values("calibration_l1_area").reset_index(drop=True)
    labels = diagnostics["result"].astype(str).tolist()
    resolved_colors = _result_colors(labels, colors)
    y = np.arange(len(diagnostics))
    height = max(5.2, 0.34 * len(diagnostics))
    figure, axis = plt.subplots(figsize=(10.5, height))
    axis.hlines(
        y,
        0,
        diagnostics["calibration_l1_area"],
        color="#D3D8DE",
        linewidth=1.0,
    )
    axis.scatter(
        diagnostics["calibration_l1_area"],
        y,
        color=[resolved_colors[label] for label in labels],
        s=42,
        zorder=2,
    )
    cohort_label = "tracked reported cohort" if cohort == "tracked" else f"reported top {k}"
    axis.set(
        title=(
            "Calibration divergence across results\n"
            f"{_score_label(score)}; {cohort_label}"
        ),
        xlabel="Area between ECDF and uniform diagonal",
        ylabel="",
        yticks=y,
        yticklabels=labels,
        xlim=(0, None),
    )
    _style_axis(axis, grid_axis="x")
    figure.tight_layout()
    return figure, axis, diagnostics


def plot_rank_divergence_boxplots(
    results: ResultCollection,
    *,
    score: str,
    k: int,
    reported_top_n: int,
    colors: Mapping[str, str] | None = None,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot signed 5th/95th posterior-rank deviations for multiple results."""
    if reported_top_n < 1:
        raise ValueError("reported_top_n must be positive.")
    items = _result_items(results)
    labels = [label for label, _ in items]
    resolved_colors = _result_colors(labels, colors)
    frames = []
    for label, result in items:
        rows = _reported_rows(
            _load_plot_data(result),
            score=score,
            k=k,
            reported_top_n=reported_top_n,
            label_column=None,
        )
        common = rows[["variant_id", "reported_rank"]].copy()
        lower = common.copy()
        lower["bound"] = "5th percentile"
        lower["signed_rank_deviation"] = (
            rows["posterior_rank_lower"] - rows["reported_rank"]
        )
        upper = common.copy()
        upper["bound"] = "95th percentile"
        upper["signed_rank_deviation"] = (
            rows["posterior_rank_upper"] - rows["reported_rank"]
        )
        for frame in (lower, upper):
            frame.insert(0, "result", label)
        frames.extend((lower, upper))
    divergence = pd.concat(frames, ignore_index=True)
    if not np.isfinite(divergence["signed_rank_deviation"]).all():
        raise ValueError("Rank deviations must be finite for the requested cohort.")

    height = max(6.0, 0.34 * len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(16.0, height), sharey=True)
    for axis, bound in zip(axes, ("5th percentile", "95th percentile")):
        distributions = [
            divergence.loc[
                divergence["result"].eq(label) & divergence["bound"].eq(bound),
                "signed_rank_deviation",
            ].to_numpy()
            for label in labels
        ]
        boxes = axis.boxplot(
            distributions,
            orientation="horizontal",
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            widths=0.58,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": "#6F7780"},
            capprops={"color": "#6F7780"},
        )
        for patch, label in zip(boxes["boxes"], labels):
            patch.set_facecolor(resolved_colors[label])
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        axis.axvline(0, color="#343A40", linestyle="--", linewidth=1.0)
        axis.set(
            title=f"{bound} − reported rank",
            xlabel="Signed rank deviation",
            ylabel="",
        )
        _style_axis(axis, grid_axis="x")
    figure.suptitle(
        (
            f"Posterior rank divergence for reported top {reported_top_n}\n"
            f"{_score_label(score)}"
        ),
        fontweight="bold",
    )
    figure.tight_layout()
    return figure, axes, divergence


def plot_inclusion(
    result: ResultLike,
    *,
    score: str,
    k: int,
    n_variants: int = 40,
    label_column: str | None = None,
    color: str = DEFAULT_COLOR,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot variants with the largest posterior top-K inclusion probabilities."""
    if n_variants < 1:
        raise ValueError("n_variants must be positive.")
    data = _load_plot_data(result)
    summary = _selected_summary(data, score=score, k=k)
    _require_columns(
        summary,
        {"variant_id", "top_k_probability", "reported_rank"},
    )
    displayed = summary.nlargest(n_variants, "top_k_probability").copy()
    displayed = displayed.sort_values(
        ["top_k_probability", "reported_rank"],
        ascending=[True, False],
    )
    displayed["display_label"] = _variant_labels(displayed, label_column)
    reference_identifiable = bool(
        data.metadata.get("reference_top_k_identifiable", True)
    )
    displayed["reported_top_k"] = (
        displayed["reported_rank"] <= k if reference_identifiable else False
    )
    y = np.arange(len(displayed))

    height = max(5.5, 0.25 * len(displayed))
    figure, axis = plt.subplots(figsize=(10.5, height))
    axis.hlines(
        y,
        0,
        displayed["top_k_probability"],
        color="#D3D8DE",
        linewidth=1.0,
        zorder=1,
    )
    other = ~displayed["reported_top_k"]
    axis.scatter(
        displayed.loc[other, "top_k_probability"],
        y[other],
        s=42,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        label="Outside reported top K" if reference_identifiable else None,
        zorder=2,
    )
    if reference_identifiable:
        reported = displayed["reported_top_k"]
        axis.scatter(
            displayed.loc[reported, "top_k_probability"],
            y[reported],
            s=52,
            marker="D",
            color=REFERENCE_COLOR,
            edgecolor="white",
            linewidth=0.6,
            label="Reported top K",
            zorder=3,
        )
    axis.set(
        title=f"{_landscape_label(data)} top-{k} inclusion\n{_score_label(score)}",
        xlabel=f"Probability of entering the top {k}",
        ylabel="Variant",
        xlim=(-0.02, 1.02),
        yticks=y,
        yticklabels=displayed["display_label"],
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    if reference_identifiable:
        axis.legend(frameon=False, loc="lower right")
    _style_axis(axis, grid_axis="x")
    figure.tight_layout()
    return figure, axis, displayed


def plot_overlap(
    result: ResultLike,
    *,
    score: str,
    k: int,
    metric: str = "recall",
    color: str = DEFAULT_COLOR,
) -> tuple[Figure, Axes]:
    """Plot sample-to-sample and sample-to-reference top-K overlap."""
    data = _load_plot_data(result)
    _selected_summary(data, score=score, k=k)
    _require_columns(data.overlaps, {"score", "k", "comparison", metric})
    selected = data.overlaps[data.overlaps["score"].eq(score) & data.overlaps["k"].eq(k)]
    comparisons = ("sample_vs_sample", "sample_vs_reference")
    labels = ("Sample vs sample", "Sample vs reported")
    distributions = [
        selected.loc[selected["comparison"].eq(comparison), metric].dropna().to_numpy()
        for comparison in comparisons
    ]
    if any(not len(values) for values in distributions):
        raise ValueError("Both overlap comparisons must contain finite values.")

    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    boxes = axis.boxplot(
        distributions,
        tick_labels=labels,
        showfliers=False,
        widths=0.56,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": "#6F7780"},
        capprops={"color": "#6F7780"},
    )
    for patch, facecolor in zip(boxes["boxes"], (color, REFERENCE_COLOR)):
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.88)
        patch.set_edgecolor("white")
    axis.set(
        title=f"{_landscape_label(data)} top-{k} selection overlap\n{_score_label(score)}",
        xlabel="",
        ylabel=metric.replace("_", " ").capitalize(),
        ylim=(-0.02, 1.02),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axis(axis, grid_axis="y")
    figure.tight_layout()
    return figure, axis


def _reported_rows(
    data: _PlotData,
    *,
    score: str,
    k: int,
    reported_top_n: int | None,
    label_column: str | None,
) -> pd.DataFrame:
    summary = _selected_summary(data, score=score, k=k)
    _require_columns(summary, {"reported_rank", "is_tracked_reported_cohort"})
    resolved_top_n = k if reported_top_n is None else reported_top_n
    if resolved_top_n < 1:
        raise ValueError("reported_top_n must be positive.")
    displayed = summary[
        summary["is_tracked_reported_cohort"].astype(bool)
        & (summary["reported_rank"] <= resolved_top_n)
    ].sort_values("reported_rank")
    if len(displayed) != resolved_top_n:
        raise ValueError(
            f"Requested reported top {resolved_top_n}, but found {len(displayed)} rows."
        )
    displayed = displayed.copy()
    labels = _variant_labels(displayed, label_column)
    displayed["display_label"] = [
        f"{rank:g}  {label}" for rank, label in zip(displayed["reported_rank"], labels)
    ]
    return displayed


def plot_rank_intervals(
    result: ResultLike,
    *,
    score: str,
    k: int,
    reported_top_n: int | None = None,
    uncertainty: str = "both",
    rank_scale: str = "linear",
    context_multiplier: int = 6,
    label_column: str | None = None,
    color: str = DEFAULT_COLOR,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot full-landscape posterior ranks for a fixed reported cohort."""
    if uncertainty not in {"sd", "quantile", "both"}:
        raise ValueError("uncertainty must be 'sd', 'quantile', or 'both'.")
    if rank_scale not in {"linear", "log"}:
        raise ValueError("rank_scale must be 'linear' or 'log'.")
    if context_multiplier < 1:
        raise ValueError("context_multiplier must be positive.")
    data = _load_plot_data(result)
    displayed = _reported_rows(
        data,
        score=score,
        k=k,
        reported_top_n=reported_top_n,
        label_column=label_column,
    )
    _require_columns(
        displayed,
        {
            "posterior_mean_rank",
            "posterior_rank_sd",
            "posterior_rank_lower",
            "posterior_rank_upper",
        },
    )
    candidate_count = int(data.metadata.get("n_variants", displayed["posterior_mean_rank"].max()))
    mean = displayed["posterior_mean_rank"].to_numpy(dtype=float)
    rank_sd = displayed["posterior_rank_sd"].to_numpy(dtype=float)
    quantile_lower = displayed["posterior_rank_lower"].to_numpy(dtype=float)
    quantile_upper = displayed["posterior_rank_upper"].to_numpy(dtype=float)
    upper_endpoints = [mean]
    if uncertainty in {"sd", "both"}:
        upper_endpoints.append(mean + rank_sd)
    if uncertainty in {"quantile", "both"}:
        upper_endpoints.append(quantile_upper)
    endpoint_max = float(np.nanmax(np.concatenate(upper_endpoints)))
    if rank_scale == "linear":
        display_limit = min(candidate_count, max(k, min(context_multiplier * k, endpoint_max)))
        x_max = display_limit + 0.5
    else:
        display_limit = min(candidate_count, max(k, endpoint_max))
        x_max = display_limit * 1.05
    displayed_mean = np.minimum(mean, display_limit)
    y = np.arange(len(displayed))

    height = max(6.0, 0.25 * len(displayed))
    figure, axis = plt.subplots(figsize=(11.5, height))
    axis.axvspan(0.5, k + 0.5, color="#DCEAD8", alpha=0.55, zorder=0)
    axis.axvspan(k + 0.5, min(candidate_count, 2 * k) + 0.5, color="#ECEFF2", alpha=0.8, zorder=0)
    if uncertainty in {"quantile", "both"}:
        axis.hlines(
            y,
            np.minimum(quantile_lower, display_limit),
            np.minimum(quantile_upper, display_limit),
            color=INTERVAL_COLOR if uncertainty == "both" else color,
            linewidth=1.2,
            zorder=2,
        )
    if uncertainty in {"sd", "both"}:
        lower = np.maximum(1.0, mean - rank_sd)
        upper = np.minimum(candidate_count, mean + rank_sd)
        axis.errorbar(
            displayed_mean,
            y,
            xerr=np.vstack(
                [
                    displayed_mean - np.minimum(lower, display_limit),
                    np.minimum(upper, display_limit) - displayed_mean,
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=2.5,
            markersize=4.5,
            zorder=3,
        )
    elif uncertainty == "quantile":
        axis.scatter(displayed_mean, y, s=24, color=color, zorder=3)
    interval_upper = np.full(len(displayed), -np.inf)
    if uncertainty in {"sd", "both"}:
        interval_upper = np.maximum(interval_upper, mean + rank_sd)
    if uncertainty in {"quantile", "both"}:
        interval_upper = np.maximum(interval_upper, quantile_upper)
    clipped_mean = mean > display_limit
    clipped_interval = (interval_upper > display_limit) & ~clipped_mean
    if clipped_interval.any():
        axis.scatter(
            np.full(clipped_interval.sum(), display_limit),
            y[clipped_interval],
            marker=">",
            s=34,
            color=color,
            zorder=4,
            label="Interval extends beyond display",
        )
    if clipped_mean.any():
        axis.scatter(
            np.full(clipped_mean.sum(), display_limit),
            y[clipped_mean],
            marker=">",
            s=38,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=5,
            label="Mean rank beyond display",
        )
    axis.axvline(k + 0.5, color="#77936F", linewidth=1.0)
    title = (
        f"{_landscape_label(data)} reported top-{len(displayed)} ranks\n"
        f"{_score_label(score)}"
    )
    axis.set(
        title=title,
        xlabel="Posterior rank (1 = highest fitness)",
        ylabel="Reported rank and variant",
        xlim=(0.8 if rank_scale == "log" else 0.5, x_max),
        xscale=rank_scale,
        yticks=y,
        yticklabels=displayed["display_label"],
    )
    axis.invert_yaxis()
    if axis.get_legend_handles_labels()[0]:
        axis.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            borderaxespad=0.0,
        )
    _style_axis(axis, grid_axis="x")
    figure.tight_layout()
    return figure, axis, displayed


def plot_value_intervals(
    result: ResultLike,
    *,
    score: str,
    k: int,
    reported_top_n: int | None = None,
    uncertainty: str = "both",
    label_column: str | None = None,
    color: str = DEFAULT_COLOR,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot posterior score values for a fixed reported cohort."""
    if uncertainty not in {"sd", "quantile", "both"}:
        raise ValueError("uncertainty must be 'sd', 'quantile', or 'both'.")
    data = _load_plot_data(result)
    displayed = _reported_rows(
        data,
        score=score,
        k=k,
        reported_top_n=reported_top_n,
        label_column=label_column,
    )
    _require_columns(
        displayed,
        {
            "posterior_value_mean",
            "posterior_value_sd",
            "posterior_value_lower",
            "posterior_value_upper",
        },
    )
    mean = displayed["posterior_value_mean"].to_numpy(dtype=float)
    value_sd = displayed["posterior_value_sd"].to_numpy(dtype=float)
    y = np.arange(len(displayed))
    height = max(6.0, 0.25 * len(displayed))
    figure, axis = plt.subplots(figsize=(11.5, height))
    if uncertainty in {"quantile", "both"}:
        axis.hlines(
            y,
            displayed["posterior_value_lower"],
            displayed["posterior_value_upper"],
            color=INTERVAL_COLOR if uncertainty == "both" else color,
            linewidth=1.2,
            zorder=2,
        )
    if uncertainty in {"sd", "both"}:
        axis.errorbar(
            mean,
            y,
            xerr=value_sd,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=2.5,
            markersize=4.5,
            zorder=3,
        )
    else:
        axis.scatter(mean, y, s=24, color=color, zorder=3)
    title = (
        f"{_landscape_label(data)} reported top-{len(displayed)} values\n"
        f"{_score_label(score)}"
    )
    axis.set(
        title=title,
        xlabel=_score_label(score),
        ylabel="Reported rank and variant",
        yticks=y,
        yticklabels=displayed["display_label"],
    )
    axis.invert_yaxis()
    _style_axis(axis, grid_axis="x")
    figure.tight_layout()
    return figure, axis, displayed


def plot_rank_calibration(
    result: ResultLike,
    *,
    score: str,
    k: int,
    color: str = DEFAULT_COLOR,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot directional reported-rank percentile ECDFs for tracked variants."""
    data = _load_plot_data(result)
    summary = _selected_summary(data, score=score, k=k)
    _require_columns(
        summary,
        {
            "reported_rank",
            "reported_rank_percentile",
            "is_tracked_reported_cohort",
        },
    )
    tracked = summary[summary["is_tracked_reported_cohort"].astype(bool)]
    subsets = (
        (f"Tracked reported top {len(tracked):,}", tracked),
        (f"Reported top {k:,}", tracked[tracked["reported_rank"] <= k]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    diagnostics = []
    for axis, (label, subset) in zip(axes, subsets):
        percentiles = np.sort(subset["reported_rank_percentile"].dropna().to_numpy())
        if not len(percentiles):
            raise ValueError(f"No finite rank percentiles for {label}.")
        ecdf = np.arange(1, len(percentiles) + 1) / len(percentiles)
        axis.step(
            np.r_[0.0, percentiles, 1.0],
            np.r_[0.0, ecdf, 1.0],
            where="post",
            color=color,
            linewidth=1.8,
        )
        axis.plot([0, 1], [0, 1], color="#8A8F98", linestyle="--", linewidth=1.2)
        axis.set(
            title=label, xlabel="Directional reported-rank percentile", xlim=(0, 1), ylim=(0, 1)
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        _style_axis(axis, grid_axis="both")
        diagnostics.append(
            {
                "variant_subset": label,
                "n_variants": len(percentiles),
                "median_reported_rank_percentile": np.median(percentiles),
                "fraction_below_0.05": np.mean(percentiles < 0.05),
                "fraction_above_0.95": np.mean(percentiles > 0.95),
            }
        )
    axes[0].set_ylabel("ECDF")
    figure.suptitle(
        f"{_landscape_label(data)} reported-rank calibration\n{_score_label(score)}",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure, axes, pd.DataFrame(diagnostics)


def plot_report(
    result: ResultLike,
    *,
    k: int,
    scores: Sequence[str] | None = None,
    views: Sequence[str] = ("inclusion", "overlap", "rank", "value", "calibration"),
    n_variants: int = 40,
    reported_top_n: int | None = None,
    label_column: str | None = None,
) -> dict[str, Figure]:
    """Create a standard set of figures with one notebook command."""
    data = _load_plot_data(result)
    selected_scores = (
        tuple(_available_values(data.summary, "score")) if scores is None else tuple(scores)
    )
    valid_views = {"inclusion", "overlap", "rank", "value", "calibration"}
    unknown_views = set(views) - valid_views
    if unknown_views:
        raise ValueError(f"Unknown report views: {sorted(unknown_views)}")
    figures = {}
    for score in selected_scores:
        _selected_summary(data, score=score, k=k)
        if "inclusion" in views:
            figures[f"{score}.inclusion"] = plot_inclusion(
                data,
                score=score,
                k=k,
                n_variants=n_variants,
                label_column=label_column,
            )[0]
        if "overlap" in views:
            figures[f"{score}.overlap"] = plot_overlap(data, score=score, k=k)[0]
        if "rank" in views:
            figures[f"{score}.rank"] = plot_rank_intervals(
                data,
                score=score,
                k=k,
                reported_top_n=reported_top_n,
                label_column=label_column,
            )[0]
        if "value" in views:
            figures[f"{score}.value"] = plot_value_intervals(
                data,
                score=score,
                k=k,
                reported_top_n=reported_top_n,
                label_column=label_column,
            )[0]
        if "calibration" in views:
            figures[f"{score}.calibration"] = plot_rank_calibration(
                data,
                score=score,
                k=k,
            )[0]
    return figures
