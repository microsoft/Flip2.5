"""Compact manuscript figures from the same tables used by the notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, PercentFormatter

from analysis.examples import Example, representative_examples
from analysis.figures import (
    CHALLENGER_COLOR,
    FAMILY_LABELS,
    FAMILY_ORDER,
    MANUSCRIPT_FIGURE_WIDTH,
    MODEL_COLOR,
    _decision_rows,
    _panel_label,
    _positive_floor,
    apply_figure_typography,
    figure_summary_why_by_analysis,
    save_figure,
    set_paper_style,
)
from dms_rank_stability.tables import ecdf_to_uniform_area

SCHEMES = {
    "parameter_posterior": ("Parameter", "#276FBF", "o"),
    "mle_predictive": ("Optimized-fit counts", "#B84A48", "s"),
    "posterior_predictive": ("Posterior counts", "#208475", "^"),
}
RUNNING_EXAMPLES = (
    ("pdz-trans-1", "exponential-exponential", "PDZ trans-1"),
    ("trpb-libB", "exponential-exponential", "TrpB libB"),
    ("nuc-G3", "exponential", "NucB G3"),
)
EXAMPLE_ROLES = {
    "pdz-trans-1": "Many plausible replacements across a narrow score-width range.",
    "trpb-libB": "Low reported-rank agreement and substantial sampled-batch turnover.",
    "nuc-G3": "Uncertain internal ranks but a comparatively stable top-100 batch.",
}
CURVE_MODELS = {
    "exponential-exponential": ("E/E", "solid"),
    "exponential-sigmoid": ("E/S", (0, (5, 2))),
    "lomax-exponential": ("L/E", (0, (2, 1.5))),
    "lomax-sigmoid": ("L/S", (0, (5, 2, 1, 2))),
    "gamma-exponential": ("G/E", (0, (1, 1))),
    "gamma-sigmoid": ("G/S", (0, (3, 1, 1, 1, 1, 1))),
    "exponential": ("E", "solid"),
    "lomax": ("L", (0, (5, 2))),
}
TABLE_NAMES = (
    "runs",
    "rank_percentiles",
    "model_rank_intervals",
    "top_k_curves",
    "top_k_candidates",
    "mechanism_variants",
    "mechanism_runs",
    "mechanism_diagnostics",
    "reference_diagnostics",
)
TableCollection = Mapping[str, Mapping[str, pd.DataFrame]]


def landscape_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Give each landscape one median after combining its model specifications."""
    return frame.groupby(["family", "landscape"], as_index=False)[metric].median()


def _example(frame: pd.DataFrame, specification: tuple[str, str, str]) -> pd.DataFrame:
    landscape, model, _ = specification
    selected = frame[frame["landscape"].eq(landscape) & frame["model"].eq(model)]
    if selected.empty or selected["run_id"].nunique() != 1:
        raise ValueError(f"Expected one run for {landscape}, {model}.")
    return selected.copy()


def _finish(figure: Figure) -> Figure:
    for panel, axis in enumerate(figure.axes):
        _panel_label(axis, chr(97 + panel))
        axis.grid(alpha=0.55)
        axis.set_axisbelow(True)
    apply_figure_typography(figure)
    return figure


def _scheme_legend(figure: Figure, *, hollow_markers: bool = False) -> None:
    handles = [
        Line2D([], [], color=color, marker=marker, label=label,
               markerfacecolor="none" if hollow_markers else color)
        for label, color, marker in SCHEMES.values()
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)


def _landscape_dots(
    axis, frames: Mapping[str, pd.DataFrame], metric: str,
    *, example: Example | None = None, highlight_schemes: tuple[str, ...] | None = None,
) -> None:
    landscapes = list(dict.fromkeys(
        landscape for frame in frames.values() for landscape in frame["landscape"]
    ))
    for scheme_index, (sampling, frame) in enumerate(frames.items()):
        _, color, marker = SCHEMES[sampling]
        summary = landscape_summary(frame, metric).set_index("landscape")
        for position, landscape in enumerate(landscapes):
            if landscape not in summary.index:
                continue
            location = position + (scheme_index - 1) * 0.19
            models = frame.loc[frame["landscape"].eq(landscape)].sort_values(
                ["model", "run_id"], kind="stable"
            )
            offsets = np.linspace(-0.08, 0.08, len(models)) if len(models) > 1 else np.zeros(1)
            locations = location + offsets
            axis.scatter(
                models[metric], locations, color=color, s=8, alpha=0.8, linewidths=0,
                zorder=3, gid=f"model-points:{sampling}:{landscape}:{metric}",
            )
            axis.scatter(
                summary.loc[landscape, metric], location, marker=marker, s=32,
                facecolors="none", edgecolors=color, linewidths=0.8, zorder=2,
                gid=f"model-median:{sampling}:{landscape}:{metric}",
            )
            if (example is not None and landscape == example[0]
                    and (highlight_schemes is None or sampling in highlight_schemes)):
                selected = _example(frame, example).iloc[0]
                value = float(selected[metric])
                if np.isfinite(value):
                    example_location = float(
                        locations[models["run_id"].eq(selected["run_id"]).to_numpy()][0]
                    )
                    axis.scatter(
                        value, example_location, marker=marker, s=48, facecolors="none",
                        edgecolors="black", linewidths=0.9, zorder=5,
                        gid=f"example-point:{selected['run_id']}:{metric}",
                    )
                    axis.annotate(
                        "", xy=(value, example_location), xytext=(11, -9),
                        textcoords="offset points",
                        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
                        gid=f"example-arrow:{selected['run_id']}:{metric}",
                    )
    axis.set_yticks(range(len(landscapes)), [name.split("-", 1)[1] for name in landscapes])
    axis.invert_yaxis()


def figure_reported(
    tables: TableCollection,
    *,
    cohort_top_n: int = 100,
    examples: tuple[Example, ...],
) -> Figure:
    """Show percentile curves, their areas, and rank correlations as distinct summaries."""
    figure, axes = plt.subplots(
        3, 3, figsize=(MANUSCRIPT_FIGURE_WIDTH, 6.8), layout="constrained",
        gridspec_kw={"height_ratios": [1, 1.25, 1.25]},
    )
    for column, (family, specification) in enumerate(zip(FAMILY_ORDER, examples, strict=True)):
        frames = {}
        for sampling, scheme_tables in tables.items():
            label, color, marker = SCHEMES[sampling]
            selected = scheme_tables["rank_percentiles"]
            selected = selected[selected["cohort_top_n"].eq(cohort_top_n)]
            example = _example(selected, specification).sort_values(
                "external_rank_sample_percentile"
            )
            axes[0, column].step(
                np.r_[0, example["external_rank_sample_percentile"], 1],
                np.r_[0, example["ecdf"], 1], where="post", color=color, label=label,
            )
            frames[sampling] = selected[selected["family"].eq(family)].drop_duplicates("run_id")
        axes[0, column].plot([0, 1], [0, 1], "--", color="0.6", linewidth=0.8)
        axes[0, column].set(
            title=specification[2], xlabel="Reported-rank\nsample percentile",
            ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1),
        )
        _landscape_dots(axes[1, column], frames, "ecdf_to_uniform_area", example=specification)
        axes[1, column].set(
            xlabel="Area between curve\nand diagonal", xlim=(-0.02, 0.52),
        )
        _landscape_dots(
            axes[2, column], frames, "external_vs_posterior_median_rank_spearman",
            example=specification,
        )
        axes[2, column].axvline(0, color="0.6", linewidth=0.8)
        axes[2, column].set(
            xlabel="Reported vs. median\nsampled rank: Spearman",
            xlim=(-1.05, 1.05),
        )
    _scheme_legend(figure, hollow_markers=True)
    return _finish(figure)


def figure_ranks(
    tables: TableCollection,
    *,
    detail_top_n: int = 20,
    examples: tuple[Example, ...],
) -> Figure:
    """Pair individual rank intervals with the variation across fitted analyses."""
    figure, axes = plt.subplots(
        2, 3, figsize=(MANUSCRIPT_FIGURE_WIDTH, 6.5), layout="constrained"
    )
    parameter = tables["parameter_posterior"]["model_rank_intervals"]
    for column, (family, specification) in enumerate(zip(FAMILY_ORDER, examples, strict=True)):
        rows = _example(parameter, specification).nsmallest(detail_top_n, "model_reference_rank")
        axis = axes[0, column]
        axis.hlines(rows["model_reference_rank"], rows["posterior_rank_lower"],
                    rows["posterior_rank_upper"], color=MODEL_COLOR, alpha=0.65)
        axis.scatter(rows["posterior_median_rank"], rows["model_reference_rank"],
                     color=MODEL_COLOR, s=15, label="Median sampled rank")
        axis.scatter(rows["model_reference_rank"], rows["model_reference_rank"],
                     color="0.25", marker="|", s=30, label="Reference rank")
        axis.set(title=specification[2], xlabel="Sampled rank",
                 ylabel="Median-score reference rank")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.invert_yaxis()
        if column == 0:
            axis.legend(frameon=False)
        frames = {}
        for sampling, scheme_tables in tables.items():
            intervals = scheme_tables["model_rank_intervals"]
            intervals = intervals[intervals["family"].eq(family)]
            frames[sampling] = intervals.groupby(
                ["run_id", "family", "landscape", "model"], as_index=False,
            )["rank_interval_width"].median()
        _landscape_dots(
            axes[1, column], frames, "rank_interval_width", example=specification,
            highlight_schemes=("parameter_posterior",),
        )
        axes[1, column].axvline(100, color="0.6", linestyle="--", linewidth=0.8)
        axes[1, column].set(title=FAMILY_LABELS[family], xscale="log",
                            xlabel="Median rank-interval width\n(reference top 100)")
        axes[1, column].xaxis.set_minor_formatter(NullFormatter())
        if family == "nuc":
            axes[1, column].xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
            axes[1, column].xaxis.set_major_formatter("{x:g}")
    _scheme_legend(figure, hollow_markers=True)
    return _finish(figure)


def figure_selection(
    tables: TableCollection,
    *,
    examples: tuple[Example, ...],
) -> Figure:
    """Show boundary inclusion and agreement for batches of 10 to 1,000 variants."""
    figure, axes = plt.subplots(
        2, 3, figsize=(MANUSCRIPT_FIGURE_WIDTH, 6.8), layout="constrained"
    )
    parameter = tables["parameter_posterior"]["top_k_candidates"]
    for column, specification in enumerate(examples):
        rows = _example(parameter, specification)
        incumbents = rows[rows["is_model_reference_top_k"]].nlargest(10, "model_reference_rank")
        incumbents = incumbents.sort_values("model_reference_rank")
        challengers = rows[~rows["is_model_reference_top_k"]].nlargest(10, "top_k_probability")
        selected = pd.concat([incumbents, challengers])
        labels = [f"Ref. {int(rank)}" for rank in incumbents["model_reference_rank"]]
        labels += [f"Outside {position + 1}" for position in range(len(challengers))]
        colors = [MODEL_COLOR] * len(incumbents) + [CHALLENGER_COLOR] * len(challengers)
        axis = axes[0, column]
        axis.hlines(range(len(selected)), 0, selected["top_k_probability"], color="0.85")
        axis.scatter(selected["top_k_probability"], range(len(selected)), c=colors, s=20)
        axis.set(title=specification[2], yticks=range(len(selected)), yticklabels=labels,
                 xlabel="Top-100 inclusion", xlim=(-0.02, 1.02))
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        for sampling, scheme_tables in tables.items():
            label, color, marker = SCHEMES[sampling]
            curves = _example(scheme_tables["top_k_curves"], specification).sort_values("k")
            curves = curves[curves["k"].between(10, 1000)]
            axes[1, column].plot(curves["k"], curves["sample_self_consistency"],
                                 color=color, marker=marker, markersize=3, label=label)
        axes[1, column].axvline(100, color="0.6", linestyle="--", linewidth=0.8)
        axes[1, column].set(xscale="log", xlim=(10, 1000), ylim=(-0.02, 1.02),
                            xlabel="Batch size k",
                            ylabel="Mean shared fraction")
        axes[1, column].xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
        axes[1, column].xaxis.set_major_formatter("{x:g}")
        axes[1, column].xaxis.set_minor_formatter(NullFormatter())
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
    _scheme_legend(figure)
    return _finish(figure)


def explanation_rows(
    tables: TableCollection,
    *,
    examples: tuple[Example, ...],
) -> pd.DataFrame:
    """Keep every decision-cohort variant in three specified non-Gamma analyses."""
    if len(examples) != 3:
        raise ValueError("The explanation figure requires three distinct examples.")
    variants = tables["parameter_posterior"]["mechanism_variants"]
    selected = []
    for row_number, specification in enumerate(examples, start=1):
        if "gamma" in specification[1].lower().split("-"):
            raise ValueError("Gamma fits are excluded from the main explanation examples.")
        rows = _example(variants, specification)
        rows = rows[rows["is_primary_cohort"]].copy()
        if rows.empty or not rows["sampling"].eq("parameter_posterior").all():
            raise ValueError("Explanation examples require parameter-resampling decision cohorts.")
        rows["figure_row"] = row_number
        rows["example_label"] = specification[2]
        selected.append(rows)
    rows = pd.concat(selected, ignore_index=True)
    if rows["run_id"].nunique() != 3:
        raise ValueError("The explanation figure requires three distinct examples.")
    numeric = [
        "mechanism_input_count", "mechanism_output_count", "normalized_score_interval_width",
        "score_margin_to_top_k_boundary", "reference_score_iqr", "top_k_inclusion_probability",
    ]
    if not np.isfinite(rows[numeric]).all().all():
        raise ValueError("Explanation examples require finite coordinates for every cohort point.")
    nonnegative = numeric[:3]
    if (rows[nonnegative] < 0).any().any() or not rows["reference_score_iqr"].gt(0).all():
        raise ValueError("Explanation counts and widths must be nonnegative, with positive IQRs.")
    if not rows["top_k_inclusion_probability"].between(0, 1).all():
        raise ValueError("Explanation inclusion frequencies must lie between zero and one.")
    if not (rows["is_model_top_k"] ^ rows["is_challenger"]).all():
        raise ValueError("Explanation points must be reference members or outside challengers.")
    rows = _decision_rows({"mechanism_variants": rows}, family=None)
    rows["input_log_count"] = np.log10(rows["mechanism_input_count"] + 1)
    rows["output_log_count"] = np.log10(rows["mechanism_output_count"] + 1)
    return rows


def figure_explanation(
    tables: TableCollection,
    *,
    examples: tuple[Example, ...],
) -> Figure:
    """Show Figure 10's four point-level relationships for three non-Gamma examples."""
    rows = explanation_rows(tables, examples=examples)
    scheme_tables = {**tables["parameter_posterior"], "mechanism_variants": rows}
    return figure_summary_why_by_analysis(
        scheme_tables, run_ids=tuple(rows["run_id"].drop_duplicates())
    )


def export_explanation_tables(
    tables: TableCollection,
    output_dir: Path,
    *,
    examples: tuple[Example, ...],
) -> dict[str, object]:
    """Export every plotted variant and a run index, without aggregating points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = explanation_rows(tables, examples=examples)
    rows.to_csv(output_dir / "explanation_variants.csv", index=False)
    annotations = tables["parameter_posterior"]["mechanism_runs"].set_index("run_id")
    floor = _positive_floor(rows["normalized_score_interval_width"])
    index = []
    for run_id, group in rows.groupby("run_id", sort=False):
        first = group.iloc[0]
        annotation = annotations.loc[run_id]
        index.append({
            "figure_row": int(first["figure_row"]), "run_id": run_id,
            "family": first["family"], "landscape": first["landscape"],
            "model": first["model"], "n_samples": int(annotation["n_samples"]),
            "cohort_size": len(group), "reference_members": int(group["is_model_top_k"].sum()),
            "challengers": int(group["is_challenger"].sum()),
            "sample_self_consistency": float(annotation["sample_self_consistency"]),
            "zero_width_count": int(group["normalized_score_interval_width"].eq(0).sum()),
            "plot_width_floor": floor,
        })
    pd.DataFrame(index).to_csv(output_dir / "explanation_runs.csv", index=False)
    return {
        "sampling": "parameter_posterior", "examples": examples,
        "run_ids": rows["run_id"].drop_duplicates().tolist(), "variant_rows": len(rows),
        "excluded_model_families": ["gamma"], "aggregation": "none",
        "cohort": "reference top 100 plus outside candidates with inclusion at least 0.01",
        "axis_limits": "counts and widths shared across examples; margins shared within each row",
        "plot_width_floor": floor,
    }


def figure_count_support(tables: TableCollection) -> Figure:
    """Show within-run correlations without pooling variants across landscapes."""
    figure, axes = plt.subplots(
        1, 2, figsize=(MANUSCRIPT_FIGURE_WIDTH, 6.4), layout="constrained"
    )
    for axis, column, label in zip(
        axes,
        ("input_support_vs_score_uncertainty_spearman",
         "output_abundance_vs_score_uncertainty_spearman"),
        ("Input count vs. score width", "Output count vs. score width"),
        strict=True,
    ):
        frames = {
            sampling: scheme_tables["mechanism_diagnostics"].loc[
                scheme_tables["mechanism_diagnostics"]["stratum"].eq("primary")
            ]
            for sampling, scheme_tables in tables.items()
        }
        _landscape_dots(axis, frames, column)
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set(xlim=(-1.03, 0.1), xlabel="Within-run Spearman correlation", title=label)
    _scheme_legend(figure, hollow_markers=True)
    return _finish(figure)


def reference_comparison_rows(
    tables: TableCollection,
    *,
    examples: tuple[Example, ...],
    decision_k: int = 100,
) -> pd.DataFrame:
    """Separate fixed-reference intersections from mean resampled overlaps."""
    rows = []
    for specification in examples:
        for sampling, scheme_tables in tables.items():
            diagnostics = _example(scheme_tables["reference_diagnostics"], specification)
            curves = _example(scheme_tables["top_k_curves"], specification)
            selected = curves[curves["k"].eq(decision_k)]
            if len(selected) != 1 or not diagnostics["top_k"].eq(decision_k).all():
                raise ValueError("Reference comparison requires matching decision sizes.")
            row = selected.iloc[0]
            rows.append({
                "example": specification[2], "sampling": sampling, "k": decision_k,
                "fixed_shared_count": int(diagnostics["model_external_top_k_intersection"].iloc[0]),
                "sample_shared_count": decision_k * row["sample_self_consistency"],
                "model_shared_count": decision_k * row["sample_vs_model_reference"],
                "reported_shared_count": decision_k * row["sample_vs_external_reference"],
            })
    return pd.DataFrame(rows)


def rank_displacement_rows(
    tables: TableCollection, *, cohort_top_n: int = 100,
) -> pd.DataFrame:
    """Average absolute reported-to-median sampled rank differences within each run."""
    identity = ["family", "landscape", "model"]
    frames = []
    for sampling, collection in tables.items():
        runs = collection["runs"][["run_id", *identity]]
        if runs["run_id"].duplicated().any() or runs.duplicated(identity).any():
            raise ValueError("Displacement summaries require unique run and model identities.")
        source = collection["rank_percentiles"]
        source = source.loc[source["cohort_top_n"].eq(cohort_top_n), [
            "run_id", "variant_index", "reported_rank", "posterior_median_sampled_rank",
        ]].copy()
        if source.duplicated(["run_id", "variant_index"]).any():
            raise ValueError("Displacement summaries require one row per candidate and cohort.")
        if set(source["run_id"]) != set(runs["run_id"]):
            raise ValueError("Every declared run requires the requested reported-rank cohort.")
        finite = np.isfinite(source[["reported_rank", "posterior_median_sampled_rank"]]).all(axis=1)
        source["absolute_displacement"] = (
            source["posterior_median_sampled_rank"] - source["reported_rank"]
        ).abs().where(finite)
        summary = source.groupby("run_id", as_index=False).agg(
            mean_absolute_rank_displacement=("absolute_displacement", "mean"),
            supported_candidates=("absolute_displacement", "count"),
        )
        summary = runs.merge(summary, on="run_id", validate="one_to_one")
        summary["sampling"] = sampling
        summary["cohort_top_n"] = cohort_top_n
        frames.append(summary)
    if not frames:
        raise ValueError("No analyses are available for rank displacement.")
    return pd.concat(frames, ignore_index=True)


def export_rank_displacement(
    tables: TableCollection, output_dir: Path, *, cohort_top_n: int = 100,
) -> None:
    """Tabulate every landscape/model across the three sampling schemes without pooling."""
    rows = rank_displacement_rows(tables, cohort_top_n=cohort_top_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "rank_displacement.csv", index=False)
    wide = rows.pivot(
        index=["family", "landscape", "model"], columns="sampling",
        values="mean_absolute_rank_displacement",
    ).reindex(columns=list(SCHEMES))
    order = sorted(wide.index, key=lambda key: (FAMILY_ORDER.index(key[0]), key[1], key[2]))
    display = wide.reindex(order).reset_index()
    display["landscape"] = [
        f"{FAMILY_LABELS[row.family]} {row.landscape.removeprefix(row.family + '-')}"
        for row in display.itertuples()
    ]
    display["model"] = display["model"].map(lambda model: CURVE_MODELS[model][0])
    labels = {
        "parameter_posterior": "Parameter posterior", "mle_predictive": "Optimized-fit counts",
        "posterior_predictive": "Posterior counts",
    }
    display = display.drop(columns="family").rename(columns={
        "landscape": "Landscape", "model": "Model", **labels,
    })
    tex = display.style.hide(axis="index").format(
        {label: "{:,.1f}" for label in labels.values()}, na_rep="--", escape="latex",
    ).to_latex(
        environment="longtable", position="c", hrules=True, column_format="llrrr",
        caption=(
            f"Mean absolute rank displacement for the reported top {cohort_top_n:,}. "
            "Each row is one landscape/model specification. Values average absolute "
            "differences between reported and median sampled ranks over supported candidates, "
            "in rank positions against the full competitor pool. "
            "E, L, and G denote Exponential, Lomax, and Gamma priors; a second E or S "
            "denotes exponential or sigmoid growth. Dashes indicate no available value. "
            "Rounding is for display only."
        ), label="tab:rank-displacement",
    )
    (output_dir / "rank_displacement.tex").write_text(
        "\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}\n" + tex + "\\endgroup\n"
    )


def export_evidence_tables(
    tables: TableCollection, output_dir: Path, *, examples: tuple[Example, ...],
) -> None:
    comparison = reference_comparison_rows(tables, examples=examples)
    comparison.to_csv(output_dir / "reference_comparisons.csv", index=False)
    display = comparison.drop(columns="k").copy()
    display["sampling"] = display["sampling"].map({
        "parameter_posterior": "Parameter", "mle_predictive": "Optimized counts",
        "posterior_predictive": "Posterior counts",
    })
    display.columns = ["Example", "Scheme", "Fixed", "Sample pair", "Reference", "Reported"]
    tex = display.style.hide(axis="index").format(precision=2, escape="latex").to_latex(
        position="t", position_float="centering", hrules=True,
        caption=("Distinct top-100 comparisons for the three running examples. "
             "Fixed is the intersection of the median-score reference "
                 "and reported sets. Sample pair, Reference, and Reported are mean "
                 "numbers shared by a sampled set with another sampled set, the "
                 "median-score set, and the reported set, respectively. All lists "
                 "contain 100 variants, so counts also equal percentages."),
        label="tab:reference-comparisons",
    )
    (output_dir / "reference_comparisons.tex").write_text(
        "\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}\n" + tex + "\\endgroup\n"
    )
    summaries = []
    for sampling, scheme_tables in tables.items():
        curves = scheme_tables["top_k_curves"]
        summary = curves[curves["k"].isin([100, 1000])].groupby(
            ["family", "landscape", "k"], as_index=False,
        )["sample_self_consistency"].median()
        summary["sampling"] = sampling
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_dir / "landscape_set_sizes.csv", index=False)
    wide = combined.pivot(index="landscape", columns=["sampling", "k"],
                          values="sample_self_consistency")
    columns = pd.MultiIndex.from_product([list(SCHEMES), [100, 1000]])
    wide = (100 * wide.reindex(columns=columns)).round(1)
    wide.columns = ["P100", "P1000", "O100", "O1000", "PC100", "PC1000"]
    wide = wide.reset_index().rename(columns={"landscape": "Landscape"})
    tex = wide.style.hide(axis="index").format(precision=1, na_rep="--", escape="latex").to_latex(
        position="p", position_float="centering", hrules=True,
        caption=("Mean shared top-k fractions (percent), summarized by the median over "
                 "model specifications within each landscape. P, O, and PC denote "
                 "parameter, optimized-fit count, and posterior-count resampling; "
                 "the suffix denotes k. Missing entries indicate no analyzed posterior "
                 "fit for that landscape. This is an endpoint comparison, not a claim "
                 "of monotonicity over every intermediate k."),
        label="tab:landscape-set-sizes",
    )
    (output_dir / "landscape_set_sizes.tex").write_text(tex)


def summary_why_pages(
    tables: TableCollection, *, runs_per_page: int = 3,
) -> list[dict[str, object]]:
    """Partition every declared run into family- and scheme-specific pages."""
    if runs_per_page < 1:
        raise ValueError("runs_per_page must be positive.")
    pages = []
    for sampling, scheme_tables in tables.items():
        runs = scheme_tables["runs"]
        if runs["run_id"].duplicated().any():
            raise ValueError("Summary pages require unique run IDs within each scheme.")
        if not set(runs["family"]).issubset(FAMILY_ORDER):
            raise ValueError("Summary pages require a known protein family.")
        for family in FAMILY_ORDER:
            run_ids = runs.loc[runs["family"].eq(family), "run_id"].tolist()
            for offset in range(0, len(run_ids), runs_per_page):
                page_number = offset // runs_per_page + 1
                pages.append({
                    "sampling": sampling, "family": family, "page": page_number,
                    "run_ids": run_ids[offset:offset + runs_per_page],
                    "stem": f"summary_why_{sampling}_{family}_{page_number:02d}",
                    "label": f"fig:why-{sampling}-{family}-{page_number}",
                })
    return pages


def _save_checked(figure: Figure, output_dir: Path, stem: str) -> None:
    try:
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        if np.count_nonzero(np.any(pixels < 245, axis=2)) < 1000:
            raise ValueError(f"Figure is blank: {stem}.")
        save_figure(figure, output_dir, stem)
    finally:
        plt.close(figure)


def export_summary_why(tables: TableCollection, output_dir: Path) -> list[dict[str, object]]:
    """Export all Figure 10 pages, a complete run index, and their LaTeX captions."""
    pages = summary_why_pages(tables)
    if not pages:
        raise ValueError("No runs are available for supplementary summaries.")
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    index_rows = []
    scheme_names = {
        "parameter_posterior": "Parameter resampling",
        "mle_predictive": "Optimized-fit count resampling",
        "posterior_predictive": "Posterior count resampling",
    }
    for page in pages:
        sampling = str(page["sampling"])
        family = str(page["family"])
        run_ids = tuple(page["run_ids"])
        scheme_tables = tables[sampling]
        figure = figure_summary_why_by_analysis(scheme_tables, family=family, run_ids=run_ids)
        _save_checked(figure, output_dir, str(page["stem"]))
        runs = scheme_tables["runs"].set_index("run_id")
        variants = scheme_tables["mechanism_variants"]
        family_rows = variants[variants["family"].eq(family) & variants["is_primary_cohort"]]
        positive_widths = family_rows.loc[
            family_rows["normalized_score_interval_width"] > 0, "normalized_score_interval_width"
        ]
        floor = float(positive_widths.min()) / 2
        titles = []
        for row_number, run_id in enumerate(run_ids, start=1):
            run = runs.loc[run_id]
            cohort = family_rows[family_rows["run_id"].eq(run_id)]
            title = f"{run['landscape']}, {str(run['model']).replace('-', '--').title()}"
            titles.append(title)
            index_rows.append({
                **page, "run_id": run_id, "row": row_number,
                "landscape": run["landscape"], "model": run["model"],
                "n_samples": int(run["n_samples"]), "cohort_size": len(cohort),
                "reference_members": int(cohort["is_model_top_k"].sum()),
                "challengers": int(cohort["is_challenger"].sum()),
                "zero_width_count": int(cohort["normalized_score_interval_width"].eq(0).sum()),
                "plot_width_floor": floor,
            })
        caption = (
            f"{FAMILY_LABELS[family]} decision-cohort summaries: {scheme_names[sampling].lower()}. "
            "Rows, top to bottom: " + "; ".join(titles) + ". "
            "Columns show normalized central 90\\% score width versus input count, "
            "output count, and median-score margin from the top-100 cutoff; then "
            "top-100 inclusion versus that margin. Widths and margins use the run's "
            "top-1,000 median-score interquartile range (IQR). Blue filled points are "
            "reference top-100 members; orange open points are outside candidates with "
            "inclusion probability at least 0.01. Count and score-width limits are shared "
            "within family and scheme; margin limits are shared within a row. "
            "Dotted lines mark the cutoff. No regression lines are fitted."
        )
        captions.extend([
            r"\begin{figure}[p]", r"\centering",
            r"\includegraphics[width=\linewidth]{"
            + str(page["stem"]) + ".pdf}",
            r"\caption{" + caption + "}", r"\label{" + str(page["label"]) + "}",
            r"\end{figure}", r"\clearpage", "",
        ])
        print(f"Exported {page['stem']} ({len(run_ids)} runs)", flush=True)
    index = pd.DataFrame(index_rows).drop(columns="run_ids")
    if index["run_id"].duplicated().any():
        raise ValueError("An analysis appears on more than one supplementary page.")
    index.to_csv(output_dir / "summary_why_index.csv", index=False)
    table = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Index of per-analysis count-support and selection summaries. "
        + f"Every one of the {len(index)} analysis runs appears on exactly one page.}}",
        r"\label{tab:why-index}", r"\begin{tabular}{llrr}", r"\toprule",
        r"Scheme & Family & Runs & Figures \\", r"\midrule",
    ]
    for sampling in tables:
        for family in FAMILY_ORDER:
            group = [page for page in pages if page["sampling"] == sampling
                     and page["family"] == family]
            if not group:
                continue
            figure_range = r"\ref{" + str(group[0]["label"]) + "}"
            if len(group) > 1:
                figure_range += r"--\ref{" + str(group[-1]["label"]) + "}"
            count = sum(len(page["run_ids"]) for page in group)
            table.append(
                f"{scheme_names[sampling]} & {FAMILY_LABELS[family]} & {count} & "
                + figure_range + r" \\"
            )
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\clearpage", ""])
    (output_dir / "summary_why.tex").write_text("\n".join([*table, *captions]))
    return pages


def figure_reported_curves(
    tables: Mapping[str, pd.DataFrame], *, run_ids: tuple[str, ...],
    cohorts: tuple[int, ...] = (100, 1000), example_ids: tuple[str, ...] = (),
    landscape_colors: Mapping[str, object] | None = None,
) -> Figure:
    """Overlay complete curves within one protein system and sampling scheme."""
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Curve pages require distinct, nonempty run IDs.")
    runs = tables["runs"].set_index("run_id")
    selected = runs.loc[list(run_ids)]
    if selected["family"].nunique() != 1 or selected["sampling"].nunique() != 1:
        raise ValueError("Grouped curves require one protein system and sampling scheme.")
    family = str(selected["family"].iloc[0])
    sampling = str(selected["sampling"].iloc[0])
    landscapes = selected["landscape"].drop_duplicates().tolist()
    if landscape_colors is None:
        palette = plt.get_cmap("tab10").colors
        landscape_colors = {name: palette[position % len(palette)]
                            for position, name in enumerate(landscapes)}
    figure, axes = plt.subplots(
        1, len(cohorts), squeeze=False, figsize=(MANUSCRIPT_FIGURE_WIDTH, 4.8),
        layout="constrained",
    )
    figure.suptitle(
        f"{FAMILY_LABELS[family]} | {SCHEMES[sampling][0]} | {len(run_ids)} analyses"
    )
    for run_id in run_ids:
        run = runs.loc[run_id]
        color = landscape_colors[str(run["landscape"])]
        _, linestyle = CURVE_MODELS[str(run["model"])]
        is_example = run_id in example_ids
        for axis, cohort in zip(axes[0], cohorts, strict=True):
            data = tables["rank_percentiles"]
            data = data[data["run_id"].eq(run_id) & data["cohort_top_n"].eq(cohort)].sort_values(
                "external_rank_sample_percentile", kind="stable"
            )
            if data.empty:
                raise ValueError(f"Missing curve for {run_id}, cohort {cohort}.")
            area = float(data["ecdf_to_uniform_area"].iloc[0])
            values = data["external_rank_sample_percentile"].to_numpy()
            if not np.isclose(ecdf_to_uniform_area(values), area, atol=1e-12, rtol=0):
                raise ValueError("Stored curve area does not match its complete percentile values.")
            axis.step(
                np.r_[0, values, 1], np.r_[0, data["ecdf"], 1], where="post",
                color=color, linestyle=linestyle, linewidth=1.8 if is_example else 0.8,
                alpha=1.0 if is_example else 0.75, zorder=3 if is_example else 2,
                gid=f"reported-curve:{run_id}:{cohort}",
            )
    for column, (axis, cohort) in enumerate(zip(axes[0], cohorts, strict=True)):
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.65", linewidth=0.7, zorder=1)
        axis.set(
            xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
            xlabel="Reported-rank sample percentile", ylabel="Cumulative fraction",
            title=f"Reported top {cohort:,}", aspect="equal",
        )
        _panel_label(axis, chr(97 + column))
        axis.grid(alpha=0.55)
        axis.set_axisbelow(True)
    handles = [Line2D([], [], color=landscape_colors[name], linewidth=1.5,
                      label=name.removeprefix(f"{family}-")) for name in landscapes]
    handles.extend(
        Line2D([], [], color="0.2", linestyle=CURVE_MODELS[model][1],
               label=CURVE_MODELS[model][0]) for model in selected["model"].drop_duplicates()
    )
    if any(run_id in example_ids for run_id in run_ids):
        handles.append(Line2D([], [], color="0.2", linewidth=1.8, label="Main example"))
    figure.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)),
                  frameon=False, handlelength=3)
    apply_figure_typography(figure)
    return figure


def export_curve_statistics(
    index: pd.DataFrame, output_dir: Path, cohorts: tuple[int, ...],
) -> None:
    """Write full-precision run summaries and a compact, automatically paginated table."""
    columns = ["analysis_id", "sampling", "family", "landscape", "model", "n_samples"]
    statistics = index.drop_duplicates("run_id").set_index("run_id")[columns].copy()
    statistics["landscape_label"] = [
        f"{FAMILY_LABELS[row.family]} {row.landscape.removeprefix(row.family + '-')}"
        for row in statistics.itertuples()
    ]
    statistics["model_label"] = statistics["model"].map(lambda model: CURVE_MODELS[model][0])
    headers = {"analysis_id": "ID", "landscape_label": "Landscape", "model_label": "Model",
               "n_samples": "Samples"}
    formats = {"Samples": "{:,.0f}"}
    for cohort in cohorts:
        group = index[index["cohort_top_n"].eq(cohort)].set_index("run_id")
        for source, prefix, display in (
            ("supported_candidates", "n", "n"), ("area", "area", "A"),
            ("spearman", "spearman", r"\rho"),
        ):
            key = f"{prefix}_{cohort}"
            header = f"${display}_{{{cohort}}}$"
            statistics[key] = group[source]
            headers[key] = header
            formats[header] = "{:,.0f}" if prefix == "n" else "{:.3f}"
    statistics.reset_index().to_csv(output_dir / "reported_curve_statistics.csv", index=False)
    display = statistics[list(headers)].rename(columns=headers)
    tex = display.style.hide(axis="index").format(
        formats, na_rep="--", escape="latex",
    ).to_latex(
        environment="longtable", position="c", hrules=True,
        column_format="lll" + "r" * (1 + 3 * len(cohorts)),
        caption=(
            "Reported-rank summaries for every analysis. For each cohort, $n$ is the "
            "supported count, $A$ the area between curve and diagonal, and $\\rho$ the "
            "Spearman correlation. IDs and model codes are defined in "
            "Section~\\ref{app:all-reported-curves}; sample counts are retained draws. "
            "Statistics are rounded to three decimals; -- denotes an undefined value."
        ), label="tab:reported-curve-statistics",
    )
    (output_dir / "reported_curve_statistics.tex").write_text(
        "\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}\n" + tex + "\\endgroup\n"
    )


def export_reported_curves(
    tables: TableCollection, output_dir: Path, *, examples: tuple[Example, ...],
    cohorts: tuple[int, ...] = (100, 1000),
) -> list[dict[str, object]]:
    """Export nine system/scheme overlays and tabulate every individual analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_per_page = max(len(collection["runs"]) for collection in tables.values())
    pages = [{**page, "stem": str(page["stem"]).replace("summary_why", "reported_curves"),
              "label": str(page["label"]).replace("fig:why-", "fig:curves-")}
             for page in summary_why_pages(tables, runs_per_page=runs_per_page)]
    all_runs = pd.concat([collection["runs"] for collection in tables.values()], ignore_index=True)
    palette = plt.get_cmap("tab10").colors
    landscape_colors = {}
    for family in FAMILY_ORDER:
        landscapes = all_runs.loc[all_runs["family"].eq(family), "landscape"].drop_duplicates()
        landscape_colors.update({name: palette[position % len(palette)]
                                 for position, name in enumerate(landscapes)})
    prefixes = dict(zip(SCHEMES, ("P", "O", "C"), strict=True))
    analysis_ids = {str(run_id): f"{prefixes[sampling]}{number:02d}"
                    for sampling, collection in tables.items()
                    for number, run_id in enumerate(collection["runs"]["run_id"], start=1)}
    index = []
    tex = []
    for page in pages:
        sampling = str(page["sampling"])
        collection = tables[sampling]
        run_ids = tuple(page["run_ids"])
        runs = collection["runs"].set_index("run_id")
        example_ids = tuple(
            str(row.run_id) for row in collection["runs"].itertuples()
            if any((row.landscape, row.model) == example[:2] for example in examples)
        )
        figure = figure_reported_curves(
            collection, run_ids=run_ids, cohorts=cohorts, example_ids=example_ids,
            landscape_colors=landscape_colors,
        )
        _save_checked(figure, output_dir, str(page["stem"]))
        for curve_number, run_id in enumerate(run_ids, start=1):
            run = runs.loc[run_id]
            for column, cohort in enumerate(cohorts):
                data = collection["rank_percentiles"]
                data = data[data["run_id"].eq(run_id) & data["cohort_top_n"].eq(cohort)]
                index.append({
                    "run_id": run_id, "analysis_id": analysis_ids[run_id],
                    "sampling": sampling, "family": run["family"],
                    "landscape": run["landscape"], "model": run["model"],
                    "stem": page["stem"], "label": page["label"], "curve": curve_number,
                    "panel": chr(97 + column),
                    "cohort_top_n": cohort, "supported_candidates": len(data),
                    "n_samples": int(run["n_samples"]), "main_example": run_id in example_ids,
                    "area": data["ecdf_to_uniform_area"].iloc[0],
                    "spearman": data["external_vs_posterior_median_rank_spearman"].iloc[0],
                })
        caption = (
            f"{FAMILY_LABELS[str(page['family'])]} reported-rank percentile curves: "
            f"{SCHEMES[sampling][0].lower()}. Each of the {len(run_ids)} lines per panel "
            "is one landscape/model analysis; panels use the reported top "
            + " and top ".join(map(str, cohorts)) + ". "
            "Colors identify landscapes and line styles identify models. "
            "E, L, and G denote Exponential, Lomax, and Gamma priors; a second E or S "
            "denotes exponential or sigmoid growth. A thicker line marks the main example. "
            "All percentiles are retained; curves are overlaid, not averaged or pooled. "
            "Table~\\ref{tab:reported-curve-statistics} gives each analysis's area, "
            "correlation, and counts. The gray diagonal is descriptive, not a calibration target."
        )
        tex.extend([
            r"\begin{figure}[p]", r"\centering",
            r"\includegraphics[width=\linewidth]{" + str(page["stem"]) + ".pdf}",
            r"\caption{" + caption + "}", r"\label{" + str(page["label"]) + "}",
            r"\end{figure}", r"\clearpage", "",
        ])
        print(f"Exported {page['stem']} ({len(run_ids)} analyses)", flush=True)
    index_frame = pd.DataFrame(index)
    if index_frame.duplicated(["run_id", "cohort_top_n"]).any():
        raise ValueError("A run/cohort appears twice in the curve supplement.")
    expected_count = sum(len(group["runs"]) for group in tables.values()) * len(cohorts)
    if len(index_frame) != expected_count:
        raise ValueError("The curve supplement does not cover every declared run and cohort.")
    index_frame.to_csv(output_dir / "reported_curves_index.csv", index=False)
    export_curve_statistics(index_frame, output_dir, cohorts)
    overview = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Grouped reported-rank percentile curves. Each figure overlays one "
        + "protein system and sampling scheme, with reported cohorts in separate panels.}",
        r"\label{tab:reported-curves-index}", r"\begin{tabular}{llrr}", r"\toprule",
        r"Scheme & Family & Analyses & Figures \\", r"\midrule",
    ]
    for sampling in tables:
        for family in FAMILY_ORDER:
            group = [page for page in pages if page["sampling"] == sampling
                     and page["family"] == family]
            if group:
                references = r"\ref{" + str(group[0]["label"]) + "}"
                if len(group) > 1:
                    references += r"--\ref{" + str(group[-1]["label"]) + "}"
                count = sum(len(page["run_ids"]) for page in group)
                overview.append(f"{SCHEMES[sampling][0]} & {FAMILY_LABELS[family]} & {count} & "
                                + references + r" \\")
    overview.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\clearpage", ""])
    (output_dir / "reported_curves.tex").write_text("\n".join([*overview, *tex]))
    return pages


def export_figures(table_root: Path, output_dir: Path, config: Path) -> None:
    """Validate the paper inventory and export figures with source checksums."""
    declarations = json.loads(config.read_text())["runs"]
    tables = {}
    hashes = {}
    for sampling in SCHEMES:
        scheme_tables = {}
        for name in TABLE_NAMES:
            path = table_root / sampling / f"{name}.csv"
            scheme_tables[name] = pd.read_csv(path, low_memory=False)
            with path.open("rb") as handle:
                hashes[f"{sampling}/{name}.csv"] = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = Counter(
            (run["run_id"], run["family"], run["landscape"], run["model"])
            for run in declarations if run["sampling"] == sampling
        )
        observed = Counter(scheme_tables["runs"][[
            "run_id", "family", "landscape", "model",
        ]].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"Table inventory does not match the paper manifest: {sampling}.")
        tables[sampling] = scheme_tables
    set_paper_style()
    fits_path = Path("paper/generated/fit_inventory.csv")
    examples, audit, candidates = representative_examples(
        tables, pd.read_csv(fits_path), examples=RUNNING_EXAMPLES
    )
    for frame in (audit, candidates):
        frame["selection_reason"] = (
            frame["landscape"].map(EXAMPLE_ROLES).where(frame["selected"], "")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "example_selection_runs.csv", index=False)
    candidates.to_csv(output_dir / "example_selection_candidates.csv", index=False)
    builders = (
        ("reported_order", lambda: figure_reported(tables, examples=examples)),
        ("rank_uncertainty", lambda: figure_ranks(tables, examples=examples)),
        ("selection_stability", lambda: figure_selection(tables, examples=examples)),
        ("explaining_resolution", lambda: figure_explanation(tables, examples=examples)),
        ("reported_order_top1000", lambda: figure_reported(
            tables, cohort_top_n=1000, examples=examples)),
        ("count_support", lambda: figure_count_support(tables)),
    )
    for stem, build in builders:
        _save_checked(build(), output_dir, stem)
        print(f"Exported {stem}.pdf and {stem}.png")
    export_evidence_tables(tables, output_dir, examples=examples)
    export_rank_displacement(tables, output_dir)
    explanation = export_explanation_tables(tables, output_dir, examples=examples)
    why_pages = export_summary_why(tables, output_dir)
    curve_pages = export_reported_curves(tables, output_dir, examples=examples)
    provenance = {
        "table_root": str(table_root.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "version": "2.1", "examples": examples,
        "example_selection": {
            "metrics": ["area", "spearman", "rank_width", "shared_fraction"],
            "criterion": "explicit illustrative examples; centrality summaries are an audit",
            "roles": EXAMPLE_ROLES,
            "reference": "non-Gamma model medians within landscape, separately by family/scheme",
            "diagnostics": "omitted summaries do not exclude fits; recorded warnings are retained",
            "fit_inventory_sha256": hashlib.sha256(fits_path.read_bytes()).hexdigest(),
        },
        "table_sha256": hashes,
        "figures": [stem for stem, _ in builders],
        "evidence_tables": [
            "reference_comparisons", "landscape_set_sizes", "explanation_variants",
            "explanation_runs", "example_selection_runs", "example_selection_candidates",
            "reported_curves_index", "reported_curve_statistics", "rank_displacement",
        ],
        "explanation": explanation,
        "supplementary_why_pages": why_pages,
        "supplementary_curve_pages": curve_pages,
        "supplementary_curve_layout": "system/scheme overlays; separate cohorts; no aggregation",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=Path("analysis/tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--config", type=Path, default=Path("analysis/runs.json"))
    options = parser.parse_args()
    export_figures(options.table_root, options.output_dir, options.config)


if __name__ == "__main__":
    main()
