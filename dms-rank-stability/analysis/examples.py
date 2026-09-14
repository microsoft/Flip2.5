"""Auditable selection of non-extreme running examples from prepared summaries."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from analysis.figures import FAMILY_LABELS, FAMILY_ORDER

Example = tuple[str, str, str]
TableCollection = Mapping[str, Mapping[str, pd.DataFrame]]
METRICS = ("area", "spearman", "rank_width", "shared_fraction")
IDENTITY = ["family", "landscape", "model"]


def run_summaries(tables: TableCollection, *, cohort_top_n: int = 100) -> pd.DataFrame:
    """Join one area/correlation and reference-top-k width/overlap per run."""
    frames = []
    for sampling, collection in tables.items():
        percentiles = collection["rank_percentiles"]
        percentiles = percentiles[percentiles["cohort_top_n"].eq(cohort_top_n)]
        source_metrics = [
            "ecdf_to_uniform_area", "external_vs_posterior_median_rank_spearman"
        ]
        if percentiles.groupby("run_id")[source_metrics].nunique(dropna=False).gt(1).any().any():
            raise ValueError("Area and correlation must be constant within a run and cohort.")
        summary = percentiles.drop_duplicates("run_id")[["run_id", *source_metrics]].rename(
            columns=dict(zip(source_metrics, METRICS[:2], strict=True))
        )
        summary = collection["runs"].merge(summary, on="run_id", validate="one_to_one")
        if len(summary) != len(collection["runs"]):
            raise ValueError("Every declared run requires a reported-rank summary.")
        widths = collection["model_rank_intervals"].groupby("run_id")[
            "rank_interval_width"
        ].median()
        summary["rank_width"] = summary["run_id"].map(widths)
        curves = collection["top_k_curves"]
        curves = curves[curves["k"].eq(100)].set_index("run_id")
        if curves.index.duplicated().any():
            raise ValueError("Each run requires one top-100 overlap.")
        summary["shared_fraction"] = summary["run_id"].map(curves["sample_self_consistency"])
        summary["sampling"] = sampling
        summary["cohort_top_n"] = cohort_top_n
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def representative_examples(
    tables: TableCollection, fits: pd.DataFrame, *, examples: tuple[Example, ...] | None = None,
) -> tuple[tuple[Example, ...], pd.DataFrame, pd.DataFrame]:
    """Audit explicit examples, or choose central fits without penalizing omitted diagnostics."""
    requested = {(landscape, model): label for landscape, model, label in examples or ()}
    if examples is not None and (len(examples) != len(FAMILY_ORDER)
                                 or len(requested) != len(FAMILY_ORDER)):
        raise ValueError("Specify three distinct examples, one per protein system.")
    audit = run_summaries(tables)
    if audit.duplicated([*IDENTITY, "sampling"]).any():
        raise ValueError("Example identities must be unique within each sampling scheme.")
    audit["non_gamma"] = ~audit["model"].str.contains("gamma", case=False)
    position_columns = [f"{metric}_percentile" for metric in METRICS]
    audit[position_columns] = np.nan
    for (_, _), group in audit[audit["non_gamma"]].groupby(["family", "sampling"], sort=True):
        landscape_medians = group.groupby("landscape")[list(METRICS)].median()
        for metric in METRICS:
            reference = landscape_medians[metric].dropna().to_numpy()
            values = group[metric].to_numpy()[:, None]
            positions = ((values > reference).sum(axis=1)
                         + 0.5 * (values == reference).sum(axis=1)) / len(reference)
            positions[~np.isfinite(group[metric])] = np.nan
            audit.loc[group.index, f"{metric}_percentile"] = positions
    distances = (audit[position_columns] - 0.5).abs()
    audit["center_distance"] = distances.mean(axis=1)
    audit["largest_distance"] = distances.max(axis=1)
    audit["finite_metrics"] = np.isfinite(audit[list(METRICS)]).all(axis=1)
    candidates = audit.groupby(IDENTITY, as_index=False).agg(
        n_schemes=("sampling", "nunique"), non_gamma=("non_gamma", "all"),
        finite_metrics=("finite_metrics", "all"), center_distance=("center_distance", "mean"),
        largest_distance=("largest_distance", "max"),
    )
    mean_positions = audit.groupby(IDENTITY, as_index=False)[position_columns].mean()
    candidates = candidates.merge(mean_positions, on=IDENTITY, validate="one_to_one")
    candidates["max_mean_distance"] = (candidates[position_columns] - 0.5).abs().max(axis=1)
    diagnostic_columns = [
        "native_score_diagnostics_available", "divergences", "tree_depth_hits",
        "r_hat_nonfinite", "r_hat_extreme", "ess_bulk_nonfinite", "ess_tail_nonfinite",
    ]
    posterior = fits.loc[fits["kind"].eq("hmc"), [*IDENTITY, *diagnostic_columns]]
    candidates = candidates.merge(posterior, on=IDENTITY, how="left", validate="one_to_one")
    candidates["diagnostics_eligible"] = ~(
        candidates[["divergences", "tree_depth_hits", "r_hat_nonfinite",
                    "ess_bulk_nonfinite", "ess_tail_nonfinite"]].gt(0).any(axis=1)
        | candidates["r_hat_extreme"].ge(1.01)
    )
    candidates["eligible"] = (
        candidates["non_gamma"] & candidates["finite_metrics"]
        & candidates["n_schemes"].eq(len(tables)) & candidates["diagnostics_eligible"]
    )
    candidates["selected"] = False
    candidates["shortlisted"] = False
    selected = []
    for family in FAMILY_ORDER:
        eligible = candidates[candidates["family"].eq(family) & candidates["eligible"]]
        eligible = eligible.assign(
            max_mean_distance=eligible["max_mean_distance"].round(12),
            center_distance=eligible["center_distance"].round(12),
            largest_distance=eligible["largest_distance"].round(12),
        ).sort_values(
            ["max_mean_distance", "center_distance", "largest_distance", "landscape", "model"],
            kind="stable",
        )
        if eligible.empty:
            raise ValueError(f"No eligible non-Gamma example available for {family}.")
        candidates.loc[eligible.head(3).index, "shortlisted"] = True
        if examples is not None:
            keys = pd.MultiIndex.from_frame(eligible[["landscape", "model"]])
            eligible = eligible.loc[keys.isin(requested)]
            if len(eligible) != 1:
                raise ValueError(f"Expected one eligible explicit example for {family}.")
        choice = eligible.iloc[0]
        candidates.loc[choice.name, "selected"] = True
        label = f"{FAMILY_LABELS[family]} {str(choice['landscape']).removeprefix(f'{family}-')}"
        label = requested.get((str(choice["landscape"]), str(choice["model"])), label)
        selected.append((str(choice["landscape"]), str(choice["model"]), label))
    audit = audit.merge(candidates[[*IDENTITY, "eligible", "selected", "shortlisted"]],
                        on=IDENTITY, validate="many_to_one")
    return tuple(selected), audit, candidates.sort_values(IDENTITY).reset_index(drop=True)
