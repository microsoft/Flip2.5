"""Tidy numerical tables from rank results and landscape features."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .data import FEATURE_DTYPES, LandscapeFeatures
from .mechanisms import (
    boundary_difficulty,
    count_within_score_window,
    local_score_spacing,
    partial_spearman,
    score_uncertainty,
    standardized_regression,
)
from .results import IDENTITY_FIELDS, RankResult
from .runs import RunConfig, RunSpec

TABLE_NAMES = (
    "runs",
    "rank_percentiles",
    "rank_agreement",
    "model_rank_intervals",
    "top_k_curves",
    "top_k_candidates",
    "mechanism_inventory",
    "mechanism_variants",
    "mechanism_runs",
    "mechanism_effects",
    "reference_diagnostics",
    "reference_category_sensitivity",
    "mechanism_diagnostics",
    "functionality_scores",
)

REFERENCE_CATEGORY_COLUMNS = (
    *IDENTITY_FIELDS,
    "cohort",
    "top_k",
    "category_column",
    "category",
    "count",
    "fraction",
)


def _identity(result: RankResult) -> dict[str, str]:
    return {name: str(result.metadata[name]) for name in IDENTITY_FIELDS}


def _add_identity(rows: pd.DataFrame, result: RankResult) -> pd.DataFrame:
    rows = rows.copy()
    for name, value in reversed(_identity(result).items()):
        if name in rows:
            if not rows[name].eq(value).all():
                raise ValueError(f"Variant column {name!r} conflicts with result metadata.")
        else:
            rows.insert(0, name, value)
    return rows


def _variant_rows(result: RankResult, indices: np.ndarray) -> pd.DataFrame:
    resolved = np.asarray(indices, dtype=np.int32)
    rows = result.variants.iloc[resolved].copy().reset_index(drop=True)
    rows.insert(0, "variant_index", resolved)
    return _add_identity(rows, result)


def _candidate_indices(result: RankResult) -> np.ndarray:
    return np.flatnonzero(
        result.variants["is_candidate"].to_numpy(dtype=bool)
    ).astype(np.int32)


def _reference_order(result: RankResult) -> np.ndarray:
    candidate_indices = _candidate_indices(result)
    scores = result.stability["reference_score"].values.astype(float)
    supported = candidate_indices[np.isfinite(scores[candidate_indices])]
    ordering_scores = (
        -scores[supported]
        if bool(result.metadata["higher_is_better"])
        else scores[supported]
    )
    order = np.lexsort((supported, ordering_scores))
    return supported[order].astype(np.int32)


def _reference_ranks(result: RankResult) -> np.ndarray:
    order = _reference_order(result)
    ranks = np.full(len(result.variants), np.nan)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _external_order(result: RankResult) -> np.ndarray:
    column = str(result.metadata["external_rank_column"])
    if column not in result.variants:
        raise ValueError(f"Result variants are missing external rank column {column!r}.")
    ranks = result.variants[column].to_numpy(dtype=float)
    candidate_indices = _candidate_indices(result)
    supported = candidate_indices[np.isfinite(ranks[candidate_indices])]
    return supported[np.lexsort((supported, ranks[supported]))].astype(np.int32)


def _external_value_order(
    result: RankResult,
    *,
    higher_is_better: bool | None = None,
) -> np.ndarray:
    column = str(result.metadata["external_value_column"])
    if column not in result.variants:
        raise ValueError(f"Result variants are missing external value column {column!r}.")
    values = result.variants[column].to_numpy(dtype=float)
    candidate_indices = _candidate_indices(result)
    supported = candidate_indices[np.isfinite(values[candidate_indices])]
    direction = (
        bool(result.metadata["external_higher_is_better"])
        if higher_is_better is None
        else higher_is_better
    )
    ordering_values = -values[supported] if direction else values[supported]
    return supported[np.lexsort((supported, ordering_values))].astype(np.int32)


def _top_k_counts(result: RankResult, k: int) -> np.ndarray:
    selected = result.stability["selection_variant_index"].values.astype(np.int32)
    return np.bincount(
        selected[:, :k].ravel(),
        minlength=len(result.variants),
    ).astype(np.int32)


def _self_consistency(counts: np.ndarray, n_samples: int, k: int) -> float:
    if n_samples < 2:
        raise ValueError("At least two samples are required for self-consistency.")
    return float(
        np.sum(counts * (counts - 1))
        / (n_samples * (n_samples - 1) * k)
    )


def ecdf_to_uniform_area(values: np.ndarray) -> float:
    """Return the absolute area between an empirical CDF and Uniform(0, 1)."""
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Percentiles must be finite and non-empty.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Percentiles must lie between zero and one.")
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


def _safe_spearman(first: np.ndarray | pd.Series, second: np.ndarray | pd.Series) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    complete = np.isfinite(left) & np.isfinite(right)
    left = left[complete]
    right = right[complete]
    if len(left) < 3 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def _runs_table(result: RankResult) -> pd.DataFrame:
    candidate_count = int(result.variants["is_candidate"].sum())
    row = {
        **_identity(result),
        "n_samples": int(result.metadata["n_samples"]),
        "n_variants": len(result.variants),
        "n_candidates": candidate_count,
        "n_excluded_from_ranking": len(result.variants) - candidate_count,
        "retained_k": int(result.metadata["retained_k"]),
        "decision_k": int(result.metadata["decision_k"]),
        "external_top_n": int(result.metadata["external_top_n"]),
        "model_top_n": int(result.metadata["model_top_n"]),
        "higher_is_better": bool(result.metadata["higher_is_better"]),
        "challenger_probability": float(result.metadata["challenger_probability"]),
    }
    return pd.DataFrame([row])


def _rank_percentiles(result: RankResult) -> pd.DataFrame:
    cohort_indices = result.stability["cohort_variant_index"].values.astype(np.int32)
    external = result.stability["is_external_reference"].values.astype(bool)
    external_indices = cohort_indices[external]
    ranks = result.stability["cohort_rank"].values[:, external].astype(float)
    rank_column = str(result.metadata["external_rank_column"])
    external_ranks = result.variants.iloc[external_indices][rank_column].to_numpy(dtype=float)
    external_order = np.lexsort((external_indices, external_ranks))
    external_indices = external_indices[external_order]
    ranks = ranks[:, external_order]
    external_ranks = external_ranks[external_order]
    finite_count = np.isfinite(ranks).sum(axis=0)
    better = ((ranks < external_ranks[None, :]) & np.isfinite(ranks)).sum(axis=0)
    tied = ((ranks == external_ranks[None, :]) & np.isfinite(ranks)).sum(axis=0)
    percentiles = np.divide(
        better + 0.5 * tied,
        finite_count,
        out=np.full(len(external_indices), np.nan),
        where=finite_count > 0,
    )
    median_ranks = np.full(len(external_indices), np.nan)
    supported = finite_count > 0
    median_ranks[supported] = np.nanmedian(ranks[:, supported], axis=0)
    frames = []
    cohorts = tuple(
        sorted(
            {
                int(result.metadata["decision_k"]),
                int(result.metadata["external_top_n"]),
            }
        )
    )
    for cohort_top_n in cohorts:
        include = (
            np.isfinite(external_ranks)
            & np.isfinite(percentiles)
            & np.isfinite(median_ranks)
            & (external_ranks <= cohort_top_n)
        )
        if not include.any():
            raise ValueError(
                "No external-reference variants are available for "
                f"top {cohort_top_n}."
            )
        indices = external_indices[include]
        values = percentiles[include]
        selected_external_ranks = external_ranks[include]
        selected_median_ranks = median_ranks[include]
        order = np.argsort(values, kind="stable")
        rows = _variant_rows(result, indices[order])
        rows["posterior_median_sampled_rank"] = selected_median_ranks[order]
        rows["median_sampled_rank_minus_external_rank"] = (
            selected_median_ranks[order] - selected_external_ranks[order]
        )
        rows["external_vs_posterior_median_rank_spearman"] = _safe_spearman(
            selected_external_ranks,
            selected_median_ranks,
        )
        rows.insert(6, "cohort_top_n", cohort_top_n)
        rows.insert(6, "ecdf", np.arange(1, len(rows) + 1) / len(rows))
        rows.insert(6, "external_rank_sample_percentile", values[order])
        rows.insert(6, "ecdf_to_uniform_area", ecdf_to_uniform_area(values))
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _rank_agreement(result: RankResult) -> pd.DataFrame:
    indices = result.stability["reference_variant_index"].values.astype(np.int32)
    rows = _variant_rows(result, indices)
    rows["model_reference_rank"] = np.arange(1, len(rows) + 1)
    rows["model_reference_score"] = result.stability["reference_score"].values[
        indices
    ].astype(float)
    rank_column = str(result.metadata["external_rank_column"])
    rows["model_rank_minus_external_rank"] = (
        rows["model_reference_rank"] - rows[rank_column]
    )
    rows["absolute_model_external_rank_difference"] = rows[
        "model_rank_minus_external_rank"
    ].abs()
    return rows


def _model_rank_intervals(result: RankResult) -> pd.DataFrame:
    decision_k = int(result.metadata["decision_k"])
    cohort_indices = result.stability["cohort_variant_index"].values.astype(np.int32)
    model = result.stability["is_model_reference"].values.astype(bool)
    model_indices = cohort_indices[model]
    reference_ranks = _reference_ranks(result)[model_indices]
    order = np.argsort(reference_ranks, kind="stable")
    model_indices = model_indices[order][:decision_k]
    reference_ranks = reference_ranks[order][:decision_k]
    cohort_positions = pd.Index(cohort_indices).get_indexer(model_indices)
    ranks = result.stability["cohort_rank"].values[:, cohort_positions].astype(float)
    interval = tuple(float(value) for value in result.metadata["interval"])
    lower, median, upper = np.nanquantile(
        ranks,
        [interval[0], 0.5, interval[1]],
        axis=0,
    )
    rows = _variant_rows(result, model_indices)
    rows["k"] = decision_k
    rows["model_reference_rank"] = reference_ranks.astype(np.int32)
    rows["model_reference_score"] = result.stability["reference_score"].values[
        model_indices
    ].astype(float)
    rows["posterior_rank_lower"] = lower
    rows["posterior_median_rank"] = median
    rows["posterior_rank_upper"] = upper
    rows["rank_interval_width"] = upper - lower
    rows["rank_interval_leaves_top_k"] = upper > decision_k
    rows["top_k_inclusion_probability"] = np.nanmean(ranks <= decision_k, axis=0)
    rows["lower_rank_minus_model_reference"] = lower - reference_ranks
    rows["median_rank_minus_model_reference"] = median - reference_ranks
    rows["upper_rank_minus_model_reference"] = upper - reference_ranks
    return rows


def _top_k_curves(result: RankResult) -> pd.DataFrame:
    k_values = tuple(int(value) for value in result.metadata["k_values"])
    retained_k = result.stability.sizes["selection_rank"]
    if any(k < 1 or k > retained_k for k in k_values):
        raise ValueError("Configured k values must lie on the retained selection axis.")
    n_samples = result.stability.sizes["sample"]
    reference_order = _reference_order(result)
    external_order = _external_order(result)
    rows = []
    for k in k_values:
        if min(len(reference_order), len(external_order)) < k:
            raise ValueError(f"Fewer than k={k} supported reference variants.")
        counts = _top_k_counts(result, k)
        probabilities = counts / n_samples
        rows.append(
            {
                **_identity(result),
                "k": k,
                "sample_self_consistency": _self_consistency(counts, n_samples, k),
                "sample_vs_model_reference": float(
                    probabilities[reference_order[:k]].sum() / k
                ),
                "sample_vs_external_reference": float(
                    probabilities[external_order[:k]].sum() / k
                ),
                "candidate_count_q_gt_0": int(np.sum(probabilities > 0)),
                "candidate_count_q_gt_0_01": int(np.sum(probabilities > 0.01)),
                "candidate_count_q_gt_0_05": int(np.sum(probabilities > 0.05)),
            }
        )
    return pd.DataFrame(rows)


def _top_k_candidates(result: RankResult) -> pd.DataFrame:
    decision_k = int(result.metadata["decision_k"])
    threshold = float(result.metadata["challenger_probability"])
    counts = _top_k_counts(result, decision_k)
    probabilities = counts / result.stability.sizes["sample"]
    reference_ranks = np.full(len(result.variants), np.nan)
    retained_reference = result.stability["reference_variant_index"].values.astype(
        np.int32
    )
    reference_ranks[retained_reference] = np.arange(1, len(retained_reference) + 1)
    external_order = _external_order(result)
    external_top = np.zeros(len(result.variants), dtype=bool)
    external_top[external_order[:decision_k]] = True
    model_top = np.isfinite(reference_ranks) & (reference_ranks <= decision_k)
    include = (probabilities > threshold) | external_top | model_top
    indices = np.flatnonzero(include).astype(np.int32)
    rows = _variant_rows(result, indices)
    rows["k"] = decision_k
    rows["top_k_count"] = counts[indices]
    rows["top_k_probability"] = probabilities[indices]
    rows["model_reference_rank"] = reference_ranks[indices]
    rows["is_external_reference_top_k"] = external_top[indices]
    rows["is_model_reference_top_k"] = model_top[indices]
    rows["candidate_probability_threshold"] = threshold
    rank_column = str(result.metadata["external_rank_column"])
    return rows.sort_values(
        ["top_k_probability", "model_reference_rank", rank_column, "variant_index"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _mechanism_variants(
    result: RankResult,
    features: LandscapeFeatures,
) -> pd.DataFrame:
    candidate_order = _reference_order(result)
    candidate_scores = result.stability["reference_score"].values
    ordered_scores = candidate_scores[candidate_order]
    retain_k = min(int(result.metadata["retained_k"]), len(ordered_scores))
    if retain_k < 2:
        raise ValueError("At least two supported candidates are required for mechanism tables.")
    normalization_scores = ordered_scores[:retain_k]
    higher_is_better = bool(result.metadata["higher_is_better"])
    reference_ranks = _reference_ranks(result)
    decision_k = int(result.metadata["decision_k"])
    cohort_indices = result.stability["cohort_variant_index"].values.astype(np.int32)
    cohort_scores = result.stability["cohort_score"].values.astype(float)
    finite_count = np.isfinite(cohort_scores).sum(axis=0)
    model_top = reference_ranks[cohort_indices] <= decision_k
    challenger = result.stability["is_challenger"].values.astype(bool)
    external = result.stability["is_external_reference"].values.astype(bool)
    primary = model_top | challenger
    if np.any(primary & (finite_count < 2)):
        raise ValueError("Every model-top or challenger variant requires two finite scores.")
    include = primary | (external & (finite_count >= 2))
    cohort_indices = cohort_indices[include]
    cohort_scores = cohort_scores[:, include]
    cohort_ranks = result.stability["cohort_rank"].values[:, include].astype(float)
    model_top = model_top[include]
    challenger = challenger[include]
    external = external[include]
    finite_count = finite_count[include]

    uncertainty = score_uncertainty(
        cohort_scores,
        reference_scores=normalization_scores,
        interval=tuple(float(value) for value in result.metadata["interval"]),
        higher_is_better=higher_is_better,
    )
    spacing = local_score_spacing(
        ordered_scores,
        neighborhood=5,
        normalization_scores=normalization_scores,
        higher_is_better=higher_is_better,
    )
    crowd_01 = count_within_score_window(
        ordered_scores,
        window_fraction=0.01,
        normalization_scores=normalization_scores,
        higher_is_better=higher_is_better,
    )
    crowd_02 = count_within_score_window(
        ordered_scores,
        window_fraction=0.02,
        normalization_scores=normalization_scores,
        higher_is_better=higher_is_better,
    )
    interval = tuple(float(value) for value in result.metadata["interval"])
    rank_lower, rank_median, rank_upper = np.nanquantile(
        cohort_ranks,
        [interval[0], 0.5, interval[1]],
        axis=0,
    )
    reference_rank = reference_ranks[cohort_indices].astype(np.int32)
    order_position = reference_rank - 1
    counts = result.stability["top_k_count"].values.astype(np.int32)
    probability = result.stability["top_k_probability"].values.astype(float)
    rows = _variant_rows(result, cohort_indices)
    rows["model_reference_rank"] = reference_rank
    rows["model_reference_score"] = candidate_scores[cohort_indices]
    for name in (
        "score_lower",
        "score_median",
        "score_upper",
        "score_interval_width",
        "normalized_score_interval_width",
        "n_finite_samples",
    ):
        rows[name] = uncertainty[name]
    rows["reference_score_iqr"] = uncertainty["reference_score_iqr"]
    rows["local_score_spacing"] = spacing["score_spacing"][order_position]
    rows["normalized_local_score_spacing"] = spacing[
        "normalized_score_spacing"
    ][order_position]
    rows["score_crowding"] = spacing["score_crowding"][order_position]
    rows["competitors_within_1pct_iqr"] = crowd_01["competitor_count"][order_position]
    rows["competitors_within_2pct_iqr"] = crowd_02["competitor_count"][order_position]
    rows["rank_lower"] = rank_lower
    rows["rank_median"] = rank_median
    rows["rank_upper"] = rank_upper
    rows["rank_interval_width"] = rank_upper - rank_lower
    rows["top_k_count"] = counts[cohort_indices]
    rows["top_k_inclusion_probability"] = probability[cohort_indices]
    rows["rank_interval_leaves_top_k"] = rank_upper > decision_k
    direction = 1.0 if higher_is_better else -1.0
    boundary_score = candidate_scores[candidate_order[decision_k - 1]]
    rows["score_margin_to_top_k_boundary"] = direction * (
        rows["model_reference_score"] - boundary_score
    )
    rows["margin_over_score_uncertainty"] = np.divide(
        rows["score_margin_to_top_k_boundary"],
        rows["score_interval_width"],
        out=np.full(len(rows), np.nan),
        where=rows["score_interval_width"].to_numpy() > 0,
    )
    rows["is_model_top_k"] = model_top
    rows["is_challenger"] = challenger
    rows["is_external_reference"] = external
    rows["is_primary_cohort"] = model_top | challenger
    challenger_indices = cohort_indices[challenger]
    strongest = challenger_indices[
        np.argsort(probability[challenger_indices], kind="stable")[::-1][:10]
    ]
    rows["is_strongest_challenger"] = np.isin(cohort_indices, strongest)

    feature_indices = features.indices_for(result.variants)
    aligned_feature_indices = feature_indices[cohort_indices]
    for name in FEATURE_DTYPES:
        rows[f"mechanism_{name}"] = features.dataset[name].values[
            aligned_feature_indices
        ]
    return rows.sort_values(
        ["model_reference_rank", "variant_index"],
        kind="stable",
    ).reset_index(drop=True)


def _mechanism_run(result: RankResult, variants: pd.DataFrame) -> pd.DataFrame:
    decision_k = int(result.metadata["decision_k"])
    reference_order = _reference_order(result)
    reference_scores = result.stability["reference_score"].values.astype(float)[
        reference_order
    ]
    if len(reference_order) <= decision_k:
        raise ValueError("Mechanism boundary summaries require a candidate below decision_k.")
    cohort_indices = result.stability["cohort_variant_index"].values.astype(np.int32)
    top_columns = pd.Index(cohort_indices).get_indexer(reference_order[:decision_k])
    if np.any(top_columns < 0):
        raise ValueError("RankResult does not retain every model top-k score trajectory.")
    counts = _top_k_counts(result, decision_k)
    consistency = _self_consistency(
        counts,
        result.stability.sizes["sample"],
        decision_k,
    )
    normalization_count = min(int(result.metadata["retained_k"]), len(reference_scores))
    boundary = boundary_difficulty(
        reference_scores,
        result.stability["cohort_score"].values[:, top_columns],
        sample_self_consistency=consistency,
        k=decision_k,
        interval=tuple(float(value) for value in result.metadata["interval"]),
        normalization_scores=reference_scores[:normalization_count],
        higher_is_better=bool(result.metadata["higher_is_better"]),
    )
    primary = variants["is_primary_cohort"].astype(bool)
    finite_count = np.isfinite(result.stability["cohort_score"].values).sum(axis=0)
    omitted = int(
        np.sum(
            result.stability["is_external_reference"].values.astype(bool)
            & ~result.stability["is_model_reference"].values.astype(bool)
            & ~result.stability["is_challenger"].values.astype(bool)
            & (finite_count < 2)
        )
    )
    row = {
        **_identity(result),
        "n_samples": result.stability.sizes["sample"],
        "n_candidates": len(_candidate_indices(result)),
        "mechanism_cohort_count": len(variants),
        "model_top_k_count": int(variants["is_model_top_k"].sum()),
        "challenger_count": int(variants["is_challenger"].sum()),
        "candidate_cloud_count": int(
            np.sum(
                result.stability["top_k_probability"].values
                >= float(result.metadata["challenger_probability"])
            )
        ),
        "external_context_omitted_low_support": omitted,
        "challenger_probability": float(result.metadata["challenger_probability"]),
        "reference_scale_k": normalization_count,
        "reference_score_iqr": variants["reference_score_iqr"].iloc[0],
        "median_normalized_score_interval_width": float(
            variants.loc[primary, "normalized_score_interval_width"].median()
        ),
        "median_rank_interval_width": float(
            variants.loc[primary, "rank_interval_width"].median()
        ),
        "feature_path": result.metadata["feature_path"],
        **boundary,
    }
    return pd.DataFrame([row])


def _functionality_scores(result: RankResult) -> pd.DataFrame:
    order = _reference_order(result)
    rows = _variant_rows(result, order)
    rows["score_median"] = result.stability["reference_score"].values[order].astype(
        float
    )
    rows["model_reference_rank"] = np.arange(1, len(order) + 1, dtype=np.int32)
    rows["n_finite_samples"] = result.stability[
        "reference_n_finite_samples"
    ].values[order]
    return rows


def _reference_diagnostic(result: RankResult) -> pd.DataFrame:
    candidate_indices = _candidate_indices(result)
    model_scores = result.stability["reference_score"].values.astype(float)
    value_column = str(result.metadata["external_value_column"])
    rank_column = str(result.metadata["external_rank_column"])
    if value_column not in result.variants:
        raise ValueError(f"Result variants are missing external value column {value_column!r}.")
    external_values = result.variants[value_column].to_numpy(dtype=float)
    finite = candidate_indices[
        np.isfinite(model_scores[candidate_indices])
        & np.isfinite(external_values[candidate_indices])
    ]
    external_direction = 1.0 if bool(result.metadata["external_higher_is_better"]) else -1.0
    model_direction = 1.0 if bool(result.metadata["higher_is_better"]) else -1.0

    def correlations(indices: np.ndarray) -> tuple[float, float]:
        if len(indices) < 3:
            return np.nan, np.nan
        model = model_direction * model_scores[indices]
        external = external_direction * external_values[indices]
        if np.ptp(model) <= 0 or np.ptp(external) <= 0:
            return np.nan, np.nan
        return (
            float(spearmanr(model, external).statistic),
            float(kendalltau(model, external).statistic),
        )

    all_spearman, all_kendall = correlations(finite)
    positive = finite[external_values[finite] > 0]
    positive_spearman, positive_kendall = correlations(positive)
    evaluated_k = min(int(result.metadata["decision_k"]), len(finite))
    reference_order = _reference_order(result)
    external_ordering_values = external_direction * external_values[finite]
    external_order = finite[
        np.lexsort((finite, -external_ordering_values))
    ]
    reversed_order = finite[
        np.lexsort((finite, external_ordering_values))
    ]
    model_top = set(reference_order[:evaluated_k].tolist())
    external_top = set(external_order[:evaluated_k].tolist())
    reversed_top = set(reversed_order[:evaluated_k].tolist())
    model_ranks = _reference_ranks(result)
    wild_type = np.flatnonzero(
        result.variants.get(
            "is_wild_type",
            pd.Series(False, index=result.variants.index),
        ).to_numpy(dtype=bool)
    )
    wild_type_index = int(wild_type[0]) if len(wild_type) == 1 else None
    source_controls = result.variants.get(
        "is_source_control",
        pd.Series(False, index=result.variants.index),
    ).to_numpy(dtype=bool)
    row = {
        **_identity(result),
        "n_variants": len(result.variants),
        "n_ranking_candidates": len(candidate_indices),
        "n_score_supported_candidates": int(
            np.isfinite(model_scores[candidate_indices]).sum()
        ),
        "n_excluded_from_ranking": len(result.variants) - len(candidate_indices),
        "n_source_controls": int(source_controls.sum()),
        "n_finite_external": len(finite),
        "n_positive_external": len(positive),
        "all_finite_spearman": all_spearman,
        "all_finite_kendall": all_kendall,
        "positive_external_spearman": positive_spearman,
        "positive_external_kendall": positive_kendall,
        "top_k": evaluated_k,
        "model_external_top_k_intersection": len(model_top & external_top),
        "model_external_top_k_fraction": (
            len(model_top & external_top) / evaluated_k if evaluated_k else np.nan
        ),
        "model_reversed_external_top_k_intersection": len(model_top & reversed_top),
        "model_reversed_external_top_k_fraction": (
            len(model_top & reversed_top) / evaluated_k if evaluated_k else np.nan
        ),
        "wild_type_variant_index": wild_type_index,
        "wild_type_model_rank": (
            model_ranks[wild_type_index] if wild_type_index is not None else np.nan
        ),
        "wild_type_external_rank": (
            result.variants.iloc[wild_type_index][rank_column]
            if wild_type_index is not None
            else np.nan
        ),
        "external_value_column": value_column,
        "within_landscape_model_top_k_pair_count": 0,
        "within_landscape_model_top_k_overlap_mean": np.nan,
        "within_landscape_model_top_k_overlap_min": np.nan,
    }
    return pd.DataFrame([row])


def _reference_category_sensitivity(result: RankResult) -> pd.DataFrame:
    category_column = next(
        (name for name in ("reported_activity_level",) if name in result.variants),
        None,
    )
    if category_column is None:
        return pd.DataFrame(columns=REFERENCE_CATEGORY_COLUMNS)
    top_k = int(result.metadata["decision_k"])
    rows = []
    for cohort_name, indices in (
        ("model_top_k", _reference_order(result)[:top_k]),
        ("external_top_k", _external_value_order(result)[:top_k]),
    ):
        categories = result.variants.iloc[indices][category_column].fillna("unavailable")
        for category, count in categories.value_counts(dropna=False, sort=False).items():
            rows.append(
                {
                    **_identity(result),
                    "cohort": cohort_name,
                    "top_k": top_k,
                    "category_column": category_column,
                    "category": category,
                    "count": int(count),
                    "fraction": float(count / top_k),
                }
            )
    return pd.DataFrame(rows, columns=REFERENCE_CATEGORY_COLUMNS)


def _mechanism_diagnostics(variants: pd.DataFrame) -> pd.DataFrame:
    strata = (
        ("primary", variants["is_primary_cohort"].astype(bool)),
        ("model_top_k", variants["is_model_top_k"].astype(bool)),
        ("challengers", variants["is_challenger"].astype(bool)),
    )
    rows = []
    identity = {name: variants[name].iloc[0] for name in IDENTITY_FIELDS}
    for stratum, include in strata:
        selected = variants[include]
        partial = np.nan
        if len(selected):
            try:
                partial = partial_spearman(
                    selected["score_margin_to_top_k_boundary"].to_numpy(dtype=float),
                    selected["normalized_score_interval_width"].to_numpy(dtype=float),
                    {
                        "input_support": np.log1p(
                            selected["mechanism_input_count"].to_numpy(dtype=float)
                        )
                    },
                )
            except ValueError:
                pass
        rows.append(
            {
                **identity,
                "stratum": stratum,
                "n_variants": len(selected),
                "input_support_vs_score_uncertainty_spearman": _safe_spearman(
                    selected["mechanism_input_count"],
                    selected["normalized_score_interval_width"],
                ),
                "output_abundance_vs_score_uncertainty_spearman": _safe_spearman(
                    selected["mechanism_output_count"],
                    selected["normalized_score_interval_width"],
                ),
                "decision_margin_vs_score_uncertainty_spearman": _safe_spearman(
                    selected["score_margin_to_top_k_boundary"],
                    selected["normalized_score_interval_width"],
                ),
                "decision_margin_vs_inclusion_probability_spearman": _safe_spearman(
                    selected["score_margin_to_top_k_boundary"],
                    selected["top_k_inclusion_probability"],
                ),
                "decision_margin_vs_score_uncertainty_partial_spearman_input_adjusted": (
                    partial
                ),
                "source_control_rows": int(
                    selected.get(
                        "is_source_control",
                        pd.Series(False, index=selected.index),
                    ).astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_effects(variants: pd.DataFrame) -> pd.DataFrame:
    primary = variants[variants["is_primary_cohort"]].copy()
    common = {
        **{name: primary[name].iloc[0] for name in IDENTITY_FIELDS},
        "scope": "run",
    }
    effects = []
    stage_a_predictors = {
        "log_input_support": primary["mechanism_log_input_support"].to_numpy(),
        "log_reference_rank": np.log1p(primary["model_reference_rank"].to_numpy()),
    }
    if primary["mechanism_reduced_heterogeneity"].notna().sum() > 3:
        stage_a_predictors["replicate_discordance"] = primary[
            "mechanism_reduced_heterogeneity"
        ].to_numpy()
    try:
        effects.append(
            {
                **common,
                "stage": "score_precision",
                **standardized_regression(
                    np.log1p(primary["normalized_score_interval_width"].to_numpy()),
                    stage_a_predictors,
                ),
            }
        )
    except ValueError as error:
        effects.append({**common, "stage": "score_precision", "error": str(error)})
    uncertainty = primary["normalized_score_interval_width"].to_numpy()
    crowding = primary["score_crowding"].to_numpy()
    try:
        effects.append(
            {
                **common,
                "stage": "rank_propagation",
                **standardized_regression(
                    np.log1p(primary["rank_interval_width"].to_numpy()),
                    {
                        "score_uncertainty": uncertainty,
                        "crowding": crowding,
                        "uncertainty_x_crowding": uncertainty * crowding,
                        "log_reference_rank": np.log1p(
                            primary["model_reference_rank"].to_numpy()
                        ),
                    },
                ),
            }
        )
    except ValueError as error:
        effects.append({**common, "stage": "rank_propagation", "error": str(error)})
    return pd.DataFrame(effects)


def _mechanism_inventory(
    result: RankResult,
    features: LandscapeFeatures,
) -> pd.DataFrame:
    row: dict[str, object] = {
        "sampling": result.metadata["sampling"],
        "family": result.metadata["family"],
        "landscape": result.metadata["landscape"],
        "feature_path": result.metadata["feature_path"],
        "feature_variant_count": len(features.variants),
        "feature_candidate_count": int(features.variants["is_candidate"].sum()),
        "analysis_run_count": 1,
    }
    for name in FEATURE_DTYPES:
        row[f"finite_{name}_count"] = int(np.isfinite(features.dataset[name].values).sum())
    return pd.DataFrame([row])


def build_run_tables(
    result: RankResult,
    features: LandscapeFeatures,
) -> dict[str, pd.DataFrame]:
    """Build all fourteen table products for one validated analysis result."""
    result.validate()
    features.validate()
    mechanism_variants = _mechanism_variants(result, features)
    return {
        "runs": _runs_table(result),
        "rank_percentiles": _rank_percentiles(result),
        "rank_agreement": _rank_agreement(result),
        "model_rank_intervals": _model_rank_intervals(result),
        "top_k_curves": _top_k_curves(result),
        "top_k_candidates": _top_k_candidates(result),
        "mechanism_inventory": _mechanism_inventory(result, features),
        "mechanism_variants": mechanism_variants,
        "mechanism_runs": _mechanism_run(result, mechanism_variants),
        "mechanism_effects": _mechanism_effects(mechanism_variants),
        "reference_diagnostics": _reference_diagnostic(result),
        "reference_category_sensitivity": _reference_category_sensitivity(result),
        "mechanism_diagnostics": _mechanism_diagnostics(mechanism_variants),
        "functionality_scores": _functionality_scores(result),
    }


def combine_run_tables(
    table_sets: Sequence[Mapping[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Combine per-run products and add cross-run landscape summaries."""
    if not table_sets:
        raise ValueError("At least one run table set is required.")
    if any(set(tables) != set(TABLE_NAMES) for tables in table_sets):
        raise ValueError("Every run table set must contain all table products.")
    combined = {
        name: pd.concat([tables[name] for tables in table_sets], ignore_index=True)
        for name in TABLE_NAMES
    }
    inventory = combined["mechanism_inventory"]
    inventory_keys = ["sampling", "family", "landscape", "feature_path"]
    value_columns = [
        name
        for name in inventory.columns
        if name not in {*inventory_keys, "analysis_run_count"}
    ]
    for _, group in inventory.groupby(inventory_keys, sort=False):
        if any(group[name].nunique(dropna=False) != 1 for name in value_columns):
            raise ValueError("Runs sharing a feature path disagree on feature summaries.")
    inventory = inventory.drop_duplicates(inventory_keys).copy()
    run_counts = combined["mechanism_inventory"].groupby(
        inventory_keys,
        sort=False,
    )["analysis_run_count"].sum()
    inventory["analysis_run_count"] = [
        run_counts.loc[tuple(row)]
        for row in inventory[inventory_keys].itertuples(index=False, name=None)
    ]
    combined["mechanism_inventory"] = inventory.reset_index(drop=True)

    effects = combined["mechanism_effects"]
    numeric_columns = effects.select_dtypes(include="number").columns.tolist()
    landscape_effects = effects.groupby(
        ["family", "landscape", "sampling", "stage"],
        as_index=False,
        dropna=False,
    )[numeric_columns].median(numeric_only=True)
    landscape_effects["scope"] = "landscape_median"
    for name in ("run_id", "model", "score"):
        landscape_effects[name] = pd.NA
    combined["mechanism_effects"] = pd.concat(
        [effects, landscape_effects],
        ignore_index=True,
        sort=False,
    )

    diagnostics = combined["reference_diagnostics"].copy()
    agreement = combined["rank_agreement"]
    for index, row in diagnostics.iterrows():
        own = set(
            agreement.loc[
                agreement["run_id"].eq(row["run_id"])
                & agreement["model_reference_rank"].le(row["top_k"]),
                "variant_index",
            ].astype(int)
        )
        peers = diagnostics.loc[
            diagnostics["sampling"].eq(row["sampling"])
            & diagnostics["landscape"].eq(row["landscape"])
            & diagnostics["run_id"].ne(row["run_id"]),
            "run_id",
        ]
        overlaps = []
        for peer_id in peers:
            peer = set(
                agreement.loc[
                    agreement["run_id"].eq(peer_id)
                    & agreement["model_reference_rank"].le(row["top_k"]),
                    "variant_index",
                ].astype(int)
            )
            overlaps.append(len(own & peer) / int(row["top_k"]))
        if overlaps:
            diagnostics.loc[index, "within_landscape_model_top_k_pair_count"] = len(
                overlaps
            )
            diagnostics.loc[index, "within_landscape_model_top_k_overlap_mean"] = (
                np.mean(overlaps)
            )
            diagnostics.loc[index, "within_landscape_model_top_k_overlap_min"] = (
                np.min(overlaps)
            )
    combined["reference_diagnostics"] = diagnostics
    return combined


def build_tables_for_runs(
    config: RunConfig,
    runs: Sequence[RunSpec],
    *,
    result_root: str | Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load selected results and features, then combine tables by sampling method."""
    root = Path(result_root).resolve()
    grouped: dict[str, list[dict[str, pd.DataFrame]]] = {}
    for run in runs:
        result_path = root / run.sampling / run.run_id
        feature_path = config.input_path(run.feature_path)
        result = RankResult.load(result_path)
        try:
            features = LandscapeFeatures.load(feature_path)
            try:
                expected_identity = run.identity()
                if any(
                    result.metadata.get(name) != value
                    for name, value in expected_identity.items()
                ):
                    raise ValueError(f"Result identity does not match run {run.run_id}.")
                if result.metadata.get("feature_path") != run.feature_path.as_posix():
                    raise ValueError(f"Result feature path does not match run {run.run_id}.")
                grouped.setdefault(run.sampling, []).append(
                    build_run_tables(result, features)
                )
            finally:
                features.close()
        finally:
            result.close()
    if not grouped:
        raise ValueError("At least one run is required to build tables.")
    return {
        sampling: combine_run_tables(table_sets)
        for sampling, table_sets in grouped.items()
    }


def write_tables(
    tables_by_sampling: Mapping[str, Mapping[str, pd.DataFrame]],
    path: str | Path,
) -> tuple[Path, ...]:
    """Atomically publish exactly fourteen table files per sampling method."""
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    published = []
    for sampling, tables in tables_by_sampling.items():
        if set(tables) != set(TABLE_NAMES):
            raise ValueError(f"{sampling}: table set does not contain all products.")
        temporary = Path(tempfile.mkdtemp(prefix=f".{sampling}-", dir=root))
        try:
            for name in TABLE_NAMES:
                compressed = name == "functionality_scores"
                filename = f"{name}.csv.gz" if compressed else f"{name}.csv"
                tables[name].to_csv(
                    temporary / filename,
                    index=False,
                    compression="gzip" if compressed else None,
                )
            output = root / sampling
            if output.exists():
                shutil.rmtree(output)
            os.replace(temporary, output)
            published.append(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return tuple(published)
