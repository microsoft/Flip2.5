"""Paper-facing summaries from compact stability result bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from .candidates import ranking_candidate_indices
from .results import StabilityResult

PAPER_TABLE_NAMES = (
    "inventory",
    "rank_percentiles",
    "rank_agreement",
    "model_rank_intervals",
    "top_k_curves",
    "top_k_candidates",
)

ResultLike = StabilityResult | str | Path


def _load_result(result: ResultLike) -> StabilityResult:
    return result if isinstance(result, StabilityResult) else StabilityResult.load(result)


def _metadata_columns(metadata: dict[str, object]) -> dict[str, object]:
    fields = (
        "analysis_id",
        "protein_family",
        "landscape",
        "inference_method",
        "model_spec",
        "fit_id",
    )
    return {field: metadata[field] for field in fields}


def _score_selection(selection: xr.Dataset, score: str) -> xr.Dataset:
    if "score_name" not in selection.coords:
        raise ValueError("Result selection must contain a score_name coordinate.")
    available = selection["score_name"].values.astype(str).tolist()
    if score not in available:
        raise ValueError(f"Unknown score {score!r}; choose from {available}.")
    return selection.sel(score_name=score, drop=True)


def _require_paper_result(
    result: StabilityResult,
    *,
    score: str,
    main_k: int,
    k_values: tuple[int, ...],
) -> xr.Dataset:
    metadata = result.metadata
    if metadata.get("schema_version") != 2:
        raise ValueError("Paper reporting requires result schema_version 2.")
    _metadata_columns(metadata)
    tracked_model_top_n = int(metadata.get("tracked_model_top_n", 0))
    if tracked_model_top_n < main_k:
        raise ValueError(
            f"tracked_model_top_n must be at least main_k={main_k}, "
            f"not {tracked_model_top_n}."
        )
    selected = _score_selection(result.selection, score)
    required = {
        "variant_index",
        "tracked_rank",
        "model_reference_variant_index",
        "model_reference_score",
        "model_tracked_variant_index",
        "model_tracked_rank",
    }
    missing = required - set(selected.data_vars)
    if missing:
        raise ValueError(f"Result is missing paper selection variables: {sorted(missing)}")
    retained_k = selected.sizes["selection_rank"]
    if any(k < 1 or k > retained_k for k in k_values):
        raise ValueError(f"k_values must be between 1 and retained_k={retained_k}.")
    if main_k not in k_values:
        raise ValueError("k_values must include main_k.")
    return selected


def ecdf_to_uniform_area(values: np.ndarray) -> float:
    """Return the absolute area between an empirical CDF and Uniform(0, 1)."""
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Percentiles must be finite and non-empty.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Percentiles must be between 0 and 1.")

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


def _reported_reference_indices(variants: pd.DataFrame, k: int) -> np.ndarray:
    if "reported_rank" not in variants:
        raise ValueError("Variant table must contain reported_rank.")
    ranks = variants["reported_rank"].to_numpy(dtype=float)
    candidate_indices = ranking_candidate_indices(variants)
    finite_indices = candidate_indices[np.isfinite(ranks[candidate_indices])]
    if len(finite_indices) < k:
        raise ValueError(f"Fewer than k={k} variants have finite reported ranks.")
    order = np.lexsort((finite_indices, ranks[finite_indices]))
    return finite_indices[order[:k]].astype(np.int32)


def _variant_rows(variants: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    rows = variants.iloc[np.asarray(indices, dtype=np.int64)].copy()
    rows.insert(0, "variant_index", np.asarray(indices, dtype=np.int32))
    return rows.reset_index(drop=True)


def _inventory(
    result: StabilityResult,
    *,
    score: str,
) -> pd.DataFrame:
    metadata = result.metadata
    row = {
        **_metadata_columns(metadata),
        "job_id": metadata.get("job_id"),
        "artifact_kind": metadata.get("artifact_kind", "unrecorded"),
        "artifact_validation_status": metadata.get(
            "artifact_validation_status",
            "unrecorded",
        ),
        "primary_score": metadata.get("primary_score", score),
        "score": score,
        "n_samples": int(metadata["n_samples"]),
        "n_variants": int(metadata["n_variants"]),
        "n_fitted_variants": int(
            metadata.get("n_fitted_variants", metadata["n_variants"])
        ),
        "n_excluded_from_ranking": int(metadata.get("n_excluded_from_ranking", 0)),
        "retained_k": int(metadata["retained_k"]),
        "tracked_reported_top_n": int(metadata["tracked_reported_top_n"]),
        "tracked_model_top_n": int(metadata["tracked_model_top_n"]),
        "artifact_status": metadata.get("artifact_status", "unrecorded"),
        "warnings": json.dumps(metadata.get("warnings", [])),
    }
    return pd.DataFrame([row])


def _rank_percentiles(
    result: StabilityResult,
    selected: xr.Dataset,
    *,
    percentile_cohorts: tuple[int, ...],
) -> pd.DataFrame:
    if not bool(result.metadata.get("reference_top_k_identifiable", True)):
        rows = _variant_rows(result.variants, np.empty(0, dtype=np.int32))
        columns = (
            "reported_rank_sample_percentile",
            "ecdf_to_uniform_area",
            "ecdf",
            "cohort_top_n",
            "posterior_median_sampled_rank",
            "median_sampled_rank_minus_reported_rank",
            "reported_vs_posterior_median_rank_spearman",
        )
        for column in columns:
            rows[column] = pd.Series(dtype=float)
        for field, value in reversed(_metadata_columns(result.metadata).items()):
            rows.insert(0, field, value)
        return rows

    tracked_indices = np.asarray(
        result.selection["tracked_variant_index"].values,
        dtype=np.int32,
    )
    ranks = np.asarray(selected["tracked_rank"].values, dtype=float)
    reported_ranks = result.variants.iloc[tracked_indices]["reported_rank"].to_numpy(
        dtype=float
    )
    candidate_indices = ranking_candidate_indices(result.variants)
    is_candidate = np.isin(tracked_indices, candidate_indices)
    valid = np.isfinite(ranks)
    finite_count = valid.sum(axis=0)
    better = ((ranks < reported_ranks[None, :]) & valid).sum(axis=0)
    tied = ((ranks == reported_ranks[None, :]) & valid).sum(axis=0)
    percentiles = np.divide(
        better + 0.5 * tied,
        finite_count,
        out=np.full(len(tracked_indices), np.nan),
        where=finite_count > 0,
    )
    posterior_median_ranks = np.full(len(tracked_indices), np.nan)
    rank_supported = finite_count > 0
    posterior_median_ranks[rank_supported] = np.nanmedian(
        ranks[:, rank_supported],
        axis=0,
    )

    frames = []
    metadata = _metadata_columns(result.metadata)
    for cohort_top_n in percentile_cohorts:
        include = (
            is_candidate
            & np.isfinite(percentiles)
            & np.isfinite(posterior_median_ranks)
            & (reported_ranks <= cohort_top_n)
        )
        cohort_indices = tracked_indices[include]
        cohort_percentiles = percentiles[include]
        cohort_reported_ranks = reported_ranks[include]
        cohort_median_ranks = posterior_median_ranks[include]
        if not len(cohort_indices):
            raise ValueError(f"No reported variants are available for top {cohort_top_n}.")
        order = np.argsort(cohort_percentiles, kind="stable")
        rows = _variant_rows(result.variants, cohort_indices[order])
        rows["posterior_median_sampled_rank"] = cohort_median_ranks[order]
        rows["median_sampled_rank_minus_reported_rank"] = (
            cohort_median_ranks[order] - cohort_reported_ranks[order]
        )
        rows["reported_vs_posterior_median_rank_spearman"] = spearmanr(
            cohort_reported_ranks,
            cohort_median_ranks,
        ).statistic
        rows.insert(0, "cohort_top_n", cohort_top_n)
        rows.insert(0, "ecdf", np.arange(1, len(rows) + 1) / len(rows))
        rows.insert(0, "reported_rank_sample_percentile", cohort_percentiles[order])
        rows.insert(0, "ecdf_to_uniform_area", ecdf_to_uniform_area(cohort_percentiles))
        for field, value in reversed(metadata.items()):
            rows.insert(0, field, value)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _rank_agreement(
    result: StabilityResult,
    selected: xr.Dataset,
) -> pd.DataFrame:
    reference_indices = np.asarray(
        selected["model_reference_variant_index"].values,
        dtype=np.int32,
    )
    rows = _variant_rows(result.variants, reference_indices)
    rows["model_reference_rank"] = np.arange(1, len(rows) + 1)
    rows["model_reference_value"] = np.asarray(
        selected["model_reference_score"].values,
        dtype=float,
    )
    rows["model_rank_minus_reported_rank"] = (
        rows["model_reference_rank"] - rows["reported_rank"]
    )
    rows["absolute_model_reported_rank_difference"] = rows[
        "model_rank_minus_reported_rank"
    ].abs()
    for field, value in reversed(_metadata_columns(result.metadata).items()):
        rows.insert(0, field, value)
    return rows


def _model_rank_intervals(
    result: StabilityResult,
    selected: xr.Dataset,
    *,
    main_k: int,
) -> pd.DataFrame:
    tracked_indices = np.asarray(
        selected["model_tracked_variant_index"].values,
        dtype=np.int32,
    )[:main_k]
    ranks = np.asarray(selected["model_tracked_rank"].values, dtype=float)[:, :main_k]
    reference_values = np.asarray(
        selected["model_reference_score"].values,
        dtype=float,
    )[:main_k]
    lower, median, upper = np.nanquantile(ranks, [0.05, 0.5, 0.95], axis=0)
    rows = _variant_rows(result.variants, tracked_indices)
    rows["model_reference_rank"] = np.arange(1, len(rows) + 1)
    rows["model_reference_value"] = reference_values
    rows["posterior_rank_lower"] = lower
    rows["posterior_median_rank"] = median
    rows["posterior_rank_upper"] = upper
    rows["rank_interval_width"] = upper - lower
    rows["rank_interval_leaves_top_k"] = upper > main_k
    rows["top_k_inclusion_probability"] = np.nanmean(ranks <= main_k, axis=0)
    rows["lower_rank_minus_model_reference"] = lower - rows["model_reference_rank"]
    rows["median_rank_minus_model_reference"] = median - rows["model_reference_rank"]
    rows["upper_rank_minus_model_reference"] = upper - rows["model_reference_rank"]
    rows.insert(0, "k", main_k)
    for field, value in reversed(_metadata_columns(result.metadata).items()):
        rows.insert(0, field, value)
    return rows


def _top_k_counts(
    selected_indices: np.ndarray,
    *,
    k: int,
    n_variants: int,
) -> np.ndarray:
    return np.bincount(selected_indices[:, :k].ravel(), minlength=n_variants)


def _top_k_curves(
    result: StabilityResult,
    selected: xr.Dataset,
    *,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    selected_indices = np.asarray(selected["variant_index"].values, dtype=np.int32)
    n_samples = selected_indices.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required for self-consistency.")
    n_variants = len(result.variants)
    reference_indices = np.asarray(
        selected["model_reference_variant_index"].values,
        dtype=np.int32,
    )
    reference_top_k_identifiable = bool(
        result.metadata.get("reference_top_k_identifiable", True)
    )
    metadata = _metadata_columns(result.metadata)
    rows = []
    for k in k_values:
        counts = _top_k_counts(selected_indices, k=k, n_variants=n_variants)
        probabilities = counts / n_samples
        reported_indices = (
            _reported_reference_indices(result.variants, k)
            if reference_top_k_identifiable
            else None
        )
        rows.append(
            {
                **metadata,
                "k": k,
                "sample_self_consistency": (
                    np.sum(counts * (counts - 1))
                    / (n_samples * (n_samples - 1) * k)
                ),
                "sample_vs_model_reference": probabilities[
                    reference_indices[:k]
                ].sum()
                / k,
                "sample_vs_reported": (
                    probabilities[reported_indices].sum() / k
                    if reported_indices is not None
                    else np.nan
                ),
                "reported_top_k_identifiable": reference_top_k_identifiable,
                "candidate_count_q_gt_0": int(np.sum(probabilities > 0)),
                "candidate_count_q_gt_0_01": int(np.sum(probabilities > 0.01)),
                "candidate_count_q_gt_0_05": int(np.sum(probabilities > 0.05)),
            }
        )
    return pd.DataFrame(rows)


def _top_k_candidates(
    result: StabilityResult,
    selected: xr.Dataset,
    *,
    main_k: int,
    candidate_probability: float,
) -> pd.DataFrame:
    selected_indices = np.asarray(selected["variant_index"].values, dtype=np.int32)
    counts = _top_k_counts(
        selected_indices,
        k=main_k,
        n_variants=len(result.variants),
    )
    probabilities = counts / len(selected_indices)
    reference_indices = np.asarray(
        selected["model_reference_variant_index"].values,
        dtype=np.int32,
    )
    model_ranks = np.full(len(result.variants), np.nan)
    model_ranks[reference_indices] = np.arange(1, len(reference_indices) + 1)
    reference_top_k_identifiable = bool(
        result.metadata.get("reference_top_k_identifiable", True)
    )
    reported_top = np.zeros(len(result.variants), dtype=bool)
    if reference_top_k_identifiable:
        reported_indices = _reported_reference_indices(result.variants, main_k)
        reported_top[reported_indices] = True
    model_top = np.isfinite(model_ranks) & (model_ranks <= main_k)
    include = (probabilities > candidate_probability) | reported_top | model_top
    indices = np.flatnonzero(include)
    rows = _variant_rows(result.variants, indices)
    rows["k"] = main_k
    rows["top_k_count"] = counts[indices]
    rows["top_k_probability"] = probabilities[indices]
    rows["model_reference_rank"] = model_ranks[indices]
    if reference_top_k_identifiable:
        rows["is_reported_top_k"] = reported_top[indices]
    else:
        rows["is_reported_top_k"] = pd.array(
            [pd.NA] * len(rows),
            dtype="boolean",
        )
    rows["reported_top_k_identifiable"] = reference_top_k_identifiable
    rows["is_model_reference_top_k"] = model_top[indices]
    rows["candidate_probability_threshold"] = candidate_probability
    rows = rows.sort_values(
        ["top_k_probability", "model_reference_rank", "reported_rank", "variant_index"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    for field, value in reversed(_metadata_columns(result.metadata).items()):
        rows.insert(0, field, value)
    return rows


def build_paper_tables(
    result: ResultLike,
    *,
    score: str = "log_mean_growth",
    main_k: int = 100,
    k_values: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    percentile_cohorts: tuple[int, ...] = (100, 1000),
    candidate_probability: float = 0.01,
) -> dict[str, pd.DataFrame]:
    """Build the six tidy paper tables for one compact analysis result."""
    loaded = _load_result(result)
    resolved_k = tuple(sorted(set(int(k) for k in k_values)))
    selected = _require_paper_result(
        loaded,
        score=score,
        main_k=main_k,
        k_values=resolved_k,
    )
    if not 0 <= candidate_probability <= 1:
        raise ValueError("candidate_probability must be between 0 and 1.")
    tables = {
        "inventory": _inventory(loaded, score=score),
        "rank_percentiles": _rank_percentiles(
            loaded,
            selected,
            percentile_cohorts=percentile_cohorts,
        ),
        "rank_agreement": _rank_agreement(loaded, selected),
        "model_rank_intervals": _model_rank_intervals(
            loaded,
            selected,
            main_k=main_k,
        ),
        "top_k_curves": _top_k_curves(
            loaded,
            selected,
            k_values=resolved_k,
        ),
        "top_k_candidates": _top_k_candidates(
            loaded,
            selected,
            main_k=main_k,
            candidate_probability=candidate_probability,
        ),
    }
    for table in tables.values():
        if "score" not in table:
            table.insert(
                min(6, len(table.columns)),
                "score",
                score,
            )
    return tables
