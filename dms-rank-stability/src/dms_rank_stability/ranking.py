"""Bounded-memory rank analysis over validated sampled score matrices."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from .data import SampledScores
from .results import IDENTITY_FIELDS, RankResult
from .streaming import (
    accumulate_tracked_ranks,
    empty_top_k,
    empty_tracked_ranks,
    merge_top_k,
    select_reference_top_k,
)


def _score_block(
    scores: SampledScores,
    start: int,
    stop: int,
    sample_positions: np.ndarray | None = None,
) -> np.ndarray:
    selected = scores.score.isel(variant=slice(start, stop))
    if sample_positions is not None:
        selected = selected.isel(sample=sample_positions)
    return np.asarray(
        selected.values,
        dtype=np.float32,
    )


def _selection_blocks(
    scores: SampledScores,
    variant_chunk_size: int,
) -> tuple[tuple[int, int], ...]:
    if "selection_block" not in scores.dataset.coords:
        return tuple(
            (start, min(start + variant_chunk_size, len(scores.variants)))
            for start in range(0, len(scores.variants), variant_chunk_size)
        )
    labels = np.asarray(scores.dataset["selection_block"].values, dtype=np.int64)
    return tuple(
        (int(positions[0]), int(positions[-1]) + 1)
        for label in range(int(labels.max()) + 1)
        if len(positions := np.flatnonzero(labels == label))
    )


def eligible_sample_positions(
    scores: SampledScores,
    *,
    retain_k: int,
    variant_chunk_size: int,
) -> np.ndarray:
    """Return samples with enough finite candidate scores for top-k selection."""
    if retain_k < 1 or variant_chunk_size < 1:
        raise ValueError("retain_k and variant_chunk_size must be positive.")
    candidate = scores.variants["is_candidate"].to_numpy(dtype=bool)
    candidate_indices = np.flatnonzero(candidate).astype(np.int32)
    if retain_k > len(candidate_indices):
        raise ValueError("retain_k cannot exceed the candidate count.")
    finite_counts = np.zeros(scores.score.sizes["sample"], dtype=np.int32)
    for start, stop in _selection_blocks(scores, variant_chunk_size):
        first = int(np.searchsorted(candidate_indices, start, side="left"))
        last = int(np.searchsorted(candidate_indices, stop, side="left"))
        block_indices = candidate_indices[first:last]
        if len(block_indices):
            block = _score_block(scores, start, stop)[:, block_indices - start]
            finite_counts += np.isfinite(block).sum(axis=1).astype(np.int32)
    return np.flatnonzero(finite_counts >= retain_k).astype(np.int64)


def _external_reference_indices(
    scores: SampledScores,
    *,
    column: str | None,
    count: int,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int32)
    if column is None or column not in scores.variants:
        raise ValueError("An external rank column is required when external_top_n is positive.")
    ranks = scores.variants[column].to_numpy(dtype=float)
    candidate = scores.variants["is_candidate"].to_numpy(dtype=bool)
    eligible = np.flatnonzero(candidate & np.isfinite(ranks))
    if len(eligible) < count:
        raise ValueError(f"Fewer than external_top_n={count} candidates have finite ranks.")
    order = np.lexsort((eligible, ranks[eligible]))
    return eligible[order[:count]].astype(np.int32)


def _collect_scores(
    scores: SampledScores,
    indices: np.ndarray,
    *,
    variant_chunk_size: int,
    sample_positions: np.ndarray,
) -> np.ndarray:
    values = np.full(
        (len(sample_positions), len(indices)),
        np.nan,
        dtype=np.float32,
    )
    for start, stop in _selection_blocks(scores, variant_chunk_size):
        first = int(np.searchsorted(indices, start, side="left"))
        last = int(np.searchsorted(indices, stop, side="left"))
        if first == last:
            continue
        block = _score_block(scores, start, stop, sample_positions)
        values[:, first:last] = block[:, indices[first:last] - start]
    return values


def analyze_scores(
    scores: SampledScores,
    *,
    metadata: Mapping[str, object],
    retain_k: int,
    decision_k: int,
    external_top_n: int,
    model_top_n: int,
    external_rank_column: str | None = "reported_rank",
    challenger_probability: float = 0.01,
    higher_is_better: bool = True,
    variant_chunk_size: int = 4096,
    min_reference_samples: int = 1,
) -> RankResult:
    """Analyze one score matrix without loading its full variant axis into memory."""
    scores.validate()
    missing_identity = [name for name in IDENTITY_FIELDS if name not in metadata]
    if missing_identity:
        raise ValueError(f"Run metadata are missing fields: {missing_identity}")
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be boolean.")
    if variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive.")
    candidate = scores.variants["is_candidate"].to_numpy(dtype=bool)
    candidate_indices = np.flatnonzero(candidate).astype(np.int32)
    if not 1 <= decision_k <= model_top_n <= retain_k <= len(candidate_indices):
        raise ValueError(
            "Rank counts must satisfy 1 <= decision_k <= model_top_n <= "
            "retain_k <= candidate count."
        )
    if not 0 <= external_top_n <= len(candidate_indices):
        raise ValueError("external_top_n must lie between zero and the candidate count.")
    if not 0 < challenger_probability <= 1:
        raise ValueError("challenger_probability must lie in (0, 1].")
    sample_positions = eligible_sample_positions(
        scores,
        retain_k=retain_k,
        variant_chunk_size=variant_chunk_size,
    )
    source_sample_count = scores.score.sizes["sample"]
    n_samples = len(sample_positions)
    if n_samples < 2:
        raise ValueError(
            f"Fewer than two samples contain retain_k={retain_k} finite candidates."
        )
    if not 1 <= min_reference_samples <= n_samples:
        raise ValueError("min_reference_samples must lie on the eligible sample axis.")
    external_indices = _external_reference_indices(
        scores,
        column=external_rank_column,
        count=external_top_n,
    )

    top_indices, top_scores = empty_top_k(
        n_samples,
        retain_k,
        higher_is_better=higher_is_better,
    )
    reference_scores = np.full(len(scores.variants), np.nan, dtype=np.float32)
    reference_counts = np.zeros(len(scores.variants), dtype=np.int32)
    for start, stop in _selection_blocks(scores, variant_chunk_size):
        first = int(np.searchsorted(candidate_indices, start, side="left"))
        last = int(np.searchsorted(candidate_indices, stop, side="left"))
        block_indices = candidate_indices[first:last]
        if not len(block_indices):
            continue
        block = _score_block(
            scores,
            start,
            stop,
            sample_positions,
        )[:, block_indices - start]
        top_indices, top_scores = merge_top_k(
            top_indices,
            top_scores,
            block_indices,
            block,
            higher_is_better=higher_is_better,
        )
        finite = np.isfinite(block)
        finite_count = finite.sum(axis=0).astype(np.int32)
        reference_counts[block_indices] = finite_count
        usable = finite_count >= min_reference_samples
        if usable.any():
            reference_scores[block_indices[usable]] = np.nanmedian(
                block[:, usable],
                axis=0,
            ).astype(np.float32)
    if not np.isfinite(top_scores).all():
        raise ValueError("Fewer than retain_k candidates have finite scores in a sample.")

    reference_indices, _ = select_reference_top_k(
        reference_scores[None, :],
        k=retain_k,
        higher_is_better=higher_is_better,
    )
    counts = np.bincount(
        top_indices[:, :decision_k].ravel(),
        minlength=len(scores.variants),
    ).astype(np.int32)
    probabilities = counts.astype(np.float64) / n_samples
    model_decision_indices = reference_indices[:decision_k]
    challengers = candidate_indices[probabilities[candidate_indices] >= challenger_probability]
    challengers = challengers[~np.isin(challengers, model_decision_indices)]
    model_indices = reference_indices[:model_top_n]
    cohort_indices = np.unique(
        np.concatenate([external_indices, model_indices, challengers])
    ).astype(np.int32)

    cohort_scores = _collect_scores(
        scores,
        cohort_indices,
        variant_chunk_size=variant_chunk_size,
        sample_positions=sample_positions,
    )
    cohort_ranks = empty_tracked_ranks(n_samples, len(cohort_indices))
    for start in range(0, len(scores.variants), variant_chunk_size):
        stop = min(start + variant_chunk_size, len(scores.variants))
        first = int(np.searchsorted(candidate_indices, start, side="left"))
        last = int(np.searchsorted(candidate_indices, stop, side="left"))
        block_indices = candidate_indices[first:last]
        if not len(block_indices):
            continue
        block = _score_block(
            scores,
            start,
            stop,
            sample_positions,
        )[:, block_indices - start]
        accumulate_tracked_ranks(
            cohort_ranks,
            block,
            cohort_scores,
            higher_is_better=higher_is_better,
        )
    cohort_ranks[~np.isfinite(cohort_scores)] = np.nan

    coordinates: dict[str, object] = {
        "sample": np.arange(n_samples, dtype=np.int32),
        "variant": np.arange(len(scores.variants), dtype=np.int32),
        "selection_rank": np.arange(1, retain_k + 1, dtype=np.int32),
        "reference_rank": np.arange(1, retain_k + 1, dtype=np.int32),
        "cohort": np.arange(len(cohort_indices), dtype=np.int32),
        "cohort_variant_index": ("cohort", cohort_indices),
    }
    for name in ("source_sample_id", "source_chain", "source_draw"):
        if name in scores.dataset.coords:
            coordinates[name] = (
                "sample",
                np.asarray(scores.dataset[name].values)[sample_positions],
            )
    stability = xr.Dataset(
        {
            "selection_variant_index": (
                ("sample", "selection_rank"),
                top_indices.astype(np.int32, copy=False),
            ),
            "selection_score": (
                ("sample", "selection_rank"),
                top_scores.astype(np.float32, copy=False),
            ),
            "reference_score": ("variant", reference_scores),
            "reference_n_finite_samples": ("variant", reference_counts),
            "reference_variant_index": (
                "reference_rank",
                reference_indices.astype(np.int32, copy=False),
            ),
            "top_k_count": ("variant", counts),
            "top_k_probability": ("variant", probabilities),
            "cohort_score": (("sample", "cohort"), cohort_scores),
            "cohort_rank": (("sample", "cohort"), cohort_ranks),
            "is_external_reference": (
                "cohort",
                np.isin(cohort_indices, external_indices),
            ),
            "is_model_reference": (
                "cohort",
                np.isin(cohort_indices, model_indices),
            ),
            "is_challenger": ("cohort", np.isin(cohort_indices, challengers)),
        },
        coords=coordinates,
    )
    result_metadata = {
        **dict(metadata),
        "higher_is_better": higher_is_better,
        "n_samples": n_samples,
        "n_source_samples": source_sample_count,
        "n_excluded_samples": source_sample_count - n_samples,
        "n_variants": len(scores.variants),
        "n_candidates": len(candidate_indices),
        "retained_k": retain_k,
        "decision_k": decision_k,
        "external_top_n": external_top_n,
        "model_top_n": model_top_n,
        "challenger_probability": challenger_probability,
        "cohort_size": len(cohort_indices),
        "challenger_count": len(challengers),
        "variant_chunk_size": variant_chunk_size,
        "min_reference_samples": min_reference_samples,
        "external_rank_column": external_rank_column,
    }
    result = RankResult(result_metadata, scores.variants.copy(), stability)
    result.validate()
    return result
