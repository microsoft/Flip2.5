"""Bounded-memory primitives for large fitness landscapes."""

from __future__ import annotations

import numpy as np


def select_reference_top_k(
    scores: np.ndarray,
    *,
    k: int,
    higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Select variants by median sample score with stable index ties."""
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must be sample-by-variant.")
    if not 1 <= k <= values.shape[1]:
        raise ValueError(f"k must be between 1 and the {values.shape[1]} variants.")

    finite = np.isfinite(values)
    reference_scores = np.full(values.shape[1], np.nan, dtype=np.float32)
    usable = finite.any(axis=0)
    if usable.sum() < k:
        raise ValueError(f"Fewer than k={k} variants have finite sample scores.")
    reference_scores[usable] = np.nanmedian(values[:, usable], axis=0)
    variant_indices = np.arange(values.shape[1], dtype=np.int32)
    ordering_scores = -reference_scores if higher_is_better else reference_scores
    order = np.lexsort((variant_indices, ordering_scores))
    selected = order[:k].astype(np.int32, copy=False)
    return selected, reference_scores[selected]


def empty_top_k(
    n_samples: int,
    k: int,
    *,
    higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an empty running top-K frontier."""
    if n_samples < 1 or k < 1:
        raise ValueError("n_samples and k must be positive.")
    return (
        np.full((n_samples, k), -1, dtype=np.int32),
        np.full(
            (n_samples, k),
            -np.inf if higher_is_better else np.inf,
            dtype=np.float32,
        ),
    )


def merge_top_k(
    current_indices: np.ndarray,
    current_scores: np.ndarray,
    block_indices: np.ndarray,
    block_scores: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge one score block into an exact ordered sample-by-K frontier."""
    if current_indices.shape != current_scores.shape:
        raise ValueError("Current top-K indices and scores must have the same shape.")
    if block_scores.ndim != 2 or block_scores.shape[0] != current_scores.shape[0]:
        raise ValueError("block_scores must be sample-by-variant.")
    block_indices = np.asarray(block_indices, dtype=np.int32)
    if block_indices.ndim != 1 or len(block_indices) != block_scores.shape[1]:
        raise ValueError("block_indices must identify every block score column.")

    k = current_scores.shape[1]
    finite_block_scores = np.where(
        np.isfinite(block_scores),
        block_scores,
        -np.inf if higher_is_better else np.inf,
    )
    expanded_indices = np.broadcast_to(block_indices[None, :], finite_block_scores.shape)
    combined_scores = np.concatenate([current_scores, finite_block_scores], axis=1)
    combined_indices = np.concatenate([current_indices, expanded_indices], axis=1)
    if higher_is_better:
        selected = np.argpartition(combined_scores, -k, axis=1)[:, -k:]
    else:
        selected = np.argpartition(combined_scores, k - 1, axis=1)[:, :k]
    selected_scores = np.take_along_axis(combined_scores, selected, axis=1)
    selected_indices = np.take_along_axis(combined_indices, selected, axis=1)
    order = np.argsort(selected_scores, axis=1)
    if higher_is_better:
        order = order[:, ::-1]
    return (
        np.take_along_axis(selected_indices, order, axis=1).astype(np.int32),
        np.take_along_axis(selected_scores, order, axis=1).astype(np.float32),
    )


def empty_tracked_ranks(n_samples: int, n_tracked: int) -> np.ndarray:
    """Create average-rank accumulators for a fixed tracked cohort."""
    if n_samples < 1 or n_tracked < 0:
        raise ValueError("n_samples must be positive and n_tracked nonnegative.")
    return np.full((n_samples, n_tracked), 0.5, dtype=np.float32)


def accumulate_tracked_ranks(
    ranks: np.ndarray,
    block_scores: np.ndarray,
    tracked_scores: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> None:
    """Add one score block's contribution to exact average ranks."""
    if ranks.shape != tracked_scores.shape:
        raise ValueError("ranks and tracked_scores must have the same shape.")
    if block_scores.ndim != 2 or block_scores.shape[0] != ranks.shape[0]:
        raise ValueError("block_scores must share the tracked sample dimension.")

    sorted_scores = np.sort(block_scores, axis=1)
    for sample_index in range(len(sorted_scores)):
        finite_block = sorted_scores[sample_index]
        finite_block = finite_block[np.isfinite(finite_block)]
        tracked = tracked_scores[sample_index]
        finite_tracked = np.isfinite(tracked)
        if not len(finite_block) or not finite_tracked.any():
            continue
        tracked_values = tracked[finite_tracked]
        left = np.searchsorted(finite_block, tracked_values, side="left")
        right = np.searchsorted(finite_block, tracked_values, side="right")
        strictly_better = len(finite_block) - right if higher_is_better else left
        ranks[sample_index, finite_tracked] += strictly_better + 0.5 * (right - left)
