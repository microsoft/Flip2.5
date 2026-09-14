"""Top-K selection stability calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _selection_size(selection: xr.Dataset, k: int | None) -> int:
    retained_k = selection.sizes["selection_rank"]
    selected_k = retained_k if k is None else k
    if not 1 <= selected_k <= retained_k:
        raise ValueError(f"k must be between 1 and the retained K={retained_k}.")
    return selected_k


def select_top_k(
    scores: xr.DataArray,
    *,
    k: int,
    higher_is_better: bool = True,
) -> xr.Dataset:
    """Select each sample's ordered top-K variants from a score matrix."""
    if set(scores.dims) != {"sample", "variant"} or scores.ndim != 2:
        raise ValueError("scores must have exactly the dimensions 'sample' and 'variant'.")
    if not 1 <= k <= scores.sizes["variant"]:
        raise ValueError(f"k must be between 1 and the {scores.sizes['variant']} variants.")

    values = np.asarray(scores.transpose("sample", "variant").values)
    finite_counts = np.isfinite(values).sum(axis=1)
    if np.any(finite_counts < k):
        failed_sample = int(np.flatnonzero(finite_counts < k)[0])
        raise ValueError(f"Sample {failed_sample} has fewer than k={k} finite variant scores.")

    ordering_values = values if higher_is_better else -values
    ordering_values = np.where(np.isfinite(ordering_values), ordering_values, -np.inf)
    indices = np.argpartition(ordering_values, -k, axis=1)[:, -k:]
    selected_ordering_values = np.take_along_axis(ordering_values, indices, axis=1)
    order = np.argsort(selected_ordering_values, axis=1)[:, ::-1]
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int32)
    selected_scores = np.take_along_axis(values, indices, axis=1).astype(np.float32)

    return xr.Dataset(
        {
            "variant_index": (("sample", "selection_rank"), indices),
            "score": (("sample", "selection_rank"), selected_scores),
        },
        coords={
            "sample": scores.coords.get("sample", np.arange(len(values))),
            "selection_rank": np.arange(1, k + 1, dtype=np.int32),
        },
        attrs={
            "higher_is_better": str(higher_is_better).lower(),
            "retained_k": k,
        },
    )


def summarize_inclusion(
    selection: xr.Dataset,
    variants: pd.DataFrame,
    *,
    k: int | None = None,
) -> pd.DataFrame:
    """Summarize variant inclusion probabilities at K or below."""
    selected_k = _selection_size(selection, k)
    indices = selection["variant_index"].isel(selection_rank=slice(0, selected_k))
    index_values = np.asarray(indices.values)
    if index_values.size == 0:
        raise ValueError("selection must contain at least one sample.")
    if index_values.min() < 0 or index_values.max() >= len(variants):
        raise ValueError("selection contains a variant index outside the variant table.")

    counts = np.bincount(index_values.ravel(), minlength=len(variants))
    selected = np.flatnonzero(counts)
    summary = variants.iloc[selected].copy()
    summary.insert(0, "variant_index", selected)
    summary["k"] = selected_k
    summary["top_k_count"] = counts[selected]
    summary["top_k_probability"] = counts[selected] / selection.sizes["sample"]
    return summary.sort_values(
        ["top_k_probability", "variant_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def pairwise_overlap(
    selection: xr.Dataset,
    *,
    k: int | None = None,
    n_pairs: int = 10_000,
    seed: int = 23,
) -> pd.DataFrame:
    """Estimate overlap between reproducible pairs of distinct samples."""
    selected_k = _selection_size(selection, k)
    indices = np.asarray(
        selection["variant_index"].isel(selection_rank=slice(0, selected_k)).values
    )
    if len(indices) < 2:
        raise ValueError("Pairwise overlap requires at least two samples.")
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1.")

    rng = np.random.default_rng(seed)
    left = rng.integers(len(indices), size=n_pairs)
    right = rng.integers(len(indices) - 1, size=n_pairs)
    right += right >= left
    intersection = np.fromiter(
        (
            np.intersect1d(indices[left_sample], indices[right_sample]).size
            for left_sample, right_sample in zip(left, right)
        ),
        dtype=np.int32,
        count=n_pairs,
    )
    return pd.DataFrame(
        {
            "comparison": "sample_vs_sample",
            "left_sample": left,
            "right_sample": right,
            "k": selected_k,
            "intersection": intersection,
            "recall": intersection / selected_k,
            "jaccard": intersection / (2 * selected_k - intersection),
        }
    )


def reference_overlap(
    selection: xr.Dataset,
    reference_fitness: np.ndarray | pd.Series,
    *,
    k: int | None = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compare each sample's top-K set with the fixed reference top-K set."""
    selected_k = _selection_size(selection, k)
    indices = np.asarray(
        selection["variant_index"].isel(selection_rank=slice(0, selected_k)).values
    )
    reference = np.asarray(reference_fitness, dtype=float)
    if len(reference) <= indices.max():
        raise ValueError("reference_fitness does not cover every selected variant.")
    if np.isfinite(reference).sum() < selected_k:
        raise ValueError(f"Fewer than k={selected_k} variants have finite reference fitness.")

    ordering_values = reference if higher_is_better else -reference
    ordering_values = np.where(np.isfinite(ordering_values), ordering_values, -np.inf)
    reference_indices = np.argpartition(ordering_values, -selected_k)[-selected_k:]
    intersection = np.isin(indices, reference_indices).sum(axis=1)
    return pd.DataFrame(
        {
            "comparison": "sample_vs_reference",
            "sample": np.arange(len(indices)),
            "k": selected_k,
            "intersection": intersection,
            "recall": intersection / selected_k,
            "jaccard": intersection / (2 * selected_k - intersection),
        }
    )
