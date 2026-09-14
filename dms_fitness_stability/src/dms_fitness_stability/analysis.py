"""End-to-end stability analysis from aligned score samples and raw counts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .identity import AnalysisIdentity
from .results import StabilityResult
from .stability import (
    pairwise_overlap,
    reference_overlap,
    select_top_k,
    summarize_inclusion,
)

RESULT_SCHEMA_VERSION = 2


def _top_k_values(top_k: int | Sequence[int]) -> tuple[int, ...]:
    values = (int(top_k),) if isinstance(top_k, (int, np.integer)) else tuple(top_k)
    if not values or any(not isinstance(value, (int, np.integer)) or value < 1 for value in values):
        raise ValueError("top_k must contain positive integers.")
    return tuple(sorted(set(int(value) for value in values)))


def analyze_stability(
    scores: xr.DataArray,
    variants: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    score_name: str,
    top_k: int | Sequence[int],
    higher_is_better: bool = True,
    n_pairs: int = 10_000,
    seed: int = 23,
) -> StabilityResult:
    """Calculate and package top-K stability for one fitted score."""
    required_columns = {
        "variant_id",
        "reported_fitness",
        "reference_fitness",
        "reference_fitness_source",
    }
    missing = required_columns - set(variants.columns)
    if missing:
        raise ValueError(f"Variant table is missing required columns: {sorted(missing)}")
    if variants["variant_id"].duplicated().any():
        raise ValueError("variant_id values must be unique after pooling counts.")
    if "variant" not in scores.coords:
        raise ValueError("scores must have a 'variant' coordinate for alignment.")
    fitted_ids = np.asarray(scores.coords["variant"].values)
    expected_ids = variants["variant_id"].to_numpy()
    if not np.array_equal(fitted_ids, expected_ids):
        raise ValueError("Fitted variant IDs and count-table variant IDs are not aligned.")

    top_k_values = _top_k_values(top_k)
    if top_k_values[-1] > len(variants):
        raise ValueError("top_k cannot exceed the number of variants.")
    reference_sources = variants["reference_fitness_source"].drop_duplicates().tolist()
    if len(reference_sources) != 1:
        raise ValueError("Every variant must use the same reference fitness source.")
    reference_source = reference_sources[0]

    selection = select_top_k(
        scores,
        k=top_k_values[-1],
        higher_is_better=higher_is_better,
    )
    selection.attrs.update(
        {
            **identity.metadata(),
            "fit_method": identity.inference_method,
            "score_name": score_name,
        }
    )

    summary_frames = []
    overlap_frames = []
    for selected_k in top_k_values:
        summary = summarize_inclusion(selection, variants, k=selected_k)
        summary.insert(0, "score", score_name)
        summary.insert(0, "fit_method", identity.inference_method)
        for field, value in reversed(identity.metadata().items()):
            summary.insert(0, field, value)
        summary_frames.append(summary)

        pairwise = pairwise_overlap(
            selection,
            k=selected_k,
            n_pairs=n_pairs,
            seed=seed,
        )
        reference = reference_overlap(
            selection,
            variants["reference_fitness"],
            k=selected_k,
        )
        overlap = pd.concat([pairwise, reference], ignore_index=True)
        overlap.insert(0, "score", score_name)
        overlap.insert(0, "fit_method", identity.inference_method)
        for field, value in reversed(identity.metadata().items()):
            overlap.insert(0, field, value)
        overlap_frames.append(overlap)

    warning_messages = []
    if reference_source == "count_ratio_fallback":
        warning_messages.append(
            "Reported fitness was unavailable; pooled output/input count ratio "
            "was used as the ranking reference."
        )
    metadata: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **identity.metadata(),
        "fit_method": identity.inference_method,
        "score": score_name,
        "higher_is_better": higher_is_better,
        "n_samples": scores.sizes["sample"],
        "n_variants": scores.sizes["variant"],
        "top_k": list(top_k_values),
        "seed": seed,
        "reference_fitness_source": reference_source,
        "reported_fitness_available": reference_source == "reported",
        "warnings": warning_messages,
    }
    return StabilityResult(
        metadata=metadata,
        variants=variants.copy(),
        selection=selection,
        summary=pd.concat(summary_frames, ignore_index=True),
        overlaps=pd.concat(overlap_frames, ignore_index=True),
    )
