"""Mechanism features linking count support, score uncertainty, and rank resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy.stats import rankdata

from .streaming import accumulate_tracked_ranks, empty_tracked_ranks


def _count_array(values: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite non-negative counts.")
    return array


def _validate_pseudocount(counts: np.ndarray, pseudocount: float) -> None:
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and non-negative.")
    if pseudocount == 0 and np.any(counts == 0):
        raise ValueError("pseudocount must be positive when counts contain zeros.")


def _log_proportions(counts: np.ndarray, *, pseudocount: float) -> np.ndarray:
    _validate_pseudocount(counts, pseudocount)
    depth = counts.sum(axis=-1, keepdims=True)
    denominator = depth + pseudocount * counts.shape[-1]
    if np.any(denominator <= 0):
        raise ValueError("Every count library must have positive total support.")
    return np.log((counts + pseudocount) / denominator)


def _weighted_heterogeneity(
    values: np.ndarray,
    variances: np.ndarray,
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != variances.shape:
        raise ValueError("values and variances must have the same shape.")
    weights = np.divide(
        1.0,
        variances,
        out=np.zeros_like(variances, dtype=float),
        where=np.isfinite(variances) & (variances > 0),
    )
    weight_sum = weights.sum(axis=axis)
    if np.any(weight_sum <= 0):
        raise ValueError("Every variant must have positive precision weight.")
    weighted_mean = (weights * values).sum(axis=axis) / weight_sum
    residual = values - np.expand_dims(weighted_mean, axis=axis)
    heterogeneity = (weights * residual**2).sum(axis=axis)
    return weighted_mean, heterogeneity


def _read_hmc_samples(
    variable,
    sample_ids: np.ndarray,
    *,
    variant_start: int,
    variant_stop: int,
) -> np.ndarray:
    draws_per_chain = variable.shape[1]
    total_samples = variable.shape[0] * draws_per_chain
    if np.array_equal(sample_ids, np.arange(total_samples, dtype=np.int64)):
        values = np.asarray(variable[:, :, variant_start:variant_stop])
        return values.reshape(total_samples, variant_stop - variant_start)

    chains = sample_ids // draws_per_chain
    draws = sample_ids % draws_per_chain
    result = np.empty(
        (len(sample_ids), variant_stop - variant_start),
        dtype=variable.dtype,
    )
    for chain in np.unique(chains):
        positions = np.flatnonzero(chains == chain)
        chain_draws = draws[positions]
        order = np.argsort(chain_draws)
        sorted_positions = positions[order]
        result[sorted_positions] = variable[
            int(chain),
            chain_draws[order],
            variant_start:variant_stop,
        ]
    return result


def _transform_score(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform == "positive":
            values = np.log(values)
        elif transform == "inverse":
            values = -np.log(values)
        elif transform not in {"identity", "log"}:
            raise ValueError(
                "transform must be 'identity', 'log', 'positive', or 'inverse'."
            )
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32, copy=False)


def _read_proxy_samples(
    variable,
    sample_ids: np.ndarray,
    *,
    variant_start: int,
    variant_stop: int,
) -> np.ndarray:
    total_samples = variable.shape[0]
    if np.array_equal(sample_ids, np.arange(total_samples, dtype=np.int64)):
        return np.asarray(variable[:, variant_start:variant_stop])

    order = np.argsort(sample_ids)
    result = np.empty(
        (len(sample_ids), variant_stop - variant_start),
        dtype=variable.dtype,
    )
    result[order] = variable[
        sample_ids[order],
        variant_start:variant_stop,
    ]
    return result


def _scan_fixed_cohort_blocks(
    *,
    n_source_samples: int,
    n_source_variants: int,
    source_variant_indices: np.ndarray,
    sample_ids: np.ndarray,
    candidate_variant_indices: np.ndarray,
    cohort_variant_indices: np.ndarray,
    variant_chunk_size: int,
    read_samples: Callable[[np.ndarray, int, int], np.ndarray],
    min_reference_samples: int = 1,
    higher_is_better: bool = True,
) -> Mapping[str, np.ndarray]:
    source_indices = np.asarray(source_variant_indices, dtype=np.int64)
    if source_indices.ndim != 1 or not len(source_indices):
        raise ValueError("source_variant_indices must be a non-empty one-dimensional array.")
    if np.any(np.diff(source_indices) <= 0):
        raise ValueError("source_variant_indices must be strictly increasing.")
    if source_indices[0] < 0 or source_indices[-1] >= n_source_variants:
        raise ValueError("source_variant_indices lie outside the sampled score target.")
    candidate_rows = np.asarray(candidate_variant_indices, dtype=np.int32)
    if (
        candidate_rows.ndim != 1
        or not len(candidate_rows)
        or np.any(np.diff(candidate_rows) <= 0)
        or candidate_rows[0] < 0
        or candidate_rows[-1] >= len(source_indices)
    ):
        raise ValueError(
            "candidate_variant_indices must be unique increasing rows in the variant table."
        )
    candidate_source_indices = source_indices[candidate_rows]
    cohort = np.asarray(cohort_variant_indices, dtype=np.int32)
    if cohort.ndim != 1 or not len(cohort) or len(np.unique(cohort)) != len(cohort):
        raise ValueError("cohort_variant_indices must be non-empty and unique.")
    if cohort.min() < 0 or cohort.max() >= len(source_indices):
        raise ValueError("cohort_variant_indices lie outside the variant table.")
    if not np.isin(cohort, candidate_rows).all():
        raise ValueError("Every cohort variant must be an eligible ranking candidate.")
    samples = np.asarray(sample_ids, dtype=np.int64)
    if (
        samples.ndim != 1
        or not len(samples)
        or len(np.unique(samples)) != len(samples)
        or samples.min() < 0
        or samples.max() >= n_source_samples
    ):
        raise ValueError("sample_ids must be unique valid source-sample indices.")

    if variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive.")
    if not 1 <= min_reference_samples <= len(samples):
        raise ValueError(
            "min_reference_samples must be between one and the selected sample count."
        )
    block_starts = range(0, n_source_variants, variant_chunk_size)
    cohort_source_indices = source_indices[cohort]
    tracked_score = np.full((len(samples), len(cohort)), np.nan, dtype=np.float32)

    target_blocks = np.unique(
        (cohort_source_indices // variant_chunk_size) * variant_chunk_size
    )
    for variant_start in target_blocks:
        variant_stop = min(variant_start + variant_chunk_size, n_source_variants)
        cohort_columns = np.flatnonzero(
            (cohort_source_indices >= variant_start)
            & (cohort_source_indices < variant_stop)
        )
        values = read_samples(
            samples,
            int(variant_start),
            int(variant_stop),
        )
        expected_shape = (len(samples), int(variant_stop - variant_start))
        if values.shape != expected_shape:
            raise ValueError(
                f"Score block reader returned {values.shape}, expected {expected_shape}."
            )
        local_indices = cohort_source_indices[cohort_columns] - variant_start
        tracked_score[:, cohort_columns] = values[:, local_indices]

    n_finite_samples = np.isfinite(tracked_score).sum(axis=0)
    if np.any(n_finite_samples < 2):
        raise ValueError("Every tracked candidate requires at least two finite scores.")
    tracked_rank = empty_tracked_ranks(len(samples), len(cohort))
    tracked_reference_score = np.nanmedian(tracked_score, axis=0).astype(np.float32)
    candidate_reference_score = np.full(len(candidate_rows), np.nan, dtype=np.float32)
    candidate_n_finite_samples = np.zeros(len(candidate_rows), dtype=np.int32)
    for variant_start in block_starts:
        variant_stop = min(variant_start + variant_chunk_size, n_source_variants)
        candidate_start = int(
            np.searchsorted(candidate_source_indices, variant_start, side="left")
        )
        candidate_stop = int(
            np.searchsorted(candidate_source_indices, variant_stop, side="left")
        )
        if candidate_start == candidate_stop:
            continue
        values = read_samples(samples, variant_start, variant_stop)
        expected_shape = (len(samples), variant_stop - variant_start)
        if values.shape != expected_shape:
            raise ValueError(
                f"Score block reader returned {values.shape}, expected {expected_shape}."
            )
        values = values[
            :,
            candidate_source_indices[candidate_start:candidate_stop] - variant_start,
        ]
        accumulate_tracked_ranks(
            tracked_rank,
            values,
            tracked_score,
            higher_is_better=higher_is_better,
        )
        finite_count = np.isfinite(values).sum(axis=0)
        candidate_n_finite_samples[candidate_start:candidate_stop] = finite_count
        usable = finite_count >= min_reference_samples
        block_reference = np.full(values.shape[1], np.nan, dtype=np.float32)
        block_reference[usable] = np.nanmedian(values[:, usable], axis=0)
        candidate_reference_score[candidate_start:candidate_stop] = block_reference

    tracked_rank[~np.isfinite(tracked_score)] = np.nan
    sortable_reference_score = np.where(
        np.isfinite(candidate_reference_score),
        candidate_reference_score,
        -np.inf if higher_is_better else np.inf,
    )
    ordering_scores = (
        -sortable_reference_score
        if higher_is_better
        else sortable_reference_score
    )
    reference_positions = np.lexsort((candidate_rows, ordering_scores)).astype(
        np.int32,
        copy=False,
    )
    reference_order = candidate_rows[reference_positions]
    reference_rank = np.full(len(source_indices), -1, dtype=np.int32)
    reference_rank[reference_order] = np.arange(
        1,
        len(candidate_rows) + 1,
        dtype=np.int32,
    )

    return {
        "variant_index": cohort,
        "score": tracked_score,
        "rank": tracked_rank,
        "reference_score": tracked_reference_score,
        "reference_rank": reference_rank[cohort],
        "n_finite_samples": n_finite_samples.astype(np.int32),
        "candidate_variant_index": candidate_rows,
        "candidate_reference_score": candidate_reference_score,
        "candidate_n_finite_samples": candidate_n_finite_samples,
        "reference_order": reference_order,
    }


def scan_fixed_cohort(
    posterior_variable,
    *,
    sample_ids: np.ndarray,
    hmc_variant_indices: np.ndarray,
    candidate_variant_indices: np.ndarray,
    cohort_variant_indices: np.ndarray,
    transform: str,
    variant_chunk_size: int | None = None,
    higher_is_better: bool = True,
) -> Mapping[str, np.ndarray]:
    """Track exact sampled and median-score ranks from an HMC target."""
    if getattr(posterior_variable, "ndim", None) != 3:
        raise ValueError("posterior_variable must be chain-by-draw-by-variant.")
    native_chunk = getattr(posterior_variable, "chunks", None)
    resolved_chunk = variant_chunk_size or int(
        (native_chunk or posterior_variable.shape)[-1]
    )

    def read_samples(
        selected_samples: np.ndarray,
        variant_start: int,
        variant_stop: int,
    ) -> np.ndarray:
        return _transform_score(
            _read_hmc_samples(
                posterior_variable,
                selected_samples,
                variant_start=variant_start,
                variant_stop=variant_stop,
            ),
            transform,
        )

    return _scan_fixed_cohort_blocks(
        n_source_samples=posterior_variable.shape[0] * posterior_variable.shape[1],
        n_source_variants=posterior_variable.shape[-1],
        source_variant_indices=hmc_variant_indices,
        sample_ids=sample_ids,
        candidate_variant_indices=candidate_variant_indices,
        cohort_variant_indices=cohort_variant_indices,
        variant_chunk_size=resolved_chunk,
        read_samples=read_samples,
        higher_is_better=higher_is_better,
    )


def scan_proxy_fixed_cohort(
    proxy_variable,
    *,
    sample_ids: np.ndarray,
    proxy_variant_indices: np.ndarray,
    candidate_variant_indices: np.ndarray,
    cohort_variant_indices: np.ndarray,
    variant_chunk_size: int | None = None,
    min_reference_samples: int = 2,
    higher_is_better: bool = True,
) -> Mapping[str, np.ndarray]:
    """Track exact sampled and median-score ranks from persisted proxy draws."""
    if getattr(proxy_variable, "ndim", None) != 2:
        raise ValueError("proxy_variable must be sample-by-variant.")
    native_chunk = getattr(proxy_variable, "chunks", None)
    resolved_chunk = variant_chunk_size or int(
        (native_chunk or proxy_variable.shape)[-1]
    )

    def read_samples(
        selected_samples: np.ndarray,
        variant_start: int,
        variant_stop: int,
    ) -> np.ndarray:
        values = _read_proxy_samples(
            proxy_variable,
            selected_samples,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )
        return np.where(np.isfinite(values), values, np.nan).astype(
            np.float32,
            copy=False,
        )

    return _scan_fixed_cohort_blocks(
        n_source_samples=proxy_variable.shape[0],
        n_source_variants=proxy_variable.shape[-1],
        source_variant_indices=proxy_variant_indices,
        sample_ids=sample_ids,
        candidate_variant_indices=candidate_variant_indices,
        cohort_variant_indices=cohort_variant_indices,
        variant_chunk_size=resolved_chunk,
        read_samples=read_samples,
        min_reference_samples=min_reference_samples,
        higher_is_better=higher_is_better,
    )


def paired_replicate_features(
    input_counts: np.ndarray,
    output_counts: np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> Mapping[str, np.ndarray]:
    """Summarize paired input/output replicates for every variant.

    Enrichments use smoothed within-library proportions. ``reduced_heterogeneity``
    is Cochran-style weighted residual heterogeneity divided by replicate degrees
    of freedom; values near one indicate scatter of roughly the size expected
    from the Poisson log-count variance approximation.
    """
    starting = _count_array(input_counts, name="input_counts", ndim=2)
    selected = _count_array(output_counts, name="output_counts", ndim=2)
    if starting.shape != selected.shape:
        raise ValueError("input_counts and output_counts must have the same two-dimensional shape.")
    if starting.shape[0] < 2:
        raise ValueError("paired replicate features require at least two replicates.")

    log_enrichment = _log_proportions(selected, pseudocount=pseudocount) - _log_proportions(
        starting,
        pseudocount=pseudocount,
    )
    variance = 1.0 / (starting + pseudocount) + 1.0 / (selected + pseudocount)
    mean, heterogeneity = _weighted_heterogeneity(log_enrichment, variance, axis=0)
    median = np.median(log_enrichment, axis=0)
    return {
        "observation_count": np.full(starting.shape[1], starting.shape[0], dtype=np.int32),
        "input_count": starting.sum(axis=0),
        "output_count": selected.sum(axis=0),
        "log_input_support": np.log1p(starting.sum(axis=0)),
        "zero_input_fraction": np.mean(starting == 0, axis=0),
        "zero_output_fraction": np.mean(selected == 0, axis=0),
        "log_enrichment": log_enrichment,
        "weighted_mean_log_enrichment": mean,
        "reduced_heterogeneity": heterogeneity / (starting.shape[0] - 1),
        "enrichment_range": np.ptp(log_enrichment, axis=0),
        "enrichment_mad": np.median(np.abs(log_enrichment - median), axis=0),
    }


def grouped_output_disagreement(
    output_counts: np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> Mapping[str, np.ndarray]:
    """Summarize replicate disagreement within each output timepoint or group.

    ``output_counts`` must be ``group x replicate x variant``. Group-specific
    means are removed before heterogeneity is pooled, so a biological time trend
    is not mislabeled as replicate disagreement. A shared input library is not
    needed because it cancels from within-group replicate contrasts.
    """
    selected = _count_array(output_counts, name="output_counts", ndim=3)
    n_group, n_replicate, n_variant = selected.shape
    if n_replicate < 2:
        raise ValueError("grouped output disagreement requires at least two replicates.")

    log_proportion = _log_proportions(
        selected.reshape(n_group * n_replicate, n_variant),
        pseudocount=pseudocount,
    ).reshape(selected.shape)
    variance = 1.0 / (selected + pseudocount)
    _, group_heterogeneity = _weighted_heterogeneity(log_proportion, variance, axis=1)
    group_range = np.ptp(log_proportion, axis=1)
    group_median = np.median(log_proportion, axis=1, keepdims=True)
    group_mad = np.median(np.abs(log_proportion - group_median), axis=1)
    return {
        "observation_count": np.full(n_variant, n_group * n_replicate, dtype=np.int32),
        "group_count": np.full(n_variant, n_group, dtype=np.int32),
        "replicate_count": np.full(n_variant, n_replicate, dtype=np.int32),
        "output_count": selected.sum(axis=(0, 1)),
        "zero_output_fraction": np.mean(selected == 0, axis=(0, 1)),
        "reduced_heterogeneity": group_heterogeneity.sum(axis=0)
        / (n_group * (n_replicate - 1)),
        "enrichment_range": np.median(group_range, axis=0),
        "enrichment_mad": np.median(group_mad, axis=0),
    }


def _score_vector(
    values: np.ndarray,
    *,
    name: str,
    higher_is_better: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two scores.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    differences = np.diff(array)
    if (higher_is_better and np.any(differences > 0)) or (
        not higher_is_better and np.any(differences < 0)
    ):
        direction = "highest to lowest" if higher_is_better else "lowest to highest"
        raise ValueError(f"{name} must be ordered from {direction} score.")
    return array


def _score_iqr(reference_scores: np.ndarray) -> float:
    scale = float(np.quantile(reference_scores, 0.75) - np.quantile(reference_scores, 0.25))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference score IQR must be positive.")
    return scale


def score_uncertainty(
    score_samples: np.ndarray,
    *,
    reference_scores: np.ndarray,
    interval: tuple[float, float] = (0.05, 0.95),
    higher_is_better: bool = True,
) -> Mapping[str, np.ndarray | float]:
    """Return sampled score intervals and widths normalized to a fixed run scale."""
    samples = np.asarray(score_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("score_samples must be sample-by-variant with at least two samples.")
    n_finite_samples = np.isfinite(samples).sum(axis=0)
    if np.any(n_finite_samples < 2):
        raise ValueError("Every score variant requires at least two finite samples.")
    lower_probability, upper_probability = interval
    if not 0 <= lower_probability < 0.5 < upper_probability <= 1:
        raise ValueError("interval must straddle 0.5 with probabilities in [0, 1].")
    reference = _score_vector(
        reference_scores,
        name="reference_scores",
        higher_is_better=higher_is_better,
    )
    scale = _score_iqr(reference)
    lower, median, upper = np.nanquantile(
        samples,
        [lower_probability, 0.5, upper_probability],
        axis=0,
    )
    width = upper - lower
    return {
        "score_lower": lower,
        "score_median": median,
        "score_upper": upper,
        "score_interval_width": width,
        "normalized_score_interval_width": width / scale,
        "reference_score_iqr": scale,
        "n_finite_samples": n_finite_samples,
    }


def local_score_spacing(
    reference_scores: np.ndarray,
    *,
    neighborhood: int = 5,
    normalization_scores: np.ndarray | None = None,
) -> Mapping[str, np.ndarray | float]:
    """Measure median-score drop per rank in a fixed local neighborhood."""
    scores = _score_vector(reference_scores, name="reference_scores")
    if neighborhood < 1:
        raise ValueError("neighborhood must be positive.")
    normalization = (
        scores
        if normalization_scores is None
        else _score_vector(normalization_scores, name="normalization_scores")
    )
    scale = _score_iqr(normalization)
    spacing = np.empty(len(scores), dtype=float)
    for index in range(len(scores)):
        lower_index = max(0, index - neighborhood)
        upper_index = min(len(scores) - 1, index + neighborhood)
        spacing[index] = (
            scores[lower_index] - scores[upper_index]
        ) / (upper_index - lower_index)
    normalized = spacing / scale
    return {
        "score_spacing": spacing,
        "normalized_score_spacing": normalized,
        "score_crowding": -np.log10(np.maximum(normalized, np.finfo(float).tiny)),
        "reference_score_iqr": scale,
    }


def count_within_score_window(
    reference_scores: np.ndarray,
    *,
    window_fraction: float = 0.01,
    normalization_scores: np.ndarray | None = None,
) -> Mapping[str, np.ndarray | float]:
    """Count other median scores within a fixed fraction of the run score IQR."""
    scores = _score_vector(reference_scores, name="reference_scores")
    if not np.isfinite(window_fraction) or window_fraction <= 0:
        raise ValueError("window_fraction must be positive and finite.")
    normalization = (
        scores
        if normalization_scores is None
        else _score_vector(normalization_scores, name="normalization_scores")
    )
    scale = _score_iqr(normalization)
    window = window_fraction * scale
    ascending = scores[::-1]
    left = np.searchsorted(ascending, scores - window, side="left")
    right = np.searchsorted(ascending, scores + window, side="right")
    count = right - left - 1
    return {
        "competitor_count": count.astype(np.int32),
        "score_window": float(window),
        "reference_score_iqr": scale,
    }


def boundary_difficulty(
    reference_scores: np.ndarray,
    top_score_samples: np.ndarray,
    *,
    sample_self_consistency: float,
    k: int,
    interval: tuple[float, float] = (0.05, 0.95),
    boundary_window: int = 20,
    normalization_scores: np.ndarray | None = None,
) -> Mapping[str, float]:
    """Compare score uncertainty near a top-K boundary with its median-score gap."""
    scores = _score_vector(reference_scores, name="reference_scores")
    samples = np.asarray(top_score_samples, dtype=float)
    if not 1 <= k < len(scores):
        raise ValueError("k must leave at least one retained candidate below the boundary.")
    if samples.ndim != 2 or samples.shape[1] < k:
        raise ValueError("top_score_samples must contain at least the reference top-k variants.")
    if boundary_window < 1:
        raise ValueError("boundary_window must be positive.")
    if not 0 <= sample_self_consistency <= 1:
        raise ValueError("sample_self_consistency must be between 0 and 1.")

    lower_probability, upper_probability = interval
    if not 0 <= lower_probability < 0.5 < upper_probability <= 1:
        raise ValueError("interval must straddle 0.5 with probabilities in [0, 1].")
    start = max(0, k - boundary_window)
    if np.any(np.isfinite(samples[:, start:k]).sum(axis=0) < 2):
        raise ValueError("Boundary variants require at least two finite score samples.")
    lower, upper = np.nanquantile(
        samples[:, start:k],
        [lower_probability, upper_probability],
        axis=0,
    )
    uncertainty = float(np.median(upper - lower))
    gap = float(scores[k - 1] - scores[k])
    spacing_start = max(0, k - boundary_window)
    spacing_stop = min(len(scores) - 1, k + boundary_window - 1)
    local_spacing = float(
        (scores[spacing_start] - scores[spacing_stop])
        / (spacing_stop - spacing_start)
    )
    if local_spacing <= 0:
        raise ValueError("Boundary local score spacing must be positive.")
    normalization = (
        scores
        if normalization_scores is None
        else _score_vector(normalization_scores, name="normalization_scores")
    )
    scale = _score_iqr(normalization)
    ratio = uncertainty / local_spacing
    return {
        "boundary_score_gap": gap,
        "normalized_boundary_score_gap": gap / scale,
        "boundary_local_score_spacing": local_spacing,
        "normalized_boundary_local_score_spacing": local_spacing / scale,
        "boundary_score_uncertainty": uncertainty,
        "normalized_boundary_score_uncertainty": uncertainty / scale,
        "uncertainty_to_spacing_ratio": ratio,
        "sample_self_consistency": float(sample_self_consistency),
        "top_k_instability": float(1 - sample_self_consistency),
    }


def standardized_regression(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
) -> Mapping[str, float | int]:
    """Fit a complete-case linear model after standardizing response and predictors."""
    outcome = np.asarray(response, dtype=float)
    if outcome.ndim != 1:
        raise ValueError("response must be one-dimensional.")
    if not predictors:
        raise ValueError("predictors must not be empty.")
    names = list(predictors)
    columns = [np.asarray(predictors[name], dtype=float) for name in names]
    if any(column.ndim != 1 or len(column) != len(outcome) for column in columns):
        raise ValueError("Every predictor must be one-dimensional and match response length.")
    matrix = np.column_stack(columns)
    complete = np.isfinite(outcome) & np.isfinite(matrix).all(axis=1)
    outcome = outcome[complete]
    matrix = matrix[complete]
    if len(outcome) <= len(names) + 1:
        raise ValueError("Too few complete observations for the requested predictors.")
    outcome_scale = outcome.std(ddof=0)
    predictor_scale = matrix.std(axis=0, ddof=0)
    if outcome_scale <= 0 or np.any(predictor_scale <= 0):
        raise ValueError("Response and every predictor must vary among complete cases.")

    standardized_outcome = (outcome - outcome.mean()) / outcome_scale
    standardized_matrix = (matrix - matrix.mean(axis=0)) / predictor_scale
    design = np.column_stack([np.ones(len(outcome)), standardized_matrix])
    coefficients, _, _, _ = np.linalg.lstsq(design, standardized_outcome, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((standardized_outcome - fitted) ** 2))
    total_sum = float(np.sum((standardized_outcome - standardized_outcome.mean()) ** 2))
    result: dict[str, float | int] = {
        "n_observations": int(len(outcome)),
        "r_squared": 1.0 - residual_sum / total_sum,
    }
    result.update(
        {
            f"coefficient_{name}": float(coefficient)
            for name, coefficient in zip(names, coefficients[1:])
        }
    )
    return result


def partial_spearman(
    first: np.ndarray,
    second: np.ndarray,
    controls: Mapping[str, np.ndarray],
) -> float:
    """Return a rank-residual partial Spearman correlation."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError("first and second must be matching one-dimensional arrays.")
    if not controls:
        raise ValueError("controls must not be empty.")
    control_columns = [np.asarray(values, dtype=float) for values in controls.values()]
    if any(values.ndim != 1 or len(values) != len(left) for values in control_columns):
        raise ValueError("Every control must be one-dimensional and match the inputs.")
    control_matrix = np.column_stack(control_columns)
    complete = np.isfinite(left) & np.isfinite(right) & np.isfinite(control_matrix).all(axis=1)
    left = left[complete]
    right = right[complete]
    control_matrix = control_matrix[complete]
    if len(left) <= len(control_columns) + 2:
        raise ValueError("Too few complete observations for partial correlation.")

    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    control_ranks = np.column_stack(
        [rankdata(control_matrix[:, index], method="average") for index in range(len(controls))]
    )
    if (
        left_rank.std(ddof=0) <= 0
        or right_rank.std(ddof=0) <= 0
        or np.any(control_ranks.std(axis=0, ddof=0) <= 0)
    ):
        raise ValueError("Inputs and controls must vary among complete cases.")
    design = np.column_stack([np.ones(len(left_rank)), control_ranks])
    left_residual = left_rank - design @ np.linalg.lstsq(
        design,
        left_rank,
        rcond=None,
    )[0]
    right_residual = right_rank - design @ np.linalg.lstsq(
        design,
        right_rank,
        rcond=None,
    )[0]
    if left_residual.std(ddof=0) <= 0 or right_residual.std(ddof=0) <= 0:
        raise ValueError("Rank residuals must vary after adjusting for controls.")
    return float(np.corrcoef(left_residual, right_residual)[0, 1])
