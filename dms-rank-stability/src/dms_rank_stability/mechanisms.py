"""Numerical summaries linking score uncertainty and rank stability."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata


def _ordered_scores(
    values: np.ndarray,
    *,
    name: str,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two scores.")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain only finite values.")
    oriented = scores if higher_is_better else -scores
    if np.any(np.diff(oriented) > 0):
        direction = "highest to lowest" if higher_is_better else "lowest to highest"
        raise ValueError(f"{name} must be ordered from {direction} score.")
    return scores, oriented


def _score_iqr(scores: np.ndarray) -> float:
    scale = float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference score IQR must be positive.")
    return scale


def score_uncertainty(
    score_samples: np.ndarray,
    *,
    reference_scores: np.ndarray,
    interval: tuple[float, float] = (0.05, 0.95),
    higher_is_better: bool = True,
) -> dict[str, np.ndarray | float]:
    """Return score intervals and widths normalized to a fixed run scale."""
    samples = np.asarray(score_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("score_samples must be sample-by-variant with at least two samples.")
    n_finite_samples = np.isfinite(samples).sum(axis=0)
    if np.any(n_finite_samples < 2):
        raise ValueError("Every score variant requires at least two finite samples.")
    lower_probability, upper_probability = interval
    if not 0 <= lower_probability < 0.5 < upper_probability <= 1:
        raise ValueError("interval must straddle 0.5 with probabilities in [0, 1].")
    reference, _ = _ordered_scores(
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
        "n_finite_samples": n_finite_samples.astype(np.int32),
    }


def local_score_spacing(
    reference_scores: np.ndarray,
    *,
    neighborhood: int = 5,
    normalization_scores: np.ndarray | None = None,
    higher_is_better: bool = True,
) -> dict[str, np.ndarray | float]:
    """Measure median-score separation per rank in a fixed neighborhood."""
    scores, oriented = _ordered_scores(
        reference_scores,
        name="reference_scores",
        higher_is_better=higher_is_better,
    )
    if neighborhood < 1:
        raise ValueError("neighborhood must be positive.")
    normalization = scores if normalization_scores is None else np.asarray(normalization_scores)
    normalization, _ = _ordered_scores(
        normalization,
        name="normalization_scores",
        higher_is_better=higher_is_better,
    )
    scale = _score_iqr(normalization)
    spacing = np.empty(len(scores), dtype=float)
    for index in range(len(scores)):
        lower_index = max(0, index - neighborhood)
        upper_index = min(len(scores) - 1, index + neighborhood)
        spacing[index] = (
            oriented[lower_index] - oriented[upper_index]
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
    higher_is_better: bool = True,
) -> dict[str, np.ndarray | float]:
    """Count other median scores within a fixed fraction of the run score IQR."""
    scores, _ = _ordered_scores(
        reference_scores,
        name="reference_scores",
        higher_is_better=higher_is_better,
    )
    if not np.isfinite(window_fraction) or window_fraction <= 0:
        raise ValueError("window_fraction must be positive and finite.")
    normalization = scores if normalization_scores is None else np.asarray(normalization_scores)
    normalization, _ = _ordered_scores(
        normalization,
        name="normalization_scores",
        higher_is_better=higher_is_better,
    )
    scale = _score_iqr(normalization)
    window = window_fraction * scale
    ascending = np.sort(scores)
    left = np.searchsorted(ascending, scores - window, side="left")
    right = np.searchsorted(ascending, scores + window, side="right")
    return {
        "competitor_count": (right - left - 1).astype(np.int32),
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
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Compare uncertainty near a top-k boundary with its median-score gap."""
    scores, oriented = _ordered_scores(
        reference_scores,
        name="reference_scores",
        higher_is_better=higher_is_better,
    )
    samples = np.asarray(top_score_samples, dtype=float)
    if not 1 <= k < len(scores):
        raise ValueError("k must leave at least one retained candidate below the boundary.")
    if samples.ndim != 2 or samples.shape[1] < k:
        raise ValueError("top_score_samples must contain the reference top-k variants.")
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
    gap = float(oriented[k - 1] - oriented[k])
    spacing_start = max(0, k - boundary_window)
    spacing_stop = min(len(scores) - 1, k + boundary_window - 1)
    local_spacing = float(
        (oriented[spacing_start] - oriented[spacing_stop])
        / (spacing_stop - spacing_start)
    )
    if local_spacing <= 0:
        raise ValueError("Boundary local score spacing must be positive.")
    normalization = scores if normalization_scores is None else np.asarray(normalization_scores)
    normalization, _ = _ordered_scores(
        normalization,
        name="normalization_scores",
        higher_is_better=higher_is_better,
    )
    scale = _score_iqr(normalization)
    return {
        "boundary_score_gap": gap,
        "normalized_boundary_score_gap": gap / scale,
        "boundary_local_score_spacing": local_spacing,
        "normalized_boundary_local_score_spacing": local_spacing / scale,
        "boundary_score_uncertainty": uncertainty,
        "normalized_boundary_score_uncertainty": uncertainty / scale,
        "uncertainty_to_spacing_ratio": uncertainty / local_spacing,
        "sample_self_consistency": float(sample_self_consistency),
        "top_k_instability": float(1 - sample_self_consistency),
    }


def standardized_regression(
    response: np.ndarray,
    predictors: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    """Fit a complete-case linear model after standardizing all variables."""
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
        [
            rankdata(control_matrix[:, index], method="average")
            for index in range(len(controls))
        ]
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
