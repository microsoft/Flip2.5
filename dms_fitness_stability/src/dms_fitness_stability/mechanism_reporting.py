"""Tidy mechanism tables from standard scores, features, and results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import ranking_candidate_indices
from .landscape_features import LANDSCAPE_FEATURE_NAMES, LandscapeFeatureBundle
from .mechanisms import scan_proxy_fixed_cohort, score_uncertainty
from .results import StabilityResult
from .score_ensembles import ScoreEnsemble

MECHANISM_TABLE_NAMES = ("mechanism_variants", "mechanism_runs")
_IDENTITY_FIELDS = (
    "analysis_id",
    "protein_family",
    "landscape",
    "inference_method",
    "model_spec",
    "fit_id",
)


def mechanism_candidate_indices(
    metadata: dict[str, object],
    variants: pd.DataFrame,
) -> np.ndarray:
    """Recover the result's candidate universe without family or method rules."""
    ranking_indices = ranking_candidate_indices(variants)
    expected_count = metadata.get("n_variants")
    if expected_count == len(variants):
        return np.arange(len(variants), dtype=np.int32)
    if expected_count is None or expected_count == len(ranking_indices):
        return ranking_indices
    raise ValueError(
        "Result n_variants matches neither the full nor ranking-candidate axis: "
        f"{expected_count!r}."
    )


def _matrix_indices(variants: pd.DataFrame) -> np.ndarray:
    mapping_columns = [
        name
        for name in ("matrix_variant_index", "hmc_variant_index")
        if name in variants
    ]
    if not mapping_columns:
        raise ValueError(
            "Result variants must contain matrix_variant_index; "
            "hmc_variant_index is accepted only for legacy results."
        )
    indices = variants[mapping_columns[0]].to_numpy(dtype=np.int64)
    if len(mapping_columns) == 2 and not np.array_equal(
        indices,
        variants[mapping_columns[1]].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "matrix_variant_index and legacy hmc_variant_index disagree."
        )
    return indices


def score_sample_positions(
    ensemble: ScoreEnsemble,
    result_sample_ids: np.ndarray,
) -> np.ndarray:
    """Map persisted result sample IDs onto score-ensemble row positions."""
    available = np.asarray(
        ensemble.scores.coords.get(
            "source_sample_id",
            np.arange(ensemble.scores.sizes["sample"], dtype=np.int64),
        ),
        dtype=np.int64,
    )
    if len(np.unique(available)) != len(available):
        raise ValueError("Score bundle source_sample_id values must be unique.")
    positions = pd.Index(available).get_indexer(result_sample_ids)
    if np.any(positions < 0):
        missing = np.unique(result_sample_ids[positions < 0]).tolist()
        raise ValueError(
            f"Result sample IDs are absent from the score bundle: {missing}."
        )
    return positions.astype(np.int64)


def _variant_chunk_size(
    ensemble: ScoreEnsemble,
    score_name: str,
) -> int:
    recorded = ensemble.metadata.get("variant_chunk_size")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded > 0:
        return recorded
    chunks = ensemble.fitness_score.sel(score_name=score_name).encoding.get(
        "chunksizes"
    )
    return (
        int(chunks[-1])
        if chunks is not None
        else ensemble.scores.sizes["variant"]
    )


def scan_mechanism_score_ensemble(
    ensemble: ScoreEnsemble,
    *,
    result_metadata: dict[str, object],
    result_variants: pd.DataFrame,
    result_sample_ids: np.ndarray,
    candidate_indices: np.ndarray,
    cohort_indices: np.ndarray,
    score_name: str,
    variant_chunk_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Scan one fixed mechanism cohort from a standard score ensemble."""
    ensemble.validate()
    if result_metadata.get("analysis_id") != ensemble.metadata.get("analysis_id"):
        raise ValueError("Result and score bundle analysis IDs differ.")
    available_scores = ensemble.scores["score_name"].values.astype(str).tolist()
    if score_name not in available_scores:
        raise ValueError(
            f"Score {score_name!r} is absent from the bundle; "
            f"available scores are {available_scores}."
        )
    minimum_samples = mechanism_reference_minimum(result_metadata, ensemble)
    directions = result_metadata.get("higher_is_better")
    higher_is_better = (
        bool(directions.get(score_name, True))
        if isinstance(directions, dict)
        else bool(directions)
        if isinstance(directions, bool)
        else True
    )
    target = ensemble.fitness_score.sel(score_name=score_name)
    return scan_proxy_fixed_cohort(
        target,
        sample_ids=score_sample_positions(ensemble, result_sample_ids),
        proxy_variant_indices=_matrix_indices(result_variants),
        candidate_variant_indices=np.asarray(candidate_indices, dtype=np.int32),
        cohort_variant_indices=np.asarray(cohort_indices, dtype=np.int32),
        variant_chunk_size=(
            _variant_chunk_size(ensemble, score_name)
            if variant_chunk_size is None
            else variant_chunk_size
        ),
        min_reference_samples=int(minimum_samples),
        higher_is_better=higher_is_better,
    )


def mechanism_reference_minimum(
    result_metadata: dict[str, object],
    ensemble: ScoreEnsemble,
) -> int:
    """Resolve the finite-sample threshold, including frozen-result fallback."""
    recorded = result_metadata.get("model_reference_min_finite_samples")
    if recorded is not None:
        return int(recorded)
    return (
        1
        if ensemble.metadata.get("uncertainty_target")
        == "posterior_parameter_landscape"
        else 2
    )


def _top_k_probabilities(
    sample_top_indices: np.ndarray,
    *,
    decision_k: int,
    n_variants: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(
        sample_top_indices[:, :decision_k].ravel(),
        minlength=n_variants,
    )
    return counts, counts / len(sample_top_indices)


def build_mechanism_tables(
    result: StabilityResult,
    ensemble: ScoreEnsemble,
    features: LandscapeFeatureBundle,
    *,
    score_name: str,
    direction: str,
    decision_k: int,
    challenger_probability: float = 0.01,
    reference_scale_k: int = 1000,
    interval: tuple[float, float] = (0.05, 0.95),
    variant_chunk_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build shared variant- and run-level mechanism tables."""
    ensemble.validate()
    features.validate()
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    if not 0 <= challenger_probability <= 1:
        raise ValueError("challenger_probability must be between zero and one.")
    if reference_scale_k < 2:
        raise ValueError("reference_scale_k must be at least two.")
    for field in _IDENTITY_FIELDS:
        if result.metadata.get(field) != ensemble.metadata.get(field):
            raise ValueError(
                f"Result and score bundle disagree for {field}: "
                f"{result.metadata.get(field)!r} != {ensemble.metadata.get(field)!r}."
            )
    available_scores = ensemble.scores["score_name"].values.astype(str).tolist()
    if score_name not in available_scores:
        raise ValueError(
            f"Score {score_name!r} is absent from the bundle; "
            f"available scores are {available_scores}."
        )
    if "score_name" not in result.selection.coords or score_name not in (
        result.selection["score_name"].values.astype(str).tolist()
    ):
        raise ValueError(f"Score {score_name!r} is absent from the stability result.")
    selected = result.selection.sel(score_name=score_name, drop=True)
    required_result_variables = {
        "model_reference_variant_index",
        "variant_index",
    }
    missing_variables = required_result_variables - set(selected.data_vars)
    if missing_variables:
        raise ValueError(
            "Stability result is missing mechanism variables: "
            f"{sorted(missing_variables)}."
        )
    retained_k = selected.sizes["selection_rank"]
    if not 1 <= decision_k <= retained_k:
        raise ValueError(
            f"decision_k must be between 1 and retained_k={retained_k}."
        )

    result_variants = result.variants
    feature_indices = features.align_result_variants(result_variants)
    candidate_indices = mechanism_candidate_indices(
        result.metadata,
        result_variants,
    )
    sample_top_indices = selected["variant_index"].values.astype(np.int32)
    model_reference_indices = selected[
        "model_reference_variant_index"
    ].values.astype(np.int32)
    model_top = model_reference_indices[:decision_k]
    counts, inclusion_probability = _top_k_probabilities(
        sample_top_indices,
        decision_k=decision_k,
        n_variants=len(result_variants),
    )
    challengers = np.flatnonzero(
        inclusion_probability >= challenger_probability
    ).astype(np.int32)
    challengers = challengers[~np.isin(challengers, model_top)]
    cohort_indices = np.unique(
        np.concatenate([model_top, challengers])
    ).astype(np.int32)

    result_sample_ids = np.asarray(
        selected.coords.get(
            "sample_id",
            np.arange(selected.sizes["sample"], dtype=np.int64),
        ),
        dtype=np.int64,
    )
    scan_metadata = {
        **result.metadata,
        "higher_is_better": {score_name: direction == "higher"},
    }
    tracked = scan_mechanism_score_ensemble(
        ensemble,
        result_metadata=scan_metadata,
        result_variants=result_variants,
        result_sample_ids=result_sample_ids,
        candidate_indices=candidate_indices,
        cohort_indices=cohort_indices,
        score_name=score_name,
        variant_chunk_size=variant_chunk_size,
    )
    expected_reference = model_reference_indices[: len(tracked["reference_order"])]
    observed_reference = tracked["reference_order"][: len(expected_reference)]
    if not np.array_equal(observed_reference, expected_reference):
        raise ValueError(
            "Mechanism scan and stability result have different model-reference "
            "orders. Check score direction, candidate eligibility, and sample IDs."
        )

    candidate_score_by_variant = np.full(len(result_variants), np.nan)
    candidate_score_by_variant[tracked["candidate_variant_index"]] = tracked[
        "candidate_reference_score"
    ]
    ordered_scores = candidate_score_by_variant[tracked["reference_order"]]
    ordered_scores = ordered_scores[np.isfinite(ordered_scores)]
    if len(ordered_scores) < 2:
        raise ValueError(
            "At least two candidates need finite model-reference scores for "
            "mechanism normalization."
        )
    normalization_scores = ordered_scores[: min(reference_scale_k, len(ordered_scores))]
    uncertainty = score_uncertainty(
        tracked["score"],
        reference_scores=normalization_scores,
        interval=interval,
        higher_is_better=direction == "higher",
    )
    rank_lower, rank_median, rank_upper = np.nanquantile(
        tracked["rank"],
        [interval[0], 0.5, interval[1]],
        axis=0,
    )

    variant_rows = result_variants.iloc[cohort_indices].copy().reset_index(drop=True)
    variant_rows.insert(0, "variant_index", cohort_indices)
    variant_rows.insert(0, "decision_k", decision_k)
    variant_rows.insert(0, "score", score_name)
    for field in reversed(_IDENTITY_FIELDS):
        variant_rows.insert(0, field, result.metadata[field])
    variant_rows["model_reference_rank"] = tracked["reference_rank"]
    variant_rows["model_reference_score"] = tracked["reference_score"]
    for name in (
        "score_lower",
        "score_median",
        "score_upper",
        "score_interval_width",
        "normalized_score_interval_width",
        "n_finite_samples",
    ):
        variant_rows[name] = uncertainty[name]
    variant_rows["reference_score_iqr"] = uncertainty["reference_score_iqr"]
    variant_rows["rank_lower"] = rank_lower
    variant_rows["rank_median"] = rank_median
    variant_rows["rank_upper"] = rank_upper
    variant_rows["rank_interval_width"] = rank_upper - rank_lower
    variant_rows["top_k_count"] = counts[cohort_indices]
    variant_rows["top_k_inclusion_probability"] = inclusion_probability[
        cohort_indices
    ]
    variant_rows["is_model_top_k"] = np.isin(cohort_indices, model_top)
    variant_rows["is_challenger"] = np.isin(cohort_indices, challengers)
    variant_rows["rank_interval_leaves_top_k"] = rank_upper > decision_k
    aligned_feature_indices = feature_indices[cohort_indices]
    for name in LANDSCAPE_FEATURE_NAMES:
        variant_rows[f"mechanism_{name}"] = features.features[name].values[
            aligned_feature_indices
        ]
    variant_rows = variant_rows.sort_values(
        ["model_reference_rank", "variant_index"]
    ).reset_index(drop=True)

    n_samples = len(sample_top_indices)
    self_consistency = float(
        np.sum(counts * (counts - 1))
        / (n_samples * (n_samples - 1) * decision_k)
    )
    run_row: dict[str, object] = {
        **{field: result.metadata[field] for field in _IDENTITY_FIELDS},
        "score": score_name,
        "direction": direction,
        "decision_k": decision_k,
        "n_samples": n_samples,
        "n_variants": len(candidate_indices),
        "mechanism_cohort_count": len(cohort_indices),
        "model_top_k_count": len(model_top),
        "challenger_count": len(challengers),
        "challenger_probability": challenger_probability,
        "reference_scale_k": len(normalization_scores),
        "reference_score_iqr": uncertainty["reference_score_iqr"],
        "median_normalized_score_interval_width": float(
            np.median(uncertainty["normalized_score_interval_width"])
        ),
        "median_rank_interval_width": float(
            np.median(variant_rows["rank_interval_width"])
        ),
        "sample_self_consistency": self_consistency,
        "top_k_instability": 1 - self_consistency,
        "feature_id": features.metadata["feature_id"],
        "analysis_config_sha256": result.metadata.get("analysis_config_sha256"),
    }
    return {
        "mechanism_variants": variant_rows,
        "mechanism_runs": pd.DataFrame([run_row]),
    }
