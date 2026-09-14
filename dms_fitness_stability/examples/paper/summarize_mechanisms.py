"""Build mechanism tables from paper results, features, and sampled scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau, spearmanr

from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.candidates import ranking_candidate_indices
from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.landscape_features import (
    LANDSCAPE_FEATURE_NAMES,
    LandscapeFeatureBundle,
    extract_landscape_count_features,
    landscape_feature_id,
)
from dms_fitness_stability.mechanism_reporting import (
    mechanism_candidate_indices,
    mechanism_reference_minimum,
    scan_mechanism_score_ensemble,
)
from dms_fitness_stability.mechanisms import (
    boundary_difficulty,
    count_within_score_window,
    local_score_spacing,
    partial_spearman,
    score_uncertainty,
    standardized_regression,
)
from dms_fitness_stability.results import _read_variants_csv
from dms_fitness_stability.score_ensembles import ScoreEnsemble
from dms_fitness_stability.score_paths import SCORES_RELATIVE_ROOT, score_collection_root

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
MAIN_K = 100
CHALLENGER_PROBABILITY = 0.01
REFERENCE_SCALE_K = 1000
MECHANISM_CANDIDATE_SCOPE_VERSION = 4
REFERENCE_CATEGORY_COLUMNS = (
    "analysis_id",
    "protein_family",
    "landscape",
    "model_spec",
    "cohort",
    "top_k",
    "category_column",
    "category",
    "count",
    "fraction",
)

_landscape_count_features = extract_landscape_count_features


class MechanismSpec:
    def __init__(
        self,
        identity: AnalysisIdentity,
        primary_score: str,
        *,
        source_identity: AnalysisIdentity | None = None,
    ) -> None:
        self.identity = identity
        self.source_identity = source_identity or identity
        self.primary_score = primary_score

    @property
    def analysis_id(self) -> str:
        return self.identity.analysis_id

    @property
    def source_analysis_id(self) -> str:
        return self.source_identity.analysis_id


def _manifest(path: Path) -> list[MechanismSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 1:
        identities = [AnalysisIdentity(**entry) for entry in payload["analyses"]]
        if len({identity.analysis_id for identity in identities}) != len(identities):
            raise ValueError("Analysis manifest contains duplicate analysis IDs.")
        return [
            MechanismSpec(identity=identity, primary_score="log_mean_growth")
            for identity in identities
        ]
    if payload.get("schema_version") == 2:
        specs = load_analysis_manifest(path)
        unsupported = {
            spec.identity.inference_method
            for spec in specs
            if spec.identity.inference_method
            not in {"hmc", "hmc_count_based", "mle_predictive"}
        }
        if unsupported:
            raise ValueError(
                f"Unsupported mechanism inference methods: {sorted(unsupported)}."
            )
        return [
            MechanismSpec(
                spec.score_identity(score),
                score,
                source_identity=spec.identity,
            )
            for spec in specs
            for score in spec.scores
        ]
    raise ValueError("Mechanism analysis requires manifest schema_version 1 or 2.")


def _top_k_probabilities(indices: np.ndarray, *, k: int, n_variants: int) -> np.ndarray:
    counts = np.bincount(indices[:, :k].ravel(), minlength=n_variants)
    return counts / len(indices)


def _self_consistency(probability_counts: np.ndarray, n_samples: int, k: int) -> float:
    return float(
        np.sum(probability_counts * (probability_counts - 1))
        / (n_samples * (n_samples - 1) * k)
    )


def _cache_candidate_indices(cache: xr.Dataset) -> np.ndarray:
    if "candidate_variant_index" in cache:
        return cache["candidate_variant_index"].values.astype(np.int32)
    return np.arange(cache.sizes["candidate"], dtype=np.int32)


def _candidate_score_lookup(
    cache: xr.Dataset,
    *,
    n_variant_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_indices = _cache_candidate_indices(cache)
    scores = cache["candidate_reference_score"].values.astype(float)
    if (
        len(candidate_indices) != len(scores)
        or len(np.unique(candidate_indices)) != len(candidate_indices)
        or np.any(np.diff(candidate_indices) <= 0)
        or candidate_indices[0] < 0
        or candidate_indices[-1] >= n_variant_rows
    ):
        raise ValueError("Mechanism cache candidate mapping is invalid.")
    score_by_variant = np.full(n_variant_rows, np.nan, dtype=float)
    score_by_variant[candidate_indices] = scores
    return candidate_indices, scores, score_by_variant


def _cache_valid(
    cache: xr.Dataset,
    *,
    metadata: dict[str, object],
    cohort_indices: np.ndarray,
    candidate_indices: np.ndarray,
    score_name: str,
    source: dict[str, object],
) -> bool:
    raw_scope_version = cache.attrs.get("candidate_scope_version")
    scope_version = (
        None
        if raw_scope_version is None
        else int(np.asarray(raw_scope_version).reshape(-1)[0])
    )
    legacy_scope = scope_version is None or (
        0 < scope_version < MECHANISM_CANDIDATE_SCOPE_VERSION
    )
    common = (
        cache.attrs.get("analysis_id") == metadata["analysis_id"]
        and cache.attrs.get("score_name") == score_name
        and (legacy_scope or scope_version == MECHANISM_CANDIDATE_SCOPE_VERSION)
        and np.array_equal(cache["variant_index"].values, cohort_indices)
        and np.array_equal(_cache_candidate_indices(cache), candidate_indices)
    )
    if not common:
        return False
    if "source_kind" in cache.attrs:
        if (
            scope_version == MECHANISM_CANDIDATE_SCOPE_VERSION
            and cache.attrs.get("source_kind") == source["kind"]
            and cache.attrs.get("source_path") == source["path"]
            and int(cache.attrs.get("source_size", -1)) == int(source["size"])
            and int(cache.attrs.get("source_mtime_ns", -1))
            == int(source["mtime_ns"])
            and cache.attrs.get("score_contract") == source["score_contract"]
        ):
            return True
        if (
            scope_version == MECHANISM_CANDIDATE_SCOPE_VERSION
            and source["kind"] == "score_ensemble"
            and cache.attrs.get("source_kind")
            in {"hmc_score_matrix", "hmc_count_proxy", "mle_proxy"}
            and cache.attrs.get("source_path") == source["path"]
            and int(cache.attrs.get("source_size", -1)) == int(source["size"])
            and int(cache.attrs.get("source_mtime_ns", -1))
            == int(source["mtime_ns"])
            and cache.attrs.get("score_contract") == source["score_contract"]
        ):
            return True
        if (
            scope_version == MECHANISM_CANDIDATE_SCOPE_VERSION
            and source["kind"] == "score_ensemble"
            and cache.attrs.get("source_kind") == "hmc_posterior"
        ):
            artifact = source.get("source_artifact")
            return (
                isinstance(artifact, dict)
                and cache.attrs.get("source_path") == artifact.get("path")
                and int(cache.attrs.get("source_size", -1))
                == int(artifact.get("size", -2))
                and int(cache.attrs.get("source_mtime_ns", -1))
                == int(artifact.get("mtime_ns", -2))
                and cache.attrs.get("score_contract")
                == (
                    f"posterior/{metadata.get('posterior_variable')}:"
                    f"{metadata.get('posterior_transform')}"
                )
            )
        return False
    if source["kind"] == "hmc_posterior":
        return (
            legacy_scope
            and int(cache.attrs.get("hmc_size", -1)) == int(source["size"])
            and int(cache.attrs.get("hmc_mtime_ns", -1))
            == int(source["mtime_ns"])
        )
    if source["kind"] != "score_ensemble":
        return False
    artifact = source.get("source_artifact")
    result_artifact = metadata.get("hmc_file")
    return (
        legacy_scope
        and isinstance(artifact, dict)
        and isinstance(result_artifact, dict)
        and all(
            artifact.get(name) == result_artifact.get(name)
            for name in ("path", "size", "mtime_ns")
        )
        and int(cache.attrs.get("hmc_size", -1)) == int(artifact["size"])
        and int(cache.attrs.get("hmc_mtime_ns", -1))
        == int(artifact["mtime_ns"])
    )


def _score_source(
    metadata: dict[str, object],
    *,
    score_name: str,
    scores_root: Path | None = None,
    legacy_score_root: Path | None = None,
    score_view: str = "full",
) -> dict[str, object]:
    analysis_id = str(metadata["analysis_id"])
    candidates = []
    inference_method = str(metadata["inference_method"])
    if scores_root is not None:
        collection_root = score_collection_root(
            scores_root,
            inference_method,
            sample_view=score_view if inference_method == "hmc" else "canonical",
        )
        candidates.append((collection_root / analysis_id, inference_method == "hmc"))
    if legacy_score_root is not None:
        candidates.append((legacy_score_root / analysis_id, False))
    matches = []
    for bundle_dir, requires_view in candidates:
        metadata_path = bundle_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        score_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            score_metadata.get("analysis_id") != analysis_id
            or score_name not in score_metadata.get("scores", [])
            or (
                requires_view
                and score_metadata.get("sample_view") != score_view
            )
        ):
            continue
        schema_version = score_metadata.get("schema_version")
        if schema_version == 2:
            score_path = bundle_dir / "fitness_scores.nc"
        elif schema_version == 1:
            score_path = bundle_dir / "fitness_proxies.nc"
        else:
            raise ValueError(f"Unsupported score bundle schema at {bundle_dir}.")
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing score matrix: {score_path}")
        matches.append((bundle_dir, score_path, score_metadata))
    if len(matches) > 1:
        raise ValueError(f"Multiple score bundles match {analysis_id}.")
    if matches:
        bundle_dir, score_path, score_metadata = matches[0]
        stat = score_path.stat()
        return {
            "kind": "score_ensemble",
            "bundle_dir": str(bundle_dir.resolve()),
            "path": str(score_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "score_contract": score_metadata["score_contracts"][score_name],
            "source_artifact": score_metadata.get("source_artifact"),
        }
    hmc_file = metadata.get("hmc_file")
    if scores_root is None and legacy_score_root is None and isinstance(hmc_file, dict):
        return {
            "kind": "hmc_posterior",
            "path": str(Path(str(hmc_file["path"])).resolve()),
            "size": int(hmc_file["size"]),
            "mtime_ns": int(hmc_file["mtime_ns"]),
            "score_contract": (
                f"posterior/{metadata['posterior_variable']}:"
                f"{metadata['posterior_transform']}"
            ),
        }
    raise FileNotFoundError(f"Missing score bundle for {analysis_id}.")


def _load_or_scan_cache(
    result_dir: Path,
    metadata: dict[str, object],
    cohort_indices: np.ndarray,
    cache_path: Path,
    *,
    score_name: str,
    scores_root: Path | None = None,
    legacy_score_root: Path | None = None,
    score_view: str = "full",
) -> xr.Dataset:
    variants = _read_variants_csv(result_dir / "variants.csv")
    candidate_indices = mechanism_candidate_indices(metadata, variants)
    source = _score_source(
        metadata,
        score_name=score_name,
        scores_root=scores_root,
        legacy_score_root=legacy_score_root,
        score_view=score_view,
    )
    if cache_path.is_file():
        cache = xr.load_dataset(cache_path, engine="h5netcdf")
        if _cache_valid(
            cache,
            metadata=metadata,
            cohort_indices=cohort_indices,
            candidate_indices=candidate_indices,
            score_name=score_name,
            source=source,
        ):
            if "n_finite_samples" not in cache:
                cache["n_finite_samples"] = (
                    "mechanism_variant",
                    np.isfinite(cache["score"].values).sum(axis=0).astype(np.int32),
                )
            if "candidate_n_finite_samples" not in cache:
                cache["candidate_n_finite_samples"] = (
                    "candidate",
                    np.full(
                        cache.sizes["candidate"],
                        cache.sizes["sample"],
                        dtype=np.int32,
                    ),
                )
            return cache
        cache.close()

    with xr.open_dataset(result_dir / "stability.nc", engine="h5netcdf") as selection:
        score_selection = selection.sel(score_name=score_name)
        sample_ids = score_selection["sample_id"].values.astype(np.int64)
    if source["kind"] == "hmc_posterior":
        from fit_preprocessing.mechanism_scores_from_fit import (
            scan_fit_mechanism_scores,
        )

        tracked = scan_fit_mechanism_scores(
            metadata,
            variants,
            sample_ids=sample_ids,
            candidate_indices=candidate_indices,
            cohort_indices=cohort_indices,
        )
        minimum_samples = 1
    else:
        ensemble = ScoreEnsemble.load(str(source["bundle_dir"]))
        try:
            tracked = scan_mechanism_score_ensemble(
                ensemble,
                result_metadata=metadata,
                result_variants=variants,
                result_sample_ids=sample_ids,
                candidate_indices=candidate_indices,
                cohort_indices=cohort_indices,
                score_name=score_name,
            )
            minimum_samples = mechanism_reference_minimum(metadata, ensemble)
        finally:
            ensemble.close()

    cache = xr.Dataset(
        {
            "score": (("sample", "mechanism_variant"), tracked["score"]),
            "rank": (("sample", "mechanism_variant"), tracked["rank"]),
            "variant_index": ("mechanism_variant", tracked["variant_index"]),
            "reference_score": (
                "mechanism_variant",
                tracked["reference_score"],
            ),
            "reference_rank": ("mechanism_variant", tracked["reference_rank"]),
            "n_finite_samples": (
                "mechanism_variant",
                tracked["n_finite_samples"],
            ),
            "candidate_variant_index": (
                "candidate",
                tracked["candidate_variant_index"],
            ),
            "candidate_reference_score": (
                "candidate",
                tracked["candidate_reference_score"],
            ),
            "candidate_n_finite_samples": (
                "candidate",
                tracked["candidate_n_finite_samples"],
            ),
            "reference_order": ("reference_position", tracked["reference_order"]),
        },
        coords={
            "sample": np.arange(len(sample_ids), dtype=np.int32),
            "sample_id": ("sample", sample_ids),
            "mechanism_variant": np.arange(len(cohort_indices), dtype=np.int32),
            "candidate": np.arange(len(candidate_indices), dtype=np.int32),
            "reference_position": np.arange(
                1,
                len(candidate_indices) + 1,
                dtype=np.int32,
            ),
        },
        attrs={
            "analysis_id": metadata["analysis_id"],
            "source_kind": source["kind"],
            "source_path": source["path"],
            "source_size": int(source["size"]),
            "source_mtime_ns": int(source["mtime_ns"]),
            "score_contract": source["score_contract"],
            "score_name": score_name,
            "candidate_scope_version": MECHANISM_CANDIDATE_SCOPE_VERSION,
            "reference_min_finite_samples": minimum_samples,
            "cohort_definition": (
                "model top 100 union q_i(100)>=0.01 union reported top 1000"
            ),
        },
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_netcdf(
        cache_path,
        engine="h5netcdf",
        encoding={
            "score": {"compression": "gzip", "compression_opts": 4},
            "rank": {"compression": "gzip", "compression_opts": 4},
        },
    )
    return cache


def _cohort_indices(
    variants: pd.DataFrame,
    sample_top_indices: np.ndarray,
    model_reference_indices: np.ndarray,
    *,
    k: int = MAIN_K,
    challenger_probability: float = CHALLENGER_PROBABILITY,
    reported_top_n: int = REFERENCE_SCALE_K,
    reported_supported_indices: np.ndarray | None = None,
    reference_top_k_identifiable: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n_variants = len(variants)
    probability = _top_k_probabilities(
        sample_top_indices,
        k=k,
        n_variants=n_variants,
    )
    model_top = model_reference_indices[:k]
    challenger = np.flatnonzero(probability >= challenger_probability)
    challenger = challenger[~np.isin(challenger, model_top)]
    candidates = ranking_candidate_indices(variants)
    reported_rank = variants["reported_rank"].to_numpy(dtype=float)
    reported = (
        candidates[reported_rank[candidates] <= reported_top_n]
        if reference_top_k_identifiable
        else np.empty(0, dtype=np.int32)
    )
    n_reported_context_omitted = 0
    if reported_supported_indices is not None:
        supported = np.asarray(reported_supported_indices, dtype=np.int32)
        retained = reported[np.isin(reported, supported)]
        n_reported_context_omitted = len(reported) - len(retained)
        reported = retained
    union = np.unique(np.concatenate([model_top, challenger, reported])).astype(np.int32)
    strongest = challenger[
        np.argsort(probability[challenger], kind="stable")[::-1][:10]
    ]
    return union, probability, strongest, n_reported_context_omitted


def _variant_table(
    identity: AnalysisIdentity,
    metadata: dict[str, object],
    variants: pd.DataFrame,
    raw_features: dict[str, np.ndarray],
    cache: xr.Dataset,
    inclusion_probability: np.ndarray,
    strongest_challengers: np.ndarray,
) -> pd.DataFrame:
    cohort_indices = cache["variant_index"].values.astype(np.int32)
    reference_rank = cache["reference_rank"].values.astype(np.int32)
    reference_order = cache["reference_order"].values.astype(np.int32)
    _, _, score_by_variant = _candidate_score_lookup(
        cache,
        n_variant_rows=len(variants),
    )
    ordered_scores = score_by_variant[reference_order]
    ordered_scores = ordered_scores[np.isfinite(ordered_scores)]
    if len(ordered_scores) <= MAIN_K:
        raise ValueError("Too few score-supported candidates for mechanism summaries.")
    normalization_scores = ordered_scores[: min(REFERENCE_SCALE_K, len(ordered_scores))]
    uncertainty = score_uncertainty(
        cache["score"].values,
        reference_scores=normalization_scores,
    )
    spacing = local_score_spacing(
        ordered_scores,
        neighborhood=5,
        normalization_scores=normalization_scores,
    )
    crowd_01 = count_within_score_window(
        ordered_scores,
        window_fraction=0.01,
        normalization_scores=normalization_scores,
    )
    crowd_02 = count_within_score_window(
        ordered_scores,
        window_fraction=0.02,
        normalization_scores=normalization_scores,
    )
    rank_lower, rank_median, rank_upper = np.nanquantile(
        cache["rank"].values,
        [0.05, 0.5, 0.95],
        axis=0,
    )
    hmc_indices = variants.iloc[cohort_indices]["hmc_variant_index"].to_numpy(dtype=int)
    rows = variants.iloc[cohort_indices].copy().reset_index(drop=True)
    rows.insert(0, "variant_index", cohort_indices)
    rows.insert(0, "fit_id", identity.fit_id)
    rows.insert(0, "model_spec", identity.model_spec)
    rows.insert(0, "inference_method", identity.inference_method)
    rows.insert(0, "landscape", identity.landscape)
    rows.insert(0, "protein_family", identity.protein_family)
    rows.insert(0, "analysis_id", identity.analysis_id)
    rows["model_reference_rank"] = reference_rank
    rows["model_reference_score"] = cache["reference_score"].values
    rows["score_lower"] = uncertainty["score_lower"]
    rows["score_median"] = uncertainty["score_median"]
    rows["score_upper"] = uncertainty["score_upper"]
    rows["score_interval_width"] = uncertainty["score_interval_width"]
    rows["n_finite_samples"] = cache["n_finite_samples"].values.astype(np.int32)
    rows["normalized_score_interval_width"] = uncertainty[
        "normalized_score_interval_width"
    ]
    rows["reference_score_iqr"] = uncertainty["reference_score_iqr"]
    order_position = reference_rank - 1
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
    rows["top_100_inclusion_probability"] = inclusion_probability[cohort_indices]
    rows["rank_interval_leaves_top_100"] = rank_upper > MAIN_K
    rows["score_margin_to_rank_100"] = (
        rows["model_reference_score"] - ordered_scores[MAIN_K - 1]
    )
    rows["margin_over_score_uncertainty"] = np.divide(
        rows["score_margin_to_rank_100"],
        rows["score_interval_width"],
        out=np.full(len(rows), np.nan),
        where=rows["score_interval_width"].to_numpy() > 0,
    )
    reported_rank = rows["reported_rank"].to_numpy(dtype=float)
    rows["is_model_top_100"] = reference_rank <= MAIN_K
    rows["is_challenger_q_ge_001"] = (
        (reference_rank > MAIN_K)
        & (rows["top_100_inclusion_probability"] >= CHALLENGER_PROBABILITY)
    )
    rows["is_strongest_challenger"] = np.isin(cohort_indices, strongest_challengers)
    rows["is_reported_top_100"] = reported_rank <= MAIN_K
    rows["is_reported_top_1000"] = reported_rank <= REFERENCE_SCALE_K
    rows["is_primary_cohort"] = rows["is_model_top_100"] | rows[
        "is_challenger_q_ge_001"
    ]
    for name in (
        "observation_count",
        "input_count",
        "output_count",
        "log_input_support",
        "zero_input_fraction",
        "zero_output_fraction",
        "reduced_heterogeneity",
        "enrichment_range",
        "enrichment_mad",
    ):
        if name in raw_features:
            rows[f"mechanism_{name}"] = np.asarray(raw_features[name])[hmc_indices]
    rows["artifact_status"] = metadata.get("artifact_status", "unrecorded")
    rows["artifact_warnings"] = json.dumps(metadata.get("warnings", []))
    return rows.sort_values("model_reference_rank").reset_index(drop=True)


def _functionality_table(
    identity: AnalysisIdentity,
    cache: xr.Dataset,
) -> pd.DataFrame:
    scores = cache["candidate_reference_score"].values.astype(float)
    candidate_indices = _cache_candidate_indices(cache)
    order = cache["reference_order"].values.astype(np.int32)
    n_candidates = len(scores)
    order_positions = np.searchsorted(candidate_indices, order)
    if (
        len(candidate_indices) != n_candidates
        or len(order) != n_candidates
        or np.any(order_positions >= n_candidates)
        or not np.array_equal(candidate_indices[order_positions], order)
    ):
        raise ValueError(
            f"{identity.analysis_id}: functionality cache is incomplete or invalid."
        )
    ranks = np.empty(n_candidates, dtype=np.int32)
    ranks[order_positions] = np.arange(1, n_candidates + 1, dtype=np.int32)
    supported = np.isfinite(scores)
    ordered_supported_scores = scores[order_positions]
    ordered_supported_scores = ordered_supported_scores[
        np.isfinite(ordered_supported_scores)
    ]
    if np.any(ordered_supported_scores[:-1] < ordered_supported_scores[1:]):
        raise ValueError(
            f"{identity.analysis_id}: reference scores are not sorted descending."
        )
    return pd.DataFrame(
        {
            "analysis_id": identity.analysis_id,
            "variant_index": candidate_indices[supported],
            "score_median": scores[supported],
            "model_reference_rank": ranks[supported],
            "n_finite_samples": cache["candidate_n_finite_samples"].values[
                supported
            ],
        }
    )


def _run_table(
    identity: AnalysisIdentity,
    metadata: dict[str, object],
    cache: xr.Dataset,
    inclusion_probability: np.ndarray,
    sample_top_indices: np.ndarray,
    n_reported_context_omitted: int,
) -> dict[str, object]:
    reference_order = cache["reference_order"].values.astype(np.int32)
    _, _, score_by_variant = _candidate_score_lookup(
        cache,
        n_variant_rows=len(inclusion_probability),
    )
    ordered_scores = score_by_variant[reference_order]
    ordered_scores = ordered_scores[np.isfinite(ordered_scores)]
    full_reference_rank = np.full(
        len(inclusion_probability),
        len(reference_order) + 1,
        dtype=np.int32,
    )
    full_reference_rank[reference_order] = np.arange(
        1,
        len(reference_order) + 1,
        dtype=np.int32,
    )
    top_indices = reference_order[:MAIN_K]
    cohort_indices = cache["variant_index"].values.astype(np.int32)
    column_by_variant = {int(index): column for column, index in enumerate(cohort_indices)}
    top_columns = [column_by_variant[int(index)] for index in top_indices]
    counts = np.bincount(
        sample_top_indices[:, :MAIN_K].ravel(),
        minlength=len(inclusion_probability),
    )
    self_consistency = _self_consistency(counts, len(sample_top_indices), MAIN_K)
    boundary = boundary_difficulty(
        ordered_scores,
        cache["score"].values[:, top_columns],
        sample_self_consistency=self_consistency,
        k=MAIN_K,
        normalization_scores=ordered_scores[:REFERENCE_SCALE_K],
    )
    return {
        "analysis_id": identity.analysis_id,
        "protein_family": identity.protein_family,
        "landscape": identity.landscape,
        "inference_method": identity.inference_method,
        "model_spec": identity.model_spec,
        "fit_id": identity.fit_id,
        "n_samples": int(metadata["n_samples"]),
        "n_variants": int(metadata["n_variants"]),
        "challenger_count_q_ge_001": int(
            np.sum(
                (inclusion_probability >= CHALLENGER_PROBABILITY)
                & (full_reference_rank > MAIN_K)
            )
        ),
        "candidate_cloud_count_q_ge_001": int(
            np.sum(inclusion_probability >= CHALLENGER_PROBABILITY)
        ),
        "n_reported_context_omitted_low_support": n_reported_context_omitted,
        **boundary,
        "artifact_status": metadata.get("artifact_status", "unrecorded"),
        "artifact_warnings": json.dumps(metadata.get("warnings", [])),
    }


def _safe_spearman(first: pd.Series, second: pd.Series) -> float:
    complete = np.isfinite(first.to_numpy(dtype=float)) & np.isfinite(
        second.to_numpy(dtype=float)
    )
    left = first.to_numpy(dtype=float)[complete]
    right = second.to_numpy(dtype=float)[complete]
    if len(left) < 3 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def _reference_diagnostic(
    identity: AnalysisIdentity,
    metadata: dict[str, object],
    variants: pd.DataFrame,
    cache: xr.Dataset,
    *,
    top_k: int = MAIN_K,
) -> tuple[dict[str, object], set[int]]:
    candidate_indices, model_scores, score_by_variant = _candidate_score_lookup(
        cache,
        n_variant_rows=len(variants),
    )
    reference_order = cache["reference_order"].values.astype(np.int32)
    model_rank = np.full(len(variants), np.nan)
    model_rank[reference_order] = np.arange(1, len(reference_order) + 1)
    reported = variants["reported_fitness"].to_numpy(dtype=float)
    score_supported = candidate_indices[np.isfinite(model_scores)]
    finite = candidate_indices[
        np.isfinite(reported[candidate_indices]) & np.isfinite(model_scores)
    ]
    positive = finite[reported[finite] > 0]

    def correlations(indices: np.ndarray) -> tuple[float, float]:
        if len(indices) < 3:
            return np.nan, np.nan
        model = score_by_variant[indices]
        if np.ptp(model) <= 0 or np.ptp(reported[indices]) <= 0:
            return np.nan, np.nan
        return (
            float(spearmanr(model, reported[indices]).statistic),
            float(kendalltau(model, reported[indices]).statistic),
        )

    all_spearman, all_kendall = correlations(finite)
    positive_spearman, positive_kendall = correlations(positive)
    reference_top_k_identifiable = bool(
        metadata.get("reference_top_k_identifiable", True)
    )
    evaluated_k = min(top_k, len(candidate_indices), len(finite))
    reported_order = finite[
        np.lexsort((finite, -reported[finite]))
    ]
    reversed_order = finite[
        np.lexsort((finite, reported[finite]))
    ]
    model_top = set(reference_order[:evaluated_k].tolist())
    reported_top = set(reported_order[:evaluated_k].tolist())
    reversed_top = set(reversed_order[:evaluated_k].tolist())
    wild_type = np.flatnonzero(
        variants.get("is_wild_type", pd.Series(False, index=variants.index)).to_numpy(
            dtype=bool
        )
    )
    wild_type_index = int(wild_type[0]) if len(wild_type) == 1 else None
    source_controls = variants.get(
        "is_source_control",
        pd.Series(False, index=variants.index),
    ).to_numpy(dtype=bool)
    return (
        {
            "analysis_id": identity.analysis_id,
            "protein_family": identity.protein_family,
            "landscape": identity.landscape,
            "inference_method": identity.inference_method,
            "model_spec": identity.model_spec,
            "fit_id": identity.fit_id,
            "n_fitted_variants": len(variants),
            "n_ranking_candidates": len(candidate_indices),
            "n_score_supported_candidates": len(score_supported),
            "n_excluded_from_ranking": len(variants) - len(candidate_indices),
            "n_source_controls": int(source_controls.sum()),
            "n_finite_reported": len(finite),
            "n_positive_reported": len(positive),
            "all_finite_spearman": all_spearman,
            "all_finite_kendall": all_kendall,
            "positive_reported_spearman": positive_spearman,
            "positive_reported_kendall": positive_kendall,
            "top_k": evaluated_k,
            "reference_top_k_identifiable": reference_top_k_identifiable,
            "model_reported_top_k_intersection": (
                len(model_top & reported_top)
                if reference_top_k_identifiable
                else np.nan
            ),
            "model_reported_top_k_fraction": (
                len(model_top & reported_top) / evaluated_k
                if reference_top_k_identifiable
                else np.nan
            ),
            "model_reversed_reported_top_k_intersection": (
                len(model_top & reversed_top)
                if reference_top_k_identifiable
                else np.nan
            ),
            "model_reversed_reported_top_k_fraction": (
                len(model_top & reversed_top) / evaluated_k
                if reference_top_k_identifiable
                else np.nan
            ),
            "wild_type_variant_index": wild_type_index,
            "wild_type_model_rank": (
                model_rank[wild_type_index] if wild_type_index is not None else np.nan
            ),
            "wild_type_reported_rank": (
                variants.iloc[wild_type_index]["reported_rank"]
                if wild_type_index is not None
                else np.nan
            ),
            "reference_fitness_source": metadata.get(
                "reference_fitness_source",
                "unrecorded",
            ),
        },
        model_top,
    )


def _reference_category_rows(
    identity: AnalysisIdentity,
    variants: pd.DataFrame,
    cache: xr.Dataset,
    *,
    top_k: int = MAIN_K,
) -> list[dict[str, object]]:
    category_column = next(
        (name for name in ("reported_activity_level",) if name in variants),
        None,
    )
    if category_column is None:
        return []
    candidates = _cache_candidate_indices(cache)
    reported = variants["reported_fitness"].to_numpy(dtype=float)
    finite = candidates[np.isfinite(reported[candidates])]
    evaluated_k = min(top_k, len(candidates), len(finite))
    reference_order = cache["reference_order"].values.astype(np.int32)
    reported_order = finite[np.lexsort((finite, -reported[finite]))]
    rows = []
    for cohort_name, indices in (
        ("model_top_k", reference_order[:evaluated_k]),
        ("reported_top_k", reported_order[:evaluated_k]),
    ):
        categories = variants.iloc[indices][category_column].fillna("unavailable")
        counts = categories.value_counts(dropna=False, sort=False)
        for category, count in counts.items():
            rows.append(
                {
                    "analysis_id": identity.analysis_id,
                    "protein_family": identity.protein_family,
                    "landscape": identity.landscape,
                    "model_spec": identity.model_spec,
                    "cohort": cohort_name,
                    "top_k": evaluated_k,
                    "category_column": category_column,
                    "category": category,
                    "count": int(count),
                    "fraction": float(count / evaluated_k),
                }
            )
    return rows


def _mechanism_diagnostic_rows(variant_rows: pd.DataFrame) -> list[dict[str, object]]:
    strata = (
        ("primary", variant_rows["is_primary_cohort"].astype(bool)),
        ("model_top_k", variant_rows["is_model_top_100"].astype(bool)),
        ("challengers", variant_rows["is_challenger_q_ge_001"].astype(bool)),
    )
    rows = []
    for stratum, include in strata:
        selected = variant_rows[include]
        partial = np.nan
        if len(selected):
            try:
                partial = partial_spearman(
                    selected["score_margin_to_rank_100"].to_numpy(dtype=float),
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
                "analysis_id": variant_rows["analysis_id"].iloc[0],
                "protein_family": variant_rows["protein_family"].iloc[0],
                "landscape": variant_rows["landscape"].iloc[0],
                "model_spec": variant_rows["model_spec"].iloc[0],
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
                    selected["score_margin_to_rank_100"],
                    selected["normalized_score_interval_width"],
                ),
                "decision_margin_vs_inclusion_probability_spearman": _safe_spearman(
                    selected["score_margin_to_rank_100"],
                    selected["top_100_inclusion_probability"],
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
    return rows


def _add_within_landscape_model_overlap(
    diagnostics: pd.DataFrame,
    model_top_sets: dict[str, set[int]],
) -> pd.DataFrame:
    result = diagnostics.copy()
    result["within_landscape_model_top_k_pair_count"] = 0
    result["within_landscape_model_top_k_overlap_mean"] = np.nan
    result["within_landscape_model_top_k_overlap_min"] = np.nan
    for index, row in result.iterrows():
        peer_ids = result.loc[
            (result["landscape"] == row["landscape"])
            & (result["analysis_id"] != row["analysis_id"]),
            "analysis_id",
        ]
        overlaps = [
            len(model_top_sets[row["analysis_id"]] & model_top_sets[peer_id])
            / int(row["top_k"])
            for peer_id in peer_ids
        ]
        if overlaps:
            result.loc[index, "within_landscape_model_top_k_pair_count"] = len(overlaps)
            result.loc[index, "within_landscape_model_top_k_overlap_mean"] = np.mean(
                overlaps
            )
            result.loc[index, "within_landscape_model_top_k_overlap_min"] = np.min(
                overlaps
            )
    return result


def _effect_rows(variant_rows: pd.DataFrame) -> list[dict[str, object]]:
    primary = variant_rows[variant_rows["is_primary_cohort"]].copy()
    common = {
        "analysis_id": primary["analysis_id"].iloc[0],
        "protein_family": primary["protein_family"].iloc[0],
        "landscape": primary["landscape"].iloc[0],
        "model_spec": primary["model_spec"].iloc[0],
        "scope": "analysis_run",
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

    score_uncertainty_values = primary["normalized_score_interval_width"].to_numpy()
    crowding_values = primary["score_crowding"].to_numpy()
    try:
        effects.append(
            {
                **common,
                "stage": "rank_propagation",
                **standardized_regression(
                    np.log1p(primary["rank_interval_width"].to_numpy()),
                    {
                        "score_uncertainty": score_uncertainty_values,
                        "crowding": crowding_values,
                        "uncertainty_x_crowding": (
                            score_uncertainty_values * crowding_values
                        ),
                        "log_reference_rank": np.log1p(
                            primary["model_reference_rank"].to_numpy()
                        ),
                    },
                ),
            }
        )
    except ValueError as error:
        effects.append({**common, "stage": "rank_propagation", "error": str(error)})
    return effects


def summarize_mechanisms(
    manifest_path: Path,
    *,
    result_root: Path,
    cache_root: Path,
    feature_root: Path,
    scores_root: Path | None = None,
    legacy_score_root: Path | None = None,
    score_view: str = "full",
    fit_backed: bool = False,
    selected_analysis_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if fit_backed:
        scores_root = None
        legacy_score_root = None
    elif scores_root is None and legacy_score_root is None:
        scores_root = DEFAULT_DATA_ROOT / SCORES_RELATIVE_ROOT
    specs = _manifest(manifest_path)
    if selected_analysis_ids is not None:
        specs = [
            spec
            for spec in specs
            if spec.analysis_id in selected_analysis_ids
        ]
        missing = selected_analysis_ids - {spec.analysis_id for spec in specs}
        if missing:
            raise ValueError(f"Unknown selected analysis IDs: {sorted(missing)}")
    landscape_cache: dict[
        str,
        tuple[dict[str, np.ndarray], dict[str, object]],
    ] = {}
    inventory_runs: dict[str, list[dict[str, object]]] = {}
    variant_frames = []
    functionality_frames = []
    run_rows = []
    effect_rows = []
    reference_rows = []
    reference_category_rows = []
    mechanism_diagnostic_rows = []
    model_top_sets: dict[str, set[int]] = {}

    for index, spec in enumerate(specs, start=1):
        identity = spec.identity
        result_dir = result_root / spec.source_analysis_id
        metadata_path = result_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing paper result: {result_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["analysis_id"] != spec.source_analysis_id:
            raise ValueError(f"Result identity mismatch at {result_dir}.")
        variants = _read_variants_csv(result_dir / "variants.csv")
        with xr.open_dataset(result_dir / "stability.nc", engine="h5netcdf") as ds:
            selection = ds.sel(score_name=spec.primary_score)
            sample_top_indices = selection["variant_index"].values.astype(np.int32)
            model_reference_indices = selection[
                "model_reference_variant_index"
            ].values.astype(np.int32)
            tracked_indices = selection["tracked_variant_index"].values.astype(
                np.int32
            )
            tracked_finite_counts = np.isfinite(
                selection["tracked_score"].values
            ).sum(axis=0)
            reported_supported_indices = tracked_indices[tracked_finite_counts >= 2]
        (
            cohort_indices,
            inclusion_probability,
            strongest,
            n_reported_context_omitted,
        ) = _cohort_indices(
            variants,
            sample_top_indices,
            model_reference_indices,
            reported_supported_indices=reported_supported_indices,
            reference_top_k_identifiable=bool(
                metadata.get("reference_top_k_identifiable", True)
            ),
        )
        cache = _load_or_scan_cache(
            result_dir,
            metadata,
            cohort_indices,
            cache_root / f"{identity.analysis_id}.nc",
            score_name=spec.primary_score,
            scores_root=scores_root,
            legacy_score_root=legacy_score_root,
            score_view=score_view,
        )
        if not np.array_equal(
            cache["reference_order"].values[:REFERENCE_SCALE_K],
            model_reference_indices[:REFERENCE_SCALE_K],
        ):
            raise ValueError(
                f"{identity.analysis_id}: mechanism and paper reference orders differ."
            )
        feature_score = (
            spec.primary_score if identity.protein_family == "k50" else None
        )
        feature_id = landscape_feature_id(
            spec.source_identity.protein_family,
            spec.source_identity.landscape,
            score_name=feature_score,
        )
        if feature_id not in landscape_cache:
            bundle = LandscapeFeatureBundle.load(feature_root / feature_id)
            try:
                bundle.align_result_variants(variants)
                inventory = {
                    name: bundle.metadata[name]
                    for name in bundle.metadata["inventory_fields"]
                }
                if identity.protein_family == "k50":
                    inventory["landscape"] = identity.landscape
                landscape_cache[feature_id] = (
                    {
                        name: bundle.features[name].values.copy()
                        for name in LANDSCAPE_FEATURE_NAMES
                    },
                    inventory,
                )
            finally:
                bundle.close()
        else:
            bundle = LandscapeFeatureBundle.load(feature_root / feature_id)
            try:
                bundle.align_result_variants(variants)
            finally:
                bundle.close()
        raw_features, inventory = landscape_cache[feature_id]
        inventory_runs.setdefault(feature_id, []).append(metadata)
        variant_table = _variant_table(
            identity,
            metadata,
            variants,
            raw_features,
            cache,
            inclusion_probability,
            strongest,
        )
        variant_frames.append(variant_table)
        functionality_frames.append(_functionality_table(identity, cache))
        reference_row, model_top = _reference_diagnostic(
            identity,
            metadata,
            variants,
            cache,
        )
        reference_rows.append(reference_row)
        model_top_sets[identity.analysis_id] = model_top
        reference_category_rows.extend(
            _reference_category_rows(identity, variants, cache)
        )
        mechanism_diagnostic_rows.extend(_mechanism_diagnostic_rows(variant_table))
        run_rows.append(
            _run_table(
                identity,
                metadata,
                cache,
                inclusion_probability,
                sample_top_indices,
                n_reported_context_omitted,
            )
        )
        effect_rows.extend(_effect_rows(variant_table))
        print(
            f"[{index:02d}/{len(specs):02d}] {identity.analysis_id}: "
            f"tracked {len(cohort_indices):,} variants",
            flush=True,
        )
        cache.close()

    inventory_rows = []
    for feature_id, (_, inventory) in landscape_cache.items():
        run_metadata = inventory_runs[feature_id]
        inventory_rows.append(
            {
                **inventory,
                "analysis_run_count": len(run_metadata),
                "artifact_statuses": json.dumps(
                    sorted(
                        {
                            metadata.get("artifact_status", "unrecorded")
                            for metadata in run_metadata
                        }
                    )
                ),
                "artifact_warnings": json.dumps(
                    sorted(
                        {
                            warning
                            for metadata in run_metadata
                            for warning in metadata.get("warnings", [])
                        }
                    )
                ),
            }
        )
    effects = pd.DataFrame(effect_rows)
    if not effects.empty:
        numeric_columns = effects.select_dtypes(include="number").columns.tolist()
        landscape_effects = (
            effects.groupby(
                ["protein_family", "landscape", "stage"],
                as_index=False,
            )[numeric_columns]
            .median(numeric_only=True)
        )
        landscape_effects["scope"] = "landscape_median"
        landscape_effects["analysis_id"] = pd.NA
        landscape_effects["model_spec"] = pd.NA
        effects = pd.concat([effects, landscape_effects], ignore_index=True, sort=False)
    reference_diagnostics = _add_within_landscape_model_overlap(
        pd.DataFrame(reference_rows),
        model_top_sets,
    )
    return {
        "mechanism_inventory": pd.DataFrame(inventory_rows),
        "mechanism_variants": pd.concat(variant_frames, ignore_index=True),
        "mechanism_runs": pd.DataFrame(run_rows),
        "mechanism_effects": effects,
        "reference_diagnostics": reference_diagnostics,
        "reference_category_sensitivity": pd.DataFrame(
            reference_category_rows,
            columns=REFERENCE_CATEGORY_COLUMNS,
        ),
        "mechanism_diagnostics": pd.DataFrame(mechanism_diagnostic_rows),
        "functionality_scores": pd.concat(functionality_frames, ignore_index=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/results",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/mechanisms/cache",
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=None,
        help="Global score root containing method/view collections.",
    )
    parser.add_argument(
        "--proxy-root",
        dest="legacy_score_root",
        type=Path,
        default=None,
        help="Deprecated method-specific score root.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/landscape_features",
    )
    parser.add_argument("--score-view", default="full")
    parser.add_argument(
        "--fit-backed",
        action="store_true",
        help="Read direct-HMC fit artifacts instead of persisted score matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "output/hmc/tables",
    )
    parser.add_argument("--analysis-id", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = summarize_mechanisms(
        args.manifest,
        result_root=args.result_root,
        cache_root=args.cache_root,
        feature_root=args.feature_root,
        scores_root=args.score_root,
        legacy_score_root=args.legacy_score_root,
        score_view=args.score_view,
        fit_backed=args.fit_backed,
        selected_analysis_ids=(
            None if args.analysis_id is None else set(args.analysis_id)
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        suffix = ".csv.gz" if name == "functionality_scores" else ".csv"
        output = args.output_dir / f"{name}{suffix}"
        table.to_csv(output, index=False)
        print(f"Saved {len(table):,} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
