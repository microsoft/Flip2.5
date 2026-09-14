"""Build standard analysis bundles from simple tabular research data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import is_bool_dtype

from .identity import AnalysisIdentity
from .landscape_features import (
    LANDSCAPE_FEATURE_DTYPES,
    LANDSCAPE_FEATURE_SCHEMA_VERSION,
    LandscapeFeatureBundle,
    landscape_feature_id,
)
from .mechanisms import paired_replicate_features
from .score_ensembles import SCORE_ENSEMBLE_SCHEMA_VERSION, ScoreEnsemble

TABULAR_SOURCE_SCHEMA_VERSION = 1


def _variant_table(variants: pd.DataFrame) -> pd.DataFrame:
    required = {"variant_id", "is_ranking_candidate"}
    missing = required - set(variants)
    if missing:
        raise ValueError(
            f"Variant table is missing required columns: {sorted(missing)}."
        )
    result = variants.copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("Variant table must contain at least one row.")
    if result["variant_id"].isna().any() or result["variant_id"].duplicated().any():
        raise ValueError("variant_id must contain unique, non-missing values.")
    if (
        result["is_ranking_candidate"].isna().any()
        or not is_bool_dtype(result["is_ranking_candidate"].dtype)
    ):
        raise ValueError("is_ranking_candidate must contain non-missing booleans.")
    if not result["is_ranking_candidate"].any():
        raise ValueError("At least one variant must be eligible for ranking.")
    result["variant_id"] = result["variant_id"].astype(str)
    return result


def build_score_ensemble_from_table(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    score_contracts: Mapping[str, str],
    uncertainty_target: str,
    fit_method: str,
    sample_kind: str,
    extra_metadata: Mapping[str, object] | None = None,
) -> ScoreEnsemble:
    """Build a canonical score bundle from long sample-by-variant score rows."""
    variant_table = _variant_table(variants)
    required_score_columns = {
        "sample_id",
        "variant_id",
        "score_name",
        "fitness_score",
    }
    missing = required_score_columns - set(scores)
    if missing:
        raise ValueError(
            f"Score table is missing required columns: {sorted(missing)}."
        )
    if scores.empty:
        raise ValueError("Score table must contain at least one row.")
    score_names = list(score_contracts)
    if not score_names or any(
        not isinstance(name, str) or not name.strip() for name in score_names
    ):
        raise ValueError("score_contracts must contain at least one named score.")
    if any(
        not isinstance(contract, str) or not contract.strip()
        for contract in score_contracts.values()
    ):
        raise ValueError("Every score contract must be a non-empty string.")

    rows = scores[list(required_score_columns)].copy()
    rows["sample_id"] = pd.to_numeric(rows["sample_id"], errors="raise")
    if (
        rows["sample_id"].isna().any()
        or not np.equal(rows["sample_id"], np.floor(rows["sample_id"])).all()
    ):
        raise ValueError("sample_id must contain non-missing integers.")
    rows["sample_id"] = rows["sample_id"].astype(np.int64)
    rows["variant_id"] = rows["variant_id"].astype(str)
    rows["score_name"] = rows["score_name"].astype(str)
    rows["fitness_score"] = pd.to_numeric(
        rows["fitness_score"],
        errors="raise",
    ).astype(float)
    if np.isinf(rows["fitness_score"]).any():
        raise ValueError("fitness_score may be finite or missing, but not infinite.")
    if rows.duplicated(["score_name", "sample_id", "variant_id"]).any():
        raise ValueError(
            "Score table must contain one row per score_name, sample_id, and variant_id."
        )

    observed_score_names = set(rows["score_name"])
    if observed_score_names != set(score_names):
        raise ValueError(
            "Score table names do not match score_contracts: "
            f"observed={sorted(observed_score_names)}, expected={sorted(score_names)}."
        )
    sample_ids = np.sort(rows["sample_id"].unique())
    if len(sample_ids) < 2:
        raise ValueError("Score table must contain at least two sample IDs.")
    variant_ids = variant_table["variant_id"].tolist()
    expected_index = pd.MultiIndex.from_product(
        [score_names, sample_ids, variant_ids],
        names=["score_name", "sample_id", "variant_id"],
    )
    indexed = rows.set_index(["score_name", "sample_id", "variant_id"])
    missing_rows = expected_index.difference(indexed.index)
    extra_rows = indexed.index.difference(expected_index)
    if len(missing_rows) or len(extra_rows):
        raise ValueError(
            "Score table must be a complete score-by-sample-by-variant rectangle: "
            f"missing_rows={len(missing_rows)}, unexpected_rows={len(extra_rows)}."
        )
    matrix = indexed.reindex(expected_index)["fitness_score"].to_numpy(
        dtype=np.float32
    ).reshape(len(score_names), len(sample_ids), len(variant_ids))
    score_dataset = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                matrix,
            )
        },
        coords={
            "score_name": score_names,
            "sample": np.arange(len(sample_ids), dtype=np.int32),
            "variant": np.arange(len(variant_ids), dtype=np.int32),
            "source_sample_id": ("sample", sample_ids),
        },
    )
    analysis_variants = variant_table.copy()
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        np.arange(len(variant_table), dtype=np.int64),
    )
    metadata = {
        **dict(extra_metadata or {}),
        **identity.metadata(),
        "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
        "fit_method": fit_method,
        "uncertainty_target": uncertainty_target,
        "sample_kind": sample_kind,
        "score_contracts": dict(score_contracts),
        "scores": score_names,
        "n_samples": len(sample_ids),
        "n_variants": len(variant_table),
        "n_analysis_variants": len(analysis_variants),
        "source_format": "long_tabular_scores_v1",
    }
    ensemble = ScoreEnsemble(
        metadata=metadata,
        variants=variant_table[["variant_id"]].copy(),
        analysis_variants=analysis_variants,
        scores=score_dataset,
    )
    ensemble.validate()
    return ensemble


def build_paired_feature_bundle_from_table(
    variants: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    extra_metadata: Mapping[str, object] | None = None,
) -> LandscapeFeatureBundle:
    """Build audited paired input/output features from long replicate rows."""
    variant_table = _variant_table(variants)
    required_count_columns = {
        "variant_id",
        "replicate_id",
        "input_count",
        "output_count",
    }
    missing = required_count_columns - set(counts)
    if missing:
        raise ValueError(
            f"Count table is missing required columns: {sorted(missing)}."
        )
    rows = counts[list(required_count_columns)].copy()
    if rows.empty or rows[["variant_id", "replicate_id"]].isna().any().any():
        raise ValueError("Count table variant_id and replicate_id cannot be missing.")
    rows["variant_id"] = rows["variant_id"].astype(str)
    if rows.duplicated(["replicate_id", "variant_id"]).any():
        raise ValueError(
            "Count table must contain one row per replicate_id and variant_id."
        )
    for name in ("input_count", "output_count"):
        rows[name] = pd.to_numeric(rows[name], errors="raise").astype(float)
        if not np.isfinite(rows[name]).all() or (rows[name] < 0).any():
            raise ValueError(f"{name} must contain finite, non-negative values.")
    replicate_ids = sorted(rows["replicate_id"].unique(), key=str)
    if len(replicate_ids) < 2:
        raise ValueError("Paired count features require at least two replicates.")
    variant_ids = variant_table["variant_id"].tolist()
    expected_index = pd.MultiIndex.from_product(
        [replicate_ids, variant_ids],
        names=["replicate_id", "variant_id"],
    )
    indexed = rows.set_index(["replicate_id", "variant_id"])
    missing_rows = expected_index.difference(indexed.index)
    extra_rows = indexed.index.difference(expected_index)
    if len(missing_rows) or len(extra_rows):
        raise ValueError(
            "Count table must be a complete replicate-by-variant rectangle: "
            f"missing_rows={len(missing_rows)}, unexpected_rows={len(extra_rows)}."
        )
    aligned = indexed.reindex(expected_index)
    shape = (len(replicate_ids), len(variant_ids))
    input_counts = aligned["input_count"].to_numpy(dtype=float).reshape(shape)
    output_counts = aligned["output_count"].to_numpy(dtype=float).reshape(shape)
    calculated = paired_replicate_features(input_counts, output_counts)
    features = xr.Dataset(
        {
            name: (
                "variant",
                np.asarray(calculated[name], dtype=dtype),
            )
            for name, dtype in LANDSCAPE_FEATURE_DTYPES.items()
        },
        coords={"variant": np.arange(len(variant_ids), dtype=np.int32)},
    )
    inventory = {
        "observation_structure": "paired_input_output_replicates",
        "group_count": 1,
        "replicate_count": len(replicate_ids),
        "discordance_available": True,
        "starting_count_shape": json.dumps(list(input_counts.shape)),
        "output_count_shape": json.dumps(list(output_counts.shape)),
        "input_library_depths": json.dumps(
            input_counts.sum(axis=-1, dtype=np.float64).tolist()
        ),
        "output_library_depths": json.dumps(
            output_counts.sum(axis=-1, dtype=np.float64).tolist()
        ),
        "zero_input_fraction_all_variants": float(np.mean(input_counts == 0)),
        "zero_output_fraction_all_variants": float(np.mean(output_counts == 0)),
    }
    metadata = {
        **dict(extra_metadata or {}),
        **inventory,
        "schema_version": LANDSCAPE_FEATURE_SCHEMA_VERSION,
        "feature_id": landscape_feature_id(
            identity.protein_family,
            identity.landscape,
        ),
        "protein_family": identity.protein_family,
        "landscape": identity.landscape,
        "score_name": None,
        "n_variants": len(variant_ids),
        "derivation_contract": "landscape-count-features-v1",
        "inventory_fields": list(inventory),
        "source_format": "long_paired_counts_v1",
    }
    bundle = LandscapeFeatureBundle(
        metadata=metadata,
        variants=pd.DataFrame(
            {
                "source_variant_index": np.arange(
                    len(variant_ids),
                    dtype=np.int64,
                ),
                "variant_id": variant_ids,
            }
        ),
        features=features,
    )
    bundle.validate()
    return bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tabular_source(
    path: str | Path,
) -> tuple[ScoreEnsemble, LandscapeFeatureBundle | None]:
    """Load a JSON-described tabular source into standard analysis bundles."""
    descriptor_path = Path(path)
    try:
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Tabular source descriptor is not valid JSON at {descriptor_path}: "
            f"{error.msg}."
        ) from error
    required = {
        "schema_version",
        "identity",
        "variants",
        "scores",
        "score_contracts",
        "uncertainty_target",
        "fit_method",
        "sample_kind",
    }
    optional = {"counts"}
    if not isinstance(payload, Mapping):
        raise ValueError("Tabular source descriptor must be a JSON object.")
    missing = required - set(payload)
    extra = set(payload) - required - optional
    if missing or extra:
        raise ValueError(
            "Tabular source descriptor fields differ from schema v1: "
            f"missing={sorted(missing)}, unknown={sorted(extra)}."
        )
    if payload["schema_version"] != TABULAR_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            "Tabular source descriptor schema_version must be "
            f"{TABULAR_SOURCE_SCHEMA_VERSION}."
        )
    identity_fields = {
        "protein_family",
        "landscape",
        "inference_method",
        "model_spec",
        "fit_id",
    }
    identity_payload = payload["identity"]
    if not isinstance(identity_payload, Mapping) or set(identity_payload) != (
        identity_fields
    ):
        observed = (
            sorted(identity_payload)
            if isinstance(identity_payload, Mapping)
            else type(identity_payload).__name__
        )
        raise ValueError(
            "identity must contain exactly protein_family, landscape, "
            f"inference_method, model_spec, and fit_id; observed={observed}."
        )
    identity = AnalysisIdentity(
        **{name: str(identity_payload[name]) for name in identity_fields}
    )
    root = descriptor_path.parent
    source_paths: dict[str, Path] = {}
    for field in ("variants", "scores", "counts"):
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty relative CSV path.")
        source_path = (root / value).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Tabular source {field} file is missing: {source_path}")
        source_paths[field] = source_path
    source_records = [
        {
            "role": role,
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
        for role, source_path in source_paths.items()
    ]
    provenance = {
        "source_descriptor": str(descriptor_path.resolve()),
        "source_descriptor_sha256": _sha256_file(descriptor_path),
        "source_files": source_records,
    }
    variants = pd.read_csv(source_paths["variants"], low_memory=False)
    score_rows = pd.read_csv(source_paths["scores"], low_memory=False)
    score_contracts = payload["score_contracts"]
    if not isinstance(score_contracts, Mapping):
        raise ValueError("score_contracts must be a JSON object.")
    ensemble = build_score_ensemble_from_table(
        variants,
        score_rows,
        identity=identity,
        score_contracts={
            str(name): str(contract) for name, contract in score_contracts.items()
        },
        uncertainty_target=str(payload["uncertainty_target"]),
        fit_method=str(payload["fit_method"]),
        sample_kind=str(payload["sample_kind"]),
        extra_metadata=provenance,
    )
    feature_bundle = None
    if "counts" in source_paths:
        count_rows = pd.read_csv(source_paths["counts"], low_memory=False)
        feature_bundle = build_paired_feature_bundle_from_table(
            variants,
            count_rows,
            identity=identity,
            extra_metadata=provenance,
        )
    return ensemble, feature_bundle
