import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.migration_audit import audit_score_ensemble_bundle
from dms_fitness_stability.score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    LEGACY_PROXY_SCHEMA_VERSION,
)
from dms_fitness_stability.score_migration import convert_legacy_score_bundle


def _legacy_bundle(tmp_path, write_schema_v1_bundle):
    identity = AnalysisIdentity(
        protein_family="synthetic",
        landscape="test",
        inference_method="mle_predictive",
        model_spec="model",
        fit_id="fit",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B"],
            "reported_fitness": [2.0, 1.0],
            "count_ratio": [2.1824644549763037, 1.0],
        }
    )
    fitness_proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.array([[[2.0, 1.0], [1.5, np.nan]]], dtype=np.float32),
            ),
            "observed_proxy": (
                ("score_name", "variant"),
                np.array([[2.0, 1.0]], dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["fitness"],
            "sample": np.arange(2),
            "variant": np.arange(2),
        },
    )
    return write_schema_v1_bundle(
        tmp_path / identity.analysis_id,
        metadata={
            **identity.metadata(),
            "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
            "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
            "fit_method": "mle",
            "score_contracts": {"fitness": "synthetic-fitness-v1"},
            "scores": ["fitness"],
            "n_samples": 2,
            "n_variants": 2,
        },
        variants=variants,
        legacy_dataset=fitness_proxies,
    )


def test_convert_legacy_score_bundle_preserves_semantics(
    tmp_path,
    write_schema_v1_bundle,
):
    bundle_dir = _legacy_bundle(tmp_path, write_schema_v1_bundle)
    before = audit_score_ensemble_bundle(bundle_dir, sample_chunk_size=1)

    record = convert_legacy_score_bundle(
        bundle_dir,
        expected_semantic_sha256=before["semantic_sha256"],
        sample_chunk_size=1,
    )
    after = audit_score_ensemble_bundle(bundle_dir, sample_chunk_size=2)

    assert record["status"] == "converted"
    assert after["source_schema_version"] == 2
    assert after["semantic"] == before["semantic"]
    assert after["semantic_sha256"] == before["semantic_sha256"]
    assert set(after["files"]) == {
        "metadata.json",
        "variants.csv",
        "analysis_variants.csv",
        "fitness_scores.nc",
    }
    assert not (bundle_dir / "fitness_proxies.nc").exists()


def test_convert_legacy_score_bundle_rejects_unfrozen_semantics(
    tmp_path,
    write_schema_v1_bundle,
):
    bundle_dir = _legacy_bundle(tmp_path, write_schema_v1_bundle)

    with pytest.raises(ValueError, match="differ from the frozen baseline"):
        convert_legacy_score_bundle(
            bundle_dir,
            expected_semantic_sha256="0" * 64,
            sample_chunk_size=1,
        )

    assert (bundle_dir / "fitness_proxies.nc").is_file()
    assert not (bundle_dir / "fitness_scores.nc").exists()
