import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dms_fitness_stability import AnalysisIdentity, ScoreEnsemble
from dms_fitness_stability.score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    LEGACY_PROXY_SCHEMA_VERSION,
    POSTERIOR_PARAMETER_TARGET,
)


def _mechanism_module():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("public_mechanisms", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mechanism_source_resolves_legacy_and_schema_v2_score_files(
    tmp_path,
    write_schema_v1_bundle,
):
    module = _mechanism_module()
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G3",
        inference_method="mle_predictive",
        model_spec="exponential",
        fit_id="fit",
    )
    variants = pd.DataFrame({"variant_id": ["A", "B"]})
    fitness_proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.ones((1, 2, 2), dtype=np.float32),
            ),
            "observed_proxy": (
                ("score_name", "variant"),
                np.ones((1, 2), dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["fitness"],
            "sample": np.arange(2),
            "variant": np.arange(2),
        },
    )
    metadata = {
        **identity.metadata(),
        "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "score_contracts": {"fitness": "test-v1"},
        "scores": ["fitness"],
        "n_samples": 2,
        "n_variants": 2,
    }
    legacy_root = tmp_path / "legacy"
    write_schema_v1_bundle(
        legacy_root / identity.analysis_id,
        metadata=metadata,
        variants=variants,
        legacy_dataset=fitness_proxies,
    )
    schema_v2_root = tmp_path / "schema-v2"
    ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=fitness_proxies.rename(
            {
                "fitness_proxy": "fitness_score",
                "observed_proxy": "observed_score",
            }
        ),
    ).save(schema_v2_root / identity.analysis_id)
    result_metadata = {
        **identity.metadata(),
        "inference_method": "mle_predictive",
    }

    legacy = module._score_source(
        result_metadata,
        score_name="fitness",
        legacy_score_root=legacy_root,
    )
    schema_v2 = module._score_source(
        result_metadata,
        score_name="fitness",
        legacy_score_root=schema_v2_root,
    )

    assert Path(legacy["path"]).name == "fitness_proxies.nc"
    assert Path(legacy["bundle_dir"]) == legacy_root / identity.analysis_id
    assert Path(schema_v2["path"]).name == "fitness_scores.nc"
    assert Path(schema_v2["bundle_dir"]) == schema_v2_root / identity.analysis_id


def test_historical_mechanism_cache_requires_matching_score_source():
    module = _mechanism_module()
    source_artifact = {
        "path": "/fits/source.nc",
        "size": 123,
        "mtime_ns": 456,
    }
    metadata = {
        "analysis_id": "analysis",
        "hmc_file": source_artifact,
    }
    source = {
        "kind": "score_ensemble",
        "path": "/scores/fitness_scores.nc",
        "size": 789,
        "mtime_ns": 1011,
        "score_contract": "score-v1",
        "source_artifact": source_artifact,
    }
    cache = xr.Dataset(
        {
            "variant_index": ("mechanism_variant", [0, 1]),
            "candidate_reference_score": ("candidate", [2.0, 1.0]),
        },
        attrs={
            "analysis_id": "analysis",
            "score_name": "score",
            "hmc_size": 123,
            "hmc_mtime_ns": 456,
        },
    )
    cohort = np.array([0, 1], dtype=np.int32)
    candidates = np.array([0, 1], dtype=np.int32)

    assert module._cache_valid(
        cache,
        metadata=metadata,
        cohort_indices=cohort,
        candidate_indices=candidates,
        score_name="score",
        source=source,
    )

    changed_source = {**source, "source_artifact": {**source_artifact, "size": 124}}
    assert not module._cache_valid(
        cache,
        metadata=metadata,
        cohort_indices=cohort,
        candidate_indices=candidates,
        score_name="score",
        source=changed_source,
    )


def test_fit_backed_v4_cache_matches_equivalent_score_source():
    module = _mechanism_module()
    artifact = {
        "path": "/fits/source.nc",
        "size": 123,
        "mtime_ns": 456,
    }
    metadata = {
        "analysis_id": "analysis",
        "posterior_variable": "log_score",
        "posterior_transform": "identity",
    }
    source = {
        "kind": "score_ensemble",
        "path": "/scores/fitness_scores.nc",
        "size": 789,
        "mtime_ns": 1011,
        "score_contract": "score-v1",
        "source_artifact": artifact,
    }
    cache = xr.Dataset(
        {
            "variant_index": ("mechanism_variant", [0, 1]),
            "candidate_reference_score": ("candidate", [2.0, 1.0]),
        },
        attrs={
            "analysis_id": "analysis",
            "score_name": "score",
            "candidate_scope_version": 4,
            "source_kind": "hmc_posterior",
            "source_path": artifact["path"],
            "source_size": artifact["size"],
            "source_mtime_ns": artifact["mtime_ns"],
            "score_contract": "posterior/log_score:identity",
        },
    )
    indices = np.array([0, 1], dtype=np.int32)

    assert module._cache_valid(
        cache,
        metadata=metadata,
        cohort_indices=indices,
        candidate_indices=indices,
        score_name="score",
        source=source,
    )

    cache.attrs["score_contract"] = "posterior/other:identity"
    assert not module._cache_valid(
        cache,
        metadata=metadata,
        cohort_indices=indices,
        candidate_indices=indices,
        score_name="score",
        source=source,
    )


def test_direct_hmc_mechanism_cache_rebuild_uses_only_score_bundle(tmp_path):
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G3",
        inference_method="hmc",
        model_spec="exponential",
        fit_id="missing-fit",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "A3D"],
            "hmc_variant_index": np.arange(3, dtype=np.int32),
            "is_ranking_candidate": [True, True, False],
            "reported_fitness": [0.2, 0.3, np.nan],
            "reported_rank": [2.0, 1.0, np.nan],
        }
    )
    analysis_variants = variants.copy()
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        np.arange(3, dtype=np.int64),
    )
    sampled_scores = np.array(
        [
            [1.0, 3.0, 100.0],
            [2.0, 2.5, 100.0],
            [3.0, 2.0, 100.0],
        ],
        dtype=np.float32,
    )
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                sampled_scores[None],
            )
        },
        coords={
            "score_name": ["model_fluorescence"],
            "sample": np.arange(3, dtype=np.int32),
            "variant": np.arange(3, dtype=np.int32),
            "source_sample_id": ("sample", np.arange(3, dtype=np.int64)),
            "source_chain": ("sample", np.zeros(3, dtype=np.int32)),
            "source_draw": ("sample", np.arange(3, dtype=np.int32)),
        },
    )
    scores_root = tmp_path / "scores"
    bundle_dir = scores_root / "hmc/full" / identity.analysis_id
    ensemble = ScoreEnsemble(
        metadata={
            **identity.metadata(),
            "schema_version": 2,
            "fit_method": "hmc_posterior",
            "uncertainty_target": POSTERIOR_PARAMETER_TARGET,
            "posterior_variable": "log_fluorescence",
            "posterior_transform": "identity",
            "score_contracts": {"model_fluorescence": "test-identity-v1"},
            "scores": ["model_fluorescence"],
            "n_chains": 1,
            "draws_per_chain": 3,
            "n_samples": 3,
            "n_variants": 3,
            "n_analysis_variants": 3,
            "sample_view": "full",
            "variant_chunk_size": 2,
            "source_artifact": {
                "path": str(tmp_path / "does-not-exist.nc"),
                "size": 1,
                "mtime_ns": 1,
            },
        },
        variants=variants,
        analysis_variants=analysis_variants,
        scores=scores,
    )
    ensemble.save(bundle_dir)

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    variants.to_csv(result_dir / "variants.csv", index=False)
    xr.Dataset(
        coords={
            "score_name": ["model_fluorescence"],
            "sample": np.arange(3, dtype=np.int32),
            "sample_id": ("sample", np.arange(3, dtype=np.int64)),
        }
    ).to_netcdf(result_dir / "stability.nc", engine="h5netcdf")
    metadata = {
        **identity.metadata(),
        "posterior_variable": "log_fluorescence",
        "posterior_transform": "identity",
        "hmc_file": {
            "path": str(tmp_path / "does-not-exist.nc"),
            "size": 1,
            "mtime_ns": 1,
        },
    }

    summary = _mechanism_module()
    cache = summary._load_or_scan_cache(
        result_dir,
        metadata,
        np.array([0, 1], dtype=np.int32),
        tmp_path / "cache.nc",
        score_name="model_fluorescence",
        scores_root=scores_root,
        score_view="full",
    )
    try:
        assert cache.attrs["source_kind"] == "score_ensemble"
        np.testing.assert_array_equal(cache["candidate_variant_index"], [0, 1])
        np.testing.assert_allclose(cache["candidate_reference_score"], [2.0, 2.5])
        assert not (tmp_path / "does-not-exist.nc").exists()
    finally:
        cache.close()
