import copy

import numpy as np
import pandas as pd
import xarray as xr

from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.migration_audit import (
    HMC_TABLE_FILENAMES,
    audit_result_bundle,
    audit_score_ensemble_bundle,
    build_hmc_migration_audit,
    build_score_method_baseline,
    verify_hmc_migration_audit,
    verify_score_method_conversion,
)
from dms_fitness_stability.results import StabilityResult
from dms_fitness_stability.score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    LEGACY_PROXY_SCHEMA_VERSION,
    SCORE_ENSEMBLE_SCHEMA_VERSION,
    ScoreEnsemble,
)
from dms_fitness_stability.score_migration import convert_legacy_score_bundle


def _result() -> StabilityResult:
    selection = xr.Dataset(
        data_vars={
            "variant_index": (
                ("score_name", "sample", "selection_rank"),
                np.array([[[1, 0], [0, 1], [1, 0]]], dtype=np.int32),
            ),
            "score": (
                ("score_name", "sample", "selection_rank"),
                np.array([[[0.9, 0.2], [1.0, 0.1], [0.8, 0.3]]], dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["fitness"],
            "sample": np.arange(3, dtype=np.int32),
            "sample_id": ("sample", np.array([4, 8, 12], dtype=np.int64)),
            "selection_rank": np.arange(2, dtype=np.int32),
        },
        attrs={"analysis_id": "pdz__pdz-test__hmc__model__fit"},
    )
    return StabilityResult(
        metadata={
            "analysis_id": "pdz__pdz-test__hmc__model__fit",
            "n_samples": 3,
            "retained_k": 2,
            "tracked_reported_top_n": 0,
            "tracked_model_top_n": 0,
            "top_k": [1, 2],
            "seed": 23,
            "interval": [0.05, 0.95],
        },
        variants=pd.DataFrame(
            {
                "variant_id": ["AA", "AB"],
                "hmc_variant_index": [0, 1],
            }
        ),
        selection=selection,
        summary=pd.DataFrame({"score_name": ["fitness"], "mean": [0.5]}),
        overlaps=pd.DataFrame({"score_name": ["fitness"], "k": [1]}),
    )


def test_audit_result_bundle_is_deterministic_and_tracks_semantic_arrays(tmp_path):
    result = _result()
    first_path = result.save(tmp_path / result.metadata["analysis_id"])

    first = audit_result_bundle(first_path, sample_chunk_size=1)
    second = audit_result_bundle(first_path, sample_chunk_size=2)

    assert first == second
    assert first["sample_id_sha256"]
    assert set(first["integer_selection_sha256"]) == {"variant_index"}
    assert first["selection"]["variables"]["score"]["dtype"] == "<f4"

    changed_result = copy.deepcopy(result)
    changed_result.selection["variant_index"][0, 0, 0] = 0
    changed_path = changed_result.save(
        tmp_path / "changed" / result.metadata["analysis_id"]
    )
    changed = audit_result_bundle(changed_path)

    assert (
        first["integer_selection_sha256"]["variant_index"]
        != changed["integer_selection_sha256"]["variant_index"]
    )


def test_score_bundle_audit_is_stable_across_v1_to_v2_names(
    tmp_path,
    write_schema_v1_bundle,
):
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
            "is_ranking_candidate": [True, True],
        }
    )
    fitness_proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            ),
            "observed_proxy": (
                ("score_name", "variant"),
                np.array([[2.0, 3.0]], dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["fitness"],
            "sample": np.arange(2),
            "variant": np.arange(2),
        },
    )
    common_metadata = {
        **identity.metadata(),
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "score_contracts": {"fitness": "synthetic-fitness-v1"},
        "scores": ["fitness"],
        "n_samples": 2,
        "n_variants": 2,
    }
    v1_path = write_schema_v1_bundle(
        tmp_path / "v1",
        metadata={
            **common_metadata,
            "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
        },
        variants=variants,
        legacy_dataset=fitness_proxies,
    )

    analysis_variants = variants.copy()
    analysis_variants.insert(0, "matrix_variant_index", [0, 1])
    v2 = ScoreEnsemble(
        metadata={
            **common_metadata,
            "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
            "fit_method": "mle_predictive",
            "n_analysis_variants": 2,
        },
        variants=variants,
        analysis_variants=analysis_variants,
        scores=fitness_proxies.rename(
            {
                "fitness_proxy": "fitness_score",
                "observed_proxy": "observed_score",
            }
        ),
    )
    v2_path = v2.save(tmp_path / "v2")

    v1_audit = audit_score_ensemble_bundle(v1_path, sample_chunk_size=1)
    v2_audit = audit_score_ensemble_bundle(v2_path, sample_chunk_size=2)

    assert v1_audit["source_schema_version"] == 1
    assert v2_audit["source_schema_version"] == 2
    assert set(v1_audit["files"]) != set(v2_audit["files"])
    assert v1_audit["semantic"] == v2_audit["semantic"]
    assert v1_audit["semantic_sha256"] == v2_audit["semantic_sha256"]


def test_build_hmc_migration_audit_records_all_baseline_surfaces(tmp_path):
    result = _result()
    analysis_id = result.metadata["analysis_id"]
    result.metadata.update(
        {
            "protein_family": "pdz",
            "landscape": "pdz-test",
            "inference_method": "hmc",
            "model_spec": "model",
            "fit_id": "fit",
            "job_id": 1,
            "artifact_kind": "diagnosed_hmc",
            "primary_score": "fitness",
            "posterior_variable": "log_r_mean",
            "posterior_transform": "identity",
        }
    )
    result.selection.attrs.update(
        {
            "protein_family": "pdz",
            "landscape": "pdz-test",
            "inference_method": "hmc",
            "model_spec": "model",
            "fit_id": "fit",
        }
    )
    result_root = tmp_path / "results"
    result.save(result_root / analysis_id)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    xr.Dataset(
        {"score": (("sample", "variant"), np.ones((3, 2), dtype=np.float32))},
        coords={
            "sample": np.arange(3, dtype=np.int32),
            "sample_id": ("sample", np.array([4, 8, 12], dtype=np.int64)),
            "variant": np.arange(2, dtype=np.int32),
        },
        attrs={"analysis_id": analysis_id},
    ).to_netcdf(cache_root / f"{analysis_id}.nc", engine="h5netcdf")

    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    for name in HMC_TABLE_FILENAMES:
        pd.DataFrame({"analysis_id": [analysis_id], "value": [1]}).to_csv(
            table_dir / name,
            index=False,
        )

    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "figure1.png").write_bytes(b"png")
    (figure_dir / "figure1.pdf").write_bytes(b"pdf")

    manifest_path = tmp_path / "analyses.json"
    manifest_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": 2,
                "analyses": [
                    {
                        "protein_family": "pdz",
                        "landscape": "pdz-test",
                        "inference_method": "hmc",
                        "model_spec": "model",
                        "fit_id": "fit",
                        "job_id": 1,
                        "artifact_kind": "diagnosed_hmc",
                        "primary_score": "fitness",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fit_audit_path = tmp_path / "fit_audit.json"
    fit_audit_path.write_text(
        __import__("json").dumps(
            [
                {
                    "job_id": 1,
                    "diagnosed_hmc": [
                        {
                            "path": "/fits/fit.nc",
                            "kind": "diagnosed_hmc",
                            "status": "validated",
                            "bytes": 123,
                            "score_variable": "log_r_mean",
                            "score_shape": [1, 3, 2],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    audit = build_hmc_migration_audit(
        manifest_path=manifest_path,
        fit_audit_path=fit_audit_path,
        result_root=result_root,
        cache_root=cache_root,
        table_dir=table_dir,
        figure_dir=figure_dir,
        code_paths={"analyses.json": manifest_path},
        repository_commit="abc123",
        sample_chunk_size=1,
    )

    assert audit["manifest"]["analysis_count"] == 1
    assert audit["analyses"][0]["source_score"]["score_shape"] == [1, 3, 2]
    assert set(audit["tables"]) == set(HMC_TABLE_FILENAMES)
    assert set(audit["figures"]) == {"figure1.pdf", "figure1.png"}
    assert audit["repository_commit"] == "abc123"

    changed = copy.deepcopy(audit)
    changed["analyses"][0]["result"]["run_settings"]["seed"] = 24
    verify_hmc_migration_audit(audit, copy.deepcopy(audit))
    with __import__("pytest").raises(ValueError, match="baseline changed"):
        verify_hmc_migration_audit(audit, changed)

    code_changed = copy.deepcopy(audit)
    code_changed["code_inputs"]["analyses.json"] = "changed"
    verify_hmc_migration_audit(audit, code_changed)
    with __import__("pytest").raises(ValueError, match="baseline changed"):
        verify_hmc_migration_audit(
            audit,
            code_changed,
            strict_code_inputs=True,
        )


def test_build_score_method_baseline_records_schema_v1_surfaces(
    tmp_path,
    write_schema_v1_bundle,
):
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape="pdz-test",
        inference_method="mle_predictive",
        model_spec="model",
        fit_id="fit",
    )
    analysis_id = identity.analysis_id
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B"],
            "reported_fitness": [2.0, 1.0],
            "reported_rank": [1.0, 2.0],
        }
    )
    fitness_proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.array([[[2.0, 1.0], [1.5, 1.0]]], dtype=np.float32),
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
    score_root = tmp_path / "proxies"
    write_schema_v1_bundle(
        score_root / analysis_id,
        metadata={
            **identity.metadata(),
            "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
            "job_id": 1,
            "primary_score": "fitness",
            "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
            "score_contracts": {"fitness": "test-v1"},
            "scores": ["fitness"],
            "n_samples": 2,
            "n_variants": 2,
            "source_artifact": {"path": "/retired/fit.nc", "bytes": 123},
        },
        variants=variants,
        legacy_dataset=fitness_proxies,
    )

    result = _result()
    result.metadata.update(
        {
            **identity.metadata(),
            "job_id": 1,
            "primary_score": "fitness",
        }
    )
    result.selection.attrs["analysis_id"] = analysis_id
    result_root = tmp_path / "results"
    result.save(result_root / analysis_id)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    xr.Dataset(
        {"score": (("sample", "variant"), np.ones((2, 2), dtype=np.float32))},
        attrs={"analysis_id": analysis_id},
    ).to_netcdf(cache_root / f"{analysis_id}.nc", engine="h5netcdf")
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    for name in HMC_TABLE_FILENAMES:
        pd.DataFrame({"analysis_id": [analysis_id], "value": [1]}).to_csv(
            table_dir / name,
            index=False,
        )
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "figure.png").write_bytes(b"png")
    (figure_dir / "figure.pdf").write_bytes(b"pdf")
    manifest_path = tmp_path / "analyses.json"
    manifest_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": 2,
                "analyses": [
                    {
                        "protein_family": "pdz",
                        "landscape": "pdz-test",
                        "inference_method": "mle_predictive",
                        "model_spec": "model",
                        "fit_id": "fit",
                        "job_id": 1,
                        "artifact_kind": "sampled_mle",
                        "primary_score": "fitness",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = build_score_method_baseline(
        manifest_path=manifest_path,
        score_root=score_root,
        result_root=result_root,
        cache_root=cache_root,
        table_dir=table_dir,
        figure_dir=figure_dir,
        inference_method="mle_predictive",
        repository_commit="abc123",
        sample_chunk_size=1,
    )

    assert audit["manifest"]["analysis_count"] == 1
    assert audit["manifest"]["score_view_count"] == 1
    assert audit["analyses"][0]["score_bundle"]["source_schema_version"] == 1
    assert audit["analyses"][0]["source_artifact"]["path"] == "/retired/fit.nc"
    assert set(audit["mechanism_caches"]) == {analysis_id}
    assert set(audit["tables"]) == set(HMC_TABLE_FILENAMES)
    assert set(audit["figures"]) == {"figure.pdf", "figure.png"}

    convert_legacy_score_bundle(
        score_root / analysis_id,
        expected_semantic_sha256=audit["analyses"][0]["score_bundle"][
            "semantic_sha256"
        ],
        sample_chunk_size=1,
    )
    converted = build_score_method_baseline(
        manifest_path=manifest_path,
        score_root=score_root,
        result_root=result_root,
        cache_root=cache_root,
        table_dir=table_dir,
        figure_dir=figure_dir,
        inference_method="mle_predictive",
        repository_commit="converted",
        sample_chunk_size=1,
    )

    verify_score_method_conversion(audit, converted)
    changed = copy.deepcopy(converted)
    changed["analyses"][0]["result"]["run_settings"]["seed"] = 24
    with __import__("pytest").raises(
        ValueError,
        match="changed a frozen scientific artifact",
    ):
        verify_score_method_conversion(audit, changed)
