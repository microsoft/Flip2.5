import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import AnalysisIdentity, LandscapeFeatureBundle, ScoreEnsemble
from dms_fitness_stability.analysis_data import (
    build_analysis_data_manifest,
    retrieve_analysis_data_archives,
    validate_analysis_data_manifest,
)
from dms_fitness_stability.migration_audit import sha256_file
from dms_fitness_stability.score_ensembles import POSTERIOR_PARAMETER_TARGET


def _write_score_bundle(root: Path, identity: AnalysisIdentity) -> Path:
    variants = pd.DataFrame(
        {"variant_id": ["A", "B"], "hmc_variant_index": [0, 1]}
    )
    analysis_variants = variants.copy()
    analysis_variants.insert(0, "matrix_variant_index", [0, 1])
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                np.ones((1, 2, 2), dtype=np.float32),
            )
        },
        coords={
            "score_name": ["fitness"],
            "sample": [0, 1],
            "variant": [0, 1],
            "source_sample_id": ("sample", [0, 1]),
            "source_chain": ("sample", [0, 0]),
            "source_draw": ("sample", [0, 1]),
        },
    )
    ensemble = ScoreEnsemble(
        metadata={
            **identity.metadata(),
            "schema_version": 2,
            "fit_method": "hmc_posterior",
            "uncertainty_target": POSTERIOR_PARAMETER_TARGET,
            "score_contracts": {"fitness": "test-v1"},
            "scores": ["fitness"],
            "n_samples": 2,
            "n_variants": 2,
            "n_analysis_variants": 2,
            "n_chains": 1,
            "draws_per_chain": 2,
            "sample_view": "release_1000",
        },
        variants=variants,
        analysis_variants=analysis_variants,
        scores=scores,
    )
    path = root / "paper/analysis_data/scores/hmc/release_1000" / identity.analysis_id
    ensemble.save(path)
    return path


def _write_feature_bundle(root: Path) -> Path:
    variants = pd.DataFrame(
        {"source_variant_index": [0, 1], "variant_id": ["A", "B"]}
    )
    features = xr.Dataset(
        {
            "observation_count": ("variant", np.ones(2, dtype=np.int32)),
            **{
                name: ("variant", np.ones(2, dtype=np.float64))
                for name in (
                    "input_count",
                    "output_count",
                    "log_input_support",
                    "zero_input_fraction",
                    "zero_output_fraction",
                    "reduced_heterogeneity",
                    "enrichment_range",
                    "enrichment_mad",
                )
            },
        },
        coords={"variant": [0, 1]},
    )
    bundle = LandscapeFeatureBundle(
        metadata={
            "schema_version": 1,
            "feature_id": "pdz__pdz-test",
            "protein_family": "pdz",
            "landscape": "pdz-test",
            "score_name": None,
            "n_variants": 2,
            "derivation_contract": "landscape-count-features-v1",
            "inventory_fields": [],
            "source_files": [],
        },
        variants=variants,
        features=features,
    )
    path = root / "paper/analysis_data/landscape_features/pdz__pdz-test"
    bundle.save(path)
    return path


def test_analysis_data_manifest_tracks_and_validates_bundle_relationships(tmp_path):
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape="pdz-test",
        inference_method="hmc",
        model_spec="model",
        fit_id="fit",
    )
    _write_score_bundle(tmp_path, identity)
    _write_feature_bundle(tmp_path)
    empty_mle = tmp_path / "mle"
    empty_count = tmp_path / "count"
    empty_mle.mkdir()
    empty_count.mkdir()

    manifest = build_analysis_data_manifest(
        data_root=tmp_path,
        direct_hmc_root=tmp_path / "paper/analysis_data/scores/hmc",
        mle_score_root=empty_mle,
        hmc_count_score_root=empty_count,
        feature_root=tmp_path / "paper/analysis_data/landscape_features",
    )
    counts = validate_analysis_data_manifest(manifest, data_root=tmp_path)

    assert counts == {"score_bundles": 1, "feature_bundles": 1}
    assert manifest["publication"]["status"] == "local_staging"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    score_file = next(
        tmp_path.rglob("fitness_scores.nc")
    )
    score_file.write_bytes(score_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_analysis_data_manifest(manifest, data_root=tmp_path)


def test_retrieve_analysis_data_archives_verifies_and_extracts_local_zip(tmp_path):
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("paper/analysis_data/example.txt", "data")
    manifest = {
        "schema_version": 1,
        "publication": {"status": "published"},
        "archives": [
            {
                "name": "source.zip",
                "url": archive_path.as_uri(),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "format": "zip",
            }
        ],
    }

    retrieved = retrieve_analysis_data_archives(
        manifest,
        destination=tmp_path / "destination",
        cache_dir=tmp_path / "cache",
    )

    assert [path.name for path in retrieved] == ["source.zip"]
    assert (
        tmp_path / "destination/paper/analysis_data/example.txt"
    ).read_text(encoding="utf-8") == "data"

    manifest["publication"]["status"] = "local_staging"
    with pytest.raises(ValueError, match="requires publication"):
        retrieve_analysis_data_archives(
            manifest,
            destination=tmp_path / "destination",
            cache_dir=tmp_path / "cache",
        )


def test_tracked_analysis_data_uses_schema_v2_score_bundles():
    manifest_path = Path(__file__).parents[1] / "data/analysis_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_records = manifest["score_bundles"]

    assert len(score_records) == 208
    assert {record["schema_version"] for record in score_records} == {2}
    assert {record["storage"]["filename"] for record in score_records} == {
        "fitness_scores.nc"
    }
    assert {record["storage"]["variable"] for record in score_records} == {
        "fitness_score"
    }
    assert all(
        {file["name"] for file in record["files"]}
        == {
            "metadata.json",
            "variants.csv",
            "analysis_variants.csv",
            "fitness_scores.nc",
        }
        for record in score_records
    )


def test_score_schema_v2_activation_links_exact_audits():
    root = Path(__file__).parents[1]
    activation = json.loads(
        (root / "data/score_schema_v2_activation.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = activation["analysis_data_manifest"]
    methods = activation["methods"]
    cleanup = activation["rollback_cleanup"]
    relocation = json.loads(
        (root / "data/score_root_relocation.json").read_text(encoding="utf-8")
    )

    assert activation["status"] in {
        "active_with_schema_v1_rollback",
        "complete",
    }
    assert manifest["score_bundles"] == 208
    assert manifest["feature_bundles"] == 21
    assert manifest["score_schema_version"] == 2
    assert manifest["score_filename"] == "fitness_scores.nc"
    assert relocation["status"] == "complete"
    assert relocation["manifest_sha256_before"] == manifest["sha256"]
    assert sha256_file(root / manifest["path"]) == relocation["manifest_sha256_after"]
    assert methods["mle_predictive"]["bundle_count"] == 61
    assert methods["mle_predictive"]["score_view_count"] == 62
    assert methods["hmc_count_based"]["bundle_count"] == 49
    assert methods["hmc_count_based"]["score_view_count"] == 49
    assert all(
        sha256_file(root / method["equivalence_audit"])
        == method["equivalence_audit_sha256"]
        for method in methods.values()
    )
    assert cleanup["bundle_count"] == sum(
        method["rollback_bundle_count"] for method in methods.values()
    )
    assert cleanup["logical_bytes"] == sum(
        method["rollback_logical_bytes"] for method in methods.values()
    )
    assert cleanup["allocated_bytes"] == sum(
        method["rollback_allocated_bytes"] for method in methods.values()
    )
