import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dms_fitness_stability.analysis_manifest import (
    AnalysisSpec,
    ArtifactResolver,
    load_analysis_manifest,
)


def _spec(*, job_id=1, artifact_kind="diagnosed_hmc"):
    return AnalysisSpec.from_dict(
        {
            "protein_family": "pdz",
            "landscape": "pdz-cis",
            "inference_method": "hmc",
            "model_spec": "exponential-exponential",
            "fit_id": "small-clean_1",
            "job_id": job_id,
            "artifact_kind": artifact_kind,
            "primary_score": "log_mean_growth",
        }
    )


def _mle_predictive_spec(*, artifact_kind="sampled_mle"):
    return AnalysisSpec.from_dict(
        {
            "protein_family": "pdz",
            "landscape": "pdz-cis",
            "inference_method": "mle_predictive",
            "model_spec": "exponential-exponential",
            "fit_id": "mle_1",
            "job_id": 1,
            "artifact_kind": artifact_kind,
            "primary_score": "dimsum_fitness",
        }
    )


def _hmc_count_based_spec(*, artifact_kind="diagnosed_hmc"):
    return AnalysisSpec.from_dict(
        {
            "protein_family": "pdz",
            "landscape": "pdz-cis",
            "inference_method": "hmc_count_based",
            "model_spec": "exponential-exponential",
            "fit_id": "small-clean_1",
            "job_id": 1,
            "artifact_kind": artifact_kind,
            "primary_score": "dimsum_fitness",
        }
    )


def _audit(path, *, hmc_status="validated", raw_status="not_checked"):
    diagnosed = []
    raw = []
    if hmc_status == "validated":
        diagnosed = [
            {
                "path": str(path),
                "kind": "diagnosed_hmc",
                "status": "validated",
                "bytes": path.stat().st_size,
            }
        ]
    if raw_status == "usable_for_analysis":
        raw = [
            {
                "path": str(path),
                "kind": "raw_hmc",
                "status": "validated",
                "bytes": path.stat().st_size,
            }
        ]
    return {
        "job_id": 1,
        "dataset": "pdz",
        "subset": "cis",
        "fitness_distribution": "exponential",
        "growth_curve": "exponential",
        "hmc_job_name": "small-clean_1",
        "mle_job_name": "mle_1",
        "score_variable": "log_r_mean",
        "hmc_status": hmc_status,
        "raw_status": raw_status,
        "mle_status": "validated",
        "diagnosed_hmc": diagnosed,
        "raw_hmc": raw,
        "sampled_mle": [],
    }


def test_resolver_returns_exact_validated_diagnosed_artifact(tmp_path):
    artifact = tmp_path / "amlt/small-clean/small-clean_1/fit_diagnosed.nc"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fit")
    resolved = ArtifactResolver(
        [_audit(artifact)],
        artifact_roots=(tmp_path / "amlt",),
    ).resolve(_spec())

    assert resolved.path == artifact.resolve()
    assert resolved.kind == "diagnosed_hmc"
    assert resolved.score_variable == "log_r_mean"
    assert resolved.bytes == 3


def test_resolver_accepts_only_explicit_usable_raw_artifact(tmp_path):
    artifact = tmp_path / "amlt/small-clean/small-clean_1/fit-raw.nc"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"raw")
    resolved = ArtifactResolver(
        [
            _audit(
                artifact,
                hmc_status="removed_after_failed_validation",
                raw_status="usable_for_analysis",
            )
        ],
        artifact_roots=(tmp_path / "amlt",),
    ).resolve(_spec(artifact_kind="raw_hmc"))

    assert resolved.path == artifact.resolve()
    assert resolved.kind == "raw_hmc"


def test_resolver_rejects_manifest_and_audit_model_drift(tmp_path):
    artifact = tmp_path / "amlt/fit.nc"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fit")
    record = _audit(artifact)
    record["growth_curve"] = "sigmoid"

    with pytest.raises(ValueError, match="model does not match audit"):
        ArtifactResolver([record], artifact_roots=(tmp_path / "amlt",)).resolve(
            _spec()
        )


def test_resolver_rejects_artifact_outside_root(tmp_path):
    artifact = tmp_path / "outside.nc"
    artifact.write_bytes(b"fit")

    with pytest.raises(ValueError, match="outside"):
        ArtifactResolver(
            [_audit(artifact)],
            artifact_roots=(tmp_path / "amlt",),
        ).resolve(_spec())


def test_resolver_accepts_artifacts_from_either_approved_root(tmp_path):
    canonical = tmp_path / "data/pdz_cis/fits/fit.nc"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"fit")

    resolved = ArtifactResolver(
        [_audit(canonical)],
        artifact_roots=(tmp_path / "amlt", tmp_path / "data"),
    ).resolve(_spec())

    assert resolved.path == canonical.resolve()


def test_manifest_rejects_duplicate_jobs(tmp_path):
    entry = _spec().to_dict()
    duplicate_job = {
        **entry,
        "model_spec": "exponential-sigmoid",
        "fit_id": "small-clean_4",
    }
    path = tmp_path / "analyses.json"
    path.write_text(
        json.dumps({"schema_version": 2, "analyses": [entry, duplicate_job]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate job IDs"):
        load_analysis_manifest(path)


def test_hmc_manifest_rejects_sampled_mle_artifact():
    with pytest.raises(ValueError, match="HMC analyses require"):
        _spec(artifact_kind="sampled_mle")


def test_hmc_count_based_manifest_uses_hmc_artifacts_and_fit_identity(tmp_path):
    artifact = tmp_path / "amlt/small-clean/small-clean_1/fit_diagnosed.nc"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fit")

    resolved = ArtifactResolver(
        [_audit(artifact)],
        artifact_roots=(tmp_path / "amlt",),
    ).resolve(_hmc_count_based_spec())

    assert resolved.path == artifact.resolve()
    assert resolved.kind == "diagnosed_hmc"
    with pytest.raises(ValueError, match="HMC analyses require"):
        _hmc_count_based_spec(artifact_kind="sampled_mle")


def test_mle_predictive_manifest_requires_sampled_mle_artifact():
    with pytest.raises(ValueError, match="MLE analyses require"):
        _mle_predictive_spec(artifact_kind="diagnosed_hmc")


def test_resolver_accepts_sampled_mle_for_predictive_analysis(tmp_path):
    artifact = tmp_path / "amlt/mle/mle_1/fit_samples.nc"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"mle")
    record = _audit(artifact)
    record["sampled_mle"] = [
        {
            "path": str(artifact),
            "kind": "sampled_mle",
            "status": "validated",
            "bytes": artifact.stat().st_size,
        }
    ]

    resolved = ArtifactResolver(
        [record],
        artifact_roots=(tmp_path / "amlt",),
    ).resolve(_mle_predictive_spec())

    assert resolved.path == artifact.resolve()
    assert resolved.kind == "sampled_mle"


def test_hmc_manifest_locks_curated_family_and_artifact_counts():
    manifest = Path(__file__).parents[1] / "examples/paper/analyses_hmc.json"
    specs = load_analysis_manifest(manifest)

    assert len(specs) == 49
    assert Counter(spec.identity.protein_family for spec in specs) == {
        "pdz": 14,
        "trpb": 28,
        "nuc": 7,
    }
    assert Counter(spec.artifact_kind for spec in specs) == {
        "diagnosed_hmc": 40,
        "raw_hmc": 9,
    }
    assert {spec.identity.inference_method for spec in specs} == {"hmc"}


def test_hmc_count_based_manifest_locks_hmc_scope_and_proxy_scores():
    manifest = (
        Path(__file__).parents[1]
        / "examples/paper/analyses_hmc_count_based.json"
    )
    specs = load_analysis_manifest(manifest)

    assert len(specs) == 49
    assert Counter(spec.identity.protein_family for spec in specs) == {
        "pdz": 14,
        "trpb": 28,
        "nuc": 7,
    }
    assert Counter(spec.artifact_kind for spec in specs) == {
        "diagnosed_hmc": 40,
        "raw_hmc": 9,
    }
    assert {spec.identity.inference_method for spec in specs} == {
        "hmc_count_based"
    }
    assert {
        family: {spec.primary_score for spec in specs if spec.identity.protein_family == family}
        for family in ("pdz", "trpb", "nuc")
    } == {
        "pdz": {"dimsum_fitness"},
        "trpb": {"johnston_fitness"},
        "nuc": {"highest_gate_enrichment"},
    }
    assert all("__hmc-count-based__" in spec.analysis_id for spec in specs)

    audit_records = json.loads(
        (Path(__file__).parents[1] / "data/fit_audit.json").read_text(
            encoding="utf-8"
        )
    )
    audit_by_job = {record["job_id"]: record for record in audit_records}
    selected_shapes = [
        audit_by_job[spec.job_id][spec.artifact_kind][0]["score_shape"]
        for spec in specs
    ]
    dense_bytes = sum(
        shape[0] * shape[1] * shape[-1] * np.dtype(np.float32).itemsize
        for shape in selected_shapes
    )
    assert dense_bytes == 144_283_328_000


def test_mle_predictive_manifest_locks_retained_source_jobs_and_scores():
    manifest = Path(__file__).parents[1] / "examples/paper/analyses_mle_pilot.json"
    specs = load_analysis_manifest(manifest)

    assert [spec.job_id for spec in specs] == [3, 64, 59]
    assert {spec.identity.inference_method for spec in specs} == {"mle_predictive"}
    assert {spec.artifact_kind for spec in specs} == {"sampled_mle"}
    assert [spec.primary_score for spec in specs] == [
        "dimsum_fitness",
        "johnston_fitness",
        "highest_gate_enrichment",
    ]


def test_analysis_manifest_supports_multiple_declared_scores(tmp_path):
    payload = {
        "schema_version": 2,
        "analyses": [
            {
                **_mle_predictive_spec().to_dict(),
                "primary_score": "score-a",
                "scores": ["score-a", "score-b"],
            }
        ],
    }
    path = tmp_path / "analyses.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    spec = load_analysis_manifest(path)[0]

    assert spec.primary_score == "score-a"
    assert spec.scores == ("score-a", "score-b")
    assert spec.to_dict()["scores"] == ["score-a", "score-b"]
    assert spec.score_identity("score-a").landscape == "pdz-cis-score-a"


def test_full_mle_predictive_manifest_locks_all_families():
    manifest = Path(__file__).parents[1] / "examples/paper/analyses_mle.json"
    specs = load_analysis_manifest(manifest)

    assert len(specs) == 61
    assert Counter(spec.identity.protein_family for spec in specs) == {
        "pdz": 19,
        "trpb": 34,
        "nuc": 7,
        "k50": 1,
    }
    assert [
        spec.job_id for spec in specs if spec.identity.protein_family == "pdz"
    ] == [1, 3, 4, 70, 71, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 88, 90, 92, 93]
    assert [
        spec.job_id for spec in specs if spec.identity.protein_family == "trpb"
    ] == [
        7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 25, 26, 27, 31, 32,
        33, 34, 37, 38, 39, 43, 44, 45, 49, 50, 51, 64, 65, 66, 67, 68, 69,
    ]
    assert [
        spec.job_id for spec in specs if spec.identity.protein_family == "nuc"
    ] == [55, 56, 57, 58, 59, 61, 62]
    k50 = next(spec for spec in specs if spec.identity.protein_family == "k50")
    assert k50.job_id == 63
    assert k50.scores == (
        "mean_log_chymotrypsin_enrichment",
        "mean_log_trypsin_enrichment",
    )
    assert {spec.identity.inference_method for spec in specs} == {"mle_predictive"}
    assert {spec.artifact_kind for spec in specs} == {"sampled_mle"}


def test_k50_mle_predictive_manifest_is_separate():
    manifest = Path(__file__).parents[1] / "examples/paper/analyses_mle_k50.json"
    specs = load_analysis_manifest(manifest)

    assert [spec.job_id for spec in specs] == [63]
    assert specs[0].identity.protein_family == "k50"


def test_mle_source_retirement_keeps_exactly_three_source_exemplars():
    root = Path(__file__).parents[1]
    canonical = load_analysis_manifest(root / "examples/paper/analyses_mle.json")
    pilots = load_analysis_manifest(root / "examples/paper/analyses_mle_pilot.json")
    retirement = json.loads(
        (root / "data/mle_source_retirement.json").read_text(encoding="utf-8")
    )

    canonical_jobs = {spec.job_id for spec in canonical}
    pilot_jobs = {spec.job_id for spec in pilots}
    retired_jobs = set(retirement["retired_sources"]["job_ids"])
    retained_jobs = {
        record["job_id"] for record in retirement["retained_sources"]
    }

    assert pilot_jobs == retained_jobs == {3, 59, 64}
    assert canonical_jobs == retained_jobs | retired_jobs
    assert not retained_jobs & retired_jobs
    assert len(retired_jobs) == retirement["retired_sources"]["files"] == 58
    assert 63 in retired_jobs
    assert retirement["retained_source_totals"] == {
        "files": 3,
        "logical_bytes": 15_596_907_437,
        "allocated_bytes": 15_596_929_024,
    }
    assert retirement["retired_sources"]["logical_bytes"] == 661_756_469_404
    assert retirement["retired_sources"]["allocated_bytes"] == 661_756_895_232
    assert (
        retirement["retained_source_totals"]["logical_bytes"]
        + retirement["retired_sources"]["logical_bytes"]
        == retirement["source_inventory_before_retirement"]["logical_bytes"]
    )
    assert (
        retirement["retained_source_totals"]["allocated_bytes"]
        + retirement["retired_sources"]["allocated_bytes"]
        == retirement["source_inventory_before_retirement"]["allocated_bytes"]
    )
    assert retirement["validated_products_after_retirement"] == {
        "proxy_bundles": 61,
        "compact_results": 61,
        "paper_score_views": 62,
        "mechanism_caches": 62,
        "paper_table_files": 14,
        "figure_png_files": 29,
        "figure_pdf_files": 29,
        "matrix_only_audit_status": "61 valid",
        "mechanism_table_validation_status": "valid",
    }
