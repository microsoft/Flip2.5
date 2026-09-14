import json

import pytest

from dms_fitness_stability.fit_retirement import (
    archive_and_remove,
    archive_raw_and_remove,
    build_product_fit_retirement_action,
    build_raw_retirement_actions,
    build_retirement_actions,
    prepare_product_fit_retirement,
    remove_prepared_product_fit,
)


def test_archive_is_written_before_problematic_diagnosed_file_is_removed(tmp_path):
    amlt_root = tmp_path / "amlt"
    diagnosed = amlt_root / "large-clean/large-clean_71/fit_diagnosed.nc"
    diagnosed.parent.mkdir(parents=True)
    diagnosed.write_bytes(b"diagnosed")
    validation = {
        "path": str(diagnosed),
        "kind": "diagnosed_hmc",
        "status": "unusable",
        "bytes": 9,
        "issues": ["predictive_variable_missing:starting_counts"],
        "errors": [],
        "groups": ["posterior"],
        "score_variable": "log_r_mean",
        "score_shape": [8, 2000, 4],
        "predictive_shapes": {},
        "observed_shapes": {},
        "h5py_compatible": True,
        "h5netcdf_compatible": False,
    }
    audit = [
        {
            "job_id": 71,
            "job_name": "large_clean_71",
            "hmc_status": "unusable",
            "diagnosed_hmc": [validation],
        }
    ]
    raw = [
        {
            "job_id": 71,
            "job_name": "large_clean_71",
            "issues": [],
            "artifacts": [
                {
                    "kind": "raw_hmc",
                    "experiment": "large-clean",
                    "job_name": "large-clean_71",
                    "name": "pdz/cript-c/fit-20260403053144.nc",
                    "size": "265.2 GB",
                }
            ],
        }
    ]

    actions = build_retirement_actions(
        audit,
        raw,
        amlt_root=amlt_root,
        job_ids={71},
    )
    archive = tmp_path / "retired.json"
    archive_and_remove(actions, archive_path=archive)

    assert not diagnosed.exists()
    records = json.loads(archive.read_text(encoding="utf-8"))
    assert records[0]["job_name"] == "large_clean_71"
    assert records[0]["artifacts"] == [validation]
    assert records[0]["replacement"]["name"].endswith("20260403053144.nc")


def test_unusable_raw_is_archived_before_exact_file_is_removed(tmp_path):
    amlt_root = tmp_path / "amlt"
    raw = amlt_root / "large-clean/large-clean_76/pdz/cript-n/fit-20260407.nc"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"corrupt raw")
    validation = {
        "path": str(raw),
        "kind": "raw_hmc",
        "status": "unusable",
        "bytes": 11,
        "issues": ["h5py_open_failure"],
        "errors": ["RuntimeError: bad object header"],
        "groups": [],
        "score_variable": "log_r_mean",
        "score_shape": None,
        "predictive_shapes": {},
        "observed_shapes": {},
        "h5py_compatible": False,
        "h5netcdf_compatible": False,
    }
    audit = [
        {
            "job_id": 76,
            "job_name": "large_clean_76",
            "raw_status": "unusable",
            "raw_hmc": [validation],
        }
    ]
    preflight = [
        {
            "job_id": 76,
            "job_name": "large_clean_76",
            "issues": [],
            "artifacts": [
                {
                    "kind": "raw_hmc",
                    "experiment": "large-clean",
                    "job_name": "large-clean_76",
                    "name": "pdz/cript-n/fit-20260407.nc",
                    "size": "153.0 GB",
                }
            ],
        }
    ]

    actions = build_raw_retirement_actions(
        audit,
        preflight,
        amlt_root=amlt_root,
        job_ids={76},
    )
    archive = tmp_path / "retired_raw.json"
    archive_raw_and_remove(actions, archive_path=archive)

    assert not raw.exists()
    records = json.loads(archive.read_text(encoding="utf-8"))
    assert records[0]["job_name"] == "large_clean_76"
    assert records[0]["artifacts"] == [validation]
    assert records[0]["source"]["name"] == "pdz/cript-n/fit-20260407.nc"


def test_raw_retirement_rejects_path_that_does_not_match_preflight(tmp_path):
    amlt_root = tmp_path / "amlt"
    raw = amlt_root / "large-clean/large-clean_76/pdz/cript-n/local.nc"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"corrupt raw")
    audit = [
        {
            "job_id": 76,
            "job_name": "large_clean_76",
            "raw_status": "unusable",
            "raw_hmc": [
                {
                    "path": str(raw),
                    "kind": "raw_hmc",
                    "status": "unusable",
                }
            ],
        }
    ]
    preflight = [
        {
            "job_id": 76,
            "job_name": "large_clean_76",
            "issues": [],
            "artifacts": [
                {
                    "kind": "raw_hmc",
                    "experiment": "large-clean",
                    "job_name": "large-clean_76",
                    "name": "pdz/cript-n/remote.nc",
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="does not match preflight"):
        build_raw_retirement_actions(
            audit,
            preflight,
            amlt_root=amlt_root,
            job_ids={76},
        )

    assert raw.exists()


def test_product_fit_requires_prepared_evidence_before_removal(tmp_path):
    amlt_root = tmp_path / "amlt"
    fit = amlt_root / "k50-mle/k50_job/k50/fit_samples.nc"
    fit.parent.mkdir(parents=True)
    fit.write_bytes(b"validated fit")
    validation = {
        "path": str(fit),
        "kind": "sampled_mle",
        "status": "validated",
        "bytes": fit.stat().st_size,
    }
    audit = [
        {
            "job_id": 63,
            "dataset": "k50",
            "mle_status": "validated",
            "sampled_mle": [validation],
        }
    ]
    preflight = [
        {
            "job_id": 63,
            "issues": [],
            "artifacts": [
                {
                    "kind": "sampled_mle",
                    "experiment": "k50-mle",
                    "job_name": "k50_job",
                    "name": "k50/fit_samples.nc",
                    "size": "13 B",
                }
            ],
        }
    ]
    baseline = {
        "baseline_kind": "mle_predictive_schema_v1_pre_conversion",
        "repository_commit": "baseline-commit",
        "analyses": [
            {
                "manifest": {"job_id": 63, "fit_id": "k50-fit"},
                "source_artifact": {
                    "path": str(fit),
                    "size": fit.stat().st_size,
                    "mtime_ns": fit.stat().st_mtime_ns,
                },
                "score_bundle": {
                    "analysis_id": "k50__k50__mle-predictive__model__fit",
                    "semantic_sha256": "a" * 64,
                },
                "result": {"metadata_sha256": "b" * 64},
                "mechanism_cache_ids": ["k50-chymotrypsin", "k50-trypsin"],
            }
        ]
    }
    action = build_product_fit_retirement_action(
        audit,
        preflight,
        baseline,
        amlt_root=amlt_root,
        amlt_executable="/tools/amlt",
        job_id=63,
        sha256="c" * 64,
    )
    archive = tmp_path / "retired_product_fits.json"

    with pytest.raises(ValueError, match="has not been prepared"):
        remove_prepared_product_fit(action, archive_path=archive)
    assert fit.exists()

    prepare_product_fit_retirement(
        action,
        archive_path=archive,
        reason="storage",
    )
    prepared = json.loads(archive.read_text(encoding="utf-8"))[0]
    assert prepared["status"] == "approved_pending_removal"
    assert prepared["artifact"]["sha256"] == "c" * 64
    assert prepared["source"]["name"] == "k50/fit_samples.nc"
    assert prepared["downstream"] == {
        "baseline_kind": "mle_predictive_schema_v1_pre_conversion",
        "baseline_repository_commit": "baseline-commit",
        "analysis_id": "k50__k50__mle-predictive__model__fit",
        "score_semantic_sha256": "a" * 64,
        "result_metadata_sha256": "b" * 64,
        "mechanism_cache_ids": ["k50-chymotrypsin", "k50-trypsin"],
    }
    assert prepared["recovery_command"] == [
        "/tools/amlt",
        "results",
        "download",
        "k50-mle",
        ":k50_job",
        "--include",
        "k50/fit_samples.nc",
        "--output",
        str(amlt_root.resolve()),
        "--n-workers",
        "1",
    ]
    assert fit.exists()

    remove_prepared_product_fit(action, archive_path=archive)

    assert not fit.exists()
    completed = json.loads(archive.read_text(encoding="utf-8"))[0]
    assert completed["status"] == "removed"
    assert completed["removal_started_at"]
    assert completed["removed_at"]
