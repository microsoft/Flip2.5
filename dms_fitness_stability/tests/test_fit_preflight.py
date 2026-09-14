import json
import subprocess

from dms_fitness_stability.fit_inventory import ArtifactInventory, SelectedJob
from dms_fitness_stability.fit_preflight import (
    missing_patterns,
    preflight_job,
    preflight_raw_job,
    raw_pattern_for_job,
)


def _inventory(*, hmc: bool, mle: bool) -> ArtifactInventory:
    job = SelectedJob(
        job_id=70,
        family="large_clean",
        dataset="pdz",
        subset="cript-c",
        fitness_distribution="exponential",
        growth_curve="exponential",
        result="Perfect!",
        require_hmc=True,
        require_mle=True,
    )
    return ArtifactInventory(
        job=job,
        diagnosed_hmc=("hmc.nc",) if hmc else (),
        sampled_mle=("mle.nc",) if mle else (),
        raw_hmc=(),
    )


def test_missing_patterns_only_requests_absent_preferred_artifacts():
    assert missing_patterns(_inventory(hmc=True, mle=False)) == ("*samples.nc",)
    assert missing_patterns(_inventory(hmc=False, mle=True)) == ("*diagnosed.nc",)
    assert missing_patterns(_inventory(hmc=True, mle=True)) == ()


def test_preflight_job_parses_json_and_discards_remote_url():
    def runner(command, **kwargs):
        assert "--json-tables" in command
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        rows = [
            {
                "last_modified": "4mo ago",
                "size": "9.9 GB",
                "name": "pdz/cript-c/fit_diagnosed.nc",
                "url": "https://example.invalid/secret",
            }
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")

    result = preflight_job(
        _inventory(hmc=False, mle=True),
        amlt_executable="/opt/amlt",
        runner=runner,
    )

    assert result.artifacts[0].name == "pdz/cript-c/fit_diagnosed.nc"
    assert result.artifacts[0].experiment == "large-clean"
    assert not hasattr(result.artifacts[0], "url")
    assert result.issues == ()


def test_preflight_job_records_remote_absence_from_empty_output():
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    result = preflight_job(
        _inventory(hmc=True, mle=False),
        amlt_executable="/opt/amlt",
        runner=runner,
    )

    assert result.issues == ("remote_artifact_missing:sampled_mle",)


def test_raw_pattern_is_restrictive_and_model_specific():
    assert raw_pattern_for_job(_inventory(hmc=True, mle=True).job) == (
        "pdz/cript-c/pdz_cript-c_exponential_exponential-*.nc"
    )


def test_preflight_raw_job_records_exact_candidate():
    def runner(command, **kwargs):
        row = {
            "last_modified": "4mo ago",
            "size": "265.2 GB",
            "name": "pdz/cript-c/pdz_cript-c_exponential_exponential-202604.nc",
            "url": "https://example.invalid/secret",
        }
        return subprocess.CompletedProcess(command, 0, json.dumps([row]), "")

    result = preflight_raw_job(
        _inventory(hmc=True, mle=True).job,
        amlt_executable="/opt/amlt",
        runner=runner,
    )

    assert result.issues == ()
    assert result.artifacts[0].kind == "raw_hmc"
    assert result.artifacts[0].name.endswith("-202604.nc")
