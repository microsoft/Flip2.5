from dms_fitness_stability.fit_audit import (
    JobAudit,
    render_known_issues,
    retired_diagnosed_from_records,
)
from dms_fitness_stability.fit_inventory import SelectedJob
from dms_fitness_stability.fit_validation import ArtifactValidation


def _job() -> SelectedJob:
    return SelectedJob(
        job_id=63,
        family="large_clean",
        dataset="k50",
        subset="",
        fitness_distribution="(MASSIVE, only MLE)",
        growth_curve="",
        result="",
        require_hmc=False,
        require_mle=True,
    )


def _hmc_validation(*, kind: str, path: str, status: str, issues=()):
    return ArtifactValidation(
        path=path,
        kind=kind,
        status=status,
        bytes=123,
        issues=issues,
        errors=(),
        groups=("posterior",),
        score_variable="log_r_mean",
        score_shape=(8, 2_000, 4),
        predictive_shapes=(),
        observed_shapes=(),
        h5py_compatible=True,
        h5netcdf_compatible=status == "validated",
    )


def test_render_known_issues_records_mle_only_exceptions_and_blank_rows():
    audit = JobAudit(
        job=_job(),
        score_variable="mean_dg",
        variant_count=1_824_523,
        diagnosed_hmc=(),
        sampled_mle=(),
        raw_hmc=(),
        inventory_issues=(),
    )
    blank = (
        {
            "ID": "75",
            "Dataset": "",
            "Subset": "cript-c",
            "Fitness Distribution": "gamma",
            "Growth Curve": "sigmoid",
            "Result": "",
        },
    )

    preferred = (
        {
            "job_name": "large_clean_63",
            "issues": [],
            "artifacts": [
                {
                    "kind": "sampled_mle",
                    "experiment": "k50-mle",
                    "job_name": "k50_mle_lr0-01",
                    "name": "k50/k50_None_None_None_samples.nc",
                    "size": "443.6 GB",
                }
            ],
        },
    )

    rendered = render_known_issues(
        (audit,),
        blank_rows=blank,
        preferred_preflight=preferred,
    )

    assert "`large_clean_63`" in rendered
    assert "not_required" in rendered
    assert "Sampled MLE: missing" in rendered
    assert "k50/k50_None_None_None_samples.nc" in rendered
    assert "443.6 GB" in rendered
    assert "| 75 | (blank) | cript-c | gamma | sigmoid |" in rendered
    assert audit.fallback_required is False
    assert audit.preferred_hmc_download_required is False


def test_render_known_issues_preserves_removed_diagnosed_findings():
    job = SelectedJob(
        job_id=71,
        family="large_clean",
        dataset="pdz",
        subset="cript-c",
        fitness_distribution="lomax",
        growth_curve="exponential",
        result="Perfect!",
        require_hmc=True,
        require_mle=True,
    )
    retired = _hmc_validation(
        kind="diagnosed_hmc",
        path="/data/amlt/large-clean/large-clean_71/fit_diagnosed.nc",
        status="unusable",
        issues=("predictive_variable_missing:starting_counts",),
    )
    raw = _hmc_validation(
        kind="raw_hmc",
        path="/data/amlt/large-clean/large-clean_71/fit-20260403053144.nc",
        status="validated",
    )
    audit = JobAudit(
        job=job,
        score_variable="log_r_mean",
        variant_count=4,
        diagnosed_hmc=(),
        sampled_mle=(),
        raw_hmc=(raw,),
        inventory_issues=(),
        retired_diagnosed_hmc=(retired,),
    )

    rendered = render_known_issues((audit,), blank_rows=())

    assert audit.hmc_status == "removed_after_failed_validation"
    assert audit.raw_status == "usable_for_analysis"
    assert audit.fallback_required is False
    assert audit.preferred_hmc_download_required is False
    assert "removed locally after failed validation" in rendered
    assert "predictive_variable_missing:starting_counts" in rendered
    assert "fit-20260403053144.nc" in rendered
    assert "Preferred diagnosed HMC" not in rendered


def test_render_known_issues_preserves_removed_raw_findings():
    job = SelectedJob(
        job_id=76,
        family="large_clean",
        dataset="pdz",
        subset="cript-n",
        fitness_distribution="exponential",
        growth_curve="exponential",
        result="Perfect!",
        require_hmc=True,
        require_mle=True,
    )
    retired_diagnosed = _hmc_validation(
        kind="diagnosed_hmc",
        path="/data/amlt/large-clean/large-clean_76/fit_diagnosed.nc",
        status="limited",
    )
    retired_raw = _hmc_validation(
        kind="raw_hmc",
        path="/data/amlt/large-clean/large-clean_76/fit-20260407.nc",
        status="unusable",
        issues=("h5py_open_failure",),
    )
    audit = JobAudit(
        job=job,
        score_variable="log_r_mean",
        variant_count=4,
        diagnosed_hmc=(),
        sampled_mle=(),
        raw_hmc=(),
        inventory_issues=(),
        retired_diagnosed_hmc=(retired_diagnosed,),
        retired_raw_hmc=(retired_raw,),
    )

    rendered = render_known_issues((audit,), blank_rows=())

    assert audit.raw_status == "removed_after_failed_validation"
    assert audit.fallback_required is False
    assert "no usable HMC" in rendered
    assert "raw_hmc:" in rendered
    assert "h5py_open_failure" in rendered
    assert "removed locally after failed validation" in rendered


def test_historical_diagnosed_record_preserves_unknown_reader_result():
    records = (
        {
            "job_name": "large_clean_82",
            "artifacts": [
                {
                    "path": "/data/amlt/large-clean/large-clean_82/fit_diagnosed.nc",
                    "kind": "diagnosed_hmc",
                    "status": "unusable",
                    "bytes": None,
                    "issues": ["historical_h5netcdf_open_failure"],
                    "errors": [],
                    "groups": [],
                    "score_variable": "log_r_mean",
                    "score_shape": None,
                    "predictive_shapes": {},
                    "observed_shapes": {},
                    "h5py_compatible": None,
                    "h5netcdf_compatible": False,
                }
            ],
        },
    )

    validation = retired_diagnosed_from_records(records)["large_clean_82"][0]

    assert validation.h5py_compatible is None
    assert validation.h5netcdf_compatible is False
