import csv
from pathlib import Path

from dms_fitness_stability.fit_inventory import build_inventory, load_selected_jobs

HEADERS = (
    "ID",
    "Dataset",
    "Subset",
    "Fitness Distribution",
    "Growth Curve",
    "Result",
)


def _write_summary(path: Path, *, grouped_pdz: bool = False) -> None:
    rows = []
    for job_id in range(1, 56):
        rows.append(
            {
                "ID": job_id,
                "Dataset": "pdz" if job_id == 1 else "trpb",
                "Subset": "cis" if job_id == 1 else "libA",
                "Fitness Distribution": "exponential",
                "Growth Curve": "exponential",
                "Result": "Perfect!",
            }
        )
    rows[1]["Result"] = "A few failed."
    if grouped_pdz:
        rows[2]["Result"] = "Many failed."
        rows.append(
            {
                "ID": 70,
                "Dataset": "",
                "Subset": "cript-c",
                "Fitness Distribution": "exponential",
                "Growth Curve": "exponential",
                "Result": "Perfect!",
            }
        )
    rows.append(
        {
            "ID": 63,
            "Dataset": "dataset",
            "Subset": "K50",
            "Fitness Distribution": "(MASSIVE, only MLE)",
            "Growth Curve": "",
            "Result": "",
        }
    )
    for job_id, prior, curve in (
        (64, "exponential", "exponential"),
        (65, "lomax", "exponential"),
        (66, "gamma", "exponential"),
        (67, "exponential", "sigmoid"),
        (68, "lomax", "sigmoid"),
        (69, "gamma", "sigmoid"),
    ):
        rows.append(
            {
                "ID": job_id,
                "Dataset": "trpb",
                "Subset": "four-site",
                "Fitness Distribution": prior,
                "Growth Curve": curve,
                "Result": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def test_load_selected_jobs_applies_family_boundary_and_mle_only_exceptions(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(summary_path)

    jobs = load_selected_jobs(summary_path)

    assert len(jobs) == 61
    assert 2 not in {job.job_id for job in jobs}
    assert jobs[0].family == "small_clean"
    assert jobs[0].hmc_experiment == "small-clean"
    assert jobs[0].mle_experiment == "mle"
    by_id = {job.job_id: job for job in jobs}
    k50 = by_id[63]
    assert k50.family == "large_clean"
    assert k50.hmc_experiment == "large-clean"
    assert k50.mle_experiment == "k50-mle"
    assert k50.mle_job_name == "k50_mle_lr0-01"
    assert k50.dataset == "k50"
    assert k50.subset == ""
    assert k50.require_hmc is False
    assert k50.require_mle is True

    four_site = [by_id[job_id] for job_id in range(64, 70)]
    assert all(job.dataset == "trpb" for job in four_site)
    assert all(job.subset == "four-site" for job in four_site)
    assert all(job.family == "large_clean" for job in four_site)
    assert all(job.mle_experiment == "mle" for job in four_site)
    assert all(job.require_hmc is False for job in four_site)
    assert all(job.require_mle is True for job in four_site)


def test_load_selected_jobs_restores_grouped_pdz_dataset_label(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(summary_path, grouped_pdz=True)

    jobs = load_selected_jobs(summary_path)

    job = next(job for job in jobs if job.job_id == 70)
    assert job.dataset == "pdz"


def test_build_inventory_finds_preferred_raw_and_canonical_job_one_files(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(summary_path)
    amlt_root = tmp_path / "amlt"
    data_root = tmp_path / "data"

    job_three = amlt_root / "small-clean/small-clean_3/trpb/libA"
    job_three.mkdir(parents=True)
    (job_three / "fit_diagnosed.nc").touch()
    (job_three / "fit-20260101010101.nc").touch()
    (job_three / ".azDownload-interrupted-fit-20260101010101.nc").touch()
    mle_three = amlt_root / "mle/mle_3/trpb/libA"
    mle_three.mkdir(parents=True)
    (mle_three / "fit_samples.nc").touch()

    canonical = data_root / "pdz_cis/fits"
    canonical.mkdir(parents=True)
    (canonical / "pdz_cis_exponential_exponential_diagnosed.nc").touch()
    (canonical / "pdz_cis_exponential_exponential_samples.nc").touch()

    inventory = build_inventory(
        csv_path=summary_path,
        amlt_root=amlt_root,
        data_root=data_root,
    )
    by_id = {item.job.job_id: item for item in inventory}

    assert by_id[1].missing_artifacts == ()
    assert by_id[3].missing_artifacts == ()
    assert by_id[3].raw_hmc == (str(job_three / "fit-20260101010101.nc"),)
    assert by_id[4].missing_artifacts == ("diagnosed_hmc", "sampled_mle")
    assert by_id[63].missing_artifacts == ("sampled_mle",)
