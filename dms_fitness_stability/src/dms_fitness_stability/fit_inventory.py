"""Select requested fit jobs and inventory their preferred local artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MLE_ONLY_JOB_IDS = frozenset(range(63, 70))


@dataclass(frozen=True)
class SelectedJob:
    """One fit job selected for artifact download and validation."""

    job_id: int
    family: str
    dataset: str
    subset: str
    fitness_distribution: str
    growth_curve: str
    result: str
    require_hmc: bool
    require_mle: bool

    @property
    def job_name(self) -> str:
        """Return the analysis-selection label, not a remote Amulet job name."""
        return f"{self.family}_{self.job_id}"

    @property
    def hmc_experiment(self) -> str:
        """Return the actual Amulet experiment containing the HMC job."""
        return "small-clean" if self.job_id <= 62 else "large-clean"

    @property
    def hmc_job_name(self) -> str:
        return f"{self.hmc_experiment}_{self.job_id}"

    @property
    def mle_experiment(self) -> str:
        return "k50-mle" if self.job_id == 63 else "mle"

    @property
    def mle_job_name(self) -> str:
        return "k50_mle_lr0-01" if self.job_id == 63 else f"mle_{self.job_id}"


@dataclass(frozen=True)
class ArtifactInventory:
    """Preferred artifact paths currently available for one selected job."""

    job: SelectedJob
    diagnosed_hmc: tuple[str, ...]
    sampled_mle: tuple[str, ...]
    raw_hmc: tuple[str, ...]

    @property
    def missing_artifacts(self) -> tuple[str, ...]:
        missing = []
        if self.job.require_hmc and not self.diagnosed_hmc:
            missing.append("diagnosed_hmc")
        if self.job.require_mle and not self.sampled_mle:
            missing.append("sampled_mle")
        return tuple(missing)


def load_selected_jobs(csv_path: str | Path) -> tuple[SelectedJob, ...]:
    """Load exact ``Perfect!`` rows plus the approved MLE-only jobs 63-69."""
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "ID",
            "Dataset",
            "Subset",
            "Fitness Distribution",
            "Growth Curve",
            "Result",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"Unexpected HMC summary columns: {reader.fieldnames}; expected {sorted(expected)}"
            )
        for source in reader:
            job_id = int(source["ID"])
            result = source["Result"].strip()
            mle_only = job_id in MLE_ONLY_JOB_IDS
            if result != "Perfect!" and not mle_only:
                continue
            if mle_only and result:
                raise ValueError(
                    f"MLE-only job {job_id} is expected to have a blank result."
                )
            family = "small_clean" if job_id < 62 else "large_clean"
            dataset = source["Dataset"].strip().lower()
            subset = source["Subset"].strip()
            if job_id == 63:
                dataset = "k50"
                subset = ""
            elif not dataset and subset in {
                "cript-c",
                "cript-n",
                "trans-1",
                "trans-2",
            }:
                dataset = "pdz"
            rows.append(
                SelectedJob(
                    job_id=job_id,
                    family=family,
                    dataset=dataset,
                    subset=subset,
                    fitness_distribution=source["Fitness Distribution"].strip(),
                    growth_curve=source["Growth Curve"].strip(),
                    result=result,
                    require_hmc=not mle_only,
                    require_mle=True,
                )
            )
    if len(rows) != 61:
        raise ValueError(
            f"Expected 54 Perfect jobs plus jobs 63-69, found {len(rows)} rows."
        )
    return tuple(rows)


def _matching_files(root: Path, pattern: str) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        str(path)
        for path in sorted(root.rglob(pattern))
        if path.is_file() and not path.name.startswith(".azDownload-")
    )


def inventory_job(
    job: SelectedJob,
    *,
    amlt_root: str | Path,
    data_root: str | Path,
) -> ArtifactInventory:
    """Find preferred and raw NetCDF artifacts already present for one job."""
    hmc_root = Path(amlt_root) / job.hmc_experiment / job.hmc_job_name
    mle_root = Path(amlt_root) / job.mle_experiment / job.mle_job_name
    diagnosed_hmc = _matching_files(hmc_root, "*diagnosed.nc")
    sampled_mle = _matching_files(mle_root, "*samples.nc")
    raw_hmc = tuple(
        path
        for path in _matching_files(hmc_root, "*.nc")
        if not path.endswith(("diagnosed.nc", "samples.nc"))
    )

    if job.job_id == 1:
        canonical_root = Path(data_root) / "pdz_cis/fits"
        diagnosed_hmc += _matching_files(
            canonical_root,
            "pdz_cis_exponential_exponential_diagnosed.nc",
        )
        sampled_mle += _matching_files(
            canonical_root,
            "pdz_cis_exponential_exponential_samples.nc",
        )

    return ArtifactInventory(
        job=job,
        diagnosed_hmc=tuple(sorted(set(diagnosed_hmc))),
        sampled_mle=tuple(sorted(set(sampled_mle))),
        raw_hmc=tuple(sorted(set(raw_hmc))),
    )


def build_inventory(
    *,
    csv_path: str | Path,
    amlt_root: str | Path,
    data_root: str | Path,
) -> tuple[ArtifactInventory, ...]:
    """Return the preferred-artifact inventory for all selected jobs."""
    return tuple(
        inventory_job(job, amlt_root=amlt_root, data_root=data_root)
        for job in load_selected_jobs(csv_path)
    )
