"""Source-grounded array contracts for selected fit jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from .adapters.nuclease import load_nuclease_landscape
from .adapters.pdz import load_pdz_cis, load_pdz_cript, load_pdz_trans
from .adapters.trpb import load_trpb_landscape
from .fit_inventory import SelectedJob
from .fit_validation import FitExpectation

FitMethod = Literal["hmc", "mle"]
SAMPLED_MLE_PREFIX = (1, 1_000)


def score_variable_for_job(job: SelectedJob) -> str:
    """Return the fitted score variable implied by the job's model family."""
    prior = job.fitness_distribution.lower()
    if job.dataset == "pdz":
        return "inv_r_mean" if prior == "gamma" else "log_r_mean"
    if job.dataset == "trpb":
        if job.subset in {"libA", "libB", "libC"}:
            return "r"
        return "inv_r_mean" if prior == "gamma" else "log_r_mean"
    if job.dataset == "nuc":
        return "absolute_mean_log_fluorescence" if job.subset == "G1" else "log_fluorescence"
    if job.dataset == "k50":
        return "mean_dg"
    raise ValueError(f"Unsupported fit dataset for job {job.job_id}: {job.dataset!r}")


class FitContractResolver:
    """Load each source landscape once and resolve per-job validation contracts."""

    def __init__(self, *, repository_root: str | Path, data_root: str | Path) -> None:
        self.repository_root = Path(repository_root)
        self.data_root = Path(data_root)
        self._landscapes: dict[tuple[str, str], object] = {}
        self._nuc_specs = self._job_specs("examples/nuc/fit_jobs.json")
        self._k50_specs = self._job_specs("examples/k50/fit_jobs.json")

    def _job_specs(self, relative_path: str) -> dict[int, dict[str, object]]:
        manifest = json.loads(
            (self.repository_root / relative_path).read_text(encoding="utf-8")
        )
        return {int(spec["job"]): spec for spec in manifest["fits"]}

    def _load_landscape(self, job: SelectedJob):
        key = (job.dataset, job.subset)
        if key in self._landscapes:
            return self._landscapes[key]

        if job.dataset == "trpb":
            root = self.data_root / "trpb"
            landscape = load_trpb_landscape(
                job.subset,
                count_path=root / "raw/counts" / f"{job.subset}.csv",
                reported_fitness_path=(
                    root / "raw/reported_fitness" / f"{job.subset}.csv"
                ),
            )
        elif job.dataset == "pdz" and job.subset == "cis":
            root = self.data_root / "pdz_cis/raw"
            landscape = load_pdz_cis(
                root / "cis_from_bruce_azure.tsv",
                reported_fitness_path=root / "GSE265816_cript2xa3_sub.csv.gz",
            )
        elif job.dataset == "pdz" and job.subset in {"cript-c", "cript-n"}:
            root = self.data_root / "pdz_cis/raw"
            fitness_name = job.subset.replace("-", "_")
            landscape = load_pdz_cript(
                job.subset,
                root / f"{job.subset}.tsv",
                reported_fitness_path=root / f"GSE265816_{fitness_name}_sub.csv.gz",
            )
        elif job.dataset == "pdz" and job.subset in {"trans-1", "trans-2"}:
            root = self.data_root / "pdz_trans/raw"
            landscape = load_pdz_trans(
                job.subset,
                count_paths={
                    name: root / f"{name}.tsv" for name in ("trans-1", "trans-2")
                },
                reported_fitness_path=root / "GSE265816_pdz3cript_sub.csv.gz",
            )
        elif job.dataset == "nuc":
            root = self.data_root / "nuc/raw"
            landscape = load_nuclease_landscape(
                job.subset,
                count_dir=root / "counts",
                reported_activity_path=root / "reported_fitness/landscape.csv",
            )
        else:
            raise ValueError(
                f"No source landscape loader for job {job.job_id}: "
                f"{job.dataset}/{job.subset}"
            )
        self._landscapes[key] = landscape
        return landscape

    def expectation(self, job: SelectedJob, *, method: FitMethod) -> FitExpectation:
        """Build an HMC or sampled-MLE contract for one selected job."""
        score_variable = score_variable_for_job(job)
        posterior_shape = None
        predictive_variables: tuple[str, ...] = ()
        observed_arrays: tuple[tuple[str, np.ndarray], ...] = ()

        if job.dataset == "k50":
            spec = self._k50_specs[job.job_id]
            variant_count = int(spec["mle_group_shape"][-1])
        else:
            landscape = self._load_landscape(job)
            if job.dataset == "pdz":
                variant_count = int(landscape.n_hmc_variants)
                predictive_variables = ("starting_counts", "timepoint_counts")
                observed_arrays = (
                    ("starting_counts", np.asarray(landscape.starting_counts)),
                    ("timepoint_counts", np.asarray(landscape.timepoint_counts)),
                )
            elif job.dataset == "trpb":
                variant_count = int(landscape.n_hmc_variants)
                predictive_variables = ("starting_counts", "timepoint_counts")
                observed_arrays = (
                    ("starting_counts", np.asarray(landscape.starting_counts)),
                    ("timepoint_counts", np.asarray(landscape.timepoint_counts)),
                )
            elif job.dataset == "nuc":
                variant_count = int(landscape.n_fitted_variants)
                observed_arrays = tuple(
                    (name, np.asarray(values))
                    for name, values in landscape.counts.items()
                )
                if method == "hmc":
                    spec = self._nuc_specs[job.job_id]
                    posterior_shape = tuple(int(size) for size in spec["posterior_shape"])
            else:
                raise AssertionError(f"Unreachable dataset: {job.dataset}")

        if method == "mle":
            posterior_shape = (*SAMPLED_MLE_PREFIX, variant_count)
            predictive_variables = ()
            observed_arrays = ()
        return FitExpectation(
            score_variable=score_variable,
            variant_count=variant_count,
            posterior_shape=posterior_shape,
            predictive_variables=predictive_variables,
            observed_arrays=observed_arrays,
        )
