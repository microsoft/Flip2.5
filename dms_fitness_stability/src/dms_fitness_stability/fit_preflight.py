"""Machine-readable Amulet preflight for preferred fit artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .fit_inventory import ArtifactInventory, SelectedJob


@dataclass(frozen=True)
class RemoteArtifact:
    """One non-secret row returned by ``amlt results list``."""

    kind: str
    experiment: str
    job_name: str
    name: str
    size: str
    last_modified: str


@dataclass(frozen=True)
class JobPreflight:
    """Remote preferred-artifact inventory for one selected job."""

    job: SelectedJob
    requested_patterns: tuple[str, ...]
    artifacts: tuple[RemoteArtifact, ...]
    issues: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self.job),
            "job_name": self.job.job_name,
            "requested_patterns": list(self.requested_patterns),
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "issues": list(self.issues),
            "commands": [list(command) for command in self.commands],
        }


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def raw_pattern_for_job(job: SelectedJob) -> str:
    """Return a restrictive timestamped raw-HMC pattern for one job model."""
    prior = job.fitness_distribution
    curve = job.growth_curve
    if job.dataset == "trpb":
        return f"trpb/{job.subset}/trpb_{job.subset}_{prior}_{curve}-*.nc"
    if job.dataset == "pdz":
        return f"pdz/{job.subset}/pdz_{job.subset}_{prior}_{curve}-*.nc"
    if job.dataset == "nuc":
        return f"nuc/{job.subset}/nuc_{job.subset}_{prior}-*.nc"
    raise ValueError(f"Raw-HMC fallback is unsupported for dataset {job.dataset!r}.")


def _query_remote(
    *,
    kind: str,
    experiment: str,
    job_name: str,
    pattern: str,
    amlt_executable: str | Path,
    runner: CommandRunner,
) -> tuple[tuple[RemoteArtifact, ...], tuple[str, ...]]:
    command = (
        str(Path(amlt_executable).expanduser()),
        "--json-tables",
        "results",
        "list",
        experiment,
        f":{job_name}",
        "--include",
        pattern,
    )
    completed = runner(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Amulet preflight failed for {experiment}:{job_name} "
            f"with exit code {completed.returncode}: {message}"
        )
    if completed.stdout.strip():
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Amulet returned invalid JSON for {experiment}:{job_name}: {error}"
            ) from error
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"Expected an Amulet result array for {experiment}:{job_name}.")

    artifacts = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Unexpected Amulet result row for {experiment}:{job_name}.")
        required = {"name", "size", "last_modified", "url"}
        if set(row) != required:
            raise ValueError(
                f"Unexpected Amulet result fields for {experiment}:{job_name}: "
                f"{sorted(row)}"
            )
        artifacts.append(
            RemoteArtifact(
                kind=kind,
                experiment=experiment,
                job_name=job_name,
                name=str(row["name"]),
                size=str(row["size"]),
                last_modified=str(row["last_modified"]),
            )
        )
    return tuple(artifacts), command


def missing_patterns(inventory: ArtifactInventory) -> tuple[str, ...]:
    """Return preferred remote patterns still missing locally."""
    patterns = []
    if "diagnosed_hmc" in inventory.missing_artifacts:
        patterns.append("*diagnosed.nc")
    if "sampled_mle" in inventory.missing_artifacts:
        patterns.append("*samples.nc")
    return tuple(patterns)


def preflight_job(
    inventory: ArtifactInventory,
    *,
    amlt_executable: str | Path,
    runner: CommandRunner = subprocess.run,
) -> JobPreflight:
    """List one job's missing preferred outputs through ``--json-tables``."""
    patterns = missing_patterns(inventory)
    if not patterns:
        return JobPreflight(
            job=inventory.job,
            requested_patterns=(),
            artifacts=(),
            issues=(),
            commands=(),
        )
    queries = []
    if "*diagnosed.nc" in patterns:
        queries.append(
            (
                "diagnosed_hmc",
                inventory.job.hmc_experiment,
                inventory.job.hmc_job_name,
                "*diagnosed.nc",
            )
        )
    if "*samples.nc" in patterns:
        queries.append(
            (
                "sampled_mle",
                inventory.job.mle_experiment,
                inventory.job.mle_job_name,
                "*samples.nc",
            )
        )

    artifacts = []
    issues = []
    commands = []
    for kind, experiment, job_name, pattern in queries:
        query_artifacts, command = _query_remote(
            kind=kind,
            experiment=experiment,
            job_name=job_name,
            pattern=pattern,
            amlt_executable=amlt_executable,
            runner=runner,
        )
        artifacts.extend(query_artifacts)
        commands.append(command)
        if not query_artifacts:
            issues.append(f"remote_artifact_missing:{kind}")
        elif len(query_artifacts) > 1:
            issues.append(f"multiple_remote_artifacts:{kind}")
    return JobPreflight(
        job=inventory.job,
        requested_patterns=patterns,
        artifacts=tuple(artifacts),
        issues=tuple(issues),
        commands=tuple(commands),
    )


def preflight_raw_job(
    job: SelectedJob,
    *,
    amlt_executable: str | Path,
    runner: CommandRunner = subprocess.run,
) -> JobPreflight:
    """List the exact timestamped raw-HMC candidate for one failed diagnosis."""
    pattern = raw_pattern_for_job(job)
    artifacts, command = _query_remote(
        kind="raw_hmc",
        experiment=job.hmc_experiment,
        job_name=job.hmc_job_name,
        pattern=pattern,
        amlt_executable=amlt_executable,
        runner=runner,
    )
    issues = []
    if not artifacts:
        issues.append("remote_artifact_missing:raw_hmc")
    elif len(artifacts) > 1:
        issues.append("multiple_remote_artifacts:raw_hmc")
    return JobPreflight(
        job=job,
        requested_patterns=(pattern,),
        artifacts=artifacts,
        issues=tuple(issues),
        commands=(command,),
    )
