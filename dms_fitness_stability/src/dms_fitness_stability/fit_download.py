"""Build capacity-checked exact-path downloads from fit preflight records."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

SIZE_UNITS = {
    "B": 1,
    "kB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "TB": 1_000_000_000_000,
}


@dataclass(frozen=True)
class DownloadAction:
    """One exact remote artifact download."""

    family: str
    job_id: int
    kind: str
    experiment: str
    job_name: str
    remote_name: str
    size: str
    size_bytes: int
    destination: Path
    command: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)


def parse_human_size(value: str) -> int:
    """Parse an Amulet decimal human-readable file size."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?) ([kMGT]?B)", value)
    if match is None:
        raise ValueError(f"Unsupported Amulet size: {value!r}")
    return round(float(match.group(1)) * SIZE_UNITS[match.group(2)])


def build_download_actions(
    records: list[dict[str, object]],
    *,
    amlt_executable: str | Path,
    output_root: str | Path,
    n_workers: int,
    families: set[str] | None = None,
    kinds: set[str] | None = None,
    job_ids: set[int] | None = None,
) -> tuple[DownloadAction, ...]:
    """Return exact missing downloads selected from preflight records."""
    if not 1 <= n_workers <= 50:
        raise ValueError("n_workers must be between 1 and 50.")
    output_root = Path(output_root)
    executable = str(Path(amlt_executable).expanduser())
    actions = []
    for record in records:
        family = str(record["family"])
        job_id = int(record["job_id"])
        if families and family not in families:
            continue
        if job_ids and job_id not in job_ids:
            continue
        issues = tuple(str(issue) for issue in record.get("issues", []))
        if issues:
            raise ValueError(
                f"Preflight issues block {record['job_name']}: {', '.join(issues)}"
            )
        for artifact in record["artifacts"]:
            kind = str(artifact["kind"])
            if kinds and kind not in kinds:
                continue
            experiment = str(artifact["experiment"])
            job_name = str(artifact["job_name"])
            remote_name = str(artifact["name"])
            destination = output_root / experiment / job_name / remote_name
            if destination.is_file():
                continue
            command = (
                executable,
                "results",
                "download",
                experiment,
                f":{job_name}",
                "--include",
                remote_name,
                "--output",
                str(output_root),
                "--n-workers",
                str(n_workers),
            )
            size = str(artifact["size"])
            actions.append(
                DownloadAction(
                    family=family,
                    job_id=job_id,
                    kind=kind,
                    experiment=experiment,
                    job_name=job_name,
                    remote_name=remote_name,
                    size=size,
                    size_bytes=parse_human_size(size),
                    destination=destination,
                    command=command,
                )
            )
    return tuple(actions)


def required_bytes(actions: tuple[DownloadAction, ...]) -> int:
    return sum(action.size_bytes for action in actions)


def ensure_capacity(
    actions: tuple[DownloadAction, ...],
    *,
    free_bytes: int,
    reserve_bytes: int,
) -> None:
    """Reject a selection that would consume the configured free-space reserve."""
    required = required_bytes(actions)
    available = max(0, free_bytes - reserve_bytes)
    if required > available:
        raise OSError(
            f"Selected downloads require about {required / 1e9:.1f} GB, but only "
            f"{available / 1e9:.1f} GB are available after the reserve."
        )
