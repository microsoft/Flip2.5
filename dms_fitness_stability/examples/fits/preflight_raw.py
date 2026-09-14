"""Preflight raw HMC only for diagnosed artifacts that failed validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dms_fitness_stability.fit_inventory import load_selected_jobs
from dms_fitness_stability.fit_preflight import preflight_raw_job

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CSV_PATH = DEFAULT_REPOSITORY_ROOT / "data/HMC Results Summaries.csv"
DEFAULT_AUDIT = DEFAULT_REPOSITORY_ROOT / "data/fit_audit.json"
DEFAULT_AMLT_EXECUTABLE = Path.home() / "venvs/amulet/bin/amlt"
DEFAULT_OUTPUT = DEFAULT_REPOSITORY_ROOT / "data/raw_fit_preflight.json"


def _write_output(path: Path, records: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps([records[name] for name in sorted(records)], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--amlt-executable", type=Path, default=DEFAULT_AMLT_EXECUTABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job", type=int, nargs="+")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    fallback_ids = {
        int(record["job_id"]) for record in audit if record["fallback_required"]
    }
    requested_ids = set(args.job or fallback_ids)
    invalid = requested_ids - fallback_ids
    if invalid:
        raise ValueError(
            f"Raw fallback is not currently triggered for jobs: {sorted(invalid)}"
        )
    jobs = {
        job.job_id: job
        for job in load_selected_jobs(args.csv_path)
        if job.job_id in requested_ids
    }
    if set(jobs) != requested_ids:
        raise ValueError(f"Selected jobs are unavailable: {sorted(requested_ids - set(jobs))}")

    records: dict[str, dict[str, object]] = {}
    if args.output.is_file() and not args.refresh:
        records = {
            record["job_name"]: record
            for record in json.loads(args.output.read_text(encoding="utf-8"))
        }
    for job_id in sorted(jobs):
        job = jobs[job_id]
        if job.job_name in records and not args.refresh:
            print(f"cached: {job.job_name}")
            continue
        record = preflight_raw_job(job, amlt_executable=args.amlt_executable)
        records[job.job_name] = record.to_dict()
        _write_output(args.output, records)
        artifact_text = ", ".join(
            f"{artifact.name} ({artifact.size})" for artifact in record.artifacts
        ) or "no matching remote artifacts"
        issue_text = f"; issues={record.issues}" if record.issues else ""
        print(f"raw preflight: {job.job_name}: {artifact_text}{issue_text}")
    print(f"Raw preflight records: {args.output}")


if __name__ == "__main__":
    main()
