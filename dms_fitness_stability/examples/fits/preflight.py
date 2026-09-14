"""Preflight missing preferred fit artifacts through Amulet without downloading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dms_fitness_stability.fit_inventory import build_inventory
from dms_fitness_stability.fit_preflight import preflight_job

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CSV_PATH = DEFAULT_REPOSITORY_ROOT / "data/HMC Results Summaries.csv"
DEFAULT_AMLT_ROOT = Path("/data/amlt")
DEFAULT_DATA_ROOT = Path("/data/data_dms_stability")
DEFAULT_AMLT_EXECUTABLE = Path.home() / "venvs/amulet/bin/amlt"


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
    parser.add_argument("--amlt-root", type=Path, default=DEFAULT_AMLT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--amlt-executable", type=Path, default=DEFAULT_AMLT_EXECUTABLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT / "data/fit_preflight.json",
    )
    parser.add_argument("--job", type=int, nargs="+")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    inventory = build_inventory(
        csv_path=args.csv_path,
        amlt_root=args.amlt_root,
        data_root=args.data_root,
    )
    selected_ids = set(args.job or ())
    inventory = tuple(
        item for item in inventory if not selected_ids or item.job.job_id in selected_ids
    )
    unknown_ids = selected_ids - {item.job.job_id for item in inventory}
    if unknown_ids:
        raise ValueError(f"Requested jobs are not selected: {sorted(unknown_ids)}")

    records: dict[str, dict[str, object]] = {}
    if args.output.is_file() and not args.refresh:
        records = {
            record["job_name"]: record
            for record in json.loads(args.output.read_text(encoding="utf-8"))
        }
    for item in inventory:
        if item.job.job_name in records and not args.refresh:
            print(f"cached: {item.job.job_name}")
            continue
        record = preflight_job(item, amlt_executable=args.amlt_executable)
        records[item.job.job_name] = record.to_dict()
        _write_output(args.output, records)
        artifact_text = ", ".join(
            f"{artifact.name} ({artifact.size})" for artifact in record.artifacts
        ) or "no matching remote artifacts"
        issue_text = f"; issues={record.issues}" if record.issues else ""
        print(f"preflight: {item.job.job_name}: {artifact_text}{issue_text}")
    print(f"Preflight records: {args.output}")


if __name__ == "__main__":
    main()
