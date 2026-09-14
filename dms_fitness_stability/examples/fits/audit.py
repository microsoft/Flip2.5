"""Audit selected fit artifacts and write JSON plus central known issues."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dms_fitness_stability.fit_audit import (
    audit_inventory,
    render_known_issues,
    retired_diagnosed_from_records,
    retired_raw_from_records,
)
from dms_fitness_stability.fit_contracts import FitContractResolver
from dms_fitness_stability.fit_inventory import MLE_ONLY_JOB_IDS, build_inventory

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CSV_PATH = DEFAULT_REPOSITORY_ROOT / "data/HMC Results Summaries.csv"
DEFAULT_AMLT_ROOT = Path("/data/amlt")
DEFAULT_DATA_ROOT = Path("/data/data_dms_stability")
DEFAULT_PREFLIGHT = DEFAULT_REPOSITORY_ROOT / "data/fit_preflight.json"
DEFAULT_RAW_PREFLIGHT = DEFAULT_REPOSITORY_ROOT / "data/raw_fit_preflight.json"
DEFAULT_RETIRED_DIAGNOSED = (
    DEFAULT_REPOSITORY_ROOT / "data/retired_diagnosed_hmc.json"
)
DEFAULT_RETIRED_RAW = DEFAULT_REPOSITORY_ROOT / "data/retired_raw_hmc.json"


def _blank_rows(csv_path: Path) -> tuple[dict[str, str], ...]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if not row["Result"].strip()
            and int(row["ID"]) not in MLE_ONLY_JOB_IDS
        )
    return rows


def _records(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    return tuple(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--amlt-root", type=Path, default=DEFAULT_AMLT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--raw-preflight", type=Path, default=DEFAULT_RAW_PREFLIGHT)
    parser.add_argument(
        "--retired-diagnosed",
        type=Path,
        default=DEFAULT_RETIRED_DIAGNOSED,
    )
    parser.add_argument("--retired-raw", type=Path, default=DEFAULT_RETIRED_RAW)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument(
        "--known-issues",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT / "data/KNOWN_ISSUES.md",
    )
    args = parser.parse_args()

    inventory = build_inventory(
        csv_path=args.csv_path,
        amlt_root=args.amlt_root,
        data_root=args.data_root,
    )
    resolver = FitContractResolver(
        repository_root=args.repository_root,
        data_root=args.data_root,
    )
    audits = audit_inventory(
        inventory,
        resolver=resolver,
        retired_diagnosed_hmc=retired_diagnosed_from_records(
            _records(args.retired_diagnosed)
        ),
        retired_raw_hmc=retired_raw_from_records(_records(args.retired_raw)),
    )
    payload = [audit.to_dict() for audit in audits]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.known_issues.parent.mkdir(parents=True, exist_ok=True)
    args.known_issues.write_text(
        render_known_issues(
            audits,
            blank_rows=_blank_rows(args.csv_path),
            preferred_preflight=_records(args.preflight),
            raw_preflight=_records(args.raw_preflight),
        ),
        encoding="utf-8",
    )
    fallback_count = sum(audit.fallback_required for audit in audits)
    missing_hmc_count = sum(audit.preferred_hmc_download_required for audit in audits)
    print(
        f"Audited {len(audits)} jobs; {missing_hmc_count} await diagnosed HMC and "
        f"{fallback_count} require raw-HMC fallback review."
    )
    print(f"JSON: {args.json_output}")
    print(f"Known issues: {args.known_issues}")


if __name__ == "__main__":
    main()
