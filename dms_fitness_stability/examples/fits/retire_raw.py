"""Archive unusable raw-HMC evidence, then remove exact local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dms_fitness_stability.fit_retirement import (
    archive_raw_and_remove,
    build_raw_retirement_actions,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_AUDIT = DEFAULT_REPOSITORY_ROOT / "data/fit_audit.json"
DEFAULT_RAW_PREFLIGHT = DEFAULT_REPOSITORY_ROOT / "data/raw_fit_preflight.json"
DEFAULT_ARCHIVE = DEFAULT_REPOSITORY_ROOT / "data/retired_raw_hmc.json"
DEFAULT_AMLT_ROOT = Path("/data/amlt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--raw-preflight", type=Path, default=DEFAULT_RAW_PREFLIGHT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--amlt-root", type=Path, default=DEFAULT_AMLT_ROOT)
    parser.add_argument("--job", type=int, nargs="+", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    actions = build_raw_retirement_actions(
        json.loads(args.audit.read_text(encoding="utf-8")),
        json.loads(args.raw_preflight.read_text(encoding="utf-8")),
        amlt_root=args.amlt_root,
        job_ids=set(args.job),
    )
    for action in actions:
        print(
            f"{action.job_name}: retire {action.path} "
            f"({action.validation['status']}); source={action.source['name']} "
            f"({action.source['size']})"
        )
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the selection.")
        return

    archive_raw_and_remove(actions, archive_path=args.archive)
    print(f"Archived {len(actions)} raw validation record(s): {args.archive}")
    print(f"Removed {len(actions)} raw HMC file(s).")


if __name__ == "__main__":
    main()
