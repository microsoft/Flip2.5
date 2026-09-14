"""Prepare or execute retirement of one validated fit with frozen products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dms_fitness_stability.fit_retirement import (
    build_product_fit_retirement_action,
    prepare_product_fit_retirement,
    remove_prepared_product_fit,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--reason",
        default="excluded_from_first_paper_and_removed_for_conversion_capacity",
    )
    parser.add_argument("--audit", type=Path, default=REPOSITORY_ROOT / "data/fit_audit.json")
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_preflight.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPOSITORY_ROOT / "data/mle_pre_conversion_baseline.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPOSITORY_ROOT / "data/retired_product_fits.json",
    )
    parser.add_argument("--amlt-root", type=Path, default=Path("/data/amlt"))
    parser.add_argument(
        "--amlt-executable",
        type=Path,
        default=Path.home() / "venvs/amulet/bin/amlt",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--prepare", action="store_true")
    action_group.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    action = build_product_fit_retirement_action(
        json.loads(args.audit.read_text(encoding="utf-8")),
        json.loads(args.preflight.read_text(encoding="utf-8")),
        json.loads(args.baseline.read_text(encoding="utf-8")),
        amlt_root=args.amlt_root,
        amlt_executable=args.amlt_executable,
        job_id=args.job,
        sha256=args.sha256,
    )
    print(
        f"job {action.job_id} ({action.dataset}): {action.path} "
        f"({action.size:,} bytes, sha256={action.sha256})"
    )
    print("recovery:", " ".join(action.recovery_command))
    if not args.prepare and not args.execute:
        print("Dry run only. Use --prepare, publish the evidence, then --execute.")
        return
    if args.prepare:
        prepare_product_fit_retirement(
            action,
            archive_path=args.archive,
            reason=args.reason,
        )
        print(f"Prepared retirement evidence at {args.archive}; fit remains present.")
        return
    remove_prepared_product_fit(action, archive_path=args.archive)
    print(f"Removed {action.path}; completion recorded at {args.archive}.")


if __name__ == "__main__":
    main()
