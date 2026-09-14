"""Download exact preflighted fit artifacts with a free-space guard."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from dms_fitness_stability.fit_download import (
    build_download_actions,
    ensure_capacity,
    required_bytes,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_PREFLIGHT = DEFAULT_REPOSITORY_ROOT / "data/fit_preflight.json"
DEFAULT_AMLT_EXECUTABLE = Path.home() / "venvs/amulet/bin/amlt"
DEFAULT_OUTPUT_ROOT = Path("/data/amlt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--amlt-executable", type=Path, default=DEFAULT_AMLT_EXECUTABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--family",
        choices=("small_clean", "large_clean"),
        nargs="+",
    )
    parser.add_argument(
        "--kind",
        choices=("diagnosed_hmc", "sampled_mle", "raw_hmc"),
        nargs="+",
    )
    parser.add_argument("--job", type=int, nargs="+")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--reserve-gb", type=float, default=100.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.preflight.read_text(encoding="utf-8"))
    actions = build_download_actions(
        records,
        amlt_executable=args.amlt_executable,
        output_root=args.output_root,
        n_workers=args.n_workers,
        families=set(args.family) if args.family else None,
        kinds=set(args.kind) if args.kind else None,
        job_ids=set(args.job) if args.job else None,
    )
    if any(action.kind == "raw_hmc" for action in actions) and args.n_workers != 1:
        raise ValueError("Raw HMC fallbacks must be downloaded with --n-workers 1.")
    required = required_bytes(actions)
    free = shutil.disk_usage(args.output_root).free
    reserve = round(args.reserve_gb * 1_000_000_000)
    print(
        f"Selected {len(actions)} artifact(s), about {required / 1e9:.1f} GB; "
        f"filesystem free {free / 1e9:.1f} GB; reserve {args.reserve_gb:.1f} GB."
    )
    for action in actions:
        print(f"[{action.family} {action.kind} {action.size}] {action.shell_command}")
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the selection.")
        return

    ensure_capacity(actions, free_bytes=free, reserve_bytes=reserve)
    for index, action in enumerate(actions, start=1):
        current_free = shutil.disk_usage(args.output_root).free
        ensure_capacity(
            (action,),
            free_bytes=current_free,
            reserve_bytes=reserve,
        )
        print(f"Downloading {index}/{len(actions)}: {action.remote_name}", flush=True)
        completed = subprocess.run(action.command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Amulet download failed with exit code {completed.returncode}: "
                f"{action.experiment}:{action.job_name}"
            )
        if not action.destination.is_file():
            raise FileNotFoundError(
                f"Amulet exited successfully but the expected file is missing: "
                f"{action.destination}"
            )
    print("Downloads completed. Run `uv run python examples/fits/audit.py` next.")


if __name__ == "__main__":
    main()
