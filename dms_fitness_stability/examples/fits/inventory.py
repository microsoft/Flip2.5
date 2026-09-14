"""Print the selected-job and local fit-artifact inventory as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dms_fitness_stability.fit_inventory import build_inventory

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CSV_PATH = DEFAULT_REPOSITORY_ROOT / "data/HMC Results Summaries.csv"
DEFAULT_AMLT_ROOT = Path("/data/amlt")
DEFAULT_DATA_ROOT = Path("/data/data_dms_stability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--amlt-root", type=Path, default=DEFAULT_AMLT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory = build_inventory(
        csv_path=args.csv_path,
        amlt_root=args.amlt_root,
        data_root=args.data_root,
    )
    payload = [
        {
            **asdict(item.job),
            "job_name": item.job.job_name,
            "diagnosed_hmc": list(item.diagnosed_hmc),
            "sampled_mle": list(item.sampled_mle),
            "raw_hmc": list(item.raw_hmc),
            "missing_artifacts": list(item.missing_artifacts),
        }
        for item in inventory
    ]
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
