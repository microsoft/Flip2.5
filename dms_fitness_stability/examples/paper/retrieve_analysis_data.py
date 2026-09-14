"""Download and verify published analysis-data archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.analysis_data import retrieve_analysis_data_archives

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/analysis_data_manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive", nargs="+", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/archive_cache",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    archives = manifest.get("archives", [])
    selected = [
        record
        for record in archives
        if args.archive is None or record.get("name") in set(args.archive)
    ]
    for record in selected:
        print(
            f"retrieve: {record.get('name')}: {record.get('bytes')} bytes from "
            f"{record.get('url')}"
        )
    if not args.execute:
        print(
            f"Dry run: {len(selected)} archive(s); add --execute after publication."
        )
        return
    extracted = retrieve_analysis_data_archives(
        manifest,
        destination=args.data_root,
        cache_dir=args.cache_dir,
        archive_names=None if args.archive is None else set(args.archive),
    )
    print(f"Retrieved and extracted {len(extracted)} archive(s).", flush=True)


if __name__ == "__main__":
    main()
