"""Validate local analysis-data bundles against the tracked release manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dms_fitness_stability.analysis_data import (
    load_and_validate_analysis_data_manifest,
)

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
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    counts = load_and_validate_analysis_data_manifest(
        args.manifest,
        data_root=args.data_root,
        verify_checksums=not args.skip_checksums,
    )
    print(
        f"Validated {counts['score_bundles']} score bundles and "
        f"{counts['feature_bundles']} feature bundles.",
        flush=True,
    )


if __name__ == "__main__":
    main()
