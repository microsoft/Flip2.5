"""Validate observed-count source files and publication gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.count_sources import validate_count_sources_manifest

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
        default=REPOSITORY_ROOT / "data/count_sources_manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = validate_count_sources_manifest(
        manifest,
        data_root=args.data_root,
        verify_checksums=not args.skip_checksums,
    )
    print(
        f"Validated {counts['sources']} source files; "
        f"{counts['blocked_families']} family release gate(s) remain.",
        flush=True,
    )


if __name__ == "__main__":
    main()
