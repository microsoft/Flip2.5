"""Build the observed-count source provenance and release-gate manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.count_sources import build_count_sources_manifest

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/landscape_features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/count_sources_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_count_sources_manifest(
        data_root=args.data_root,
        feature_root=args.feature_root,
        repository_root=REPOSITORY_ROOT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Saved {len(payload['sources'])} observed sources to {args.output}; "
        f"blocked families: {payload['blocked_families']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
