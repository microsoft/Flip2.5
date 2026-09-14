"""Build the checksummed local analysis-data release manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.analysis_data import build_analysis_data_manifest
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--direct-hmc-view", action="append", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/analysis_data_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_analysis_data_manifest(
        data_root=args.data_root,
        direct_hmc_root=args.data_root / "paper/analysis_data/scores/hmc",
        mle_score_root=score_bundle_root(args.data_root, "mle_predictive"),
        hmc_count_score_root=score_bundle_root(args.data_root, "hmc_count_based"),
        feature_root=args.data_root / "paper/analysis_data/landscape_features",
        direct_hmc_views=tuple(
            args.direct_hmc_view or ("full", "release_1000")
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Saved {len(payload['score_bundles'])} score and "
        f"{len(payload['feature_bundles'])} feature bundles to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
