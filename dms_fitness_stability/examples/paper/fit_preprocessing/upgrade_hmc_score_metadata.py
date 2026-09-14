"""Add deterministic reference metadata to existing direct-HMC score bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.score_ensembles import (
    ScoreEnsemble,
    direct_hmc_reference_metadata,
)

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/scores/hmc",
    )
    parser.add_argument("--sample-view", nargs="+", default=("full", "release_1000"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = []
    for sample_view in args.sample_view:
        root = args.score_root / sample_view
        if not root.is_dir():
            print(f"skip missing view: {root}")
            continue
        for bundle_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            ensemble = ScoreEnsemble.load(bundle_dir)
            try:
                derived = direct_hmc_reference_metadata(
                    ensemble.metadata,
                    ensemble.analysis_variants,
                )
                conflicts = {
                    name: (ensemble.metadata[name], value)
                    for name, value in derived.items()
                    if name in ensemble.metadata and ensemble.metadata[name] != value
                }
                if conflicts:
                    raise ValueError(
                        f"Reference metadata conflicts at {bundle_dir}: {conflicts}"
                    )
                pending = {
                    name: value
                    for name, value in derived.items()
                    if ensemble.metadata.get(name) != value
                }
            finally:
                ensemble.close()
            state = "update" if pending else "current"
            print(f"{state}: {sample_view}: {bundle_dir.name}")
            tasks.append((bundle_dir, pending))
    if not args.execute:
        print(f"Dry run: {len(tasks)} score metadata plan(s); add --execute to update.")
        return
    for index, (bundle_dir, pending) in enumerate(tasks, start=1):
        if pending:
            metadata_path = bundle_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(pending)
            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(metadata_path)
            validated = ScoreEnsemble.load(bundle_dir)
            validated.close()
        print(f"[{index}/{len(tasks)}] valid: {bundle_dir}", flush=True)


if __name__ == "__main__":
    main()
