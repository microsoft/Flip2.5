"""Freeze immutable pre-conversion matrix, result, cache, table, and figure audits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from dms_fitness_stability.migration_audit import (
    build_score_method_baseline,
    verify_score_method_conversion,
)
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
METHOD_DEFAULTS = {
    "mle_predictive": {
        "manifest": HERE / "analyses_mle.json",
        "data_dir": "mle_predictive",
        "output": REPOSITORY_ROOT / "data/mle_pre_conversion_baseline.json",
    },
    "hmc_count_based": {
        "manifest": HERE / "analyses_hmc_count_based.json",
        "data_dir": "hmc_count_based",
        "output": REPOSITORY_ROOT / "data/hmc_count_pre_conversion_baseline.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_DEFAULTS), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--score-root", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify-against", type=Path, default=None)
    parser.add_argument("--sample-chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = METHOD_DEFAULTS[args.method]
    data_dir = str(defaults["data_dir"])
    paper_data = args.data_root / "paper" / data_dir
    paper_output = HERE / "output" / data_dir
    output = args.output or Path(defaults["output"])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def progress(index: int, total: int, analysis_id: str) -> None:
        print(f"[{index:02d}/{total:02d}] audited {analysis_id}", flush=True)

    audit = build_score_method_baseline(
        manifest_path=args.manifest or Path(defaults["manifest"]),
        score_root=args.score_root or score_bundle_root(args.data_root, args.method),
        result_root=args.result_root or paper_data / "results",
        cache_root=args.cache_root or paper_data / "mechanisms/cache",
        table_dir=args.table_dir or paper_output / "tables",
        figure_dir=args.figure_dir or paper_output / "figures",
        inference_method=args.method,
        repository_commit=commit,
        sample_chunk_size=args.sample_chunk_size,
        progress=progress,
    )
    if args.verify_against is not None:
        expected = json.loads(args.verify_against.read_text(encoding="utf-8"))
        verify_score_method_conversion(expected, audit)
        print(
            f"Verified schema-v2 artifacts against {args.verify_against}.",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"Saved {len(audit['analyses'])} analyses, "
        f"{len(audit['mechanism_caches'])} caches, "
        f"{len(audit['tables'])} tables, and "
        f"{len(audit['figures'])} figure files to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
