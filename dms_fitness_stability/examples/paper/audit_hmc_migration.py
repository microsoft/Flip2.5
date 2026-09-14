"""Freeze or verify the fit-driven direct-HMC migration baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.migration_audit import (
    build_hmc_migration_audit,
    migration_artifact_sha256,
    migration_audit_sha256,
    verify_hmc_migration_audit,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
CODE_INPUTS = (
    "src/dms_fitness_stability/adapters/nuclease.py",
    "src/dms_fitness_stability/adapters/pdz.py",
    "src/dms_fitness_stability/adapters/trpb.py",
    "src/dms_fitness_stability/analysis_manifest.py",
    "src/dms_fitness_stability/hmc_scanning.py",
    "src/dms_fitness_stability/mechanisms.py",
    "src/dms_fitness_stability/migration_audit.py",
    "src/dms_fitness_stability/reporting.py",
    "src/dms_fitness_stability/results.py",
    "src/dms_fitness_stability/streaming.py",
    "examples/paper/audit_hmc_migration.py",
    "examples/paper/figures.py",
    "examples/paper/plots.ipynb",
    "examples/paper/run_hmc.py",
    "examples/paper/run_hmc_from_scores.py",
    "examples/paper/fit_preprocessing/run_hmc_from_fits.py",
    "examples/paper/summarize.py",
    "examples/paper/summarize_mechanisms.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--fit-audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/results",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/mechanisms/cache",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=HERE / "output/hmc/tables",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=HERE / "output/hmc/figures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/hmc_matrix_migration_audit.json",
    )
    parser.add_argument("--sample-chunk-size", type=int, default=4_096)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--strict-code-inputs",
        action="store_true",
        help="Also require the original repository commit and code-input hashes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _progress(index: int, total: int, analysis_id: str) -> None:
    print(f"[{index:02d}/{total:02d}] hashed {analysis_id}", flush=True)


def _build(args: argparse.Namespace) -> dict[str, object]:
    return build_hmc_migration_audit(
        manifest_path=args.manifest,
        fit_audit_path=args.fit_audit,
        result_root=args.result_root,
        cache_root=args.cache_root,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        code_paths={name: REPOSITORY_ROOT / name for name in CODE_INPUTS},
        repository_commit=_repository_commit(),
        sample_chunk_size=args.sample_chunk_size,
        progress=_progress,
    )


def main() -> None:
    args = _parse_args()
    if args.sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    if args.strict_code_inputs and not args.verify:
        raise ValueError("--strict-code-inputs requires --verify.")
    specs = load_analysis_manifest(args.manifest)
    if not args.execute and not args.verify:
        print(
            f"Dry run: freeze {len(specs)} direct-HMC analyses from "
            f"{args.result_root}; add --execute to write {args.output}."
        )
        return

    if args.execute and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Migration baseline already exists: {args.output}. "
            "Use --verify, or --overwrite only after reviewing baseline drift."
        )
    if args.verify and not args.output.is_file():
        raise FileNotFoundError(f"Migration baseline is missing: {args.output}")

    observed = _build(args)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        verify_hmc_migration_audit(
            expected,
            observed,
            strict_code_inputs=args.strict_code_inputs,
        )
        changed_code_inputs = sorted(
            name
            for name, digest in expected.get("code_inputs", {}).items()
            if observed.get("code_inputs", {}).get(name) != digest
        )
        code_note = (
            "strict code provenance matched"
            if args.strict_code_inputs
            else f"{len(changed_code_inputs)} code input(s) changed since freeze"
        )
        print(
            f"Verified {len(specs)} analyses against {args.output} "
            f"(artifact digest {migration_artifact_sha256(observed)}; {code_note})."
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Saved {len(specs)} analyses to {args.output} "
        f"({migration_audit_sha256(observed)})."
    )


if __name__ == "__main__":
    main()
