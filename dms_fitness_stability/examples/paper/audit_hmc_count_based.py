"""Audit persisted all-draw HMC count-based scores and compact results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from audit_mle import _audit_result, _audit_score

from dms_fitness_stability import ScoreEnsemble
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _audit_hmc_score(score_dir: Path, spec, *, variant_chunk_size: int) -> dict[str, object]:
    record = _audit_score(
        score_dir,
        spec,
        variant_chunk_size=variant_chunk_size,
    )
    ensemble = ScoreEnsemble.load(score_dir)
    try:
        metadata = ensemble.metadata
        if metadata.get("inference_method") != "hmc_count_based":
            raise ValueError(f"Score method mismatch at {score_dir}.")
        if metadata.get("artifact_kind") != spec.artifact_kind:
            raise ValueError(f"Score artifact kind mismatch at {score_dir}.")
        n_chains = int(metadata["n_chains"])
        draws_per_chain = int(metadata["draws_per_chain"])
        expected_samples = n_chains * draws_per_chain
        if metadata.get("n_samples") != expected_samples:
            raise ValueError(f"Canonical score omits HMC samples at {score_dir}.")
        source_sample_id = np.asarray(
            ensemble.scores["source_sample_id"].values,
            dtype=np.int64,
        )
        if not np.array_equal(source_sample_id, np.arange(expected_samples)):
            raise ValueError(f"Canonical score sample IDs are incomplete at {score_dir}.")
        expected_chain = source_sample_id // draws_per_chain
        expected_draw = source_sample_id % draws_per_chain
        if not np.array_equal(
            ensemble.scores["source_chain"].values,
            expected_chain,
        ) or not np.array_equal(
            ensemble.scores["source_draw"].values,
            expected_draw,
        ):
            raise ValueError(f"Score chain/draw coordinates disagree at {score_dir}.")
    finally:
        ensemble.close()
    return {
        **record,
        "artifact_kind": spec.artifact_kind,
        "n_chains": n_chains,
        "draws_per_chain": draws_per_chain,
        "all_source_samples_present": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "analyses_hmc_count_based.json",
    )
    parser.add_argument(
        "--score-root",
        "--proxy-root",
        dest="score_root",
        type=Path,
        default=score_bundle_root(DEFAULT_DATA_ROOT, "hmc_count_based"),
        help="Score bundle root (--proxy-root is a deprecated alias).",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc_count_based/results",
    )
    parser.add_argument("--variant-chunk-size", type=int, default=4_096)
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--skip-results", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/hmc_count_based_score_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive.")
    all_specs = load_analysis_manifest(args.manifest)
    if {spec.identity.inference_method for spec in all_specs} != {"hmc_count_based"}:
        raise ValueError("HMC count audit requires an hmc_count_based manifest.")
    selected_jobs = set(args.job or (spec.job_id for spec in all_specs))
    specs = tuple(spec for spec in all_specs if spec.job_id in selected_jobs)
    if {spec.job_id for spec in specs} != selected_jobs:
        raise ValueError("A requested job is absent from the selected manifest.")

    records = []
    for index, spec in enumerate(specs, start=1):
        record = _audit_hmc_score(
            args.score_root / spec.analysis_id,
            spec,
            variant_chunk_size=args.variant_chunk_size,
        )
        if not args.skip_results:
            _audit_result(args.result_root / spec.analysis_id, spec)
        records.append(record)
        print(f"[{index}/{len(specs)}] valid: {spec.analysis_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Audited {len(records)} HMC count-based analyses -> {args.output}")


if __name__ == "__main__":
    main()
