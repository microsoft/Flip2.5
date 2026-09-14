"""Audit persisted sampled-MLE score ensembles and compact results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from dms_fitness_stability import ScoreEnsemble, StabilityResult
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.candidates import ranking_candidate_mask
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _audit_score(
    score_dir: Path,
    spec,
    *,
    variant_chunk_size: int,
    require_source: bool = True,
) -> dict[str, object]:
    ensemble = ScoreEnsemble.load(score_dir)
    try:
        metadata = ensemble.metadata
        variants = ensemble.variants
        candidate_mask = ranking_candidate_mask(variants)
        if metadata["analysis_id"] != spec.analysis_id:
            raise ValueError(f"Score identity mismatch at {score_dir}.")
        if metadata["scores"] != list(spec.scores):
            raise ValueError(f"Score mismatch at {score_dir}.")
        if int(metadata["n_ranking_candidates"]) != int(candidate_mask.sum()):
            raise ValueError(f"Score candidate count mismatch at {score_dir}.")
        source = metadata.get("source_artifact")
        if not isinstance(source, dict):
            raise ValueError(f"Score source signature is missing at {score_dir}.")
        source_path = Path(str(source["path"]))
        source_present = source_path.is_file()
        if require_source and not source_present:
            raise ValueError(f"Score source is missing at {score_dir}.")
        if source_present and (
            source_path.stat().st_size != source["size"]
            or source_path.stat().st_mtime_ns != source["mtime_ns"]
        ):
            raise ValueError(f"Score source signature changed at {score_dir}.")
    finally:
        ensemble.close()

    if ensemble.source_schema_version == 2:
        score_path = score_dir / "fitness_scores.nc"
        sampled_name = "fitness_score"
        observed_name = "observed_score"
    else:
        score_path = score_dir / "fitness_proxies.nc"
        sampled_name = "fitness_proxy"
        observed_name = "observed_proxy"
    with h5py.File(score_path, "r") as handle:
        sampled = handle[sampled_name]
        observed = handle[observed_name]
        expected_shape = (
            len(spec.scores),
            int(metadata["n_samples"]),
            int(metadata["n_variants"]),
        )
        if sampled.shape != expected_shape or observed.shape != (
            expected_shape[0],
            expected_shape[-1],
        ):
            raise ValueError(f"Score tensor shape mismatch at {score_dir}.")
        if sampled.dtype != np.dtype("float32"):
            raise ValueError(f"Score tensor must be float32 at {score_dir}.")
        if sampled.chunks is None or sampled.compression != "gzip":
            raise ValueError("Score tensor must be chunked and gzip-compressed.")
        score_chunks = list(sampled.chunks)
        score_compression = sampled.compression

        finite_per_score_sample = np.zeros(expected_shape[:2], dtype=np.int64)
        variable_candidates = np.zeros(expected_shape[0], dtype=np.int64)
        score_supported_candidates = np.zeros(expected_shape[0], dtype=np.int64)
        for variant_start in range(0, expected_shape[-1], variant_chunk_size):
            variant_stop = min(
                variant_start + variant_chunk_size,
                expected_shape[-1],
            )
            values = np.asarray(sampled[:, :, variant_start:variant_stop])
            local_candidates = candidate_mask[variant_start:variant_stop]
            candidate_values = values[:, :, local_candidates]
            for score_index in range(expected_shape[0]):
                score_values = candidate_values[score_index]
                finite = np.isfinite(score_values)
                finite_per_score_sample[score_index] += finite.sum(axis=-1)
                finite_count = finite.sum(axis=0)
                score_supported_candidates[score_index] += np.count_nonzero(
                    finite_count
                )
                if score_values.shape[1]:
                    minimum = np.min(
                        np.where(finite, score_values, np.inf),
                        axis=0,
                    )
                    maximum = np.max(
                        np.where(finite, score_values, -np.inf),
                        axis=0,
                    )
                    variable_candidates[score_index] += np.count_nonzero(
                        maximum > minimum
                    )

    recorded_finite = np.asarray(
        metadata.get("finite_ranking_candidates_per_sample"),
        dtype=np.int64,
    )
    finite_per_sample = finite_per_score_sample.min(axis=0)
    if not np.array_equal(recorded_finite, finite_per_sample):
        raise ValueError(f"Finite candidate accounting mismatch at {score_dir}.")
    if np.any(variable_candidates == 0):
        raise ValueError(f"Score draws have no candidate variability at {score_dir}.")
    temporary_files = sorted(
        path.name
        for pattern in ("*.tmp", "*.dat")
        for path in score_dir.glob(pattern)
    )
    if temporary_files:
        raise ValueError(f"Temporary score files remain at {score_dir}: {temporary_files}")
    return {
        "analysis_id": spec.analysis_id,
        "job_id": spec.job_id,
        "protein_family": spec.identity.protein_family,
        "landscape": spec.identity.landscape,
        "status": "valid",
        "n_samples": expected_shape[1],
        "n_variants": expected_shape[2],
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_score_supported_candidates": int(score_supported_candidates.min()),
        "n_variable_candidates": int(variable_candidates.min()),
        "n_score_supported_candidates_by_score": {
            score: int(count)
            for score, count in zip(
                spec.scores,
                score_supported_candidates,
                strict=True,
            )
        },
        "n_variable_candidates_by_score": {
            score: int(count)
            for score, count in zip(
                spec.scores,
                variable_candidates,
                strict=True,
            )
        },
        "min_finite_ranking_candidates": int(finite_per_sample.min()),
        "max_finite_ranking_candidates": int(finite_per_sample.max()),
        "score_bytes": score_path.stat().st_size,
        "score_chunks": score_chunks,
        "score_compression": score_compression,
        "source_schema_version": ensemble.source_schema_version,
        "score_filename": score_path.name,
        "score_variable": sampled_name,
        "source_present": source_present,
    }


def _audit_result(result_dir: Path, spec) -> None:
    result = StabilityResult.load(result_dir)
    if result.metadata["analysis_id"] != spec.analysis_id:
        raise ValueError(f"Result identity mismatch at {result_dir}.")
    if result.metadata["primary_score"] != spec.primary_score:
        raise ValueError(f"Result score mismatch at {result_dir}.")
    if not isinstance(
        result.metadata.get("finite_ranking_candidates_per_sample"),
        list,
    ):
        raise ValueError(f"Result lacks finite candidate accounting at {result_dir}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_mle.json")
    parser.add_argument(
        "--score-root",
        "--proxy-root",
        dest="score_root",
        type=Path,
        default=score_bundle_root(DEFAULT_DATA_ROOT, "mle_predictive"),
        help="Score bundle root (--proxy-root is a deprecated alias).",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/mle_predictive/results",
    )
    parser.add_argument("--variant-chunk-size", type=int, default=4_096)
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--skip-results", action="store_true")
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help="Audit persisted scores after intentional source-file retirement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/mle_score_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive.")
    all_specs = load_analysis_manifest(args.manifest)
    selected_jobs = set(args.job or (spec.job_id for spec in all_specs))
    specs = tuple(spec for spec in all_specs if spec.job_id in selected_jobs)
    if {spec.job_id for spec in specs} != selected_jobs:
        raise ValueError("A requested job is absent from the selected manifest.")
    records = []
    for index, spec in enumerate(specs, start=1):
        record = _audit_score(
            args.score_root / spec.analysis_id,
            spec,
            variant_chunk_size=args.variant_chunk_size,
            require_source=not args.allow_missing_source,
        )
        if not args.skip_results:
            _audit_result(args.result_root / spec.analysis_id, spec)
        records.append(record)
        print(f"[{index}/{len(specs)}] valid: {spec.analysis_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Audited {len(records)} MLE analyses -> {args.output}")


if __name__ == "__main__":
    main()
