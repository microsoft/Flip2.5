"""Build compact rank-stability results from persisted sampled-MLE proxies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability import ScoreEnsemble, analyze_score_ensemble
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _score_matches(score_dir: Path, spec) -> bool:
    metadata_path = score_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("analysis_id") == spec.analysis_id
        and metadata.get("job_id") == spec.job_id
        and metadata.get("primary_score") == spec.primary_score
        and metadata.get("scores") == list(spec.scores)
        and metadata.get("artifact_kind") == "sampled_mle"
        and metadata.get("artifact_validation_status") == "validated"
        and (
            spec.identity.protein_family == "k50"
            or isinstance(
                metadata.get("finite_ranking_candidates_per_sample"),
                list,
            )
        )
    )


def _result_matches(
    result_dir: Path,
    spec,
    *,
    retain_k: int,
    tracked_top_n: int,
    tracked_model_top_n: int,
    top_k: tuple[int, ...],
    n_samples: int | None,
    seed: int,
) -> bool:
    metadata_path = result_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        **spec.identity.metadata(),
        "job_id": spec.job_id,
        "primary_score": spec.primary_score,
        "fit_method": "mle_predictive",
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "seed": seed,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        return False
    if spec.identity.protein_family != "k50" and not isinstance(
        metadata.get("finite_ranking_candidates_per_sample"),
        list,
    ):
        return False
    return n_samples is None or metadata.get("n_samples") == n_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "analyses_mle.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
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
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--variant-chunk-size", type=int, default=None)
    parser.add_argument("--retain-k", type=int, default=1000)
    parser.add_argument(
        "--tracked-top-n",
        type=int,
        default=None,
        help="Default: 0 for K50 job 63; 1,000 for other jobs.",
    )
    parser.add_argument("--tracked-model-top-n", type=int, default=100)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    )
    parser.add_argument("--n-pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = load_analysis_manifest(args.manifest)
    available_jobs = {spec.job_id for spec in specs}
    requested_jobs = set(args.job or available_jobs)
    unsupported = requested_jobs - available_jobs
    if unsupported:
        raise ValueError(
            f"Requested jobs are absent from the selected manifest: "
            f"{sorted(unsupported)}."
        )
    selected = tuple(spec for spec in specs if spec.job_id in requested_jobs)
    if not selected or {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("The requested jobs are not present in the MLE pilot manifest.")

    top_k = tuple(sorted(set(args.top_k)))
    plans = []
    for spec in selected:
        tracked_top_n = (
            args.tracked_top_n
            if args.tracked_top_n is not None
            else (0 if spec.job_id == 63 else 1000)
        )
        score_dir = args.score_root / spec.analysis_id
        if not _score_matches(score_dir, spec):
            raise FileNotFoundError(
            f"Missing or mismatched score ensemble for {spec.analysis_id}: "
            f"{score_dir}. Run preprocess_mle.py first."
            )
        result_dir = args.result_root / spec.analysis_id
        matches = _result_matches(
            result_dir,
            spec,
            retain_k=args.retain_k,
            tracked_top_n=tracked_top_n,
            tracked_model_top_n=args.tracked_model_top_n,
            top_k=top_k,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if args.overwrite:
            matches = False
        state = "reuse" if matches else "run"
        print(f"{state}: {spec.analysis_id}: {score_dir} -> {result_dir}")
        plans.append((spec, score_dir, result_dir, matches, tracked_top_n))

    if not args.execute:
        print(
            f"Dry run: {len(plans)} MLE predictive analysis plan(s); "
            "add --execute to run."
        )
        return

    for index, (
        spec,
        score_dir,
        result_dir,
        matches,
        tracked_top_n,
    ) in enumerate(plans, start=1):
        if matches:
            print(f"[{index}/{len(plans)}] Reusing {result_dir}", flush=True)
            continue
        if result_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing result does not match the requested analysis: {result_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        print(f"[{index}/{len(plans)}] Scanning {spec.analysis_id}", flush=True)
        ensemble = ScoreEnsemble.load(score_dir)
        try:
            result = analyze_score_ensemble(
                ensemble,
                top_k=top_k,
                retain_k=args.retain_k,
                tracked_top_n=tracked_top_n,
                tracked_model_top_n=args.tracked_model_top_n,
                n_samples=args.n_samples,
                variant_chunk_size=args.variant_chunk_size,
                seed=args.seed,
                n_pairs=args.n_pairs,
                show_progress=not args.no_progress,
            )
        finally:
            ensemble.close()
        result.save(result_dir)
        print(f"Saved {result_dir}", flush=True)


if __name__ == "__main__":
    main()
