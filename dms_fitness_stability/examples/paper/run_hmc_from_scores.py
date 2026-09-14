"""Build direct-HMC results from persisted posterior score matrices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability import ScoreEnsemble, analyze_score_ensemble
from dms_fitness_stability.analysis_manifest import load_analysis_manifest

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def result_directory(
    result_root: Path,
    *,
    sample_view: str,
    analysis_id: str,
) -> Path:
    """Keep the canonical full view at the historical paper result root."""
    view_root = result_root if sample_view == "full" else result_root / sample_view
    return view_root / analysis_id


def _bundle_matches(bundle_dir: Path, spec, *, sample_view: str) -> bool:
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("analysis_id") == spec.analysis_id
        and metadata.get("inference_method") == "hmc"
        and metadata.get("job_id") == spec.job_id
        and metadata.get("artifact_kind") == spec.artifact_kind
        and metadata.get("artifact_validation_status") == "validated"
        and metadata.get("primary_score") == spec.primary_score
        and metadata.get("scores") == list(spec.scores)
        and metadata.get("sample_view") == sample_view
    )


def _result_matches(
    result_dir: Path,
    spec,
    *,
    sample_view: str,
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
        "fit_method": "hmc_posterior",
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "seed": seed,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        return False
    matrix_provenance = (
        metadata.get("score_ensemble_schema_version") == 2
        and metadata.get("sample_view") == sample_view
    )
    frozen_full_provenance = (
        sample_view == "full"
        and metadata.get("score_ensemble_schema_version") is None
        and metadata.get("sample_view") is None
        and isinstance(metadata.get("hmc_file"), dict)
    )
    if not matrix_provenance and not frozen_full_provenance:
        return False
    return n_samples is None or metadata.get("n_samples") == n_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "analyses_hmc.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--score-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/scores/hmc",
    )
    parser.add_argument("--sample-view", default="full")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/results",
    )
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-id", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--variant-chunk-size", type=int, default=None)
    parser.add_argument("--retain-k", type=int, default=1000)
    parser.add_argument("--tracked-top-n", type=int, default=1000)
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
    if {spec.identity.inference_method for spec in specs} != {"hmc"}:
        raise ValueError("Matrix HMC analysis requires an HMC-only manifest.")
    families = set(args.family or ())
    requested_jobs = set(args.job or ())
    requested_ids = set(args.analysis_id or ())
    selected = tuple(
        spec
        for spec in specs
        if (not families or spec.identity.protein_family in families)
        and (not requested_jobs or spec.job_id in requested_jobs)
        and (not requested_ids or spec.analysis_id in requested_ids)
    )
    if not selected:
        raise ValueError("No matrix HMC analyses match the requested filters.")
    if requested_jobs and {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("A requested job is absent from the selected manifest or family.")
    if requested_ids and {spec.analysis_id for spec in selected} != requested_ids:
        raise ValueError("A requested analysis ID is absent from the selected manifest.")
    if args.n_samples is not None and args.n_samples < 2:
        raise ValueError("n_samples must be at least two when provided.")

    top_k = tuple(sorted(set(args.top_k)))
    plans = []
    for spec in selected:
        bundle_dir = args.score_root / args.sample_view / spec.analysis_id
        if not _bundle_matches(bundle_dir, spec, sample_view=args.sample_view):
            raise FileNotFoundError(
                f"Missing or mismatched score ensemble for {spec.analysis_id}: "
                f"{bundle_dir}."
            )
        result_dir = result_directory(
            args.result_root,
            sample_view=args.sample_view,
            analysis_id=spec.analysis_id,
        )
        matches = _result_matches(
            result_dir,
            spec,
            sample_view=args.sample_view,
            retain_k=args.retain_k,
            tracked_top_n=args.tracked_top_n,
            tracked_model_top_n=args.tracked_model_top_n,
            top_k=top_k,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if args.overwrite:
            matches = False
        state = "reuse" if matches else "run"
        print(f"{state}: {spec.analysis_id}: {bundle_dir} -> {result_dir}")
        plans.append((spec, bundle_dir, result_dir, matches))

    if not args.execute:
        print(
            f"Dry run: {len(plans)} matrix HMC analysis plan(s); "
            "add --execute to run."
        )
        return

    for index, (spec, bundle_dir, result_dir, matches) in enumerate(plans, start=1):
        if matches:
            print(f"[{index}/{len(plans)}] Reusing {result_dir}", flush=True)
            continue
        if result_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing result does not match: {result_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        print(f"[{index}/{len(plans)}] Scanning {spec.analysis_id}", flush=True)
        ensemble = ScoreEnsemble.load(bundle_dir)
        try:
            result = analyze_score_ensemble(
                ensemble,
                top_k=top_k,
                retain_k=args.retain_k,
                tracked_top_n=args.tracked_top_n,
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
