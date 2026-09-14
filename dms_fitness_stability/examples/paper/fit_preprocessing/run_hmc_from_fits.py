"""Run the legacy direct-HMC analysis from audited fit artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.adapters.nuclease import (
    NUCLEASE_SCORE_NAME,
    load_nuclease_landscape,
    scan_nuclease_hmc,
)
from dms_fitness_stability.adapters.pdz import (
    load_pdz_cis,
    load_pdz_cript,
    load_pdz_trans,
    scan_pdz_hmc,
)
from dms_fitness_stability.adapters.trpb import (
    load_trpb_landscape,
    scan_trpb_hmc,
)
from dms_fitness_stability.analysis_manifest import (
    AnalysisSpec,
    ArtifactResolver,
    ResolvedArtifact,
    load_analysis_manifest,
)
from dms_fitness_stability.results import StabilityResult

HERE = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _load_pdz_landscape(source: str, data_root: Path):
    if source in {"trans-1", "trans-2"}:
        raw_root = data_root / "pdz_trans/raw"
        return load_pdz_trans(
            source,
            count_paths={
                name: raw_root / f"{name}.tsv"
                for name in ("trans-1", "trans-2")
            },
            reported_fitness_path=raw_root / "GSE265816_pdz3cript_sub.csv.gz",
        )
    raw_root = data_root / "pdz_cis/raw"
    if source == "cis":
        return load_pdz_cis(
            raw_root / "cis_from_bruce_azure.tsv",
            reported_fitness_path=raw_root / "GSE265816_cript2xa3_sub.csv.gz",
        )
    fitness_name = source.replace("-", "_")
    return load_pdz_cript(
        source,
        raw_root / f"{source}.tsv",
        reported_fitness_path=raw_root / f"GSE265816_{fitness_name}_sub.csv.gz",
    )


def _load_landscape(spec: AnalysisSpec, data_root: Path):
    identity = spec.identity
    source = identity.landscape.removeprefix(f"{identity.protein_family}-")
    if identity.protein_family == "pdz":
        return _load_pdz_landscape(source, data_root)
    if identity.protein_family == "trpb":
        trpb_root = data_root / "trpb/raw"
        return load_trpb_landscape(
            source,
            count_path=trpb_root / "counts" / f"{source}.csv",
            reported_fitness_path=(
                trpb_root / "reported_fitness" / f"{source}.csv"
            ),
        )
    if identity.protein_family == "nuc":
        nuc_root = data_root / "nuc/raw"
        return load_nuclease_landscape(
            source,
            count_dir=nuc_root / "counts",
            reported_activity_path=nuc_root / "reported_fitness/landscape.csv",
        )
    raise ValueError(f"Unsupported HMC protein family: {identity.protein_family!r}.")


def _existing_matches(
    result_dir: Path,
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
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
        "artifact_kind": spec.artifact_kind,
        "primary_score": spec.primary_score,
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "seed": seed,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        return False
    if n_samples is not None and metadata.get("n_samples") != n_samples:
        return False
    hmc_file = metadata.get("hmc_file")
    if not isinstance(hmc_file, dict):
        return False
    return (
        Path(str(hmc_file.get("path"))).resolve() == artifact.path
        and hmc_file.get("size") == artifact.bytes
        and metadata.get("scores") == list(spec.scores)
    )


def _run_analysis(
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
    landscape,
    *,
    retain_k: int,
    tracked_top_n: int,
    tracked_model_top_n: int,
    top_k: tuple[int, ...],
    n_samples: int | None,
    n_pairs: int,
    seed: int,
    show_progress: bool,
) -> StabilityResult:
    identity = spec.identity
    common = {
        "top_k": top_k,
        "retain_k": retain_k,
        "tracked_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "seed": seed,
        "fit_id": identity.fit_id,
        "show_progress": show_progress,
    }
    if identity.protein_family == "pdz":
        growth_rate_prior, growth_curve = identity.model_spec.split("-", 1)
        result = scan_pdz_hmc(
            landscape,
            artifact.path,
            growth_rate_prior=growth_rate_prior,
            growth_curve=growth_curve,
            score_names=(spec.primary_score,),
            **common,
        )
    elif identity.protein_family == "trpb":
        growth_rate_prior, growth_curve = identity.model_spec.split("-", 1)
        result = scan_trpb_hmc(
            landscape,
            artifact.path,
            growth_rate_prior=growth_rate_prior,
            growth_curve=growth_curve,
            score_names=(spec.primary_score,),
            **common,
        )
    elif identity.protein_family == "nuc":
        if spec.primary_score != NUCLEASE_SCORE_NAME:
            raise ValueError(
                f"Nuc HMC primary_score must be {NUCLEASE_SCORE_NAME!r}."
            )
        result = scan_nuclease_hmc(
            landscape,
            artifact.path,
            fitness_distribution=identity.model_spec,
            job_id=spec.job_id,
            artifact_kind=artifact.kind,
            **common,
        )
    else:
        raise ValueError(f"Unsupported HMC protein family: {identity.protein_family!r}.")

    if result.metadata.get("analysis_id") != spec.analysis_id:
        raise ValueError(
            f"Scanner produced {result.metadata.get('analysis_id')!r}, "
            f"expected {spec.analysis_id!r}."
        )
    provenance = {
        "job_id": spec.job_id,
        "artifact_kind": artifact.kind,
        "artifact_validation_status": artifact.status,
        "primary_score": spec.primary_score,
    }
    result.metadata.setdefault("artifact_status", artifact.status)
    result.metadata.update(provenance)
    result.selection.attrs.update(provenance)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/results",
    )
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+")
    parser.add_argument("--analysis-id", nargs="+")
    parser.add_argument("--n-samples", type=int, default=None)
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
    families = set(args.family or ())
    job_ids = set(args.job or ())
    analysis_ids = set(args.analysis_id or ())
    selected = tuple(
        spec
        for spec in specs
        if (not families or spec.identity.protein_family in families)
        and (not job_ids or spec.job_id in job_ids)
        and (not analysis_ids or spec.analysis_id in analysis_ids)
    )
    if not selected:
        raise ValueError("No HMC analyses match the requested filters.")

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    resolved = resolver.resolve_all(selected)
    top_k = tuple(sorted(set(args.top_k)))
    plans = []
    for spec, artifact in zip(selected, resolved, strict=True):
        result_dir = args.result_root / spec.analysis_id
        matches = _existing_matches(
            result_dir,
            spec,
            artifact,
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
        print(
            f"{state}: {spec.analysis_id}: job {spec.job_id}, {artifact.kind}, "
            f"{artifact.path} -> {result_dir}"
        )
        plans.append((spec, artifact, result_dir, matches))

    if not args.execute:
        print(f"Dry run: {len(plans)} HMC analysis plan(s); add --execute to run.")
        return

    landscape_cache = {}
    for index, (spec, artifact, result_dir, matches) in enumerate(plans, start=1):
        if matches:
            print(f"[{index}/{len(plans)}] Reusing {result_dir}", flush=True)
            continue
        if result_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing result does not match the requested analysis: {result_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        cache_key = (spec.identity.protein_family, spec.identity.landscape)
        if cache_key not in landscape_cache:
            landscape_cache[cache_key] = _load_landscape(spec, args.data_root)
        print(f"[{index}/{len(plans)}] Running {spec.analysis_id}", flush=True)
        result = _run_analysis(
            spec,
            artifact,
            landscape_cache[cache_key],
            retain_k=args.retain_k,
            tracked_top_n=args.tracked_top_n,
            tracked_model_top_n=args.tracked_model_top_n,
            top_k=top_k,
            n_samples=args.n_samples,
            n_pairs=args.n_pairs,
            seed=args.seed,
            show_progress=not args.no_progress,
        )
        result.save(result_dir)
        print(f"Saved {result_dir}", flush=True)


if __name__ == "__main__":
    main()
