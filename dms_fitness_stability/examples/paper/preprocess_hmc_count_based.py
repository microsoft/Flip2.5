"""Persist count-derived proxy matrices from curated HMC artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Literal, cast

import h5py
from preprocess_mle import (
    DEFAULT_DATA_ROOT,
    HERE,
    REPOSITORY_ROOT,
    _load_nuclease_landscape,
    _load_pdz_landscape,
    _load_trpb_landscape,
    _publish,
)

from dms_fitness_stability.adapters.nuclease import (
    NucleaseGeneration,
    build_nuclease_hmc_count_score_ensemble,
)
from dms_fitness_stability.adapters.pdz import (
    PDZGrowthCurve,
    PDZGrowthRatePrior,
    build_pdz_hmc_count_score_ensemble,
)
from dms_fitness_stability.adapters.trpb import (
    GrowthCurve,
    GrowthRatePrior,
    build_trpb_hmc_count_score_ensemble,
)
from dms_fitness_stability.analysis_manifest import (
    AnalysisSpec,
    ArtifactResolver,
    ResolvedArtifact,
    load_analysis_manifest,
)
from dms_fitness_stability.score_paths import score_bundle_root

BYTES_PER_FLOAT32 = 4


def _existing_matches(
    output_dir: Path,
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
    *,
    n_samples: int | None,
    seed: int,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = metadata.get("source_artifact")
    return (
        metadata.get("schema_version") == 2
        and metadata.get("analysis_id") == spec.analysis_id
        and metadata.get("job_id") == spec.job_id
        and metadata.get("artifact_kind") == artifact.kind
        and metadata.get("artifact_validation_status") == artifact.status
        and metadata.get("primary_score") == spec.primary_score
        and metadata.get("scores") == list(spec.scores)
        and metadata.get("sample_selection_seed") == seed
        and (n_samples is None or metadata.get("n_samples") == n_samples)
        and (
            n_samples is not None
            or metadata.get("n_samples") == metadata.get("n_source_samples")
        )
        and isinstance(source, dict)
        and Path(str(source.get("path"))).resolve() == artifact.path
        and source.get("size") == artifact.bytes
        and source.get("mtime_ns") == artifact.path.stat().st_mtime_ns
    )


def _build(
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
    *,
    data_root: Path,
    n_samples: int | None,
    seed: int,
    sample_chunk_size: int,
    variant_chunk_size: int | None,
    scratch_dir: Path,
    show_progress: bool,
):
    family = spec.identity.protein_family
    if family == "pdz":
        landscape_name = spec.identity.landscape.removeprefix("pdz-")
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        if growth_rate_prior not in {"exponential", "gamma", "lomax"}:
            raise ValueError(f"Unsupported PDZ growth-rate prior {growth_rate_prior!r}.")
        if growth_curve not in {"exponential", "sigmoid"}:
            raise ValueError(f"Unsupported PDZ growth curve {growth_curve!r}.")
        ensemble = build_pdz_hmc_count_score_ensemble(
            _load_pdz_landscape(data_root, landscape_name),
            artifact.path,
            growth_rate_prior=cast(PDZGrowthRatePrior, growth_rate_prior),
            growth_curve=cast(PDZGrowthCurve, growth_curve),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            artifact_kind=artifact.kind,
            n_samples=n_samples,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            variant_chunk_size=variant_chunk_size,
            scratch_dir=scratch_dir,
            show_progress=show_progress,
        )
    elif family == "trpb":
        library = spec.identity.landscape.removeprefix("trpb-")
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        if growth_rate_prior not in {"exponential", "gamma", "lomax"}:
            raise ValueError(
                f"Unsupported TrpB growth-rate prior {growth_rate_prior!r}."
            )
        if growth_curve not in {"exponential", "sigmoid"}:
            raise ValueError(f"Unsupported TrpB growth curve {growth_curve!r}.")
        ensemble = build_trpb_hmc_count_score_ensemble(
            _load_trpb_landscape(data_root, library),
            artifact.path,
            growth_rate_prior=cast(GrowthRatePrior, growth_rate_prior),
            growth_curve=cast(GrowthCurve, growth_curve),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            artifact_kind=artifact.kind,
            n_samples=n_samples,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            variant_chunk_size=variant_chunk_size,
            scratch_dir=scratch_dir,
            show_progress=show_progress,
        )
    elif family == "nuc":
        generation_text = spec.identity.landscape.removeprefix("nuc-").upper()
        if generation_text not in {"G1", "G2", "G3", "G4"}:
            raise ValueError(f"Unsupported NucB generation {generation_text!r}.")
        fitness_distribution = spec.identity.model_spec
        if fitness_distribution not in {"exponential", "lomax"}:
            raise ValueError(
                f"Unsupported NucB fitness distribution {fitness_distribution!r}."
            )
        ensemble = build_nuclease_hmc_count_score_ensemble(
            _load_nuclease_landscape(
                data_root,
                cast(NucleaseGeneration, generation_text),
            ),
            artifact.path,
            fitness_distribution=cast(
                Literal["exponential", "lomax"],
                fitness_distribution,
            ),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            artifact_kind=artifact.kind,
            n_samples=n_samples,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            scratch_dir=scratch_dir,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"Unsupported HMC count-based family {family!r}.")

    if ensemble.metadata["analysis_id"] != spec.analysis_id:
        ensemble.close()
        raise ValueError(
            f"Score builder produced {ensemble.metadata['analysis_id']!r}, "
            f"expected {spec.analysis_id!r}."
        )
    if ensemble.metadata["scores"] != list(spec.scores):
        ensemble.close()
        raise ValueError("Score builder outputs do not match the analysis manifest.")
    ensemble.metadata.update(
        {
            "artifact_validation_status": artifact.status,
            "artifact_status": artifact.status,
            "primary_score": spec.primary_score,
        }
    )
    return ensemble


def _matrix_plan(
    artifact: ResolvedArtifact,
    *,
    n_samples: int | None,
) -> tuple[int, int, int]:
    with h5py.File(artifact.path, "r") as handle:
        posterior = handle["posterior"][artifact.score_variable]
        n_chains, draws_per_chain, n_variants = map(int, posterior.shape)
    selected_samples = n_chains * draws_per_chain if n_samples is None else n_samples
    if not 1 <= selected_samples <= n_chains * draws_per_chain:
        raise ValueError(
            f"n_samples must be between 1 and {n_chains * draws_per_chain:,} "
            f"for job {artifact.job_id}."
        )
    return n_chains, draws_per_chain, selected_samples * n_variants * BYTES_PER_FLOAT32


def _existing_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    while not resolved.exists() and resolved != resolved.parent:
        resolved = resolved.parent
    return resolved


def _free_bytes(path: Path) -> tuple[Path, int]:
    existing = _existing_path(path)
    return existing, shutil.disk_usage(existing).free


def _check_storage(
    output_root: Path,
    scratch_dir: Path,
    *,
    projected_output_bytes: int,
    largest_scratch_bytes: int,
) -> None:
    output_path, output_free = _free_bytes(output_root)
    scratch_path, scratch_free = _free_bytes(scratch_dir)
    same_filesystem = os.stat(output_path).st_dev == os.stat(scratch_path).st_dev
    if same_filesystem:
        required = projected_output_bytes + largest_scratch_bytes
        if output_free < required:
            raise OSError(
                f"Insufficient free space on {output_path}: "
                f"{output_free / 2**30:.2f} GiB available, "
                f"{required / 2**30:.2f} GiB required."
            )
    else:
        if output_free < projected_output_bytes:
            raise OSError(
                f"Insufficient score space on {output_path}: "
                f"{output_free / 2**30:.2f} GiB available, "
                f"{projected_output_bytes / 2**30:.2f} GiB required."
            )
        if scratch_free < largest_scratch_bytes:
            raise OSError(
                f"Insufficient scratch space on {scratch_path}: "
                f"{scratch_free / 2**30:.2f} GiB available, "
                f"{largest_scratch_bytes / 2**30:.2f} GiB required."
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "analyses_hmc_count_based.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--score-root",
        "--output-root",
        dest="score_root",
        type=Path,
        default=score_bundle_root(DEFAULT_DATA_ROOT, "hmc_count_based"),
        help="Score output root (--output-root is a deprecated alias).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc_count_based/scratch",
    )
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sample-chunk-size", type=int, default=64)
    parser.add_argument("--variant-chunk-size", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    if args.n_samples is not None and args.n_samples < 2:
        raise ValueError("n_samples must be at least two when provided.")
    specs = load_analysis_manifest(args.manifest)
    if {spec.identity.inference_method for spec in specs} != {"hmc_count_based"}:
        raise ValueError("HMC count preprocessing requires an hmc_count_based manifest.")
    families = set(args.family or ())
    requested_jobs = set(args.job or ())
    selected = tuple(
        spec
        for spec in specs
        if (not requested_jobs or spec.job_id in requested_jobs)
        and (not families or spec.identity.protein_family in families)
    )
    if requested_jobs and {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("A requested job is absent from the selected manifest or family.")

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    artifacts = resolver.resolve_all(selected)
    plans = []
    projected_output_bytes = 0
    largest_scratch_bytes = 0
    for spec, artifact in zip(selected, artifacts, strict=True):
        n_chains, draws_per_chain, dense_bytes = _matrix_plan(
            artifact,
            n_samples=args.n_samples,
        )
        output_dir = args.score_root / spec.analysis_id
        matches = _existing_matches(
            output_dir,
            spec,
            artifact,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if args.overwrite:
            matches = False
        state = "reuse" if matches else "build"
        print(
            f"{state}: {spec.analysis_id}: {n_chains}x{draws_per_chain} source "
            f"draws, {dense_bytes / 2**30:.2f} GiB dense -> {output_dir}",
            flush=True,
        )
        plans.append((spec, artifact, output_dir, matches))
        if not matches:
            projected_output_bytes += dense_bytes
            largest_scratch_bytes = max(largest_scratch_bytes, dense_bytes)

    print(
        f"Planned {len(plans)} HMC count-based scores; new dense matrices total "
        f"{projected_output_bytes:,} bytes ({projected_output_bytes / 2**30:.2f} GiB), "
        f"largest scratch matrix {largest_scratch_bytes / 2**30:.2f} GiB.",
        flush=True,
    )
    output_path, output_free = _free_bytes(args.score_root)
    scratch_path, scratch_free = _free_bytes(args.scratch_dir)
    print(
        f"Output filesystem {output_path}: {output_free / 2**30:.2f} GiB free; "
        f"scratch filesystem {scratch_path}: {scratch_free / 2**30:.2f} GiB free.",
        flush=True,
    )
    if not args.execute:
        print("Dry run; add --execute to build matrices.")
        return
    if projected_output_bytes:
        _check_storage(
            args.score_root,
            args.scratch_dir,
            projected_output_bytes=projected_output_bytes,
            largest_scratch_bytes=largest_scratch_bytes,
        )

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for index, (spec, artifact, output_dir, matches) in enumerate(plans, start=1):
        if matches:
            print(f"[{index}/{len(plans)}] Reusing {output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing proxy does not match the requested analysis: {output_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        print(f"[{index}/{len(plans)}] Building {spec.analysis_id}", flush=True)
        ensemble = _build(
            spec,
            artifact,
            data_root=args.data_root,
            n_samples=args.n_samples,
            seed=args.seed,
            sample_chunk_size=args.sample_chunk_size,
            variant_chunk_size=args.variant_chunk_size,
            scratch_dir=args.scratch_dir,
            show_progress=not args.no_progress,
        )
        try:
            _publish(ensemble, output_dir)
        finally:
            ensemble.close()
        print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
