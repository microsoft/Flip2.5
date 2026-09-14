"""Persist direct-HMC posterior score matrices from curated fit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np

from dms_fitness_stability.adapters.nuclease import (
    NucleaseGeneration,
    build_nuclease_hmc_score_ensemble,
    load_nuclease_landscape,
)
from dms_fitness_stability.adapters.pdz import (
    PDZGrowthCurve,
    PDZGrowthRatePrior,
    build_pdz_hmc_score_ensemble,
    load_pdz_cis,
    load_pdz_cript,
    load_pdz_trans,
)
from dms_fitness_stability.adapters.trpb import (
    GrowthCurve,
    GrowthRatePrior,
    build_trpb_hmc_score_ensemble,
    load_trpb_landscape,
)
from dms_fitness_stability.analysis_manifest import (
    AnalysisSpec,
    ArtifactResolver,
    ResolvedArtifact,
    load_analysis_manifest,
)
from dms_fitness_stability.hmc_score_extraction import (
    hmc_score_geometry,
    project_score_bundle_bytes,
)
from dms_fitness_stability.score_ensembles import ScoreEnsemble

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
SAMPLE_VIEWS = ("full", "release_1000")
BYTES_PER_FLOAT32 = np.dtype("float32").itemsize


@dataclass(frozen=True)
class MatrixPlan:
    """One score-bundle build or reuse decision."""

    spec: AnalysisSpec
    artifact: ResolvedArtifact
    sample_view: str
    n_chains: int
    draws_per_chain: int
    n_samples: int
    n_variants: int
    dense_bytes: int
    projected_output_bytes: int
    output_dir: Path
    reuse: bool


def _load_pdz_landscape(data_root: Path, source: str):
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
    if source in {"cript-c", "cript-n"}:
        fitness_name = source.replace("-", "_")
        return load_pdz_cript(
            source,
            raw_root / f"{source}.tsv",
            reported_fitness_path=raw_root / f"GSE265816_{fitness_name}_sub.csv.gz",
        )
    raise ValueError(f"Unknown PDZ landscape {source!r}.")


def _load_trpb_landscape(data_root: Path, library: str):
    raw_root = data_root / "trpb/raw"
    return load_trpb_landscape(
        library,
        count_path=raw_root / "counts" / f"{library}.csv",
        reported_fitness_path=raw_root / "reported_fitness" / f"{library}.csv",
    )


def _load_nuclease_landscape(
    data_root: Path,
    generation: NucleaseGeneration,
):
    raw_root = data_root / "nuc/raw"
    return load_nuclease_landscape(
        generation,
        count_dir=raw_root / "counts",
        reported_activity_path=raw_root / "reported_fitness/landscape.csv",
    )


def _selected_sample_count(
    sample_view: str,
    *,
    source_samples: int,
) -> int:
    if sample_view == "full":
        return source_samples
    if sample_view == "release_1000":
        if source_samples < 1_000:
            raise ValueError(
                "release_1000 requires at least 1,000 source HMC samples."
            )
        return 1_000
    raise ValueError(f"Unknown sample view {sample_view!r}.")


def _matrix_geometry(
    artifact: ResolvedArtifact,
    *,
    sample_view: str,
) -> tuple[int, int, int, int, int]:
    n_chains, draws_per_chain, n_variants = hmc_score_geometry(
        artifact.path,
        posterior_variable=artifact.score_variable,
    )
    n_samples = _selected_sample_count(
        sample_view,
        source_samples=n_chains * draws_per_chain,
    )
    dense_bytes = n_samples * n_variants * BYTES_PER_FLOAT32
    return n_chains, draws_per_chain, n_samples, n_variants, dense_bytes


def _existing_matches(
    output_dir: Path,
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
    *,
    sample_view: str,
    n_samples: int,
    seed: int,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = metadata.get("source_artifact")
    return (
        metadata.get("analysis_id") == spec.analysis_id
        and metadata.get("job_id") == spec.job_id
        and metadata.get("artifact_kind") == artifact.kind
        and metadata.get("artifact_validation_status") == artifact.status
        and metadata.get("primary_score") == spec.primary_score
        and metadata.get("scores") == list(spec.scores)
        and metadata.get("sample_view") == sample_view
        and metadata.get("n_samples") == n_samples
        and metadata.get("sample_selection_seed") == seed
        and isinstance(source, dict)
        and Path(str(source.get("path"))).resolve() == artifact.path
        and source.get("size") == artifact.bytes
        and source.get("mtime_ns") == artifact.path.stat().st_mtime_ns
    )


def _existing_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    while not resolved.exists() and resolved != resolved.parent:
        resolved = resolved.parent
    return resolved


def _free_bytes(path: Path) -> tuple[Path, int]:
    existing = _existing_path(path)
    return existing, shutil.disk_usage(existing).free


def _storage_requirement(
    output_root: Path,
    scratch_dir: Path,
    plans: tuple[MatrixPlan, ...],
) -> tuple[Path, int, int]:
    pending = tuple(plan for plan in plans if not plan.reuse)
    projected_output_bytes = sum(plan.projected_output_bytes for plan in pending)
    largest_scratch_bytes = max(
        (plan.dense_bytes for plan in pending),
        default=0,
    )
    output_path, output_free = _free_bytes(output_root)
    scratch_path, scratch_free = _free_bytes(scratch_dir)
    same_filesystem = os.stat(output_path).st_dev == os.stat(scratch_path).st_dev
    if same_filesystem:
        required = projected_output_bytes + largest_scratch_bytes
        available = output_free
        filesystem = output_path
    else:
        if scratch_free < largest_scratch_bytes:
            raise OSError(
                f"Insufficient scratch space on {scratch_path}: "
                f"{scratch_free / 2**30:.2f} GiB available, "
                f"{largest_scratch_bytes / 2**30:.2f} GiB required."
            )
        required = projected_output_bytes
        available = output_free
        filesystem = output_path
    return filesystem, available, required


def _check_storage(
    output_root: Path,
    scratch_dir: Path,
    plans: tuple[MatrixPlan, ...],
) -> None:
    filesystem, available, required = _storage_requirement(
        output_root,
        scratch_dir,
        plans,
    )
    if available < required:
        raise OSError(
            f"Insufficient score-matrix space on {filesystem}: "
            f"{available / 2**30:.2f} GiB available, "
            f"{required / 2**30:.2f} GiB conservatively required."
        )


def _build(
    plan: MatrixPlan,
    landscape,
    *,
    seed: int,
    sample_chunk_size: int,
    variant_chunk_size: int | None,
    scratch_dir: Path,
    show_progress: bool,
) -> ScoreEnsemble:
    spec = plan.spec
    artifact = plan.artifact
    family = spec.identity.protein_family
    requested_samples = None if plan.sample_view == "full" else plan.n_samples
    common = {
        "fit_id": spec.identity.fit_id,
        "job_id": spec.job_id,
        "artifact_kind": artifact.kind,
        "n_samples": requested_samples,
        "seed": seed,
        "sample_view": plan.sample_view,
        "sample_chunk_size": sample_chunk_size,
        "variant_chunk_size": variant_chunk_size,
        "scratch_dir": scratch_dir,
        "show_progress": show_progress,
    }
    if family == "pdz":
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        ensemble = build_pdz_hmc_score_ensemble(
            landscape,
            artifact.path,
            growth_rate_prior=cast(PDZGrowthRatePrior, growth_rate_prior),
            growth_curve=cast(PDZGrowthCurve, growth_curve),
            **common,
        )
    elif family == "trpb":
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        ensemble = build_trpb_hmc_score_ensemble(
            landscape,
            artifact.path,
            growth_rate_prior=cast(GrowthRatePrior, growth_rate_prior),
            growth_curve=cast(GrowthCurve, growth_curve),
            **common,
        )
    elif family == "nuc":
        distribution = spec.identity.model_spec
        ensemble = build_nuclease_hmc_score_ensemble(
            landscape,
            artifact.path,
            fitness_distribution=cast(
                Literal["exponential", "lomax"],
                distribution,
            ),
            **common,
        )
    else:
        raise ValueError(f"Unsupported direct-HMC family {family!r}.")
    if ensemble.metadata["analysis_id"] != spec.analysis_id:
        ensemble.close()
        raise ValueError(
            f"Score builder produced {ensemble.metadata['analysis_id']!r}, "
            f"expected {spec.analysis_id!r}."
        )
    if ensemble.metadata["scores"] != list(spec.scores):
        ensemble.close()
        raise ValueError("Score builder scores do not match the analysis manifest.")
    ensemble.metadata.update(
        {
            "artifact_validation_status": artifact.status,
            "artifact_status": artifact.status,
        }
    )
    return ensemble


def _publish(ensemble: ScoreEnsemble, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        ensemble.save(staging_dir)
        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            output_dir.replace(backup_dir)
            try:
                staging_dir.replace(output_dir)
            except Exception:
                backup_dir.replace(output_dir)
                raise
            shutil.rmtree(backup_dir)
        else:
            staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _audit_output(output_dir: Path, plan: MatrixPlan) -> None:
    ensemble = ScoreEnsemble.load(output_dir)
    try:
        if ensemble.metadata["analysis_id"] != plan.spec.analysis_id:
            raise ValueError(f"Published score identity mismatch at {output_dir}.")
        if ensemble.fitness_score.shape != (
            1,
            plan.n_samples,
            plan.n_variants,
        ):
            raise ValueError(f"Published score shape mismatch at {output_dir}.")
    finally:
        ensemble.close()
    with h5py.File(output_dir / "fitness_scores.nc", "r") as handle:
        scores = handle["fitness_score"]
        if scores.dtype != np.dtype("float32"):
            raise ValueError(f"Published scores are not float32 at {output_dir}.")
        if scores.chunks is None or scores.compression != "gzip":
            raise ValueError(
                f"Published scores are not chunked and gzip-compressed at {output_dir}."
            )
    temporary = sorted(
        path.name
        for pattern in ("*.tmp", "*.dat")
        for path in output_dir.glob(pattern)
    )
    if temporary:
        raise ValueError(f"Temporary files remain at {output_dir}: {temporary}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAPER_ROOT / "analyses_hmc.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/scores/hmc",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/scratch/hmc",
    )
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-id", nargs="+", default=None)
    parser.add_argument(
        "--sample-view",
        choices=SAMPLE_VIEWS,
        nargs="+",
        default=SAMPLE_VIEWS,
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sample-chunk-size", type=int, default=256)
    parser.add_argument("--variant-chunk-size", type=int, default=None)
    parser.add_argument(
        "--storage-reference-view",
        default="release_1000",
        help="Measured view used to project compressed full-view bytes.",
    )
    parser.add_argument("--storage-safety-factor", type=float, default=1.15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    if args.variant_chunk_size is not None and args.variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive when provided.")
    if not np.isfinite(args.storage_safety_factor) or args.storage_safety_factor < 1:
        raise ValueError("storage-safety-factor must be finite and at least one.")
    specs = load_analysis_manifest(args.manifest)
    if {spec.identity.inference_method for spec in specs} != {"hmc"}:
        raise ValueError("Direct-HMC preprocessing requires an HMC-only manifest.")
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
        raise ValueError("No direct-HMC analyses match the requested filters.")
    if requested_jobs and {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("A requested job is absent from the selected manifest or family.")
    if requested_ids and {spec.analysis_id for spec in selected} != requested_ids:
        raise ValueError("A requested analysis ID is absent from the selected manifest.")

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    artifacts = resolver.resolve_all(selected)
    plans = []
    for spec, artifact in zip(selected, artifacts, strict=True):
        for sample_view in dict.fromkeys(args.sample_view):
            geometry = _matrix_geometry(artifact, sample_view=sample_view)
            n_chains, draws_per_chain, n_samples, n_variants, dense_bytes = geometry
            output_dir = args.output_root / sample_view / spec.analysis_id
            reuse = _existing_matches(
                output_dir,
                spec,
                artifact,
                sample_view=sample_view,
                n_samples=n_samples,
                seed=args.seed,
            )
            if args.overwrite:
                reuse = False
            reference_dir = (
                args.output_root / args.storage_reference_view / spec.analysis_id
            )
            projected_output_bytes = dense_bytes
            if sample_view != args.storage_reference_view and reference_dir.is_dir():
                projected_output_bytes = project_score_bundle_bytes(
                    reference_dir,
                    target_samples=n_samples,
                    safety_factor=args.storage_safety_factor,
                )
            plan = MatrixPlan(
                spec=spec,
                artifact=artifact,
                sample_view=sample_view,
                n_chains=n_chains,
                draws_per_chain=draws_per_chain,
                n_samples=n_samples,
                n_variants=n_variants,
                dense_bytes=dense_bytes,
                projected_output_bytes=projected_output_bytes,
                output_dir=output_dir,
                reuse=reuse,
            )
            plans.append(plan)
            state = "reuse" if reuse else "build"
            print(
                f"{state}: {sample_view}: {spec.analysis_id}: "
                f"{n_chains}x{draws_per_chain} source, "
                f"{n_samples:,}x{n_variants:,} scores "
                f"({dense_bytes / 2**30:.2f} GiB dense, "
                f"{projected_output_bytes / 2**30:.2f} GiB projected) -> "
                f"{output_dir}"
            )
    plans_tuple = tuple(plans)
    filesystem, available, required = _storage_requirement(
        args.output_root,
        args.scratch_dir,
        plans_tuple,
    )
    print(
        f"storage: {filesystem}: {available / 2**30:.2f} GiB available, "
        f"{required / 2**30:.2f} GiB conservatively required",
        flush=True,
    )
    if not args.execute:
        print(
            f"Dry run: {len(plans_tuple)} score-bundle plan(s); "
            "add --execute to build."
        )
        return
    _check_storage(args.output_root, args.scratch_dir, plans_tuple)

    landscape_cache: dict[tuple[str, str], object] = {}
    for index, plan in enumerate(plans_tuple, start=1):
        if plan.reuse:
            print(f"[{index}/{len(plans_tuple)}] Reusing {plan.output_dir}", flush=True)
            _audit_output(plan.output_dir, plan)
            continue
        if plan.output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing score bundle does not match: {plan.output_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        cache_key = (
            plan.spec.identity.protein_family,
            plan.spec.identity.landscape,
        )
        if cache_key not in landscape_cache:
            family, landscape_name = cache_key
            source = landscape_name.removeprefix(f"{family}-")
            if family == "pdz":
                landscape_cache[cache_key] = _load_pdz_landscape(
                    args.data_root,
                    source,
                )
            elif family == "trpb":
                landscape_cache[cache_key] = _load_trpb_landscape(
                    args.data_root,
                    source,
                )
            elif family == "nuc":
                generation = source.upper()
                if generation not in {"G1", "G2", "G3", "G4"}:
                    raise ValueError(f"Unknown NucB generation {generation!r}.")
                landscape_cache[cache_key] = _load_nuclease_landscape(
                    args.data_root,
                    cast(NucleaseGeneration, generation),
                )
            else:
                raise ValueError(f"Unsupported direct-HMC family {family!r}.")
        print(
            f"[{index}/{len(plans_tuple)}] Building "
            f"{plan.sample_view}: {plan.spec.analysis_id}",
            flush=True,
        )
        ensemble = _build(
            plan,
            landscape_cache[cache_key],
            seed=args.seed,
            sample_chunk_size=args.sample_chunk_size,
            variant_chunk_size=args.variant_chunk_size,
            scratch_dir=args.scratch_dir,
            show_progress=not args.no_progress,
        )
        try:
            _publish(ensemble, plan.output_dir)
        finally:
            ensemble.close()
        _audit_output(plan.output_dir, plan)
        print(f"Published {plan.output_dir}", flush=True)


if __name__ == "__main__":
    main()
