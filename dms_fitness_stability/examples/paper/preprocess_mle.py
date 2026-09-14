"""Build persisted count-derived fitness proxies from sampled-MLE artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, cast

from dms_fitness_stability.adapters.k50 import (
    build_k50_mle_score_ensemble,
    load_k50_landscape,
)
from dms_fitness_stability.adapters.nuclease import (
    NucleaseGeneration,
    build_nuclease_mle_score_ensemble,
    load_nuclease_landscape,
)
from dms_fitness_stability.adapters.pdz import (
    PDZGrowthCurve,
    PDZGrowthRatePrior,
    build_pdz_mle_score_ensemble,
    load_pdz_cis,
    load_pdz_cript,
    load_pdz_trans,
)
from dms_fitness_stability.adapters.trpb import (
    GrowthCurve,
    GrowthRatePrior,
    build_trpb_mle_score_ensemble,
    load_trpb_landscape,
)
from dms_fitness_stability.analysis_manifest import (
    AnalysisSpec,
    ArtifactResolver,
    ResolvedArtifact,
    load_analysis_manifest,
)
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _publish(ensemble, output_dir: Path) -> None:
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


def _existing_matches(
    output_dir: Path,
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
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
        and metadata.get("artifact_kind") == spec.artifact_kind
        and metadata.get("scores") == list(spec.scores)
        and isinstance(source, dict)
        and Path(str(source.get("path"))).resolve() == artifact.path
        and source.get("size") == artifact.bytes
        and (
            spec.identity.protein_family == "k50"
            or isinstance(
                metadata.get("finite_ranking_candidates_per_sample"),
                list,
            )
        )
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


def _load_k50_landscape(data_root: Path):
    raw_root = data_root / "k50/raw"
    return load_k50_landscape(
        raw_root,
        reported_fitness_path=(
            raw_root
            / "reported_fitness/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
        ),
    )


def _load_pdz_landscape(data_root: Path, landscape: str):
    if landscape == "cis":
        raw_root = data_root / "pdz_cis/raw"
        return load_pdz_cis(
            raw_root / "cis_from_bruce_azure.tsv",
            reported_fitness_path=raw_root / "GSE265816_cript2xa3_sub.csv.gz",
        )
    if landscape in {"cript-c", "cript-n"}:
        raw_root = data_root / "pdz_cis/raw"
        fitness_name = landscape.replace("-", "_")
        return load_pdz_cript(
            landscape,
            raw_root / f"{landscape}.tsv",
            reported_fitness_path=(
                raw_root / f"GSE265816_{fitness_name}_sub.csv.gz"
            ),
        )
    if landscape in {"trans-1", "trans-2"}:
        raw_root = data_root / "pdz_trans/raw"
        count_paths = {
            source: raw_root / f"{source}.tsv"
            for source in ("trans-1", "trans-2")
        }
        return load_pdz_trans(
            landscape,
            count_paths=count_paths,
            reported_fitness_path=raw_root / "GSE265816_pdz3cript_sub.csv.gz",
        )
    raise ValueError(f"Unknown PDZ landscape {landscape!r}.")


def _load_trpb_landscape(data_root: Path, library: str):
    raw_root = data_root / "trpb/raw"
    return load_trpb_landscape(
        library,
        count_path=raw_root / "counts" / f"{library}.csv",
        reported_fitness_path=raw_root / "reported_fitness" / f"{library}.csv",
    )


def _build(
    spec: AnalysisSpec,
    artifact: ResolvedArtifact,
    *,
    data_root: Path,
    sample_chunk_size: int,
    variant_chunk_size: int | None,
    scratch_dir: Path | None,
):
    if spec.identity.protein_family == "pdz":
        landscape_name = spec.identity.landscape.removeprefix("pdz-")
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        if growth_rate_prior not in {"exponential", "gamma", "lomax"}:
            raise ValueError(f"Unsupported PDZ growth-rate prior {growth_rate_prior!r}.")
        if growth_curve not in {"exponential", "sigmoid"}:
            raise ValueError(f"Unsupported PDZ growth curve {growth_curve!r}.")
        landscape = _load_pdz_landscape(data_root, landscape_name)
        ensemble = build_pdz_mle_score_ensemble(
            landscape,
            artifact.path,
            growth_rate_prior=cast(PDZGrowthRatePrior, growth_rate_prior),
            growth_curve=cast(PDZGrowthCurve, growth_curve),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            variant_chunk_size=variant_chunk_size,
        )
    elif spec.identity.protein_family == "nuc":
        generation_text = spec.identity.landscape.removeprefix("nuc-").upper()
        if generation_text not in {"G1", "G2", "G3", "G4"}:
            raise ValueError(f"Unsupported NucB generation {generation_text!r}.")
        generation = cast(NucleaseGeneration, generation_text)
        fitness_distribution = str(spec.identity.model_spec)
        if fitness_distribution not in {"exponential", "lomax"}:
            raise ValueError(
                f"Unsupported NucB fitness distribution {fitness_distribution!r}."
            )
        landscape = _load_nuclease_landscape(data_root, generation)
        ensemble = build_nuclease_mle_score_ensemble(
            landscape,
            artifact.path,
            fitness_distribution=cast(
                Literal["exponential", "lomax"],
                fitness_distribution,
            ),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            sample_chunk_size=sample_chunk_size,
        )
    elif spec.identity.protein_family == "trpb":
        library = spec.identity.landscape.removeprefix("trpb-")
        growth_rate_prior, growth_curve = spec.identity.model_spec.split("-", 1)
        if growth_rate_prior not in {"exponential", "gamma", "lomax"}:
            raise ValueError(
                f"Unsupported TrpB growth-rate prior {growth_rate_prior!r}."
            )
        if growth_curve not in {"exponential", "sigmoid"}:
            raise ValueError(f"Unsupported TrpB growth curve {growth_curve!r}.")
        landscape = _load_trpb_landscape(data_root, library)
        ensemble = build_trpb_mle_score_ensemble(
            landscape,
            artifact.path,
            growth_rate_prior=cast(GrowthRatePrior, growth_rate_prior),
            growth_curve=cast(GrowthCurve, growth_curve),
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            variant_chunk_size=variant_chunk_size,
        )
    elif spec.identity.protein_family == "k50":
        landscape = _load_k50_landscape(data_root)
        ensemble = build_k50_mle_score_ensemble(
            landscape,
            artifact.path,
            fit_id=spec.identity.fit_id,
            job_id=spec.job_id,
            scratch_dir=scratch_dir,
            variant_chunk_size=variant_chunk_size,
        )
    else:
        raise NotImplementedError(
            f"Job {spec.job_id} proxy preprocessing is not implemented yet."
        )
    if ensemble.metadata["analysis_id"] != spec.analysis_id:
        raise ValueError(
            f"Score builder produced {ensemble.metadata['analysis_id']!r}, "
            f"expected {spec.analysis_id!r}."
        )
    if ensemble.metadata["scores"] != list(spec.scores):
        raise ValueError("Score builder outputs do not match the analysis manifest.")
    ensemble.metadata.update(
        {
            "artifact_validation_status": artifact.status,
            "artifact_status": artifact.status,
            "primary_score": spec.primary_score,
        }
    )
    ensemble.metadata.setdefault("warnings", [])
    return ensemble


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "analyses_mle.json",
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
        default=score_bundle_root(DEFAULT_DATA_ROOT, "mle_predictive"),
        help="Score output root (--output-root is a deprecated alias).",
    )
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--sample-chunk-size", type=int, default=64)
    parser.add_argument("--variant-chunk-size", type=int, default=None)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Temporary storage for disk-backed proxy construction (default: system temp).",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    resolved = resolver.resolve_all(selected)
    plans = []
    for spec, artifact in zip(selected, resolved, strict=True):
        output_dir = args.score_root / spec.analysis_id
        matches = _existing_matches(output_dir, spec, artifact)
        if args.overwrite:
            matches = False
        state = "reuse" if matches else "run"
        print(
            f"{state}: {spec.analysis_id}: job {spec.job_id}, {artifact.path} "
            f"-> {output_dir}"
        )
        plans.append((spec, artifact, output_dir, matches))

    if not args.execute:
        print(
            f"Dry run: {len(plans)} MLE score plan(s); add --execute to run."
        )
        return

    for index, (spec, artifact, output_dir, matches) in enumerate(plans, start=1):
        if matches:
            print(f"[{index}/{len(plans)}] Reusing {output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing output does not match the requested analysis: {output_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        print(f"[{index}/{len(plans)}] Building {spec.analysis_id}", flush=True)
        ensemble = _build(
            spec,
            artifact,
            data_root=args.data_root,
            sample_chunk_size=args.sample_chunk_size,
            variant_chunk_size=args.variant_chunk_size,
            scratch_dir=args.scratch_dir,
        )
        try:
            _publish(ensemble, output_dir)
        finally:
            ensemble.close()
        print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
