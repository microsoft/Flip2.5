"""Directly verify count readiness for the sampled-MLE analysis manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess_mle import (
    DEFAULT_DATA_ROOT,
    HERE,
    REPOSITORY_ROOT,
    _load_nuclease_landscape,
    _load_pdz_landscape,
    _load_trpb_landscape,
)

from dms_fitness_stability.analysis_manifest import ArtifactResolver, load_analysis_manifest
from dms_fitness_stability.mle_preflight import validate_sampled_mle_counts


def _counts_for_spec(spec, data_root: Path) -> dict[str, object]:
    family = spec.identity.protein_family
    if family == "pdz":
        source = spec.identity.landscape.removeprefix("pdz-")
        landscape = _load_pdz_landscape(data_root, source)
        return {
            "starting_counts": landscape.starting_counts,
            "timepoint_counts": landscape.timepoint_counts,
        }
    if family == "trpb":
        library = spec.identity.landscape.removeprefix("trpb-")
        landscape = _load_trpb_landscape(data_root, library)
        return {
            "starting_counts": landscape.starting_counts,
            "timepoint_counts": landscape.timepoint_counts,
        }
    if family == "nuc":
        generation = spec.identity.landscape.removeprefix("nuc-").upper()
        landscape = _load_nuclease_landscape(data_root, generation)
        return landscape.counts
    raise ValueError(f"MLE count preflight does not support family {family!r}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_mle.json")
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--sample-chunk-size", type=int, default=32)
    parser.add_argument("--variant-chunk-size", type=int, default=16_384)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/mle_count_preflight.json",
    )
    return parser.parse_args()


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    specs = load_analysis_manifest(args.manifest)
    selected_jobs = set(args.job or (spec.job_id for spec in specs))
    selected = tuple(spec for spec in specs if spec.job_id in selected_jobs)
    if {spec.job_id for spec in selected} != selected_jobs:
        raise ValueError("A requested job is absent from the selected manifest.")

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    artifacts = resolver.resolve_all(selected)
    existing = []
    if args.output.is_file() and not args.refresh:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    records_by_id = {
        str(record["analysis_id"]): record
        for record in existing
        if isinstance(record, dict) and "analysis_id" in record
    }
    for index, (spec, artifact) in enumerate(zip(selected, artifacts, strict=True), start=1):
        cached = records_by_id.get(spec.analysis_id)
        cached_source = cached.get("source_artifact", {}) if cached else {}
        if (
            cached is not None
            and cached.get("status") == "ready"
            and cached_source.get("path") == str(artifact.path)
            and cached_source.get("size") == artifact.bytes
            and cached_source.get("mtime_ns") == artifact.path.stat().st_mtime_ns
        ):
            print(f"[{index}/{len(selected)}] cached: job {spec.job_id}", flush=True)
            continue
        print(
            f"[{index}/{len(selected)}] checking {spec.analysis_id}",
            flush=True,
        )
        record = validate_sampled_mle_counts(
            artifact.path,
            _counts_for_spec(spec, args.data_root),
            sample_chunk_size=args.sample_chunk_size,
            variant_chunk_size=args.variant_chunk_size,
            progress=lambda message: print(f"  {message}", flush=True),
        )
        records_by_id[spec.analysis_id] = {
            "analysis_id": spec.analysis_id,
            "job_id": spec.job_id,
            "protein_family": spec.identity.protein_family,
            "landscape": spec.identity.landscape,
            "model_spec": spec.identity.model_spec,
            "artifact_status": artifact.status,
            **record,
        }
        ordered_records = [
            records_by_id[item.analysis_id]
            for item in specs
            if item.analysis_id in records_by_id
        ]
        _write_records(args.output, ordered_records)
        print(f"  ready: job {spec.job_id}", flush=True)

    final_records = [
        records_by_id[item.analysis_id]
        for item in specs
        if item.analysis_id in records_by_id
    ]
    _write_records(args.output, final_records)
    print(f"Ready: {len(final_records)} sampled-MLE artifacts -> {args.output}")


if __name__ == "__main__":
    main()
