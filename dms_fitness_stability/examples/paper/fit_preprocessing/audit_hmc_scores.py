"""Audit persisted direct-HMC scores against their source posterior targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.analysis_manifest import (
    ArtifactResolver,
    load_analysis_manifest,
)
from dms_fitness_stability.hmc_score_extraction import audit_hmc_score_source
from dms_fitness_stability.score_ensembles import ScoreEnsemble

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAPER_ROOT / "analyses_hmc.json",
    )
    parser.add_argument(
        "--fit-audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--score-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/scores/hmc",
    )
    parser.add_argument("--artifact-root", type=Path, action="append", default=None)
    parser.add_argument("--sample-view", default="release_1000")
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-id", nargs="+", default=None)
    parser.add_argument("--sample-chunk-size", type=int, default=256)
    parser.add_argument("--variant-chunk-size", type=int, default=4_096)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/hmc_score_matrix_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_chunk_size < 1 or args.variant_chunk_size < 1:
        raise ValueError("Sample and variant chunk sizes must be positive.")
    specs = load_analysis_manifest(args.manifest)
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
        raise ValueError("No direct-HMC score bundles match the requested filters.")
    if requested_jobs and {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("A requested job is absent from the selected manifest or family.")
    if requested_ids and {spec.analysis_id for spec in selected} != requested_ids:
        raise ValueError("A requested analysis ID is absent from the selected manifest.")

    roots = args.artifact_root or [Path("/data/amlt"), args.data_root]
    resolver = ArtifactResolver.from_path(args.fit_audit, artifact_roots=roots)
    artifacts = resolver.resolve_all(selected)
    records = []
    for index, (spec, artifact) in enumerate(
        zip(selected, artifacts, strict=True),
        start=1,
    ):
        bundle_dir = args.score_root / args.sample_view / spec.analysis_id
        ensemble = ScoreEnsemble.load(bundle_dir)
        try:
            record = audit_hmc_score_source(
                ensemble,
                artifact.path,
                sample_chunk_size=args.sample_chunk_size,
                variant_chunk_size=args.variant_chunk_size,
            )
        finally:
            ensemble.close()
        records.append(
            {
                "job_id": spec.job_id,
                "protein_family": spec.identity.protein_family,
                "landscape": spec.identity.landscape,
                "artifact_kind": artifact.kind,
                "artifact_validation_status": artifact.status,
                "bundle_path": str(bundle_dir.resolve()),
                "bundle_bytes": sum(
                    path.stat().st_size for path in bundle_dir.iterdir() if path.is_file()
                ),
                **record,
            }
        )
        print(
            f"[{index}/{len(selected)}] exact: {spec.analysis_id}: "
            f"{int(record['finite_values']):,} finite, "
            f"{int(record['nan_values']):,} NaN",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "sample_view": args.sample_view,
        "manifest_analysis_count": len(specs),
        "audited_analysis_count": len(records),
        "complete_manifest": len(records) == len(specs),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Saved {len(records)} exact score audit(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
