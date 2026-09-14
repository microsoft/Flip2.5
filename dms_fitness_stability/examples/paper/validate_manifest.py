"""Resolve every curated analysis against its exact audited local artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dms_fitness_stability.analysis_manifest import (
    ArtifactResolver,
    load_analysis_manifest,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "data/fit_audit.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=None,
        help="Approved artifact root; repeat for multiple roots.",
    )
    args = parser.parse_args()

    roots = args.artifact_root or [Path("/data/amlt"), Path("/data/data_dms_stability")]
    specs = load_analysis_manifest(args.manifest)
    resolver = ArtifactResolver.from_path(args.audit, artifact_roots=roots)
    artifacts = resolver.resolve_all(specs)

    family_counts = Counter(spec.identity.protein_family for spec in specs)
    kind_counts = Counter(artifact.kind for artifact in artifacts)
    print(f"Resolved {len(specs)} analyses from {args.manifest}")
    print(f"Families: {dict(sorted(family_counts.items()))}")
    print(f"Artifacts: {dict(sorted(kind_counts.items()))}")
    for spec, artifact in zip(specs, artifacts, strict=True):
        print(
            f"{spec.analysis_id}: job {spec.job_id}, {artifact.kind}, "
            f"{artifact.score_variable}, {artifact.path}"
        )


if __name__ == "__main__":
    main()
