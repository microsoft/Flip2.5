"""Compare staged matrix-driven HMC results with fit-driven references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.result_equivalence import compare_stability_result_bundles

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/matrix_staging/results/release_1000",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/fit_staging/results/release_1000",
    )
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc"), nargs="+")
    parser.add_argument("--job", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-id", nargs="+", default=None)
    parser.add_argument("--table-atol", type=float, default=5e-12)
    parser.add_argument("--sample-chunk-size", type=int, default=4_096)
    parser.add_argument(
        "--normalize-current-trpb-candidate-annotation",
        action="store_true",
        help=(
            "Ignore is_ranking_candidate added to current TrpB fit output after "
            "the frozen baseline was generated."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/hmc_matrix_equivalence_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
        raise ValueError("No HMC analyses match the requested equivalence filters.")
    if requested_jobs and {spec.job_id for spec in selected} != requested_jobs:
        raise ValueError("A requested job is absent from the selected manifest or family.")
    if requested_ids and {spec.analysis_id for spec in selected} != requested_ids:
        raise ValueError("A requested analysis ID is absent from the selected manifest.")

    records = []
    for index, spec in enumerate(selected, start=1):
        ignored_columns = ()
        if (
            args.normalize_current_trpb_candidate_annotation
            and spec.identity.protein_family == "trpb"
        ):
            ignored_columns = ("is_ranking_candidate",)
        record = compare_stability_result_bundles(
            args.matrix_root / spec.analysis_id,
            args.reference_root / spec.analysis_id,
            table_atol=args.table_atol,
            ignored_columns=ignored_columns,
            sample_chunk_size=args.sample_chunk_size,
        )
        records.append({"job_id": spec.job_id, **record})
        print(
            f"[{index}/{len(selected)}] equivalent: {spec.analysis_id}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "manifest_analysis_count": len(specs),
        "audited_analysis_count": len(records),
        "complete_manifest": len(records) == len(specs),
        "matrix_root": str(args.matrix_root.resolve()),
        "reference_root": str(args.reference_root.resolve()),
        "table_atol": args.table_atol,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Saved {len(records)} equivalence audit(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
