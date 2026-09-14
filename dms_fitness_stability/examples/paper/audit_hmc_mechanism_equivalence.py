"""Compare matrix-only and fit-backed direct-HMC mechanism outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.mechanism_equivalence import (
    compare_mechanism_cache,
    compare_mechanism_table,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
TABLE_FILENAMES = (
    "functionality_scores.csv.gz",
    "mechanism_diagnostics.csv",
    "mechanism_effects.csv",
    "mechanism_inventory.csv",
    "mechanism_runs.csv",
    "mechanism_variants.csv",
    "reference_category_sensitivity.csv",
    "reference_diagnostics.csv",
)
CACHE_ADDITIVE_VARIABLES = (
    "candidate_n_finite_samples",
    "candidate_variant_index",
    "n_finite_samples",
)
TABLE_ADDITIVE_COLUMNS = {
    "functionality_scores.csv.gz": ("n_finite_samples",),
    "mechanism_inventory.csv": ("input_replicate_count",),
    "mechanism_runs.csv": ("n_reported_context_omitted_low_support",),
    "mechanism_variants.csv": ("n_finite_samples",),
    "reference_diagnostics.csv": (
        "n_score_supported_candidates",
        "reference_top_k_identifiable",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--matrix-cache-root",
        type=Path,
        default=(
            DEFAULT_DATA_ROOT
            / "paper/hmc/matrix_staging/mechanisms/cache/release_1000"
        ),
    )
    parser.add_argument(
        "--reference-cache-root",
        type=Path,
        default=(
            DEFAULT_DATA_ROOT
            / "paper/hmc/fit_staging/mechanisms/cache/release_1000"
        ),
    )
    parser.add_argument(
        "--matrix-table-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/matrix_staging/tables/release_1000",
    )
    parser.add_argument(
        "--reference-table-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/fit_staging/tables/release_1000",
    )
    parser.add_argument("--sample-chunk-size", type=int, default=4_096)
    parser.add_argument("--table-atol", type=float, default=5e-12)
    parser.add_argument(
        "--normalize-current-trpb-candidate-annotation",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/hmc_mechanism_equivalence_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = load_analysis_manifest(args.manifest)
    caches = []
    for index, spec in enumerate(specs, start=1):
        filename = f"{spec.analysis_id}.nc"
        record = compare_mechanism_cache(
            args.matrix_cache_root / filename,
            args.reference_cache_root / filename,
            sample_chunk_size=args.sample_chunk_size,
            allowed_observed_variables=CACHE_ADDITIVE_VARIABLES,
        )
        caches.append({"analysis_id": spec.analysis_id, "job_id": spec.job_id, **record})
        print(f"[{index:02d}/{len(specs):02d}] cache exact: {spec.analysis_id}")

    tables = {
        filename: compare_mechanism_table(
            args.matrix_table_dir / filename,
            args.reference_table_dir / filename,
            atol=args.table_atol,
            normalize_current_trpb_candidate_annotation=(
                args.normalize_current_trpb_candidate_annotation
            ),
            allowed_observed_columns=TABLE_ADDITIVE_COLUMNS.get(filename, ()),
        )
        for filename in TABLE_FILENAMES
    }
    payload = {
        "schema_version": 1,
        "matrix_cache_root": str(args.matrix_cache_root.resolve()),
        "reference_cache_root": str(args.reference_cache_root.resolve()),
        "cache_comparison_mode": (
            "same_validated_inventory"
            if args.matrix_cache_root.resolve() == args.reference_cache_root.resolve()
            else "independent_roots"
        ),
        "matrix_table_dir": str(args.matrix_table_dir.resolve()),
        "reference_table_dir": str(args.reference_table_dir.resolve()),
        "manifest_analysis_count": len(specs),
        "audited_cache_count": len(caches),
        "complete_manifest": len(caches) == len(specs),
        "table_atol": args.table_atol,
        "cache_additive_variables": list(CACHE_ADDITIVE_VARIABLES),
        "table_additive_columns": TABLE_ADDITIVE_COLUMNS,
        "caches": caches,
        "tables": tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Saved {len(caches)} cache and {len(tables)} table audits to "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
