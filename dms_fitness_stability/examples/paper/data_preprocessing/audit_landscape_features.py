"""Audit materialized landscape features against observed source files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from preprocess_landscape_features import _load_landscape, _plans

from dms_fitness_stability.landscape_features import (
    LandscapeFeatureBundle,
    audit_landscape_feature_source,
)

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
        action="append",
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/landscape_features",
    )
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc", "k50"), nargs="+")
    parser.add_argument("--feature-id", nargs="+", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/landscape_feature_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = tuple(
        args.manifest
        or (
            PAPER_ROOT / "analyses_hmc.json",
            PAPER_ROOT / "analyses_mle.json",
            PAPER_ROOT / "analyses_hmc_count_based.json",
        )
    )
    plans = _plans(manifests)
    families = set(args.family or ())
    requested_ids = set(args.feature_id or ())
    selected = tuple(
        plan
        for plan in plans
        if (not families or plan.identity.protein_family in families)
        and (not requested_ids or plan.feature_id in requested_ids)
    )
    if not selected:
        raise ValueError("No feature bundles match the requested audit filters.")
    if requested_ids and {plan.feature_id for plan in selected} != requested_ids:
        raise ValueError("A requested feature ID is absent from the selected manifests.")

    landscape_cache = {}
    records = []
    for index, plan in enumerate(selected, start=1):
        cache_key = (plan.identity.protein_family, plan.identity.landscape)
        if cache_key not in landscape_cache:
            landscape_cache[cache_key] = _load_landscape(plan, args.data_root)
        bundle = LandscapeFeatureBundle.load(args.feature_root / plan.feature_id)
        try:
            record = audit_landscape_feature_source(
                bundle,
                plan.identity,
                landscape_cache[cache_key],
                score_name=plan.score_name,
            )
        finally:
            bundle.close()
        records.append(record)
        print(
            f"[{index:02d}/{len(selected):02d}] exact: {plan.feature_id}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "manifest_feature_count": len(plans),
        "audited_feature_count": len(records),
        "complete_manifest": len(records) == len(plans),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Saved {len(records)} feature audits to {args.output}", flush=True)


if __name__ == "__main__":
    main()
