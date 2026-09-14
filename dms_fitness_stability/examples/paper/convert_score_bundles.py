"""Convert audited legacy score bundles to schema 2 without changing scores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dms_fitness_stability.migration_audit import sha256_file
from dms_fitness_stability.score_migration import convert_legacy_score_bundle

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _defaults(method: str, data_root: Path) -> tuple[Path, Path, Path]:
    if method == "mle_predictive":
        return (
            REPOSITORY_ROOT / "data/mle_pre_conversion_baseline.json",
            data_root / "paper/mle_predictive/proxies",
            REPOSITORY_ROOT / "data/mle_schema_v2_conversion.json",
        )
    return (
        REPOSITORY_ROOT / "data/hmc_count_pre_conversion_baseline.json",
        data_root / "paper/hmc_count_based/proxies",
        REPOSITORY_ROOT / "data/hmc_count_schema_v2_conversion.json",
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("mle_predictive", "hmc_count_based"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--score-root", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--destination-root", type=Path)
    destination.add_argument("--in-place", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--analysis-id", action="append", default=None)
    parser.add_argument("--sample-chunk-size", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_chunk_size < 1:
        raise ValueError("sample-chunk-size must be positive.")
    default_baseline, default_root, default_report = _defaults(
        args.method,
        args.data_root,
    )
    baseline_path = (args.baseline or default_baseline).resolve()
    source_root = (args.score_root or default_root).resolve()
    destination_root = (
        source_root
        if args.in_place
        else args.destination_root.resolve()
        if args.destination_root is not None
        else None
    )
    if destination_root == source_root and not args.in_place:
        raise ValueError("Use --in-place to modify the source root explicitly.")
    report_path = (args.report or default_report).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_kind = f"{args.method}_schema_v1_pre_conversion"
    if baseline.get("baseline_kind") != expected_kind:
        raise ValueError(
            f"Expected baseline kind {expected_kind!r}, observed "
            f"{baseline.get('baseline_kind')!r}."
        )

    analyses = baseline.get("analyses")
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("Conversion baseline has no analyses.")
    expected = {}
    for analysis in analyses:
        score_bundle = analysis.get("score_bundle", {})
        analysis_id = score_bundle.get("analysis_id")
        semantic_sha256 = score_bundle.get("semantic_sha256")
        if (
            not isinstance(analysis_id, str)
            or not analysis_id
            or not isinstance(semantic_sha256, str)
            or len(semantic_sha256) != 64
            or score_bundle.get("source_schema_version") != 1
            or analysis_id in expected
        ):
            raise ValueError("Conversion baseline contains an invalid score record.")
        expected[analysis_id] = semantic_sha256

    selected_ids = set(args.analysis_id or expected)
    unknown_ids = selected_ids - set(expected)
    if unknown_ids:
        raise ValueError(f"Analysis IDs are absent from the baseline: {sorted(unknown_ids)}")
    ordered_ids = [analysis_id for analysis_id in expected if analysis_id in selected_ids]
    observed_ids = {
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    required_ids = set(expected) if args.analysis_id is None else selected_ids
    if args.analysis_id is None and observed_ids != required_ids:
        raise ValueError(
            f"Score-root inventory differs from the baseline: "
            f"missing={sorted(required_ids - observed_ids)}, "
            f"extra={sorted(observed_ids - required_ids)}."
        )
    missing_ids = required_ids - observed_ids
    if missing_ids:
        raise ValueError(f"Selected score bundles are missing: {sorted(missing_ids)}")

    schema_counts = {1: 0, 2: 0}
    legacy_bytes = []
    for analysis_id in ordered_ids:
        bundle_dir = source_root / analysis_id
        metadata = json.loads(
            (bundle_dir / "metadata.json").read_text(encoding="utf-8")
        )
        schema_version = metadata.get("schema_version")
        if schema_version not in schema_counts:
            raise ValueError(f"Unsupported schema at {bundle_dir}: {schema_version}")
        schema_counts[schema_version] += 1
        legacy_path = bundle_dir / "fitness_proxies.nc"
        if legacy_path.is_file():
            legacy_bytes.append(legacy_path.stat().st_size)

    capacity_root = destination_root.parent if destination_root else source_root
    free = shutil.disk_usage(capacity_root).free
    largest = max(legacy_bytes, default=0)
    staged_bytes = sum(legacy_bytes) if destination_root != source_root else 0
    required_free = staged_bytes + largest + 64 * 1024**2
    print(
        f"{args.method}: {len(ordered_ids)} bundles "
        f"({schema_counts[1]} schema 1, {schema_counts[2]} schema 2); "
        f"staged matrices {staged_bytes:,} bytes; "
        f"largest temporary copy {largest:,} bytes; free {free:,} bytes.",
        flush=True,
    )
    if not args.execute:
        print(
            "Dry run only. Add --destination-root and --execute to stage, or "
            "--in-place and --execute to modify the source root.",
            flush=True,
        )
        return
    if destination_root is None:
        raise ValueError("Execution requires --destination-root or --in-place.")
    if free < required_free:
        raise OSError(
            f"Insufficient conversion capacity: need {required_free:,} bytes, "
            f"have {free:,}."
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_ids = {
        path.name
        for path in destination_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    unexpected_destination_ids = destination_ids - set(expected)
    if unexpected_destination_ids:
        raise ValueError(
            "Destination contains bundles absent from the baseline: "
            f"{sorted(unexpected_destination_ids)}"
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "conversion_kind": "score_ensemble_schema_v1_to_v2",
        "status": "running",
        "inference_method": args.method,
        "baseline": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "baseline_kind": baseline["baseline_kind"],
        "baseline_repository_commit": baseline.get("repository_commit"),
        "converter_repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "in_place": destination_root == source_root,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "expected_bundles": len(ordered_ids),
        "conversions": [],
    }
    _write_json(report_path, report)
    conversions = report["conversions"]
    assert isinstance(conversions, list)
    for index, analysis_id in enumerate(ordered_ids, start=1):
        destination_bundle = destination_root / analysis_id
        if destination_root != source_root and not destination_bundle.exists():
            copy_stage = destination_root / f".{analysis_id}.copy.tmp"
            if copy_stage.exists():
                if not copy_stage.is_dir():
                    raise ValueError(
                        f"Bundle copy staging path is not a directory: {copy_stage}"
                    )
                shutil.rmtree(copy_stage)
            shutil.copytree(source_root / analysis_id, copy_stage)
            copy_stage.replace(destination_bundle)
        record = convert_legacy_score_bundle(
            destination_bundle,
            expected_semantic_sha256=expected[analysis_id],
            sample_chunk_size=args.sample_chunk_size,
        )
        conversions.append(record)
        _write_json(report_path, report)
        print(
            f"[{index}/{len(ordered_ids)}] {record['status']}: {analysis_id}",
            flush=True,
        )
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(report_path, report)
    print(f"Saved conversion audit to {report_path}", flush=True)


if __name__ == "__main__":
    main()
