"""Relocate canonical score bundles into one method/view hierarchy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py

from dms_fitness_stability.analysis_data import (
    load_and_validate_analysis_data_manifest,
)
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.migration_audit import audit_dataset, sha256_file
from dms_fitness_stability.score_ensembles import ScoreEnsemble
from dms_fitness_stability.score_paths import score_bundle_root

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = Path("/data/data_dms_stability")
METHOD_CONFIG = {
    "mle_predictive": {
        "manifest": HERE / "analyses_mle.json",
        "old_relative_root": Path("paper/mle_predictive/proxies"),
        "cache_relative_root": Path("paper/mle_predictive/mechanisms/cache"),
        "equivalence_audit": REPOSITORY_ROOT
        / "data/mle_schema_v2_post_conversion_audit.json",
    },
    "hmc_count_based": {
        "manifest": HERE / "analyses_hmc_count_based.json",
        "old_relative_root": Path("paper/hmc_count_based/proxies"),
        "cache_relative_root": Path("paper/hmc_count_based/mechanisms/cache"),
        "equivalence_audit": REPOSITORY_ROOT
        / "data/hmc_count_schema_v2_post_conversion_audit.json",
    },
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tree_record(root: Path) -> dict[str, int | str]:
    entries = [root, *root.rglob("*")]
    files = [path for path in entries if path.is_file()]
    return {
        "path": str(root.resolve()),
        "bundle_count": sum(path.is_dir() for path in root.iterdir()),
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in entries),
    }


def _expected_bundles(manifest: object, method: str) -> set[str]:
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("score_bundles"), list
    ):
        raise ValueError("Analysis-data manifest has no score_bundles list.")
    records = [
        record
        for record in manifest["score_bundles"]
        if isinstance(record, dict)
        and record.get("inference_method") == method
        and record.get("sample_view") == "canonical"
    ]
    expected = {str(record["analysis_id"]) for record in records}
    if len(expected) != len(records) or not expected:
        raise ValueError(f"Invalid {method} score inventory in the manifest.")
    return expected


def _validate_score_root(root: Path, expected_ids: set[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Expected a regular score root: {root}")
    observed_ids = {
        path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if observed_ids != expected_ids:
        raise ValueError(
            f"Score-root inventory differs at {root}: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"extra={sorted(observed_ids - expected_ids)}."
        )
    for analysis_id in sorted(expected_ids):
        ensemble = ScoreEnsemble.load(root / analysis_id)
        try:
            if ensemble.source_schema_version != 2:
                raise ValueError(f"Score bundle is not schema 2: {analysis_id}")
        finally:
            ensemble.close()


def _cache_mapping(manifest_path: Path) -> dict[str, tuple[str, str]]:
    mapping = {}
    for spec in load_analysis_manifest(manifest_path):
        for score_name in spec.scores:
            cache_id = spec.score_identity(score_name).analysis_id
            if cache_id in mapping:
                raise ValueError(f"Duplicate mechanism cache ID: {cache_id}")
            mapping[cache_id] = (spec.analysis_id, score_name)
    return mapping


def _validate_cache_arrays(cache_root: Path, baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = baseline.get("mechanism_caches")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Equivalence audit has no mechanism caches: {baseline_path}")
    observed_files = {path.stem for path in cache_root.glob("*.nc")}
    if observed_files != set(expected):
        raise ValueError(
            f"Mechanism-cache inventory differs at {cache_root}: "
            f"missing={sorted(set(expected) - observed_files)}, "
            f"extra={sorted(observed_files - set(expected))}."
        )
    for cache_id, expected_record in expected.items():
        observed = audit_dataset(cache_root / f"{cache_id}.nc")
        for surface in ("sizes", "coordinates", "variables"):
            if observed[surface] != expected_record[surface]:
                raise ValueError(
                    f"Mechanism cache {surface} changed for {cache_id}."
                )


def _relocate_roots(data_root: Path, methods: dict[str, object]) -> None:
    states = []
    for method, record in methods.items():
        old_root = Path(str(record["old_root"]))
        new_root = Path(str(record["new_root"]))
        states.append((method, old_root.exists(), new_root.exists()))
    invalid = [state for state in states if state[1] == state[2]]
    if invalid:
        raise ValueError(f"Score roots have ambiguous relocation states: {invalid}")
    moved = []
    try:
        for method, old_exists, _ in states:
            if not old_exists:
                continue
            record = methods[method]
            old_root = Path(str(record["old_root"]))
            new_root = Path(str(record["new_root"]))
            new_root.parent.mkdir(parents=True, exist_ok=True)
            old_root.replace(new_root)
            moved.append((old_root, new_root))
    except Exception:
        for old_root, new_root in reversed(moved):
            new_root.replace(old_root)
        raise


def _integer_attribute(handle: h5py.File, name: str, cache_path: Path) -> int:
    value = handle.attrs.get(name, -1)
    if getattr(value, "shape", ()):
        if getattr(value, "size", 0) != 1:
            raise ValueError(
                f"Cache attribute {name} must contain one value: {cache_path}"
            )
        value = value.item()
    return int(value)


def _update_cache(
    cache_path: Path,
    *,
    score_path: Path,
    score_contract: str,
    old_root: Path,
) -> bool:
    stat = score_path.stat()
    with h5py.File(cache_path, "r") as handle:
        current = {
            "kind": str(handle.attrs.get("source_kind", "")),
            "path": str(handle.attrs.get("source_path", "")),
            "size": _integer_attribute(handle, "source_size", cache_path),
            "mtime_ns": _integer_attribute(
                handle, "source_mtime_ns", cache_path
            ),
            "contract": str(handle.attrs.get("score_contract", "")),
        }
    expected_path = str(score_path.resolve())
    expected = {
        "kind": "score_ensemble",
        "path": expected_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "contract": score_contract,
    }
    if current == expected:
        return False
    old_prefix = str(old_root.resolve()) + "/"
    if (
        current["kind"] not in {"mle_proxy", "hmc_count_proxy", "score_ensemble"}
        or not current["path"].startswith(old_prefix)
        or current["contract"] != score_contract
    ):
        raise ValueError(f"Unexpected cache provenance at {cache_path}: {current}")
    temporary = cache_path.with_suffix(cache_path.suffix + ".relocating")
    temporary.unlink(missing_ok=True)
    shutil.copy2(cache_path, temporary)
    try:
        with h5py.File(temporary, "r+") as handle:
            handle.attrs["source_kind"] = "score_ensemble"
            handle.attrs["source_path"] = expected_path
            handle.attrs["source_size"] = stat.st_size
            handle.attrs["source_mtime_ns"] = stat.st_mtime_ns
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _update_caches(data_root: Path, methods: dict[str, object]) -> int:
    updated = 0
    for method, record in methods.items():
        config = METHOD_CONFIG[method]
        cache_root = data_root / config["cache_relative_root"]
        mapping = _cache_mapping(Path(config["manifest"]))
        if {path.stem for path in cache_root.glob("*.nc")} != set(mapping):
            raise ValueError(f"Cache inventory differs for {method}.")
        new_root = Path(str(record["new_root"]))
        old_root = Path(str(record["old_root"]))
        for cache_id, (analysis_id, score_name) in mapping.items():
            metadata = json.loads(
                (new_root / analysis_id / "metadata.json").read_text(encoding="utf-8")
            )
            updated += _update_cache(
                cache_root / f"{cache_id}.nc",
                score_path=new_root / analysis_id / "fitness_scores.nc",
                score_contract=str(metadata["score_contracts"][score_name]),
                old_root=old_root,
            )
    return updated


def _validate_cache_sources(data_root: Path, methods: dict[str, object]) -> None:
    for method, record in methods.items():
        config = METHOD_CONFIG[method]
        cache_root = data_root / config["cache_relative_root"]
        new_root = Path(str(record["new_root"]))
        for cache_id, (analysis_id, score_name) in _cache_mapping(
            Path(config["manifest"])
        ).items():
            score_path = new_root / analysis_id / "fitness_scores.nc"
            score_stat = score_path.stat()
            metadata = json.loads(
                (new_root / analysis_id / "metadata.json").read_text(encoding="utf-8")
            )
            with h5py.File(cache_root / f"{cache_id}.nc", "r") as handle:
                observed = {
                    "kind": str(handle.attrs.get("source_kind", "")),
                    "path": str(handle.attrs.get("source_path", "")),
                    "size": _integer_attribute(
                        handle, "source_size", cache_root / f"{cache_id}.nc"
                    ),
                    "mtime_ns": _integer_attribute(
                        handle,
                        "source_mtime_ns",
                        cache_root / f"{cache_id}.nc",
                    ),
                    "contract": str(handle.attrs.get("score_contract", "")),
                }
            expected = {
                "kind": "score_ensemble",
                "path": str(score_path.resolve()),
                "size": score_stat.st_size,
                "mtime_ns": score_stat.st_mtime_ns,
                "contract": str(metadata["score_contracts"][score_name]),
            }
            if observed != expected:
                raise ValueError(
                    f"Relocated cache provenance differs for {cache_id}: "
                    f"observed={observed}, expected={expected}."
                )


def _prepare(args: argparse.Namespace) -> None:
    if args.record.exists():
        raise ValueError(f"Relocation record already exists: {args.record}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    methods = {}
    for method, config in METHOD_CONFIG.items():
        expected_ids = _expected_bundles(manifest, method)
        old_root = args.data_root / config["old_relative_root"]
        new_root = score_bundle_root(args.data_root, method)
        if new_root.exists():
            raise ValueError(f"Relocation destination already exists: {new_root}")
        _validate_score_root(old_root, expected_ids)
        cache_root = args.data_root / config["cache_relative_root"]
        _validate_cache_arrays(cache_root, Path(config["equivalence_audit"]))
        methods[method] = {
            "old_root": str(old_root.resolve()),
            "new_root": str(new_root.resolve()),
            "expected_analysis_ids_sha256": hashlib.sha256(
                json.dumps(sorted(expected_ids), separators=(",", ":")).encode()
            ).hexdigest(),
            "equivalence_audit": str(
                Path(config["equivalence_audit"]).relative_to(REPOSITORY_ROOT)
            ),
            "equivalence_audit_sha256": sha256_file(config["equivalence_audit"]),
            "cache_root": str(cache_root.resolve()),
            "cache_count": len(_cache_mapping(Path(config["manifest"]))),
            "source_tree": _tree_record(old_root),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write_json(
        args.record,
        {
            "schema_version": 1,
            "status": "approved_pending_relocation",
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": commit,
            "analysis_data_manifest": str(args.manifest.relative_to(REPOSITORY_ROOT)),
            "manifest_sha256_before": sha256_file(args.manifest),
            "methods": methods,
            "condition": "publish and independently verify this record before relocation",
        },
    )
    print(f"Prepared relocation evidence at {args.record}; no data moved.")


def _execute(args: argparse.Namespace) -> None:
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record.get("status") not in {
        "approved_pending_relocation",
        "relocated_pending_cache_provenance",
        "relocated_pending_manifest",
    }:
        raise ValueError(f"Relocation is not pending: {record.get('status')!r}")
    if sha256_file(args.manifest) != record["manifest_sha256_before"]:
        raise ValueError("Analysis-data manifest changed after relocation approval.")
    methods = record["methods"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for method, config in METHOD_CONFIG.items():
        expected_ids = _expected_bundles(manifest, method)
        old_root = Path(str(methods[method]["old_root"]))
        new_root = Path(str(methods[method]["new_root"]))
        active_root = old_root if old_root.exists() else new_root
        _validate_score_root(active_root, expected_ids)
        _validate_cache_arrays(
            args.data_root / config["cache_relative_root"],
            Path(config["equivalence_audit"]),
        )
    _relocate_roots(args.data_root, methods)
    updated = _update_caches(args.data_root, methods)
    _validate_cache_sources(args.data_root, methods)
    for method, config in METHOD_CONFIG.items():
        _validate_score_root(
            Path(str(methods[method]["new_root"])),
            _expected_bundles(manifest, method),
        )
        _validate_cache_arrays(
            args.data_root / config["cache_relative_root"],
            Path(config["equivalence_audit"]),
        )
    _validate_cache_sources(args.data_root, methods)
    record["status"] = "relocated_pending_manifest"
    record["relocated_at"] = datetime.now(timezone.utc).isoformat()
    record["updated_cache_count"] = updated
    _write_json(args.record, record)
    print(f"Relocated score roots and updated {updated} cache records.")


def _complete(args: argparse.Namespace) -> None:
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record.get("status") != "relocated_pending_manifest":
        raise ValueError("Score roots must be relocated before completion.")
    counts = load_and_validate_analysis_data_manifest(
        args.manifest,
        data_root=args.data_root,
        verify_checksums=True,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for method, config in METHOD_CONFIG.items():
        expected_ids = _expected_bundles(manifest, method)
        new_root = Path(str(record["methods"][method]["new_root"]))
        old_root = Path(str(record["methods"][method]["old_root"]))
        if old_root.exists():
            raise ValueError(f"Old score root still exists: {old_root}")
        _validate_score_root(new_root, expected_ids)
        expected_prefix = str(new_root.relative_to(args.data_root)) + "/"
        method_records = [
            item
            for item in manifest["score_bundles"]
            if item["inference_method"] == method
            and item["sample_view"] == "canonical"
        ]
        if not all(
            str(item["relative_path"]).startswith(expected_prefix)
            for item in method_records
        ):
            raise ValueError(f"Manifest paths were not relocated for {method}.")
        _validate_cache_arrays(
            args.data_root / config["cache_relative_root"],
            Path(config["equivalence_audit"]),
        )
    _validate_cache_sources(args.data_root, record["methods"])
    record["status"] = "complete"
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["manifest_sha256_after"] = sha256_file(args.manifest)
    record["validated_score_bundles"] = counts["score_bundles"]
    record["validated_feature_bundles"] = counts["feature_bundles"]
    _write_json(args.record, record)
    print(f"Completed score-root relocation: {counts}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/analysis_data_manifest.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPOSITORY_ROOT / "data/score_root_relocation.json",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.prepare:
        _prepare(args)
    elif args.execute:
        _execute(args)
    else:
        _complete(args)


if __name__ == "__main__":
    main()
