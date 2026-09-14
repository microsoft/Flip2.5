"""Physical migration of legacy score bundles without numeric transcoding."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import h5py

from .migration_audit import audit_score_ensemble_bundle
from .score_ensembles import (
    LEGACY_PROXY_SCHEMA_VERSION,
    SCORE_ENSEMBLE_SCHEMA_VERSION,
    ScoreEnsemble,
    score_method_contract,
)

LEGACY_FILES = {"metadata.json", "variants.csv", "fitness_proxies.nc"}
SCHEMA_V2_FILES = {
    "metadata.json",
    "variants.csv",
    "analysis_variants.csv",
    "fitness_scores.nc",
}
_KNOWN_FILES = LEGACY_FILES | SCHEMA_V2_FILES


def _file_names(bundle_dir: Path) -> set[str]:
    return {path.name for path in bundle_dir.iterdir() if path.is_file()}


def _write_bytes_atomically(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _stage_schema_v2(bundle_dir: Path, stage_dir: Path) -> None:
    ensemble = ScoreEnsemble.load(bundle_dir)
    try:
        n_analysis_variants = len(ensemble.analysis_variants)
        metadata = {
            **ensemble.metadata,
            "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
            "fit_method": score_method_contract(ensemble.metadata)["fit_method"],
            "n_analysis_variants": n_analysis_variants,
        }
    finally:
        ensemble.close()

    stage_dir.mkdir()
    shutil.copy2(bundle_dir / "variants.csv", stage_dir / "variants.csv")
    with (bundle_dir / "variants.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source, (stage_dir / "analysis_variants.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination, lineterminator="\n")
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"Variant table is empty: {bundle_dir}") from error
        writer.writerow(["matrix_variant_index", *columns])
        rows = 0
        for rows, values in enumerate(reader, start=1):
            writer.writerow([rows - 1, *values])
        if rows != n_analysis_variants:
            raise ValueError(
                f"Variant row count changed while staging {bundle_dir}: "
                f"expected {n_analysis_variants}, observed {rows}."
            )
    shutil.copy2(
        bundle_dir / "fitness_proxies.nc",
        stage_dir / "fitness_scores.nc",
    )
    with h5py.File(stage_dir / "fitness_scores.nc", "r+") as handle:
        for legacy_name, current_name in (
            ("fitness_proxy", "fitness_score"),
            ("observed_proxy", "observed_score"),
        ):
            if legacy_name not in handle or current_name in handle:
                raise ValueError(
                    f"Cannot rename {legacy_name!r} to {current_name!r} in "
                    f"{bundle_dir}."
                )
            handle.move(legacy_name, current_name)
    (stage_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def convert_legacy_score_bundle(
    path: str | Path,
    *,
    expected_semantic_sha256: str,
    sample_chunk_size: int = 256,
) -> dict[str, object]:
    """Convert one audited schema-v1 bundle to schema v2 in its current path."""
    bundle_dir = Path(path).resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Score bundle directory is missing: {bundle_dir}")
    unknown_files = _file_names(bundle_dir) - _KNOWN_FILES
    if unknown_files:
        raise ValueError(
            f"Score bundle contains unknown files: {sorted(unknown_files)}"
        )

    before = audit_score_ensemble_bundle(
        bundle_dir,
        sample_chunk_size=sample_chunk_size,
    )
    if before["semantic_sha256"] != expected_semantic_sha256:
        raise ValueError(
            f"Score semantics differ from the frozen baseline for "
            f"{before['analysis_id']}: expected {expected_semantic_sha256}, "
            f"observed {before['semantic_sha256']}."
        )

    stage_dir = bundle_dir.with_name(f".{bundle_dir.name}.schema-v2.tmp")
    if before["source_schema_version"] == SCORE_ENSEMBLE_SCHEMA_VERSION:
        legacy_path = bundle_dir / "fitness_proxies.nc"
        legacy_path.unlink(missing_ok=True)
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)
        if _file_names(bundle_dir) != SCHEMA_V2_FILES:
            raise ValueError(f"Schema-v2 bundle inventory is incomplete: {bundle_dir}")
        return {
            "analysis_id": before["analysis_id"],
            "status": "already_converted",
            "semantic_sha256": before["semantic_sha256"],
            "files": {
                name: record
                for name, record in before["files"].items()
                if name in SCHEMA_V2_FILES
            },
        }
    if before["source_schema_version"] != LEGACY_PROXY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported source schema: {before['source_schema_version']}"
        )

    if stage_dir.exists():
        if not stage_dir.is_dir():
            raise ValueError(f"Migration staging path is not a directory: {stage_dir}")
        shutil.rmtree(stage_dir)
    required_free = (bundle_dir / "fitness_proxies.nc").stat().st_size + 64 * 1024**2
    free = shutil.disk_usage(bundle_dir.parent).free
    if free < required_free:
        raise OSError(
            f"Insufficient free space to stage {bundle_dir.name}: "
            f"need {required_free:,} bytes, have {free:,}."
        )

    original_metadata = (bundle_dir / "metadata.json").read_bytes()
    metadata_switched = False
    try:
        _stage_schema_v2(bundle_dir, stage_dir)
        after = audit_score_ensemble_bundle(
            stage_dir,
            sample_chunk_size=sample_chunk_size,
        )
        if (
            after["source_schema_version"] != SCORE_ENSEMBLE_SCHEMA_VERSION
            or after["semantic_sha256"] != expected_semantic_sha256
        ):
            raise ValueError(
                f"Staged schema-v2 semantics changed for {before['analysis_id']}."
            )

        (stage_dir / "analysis_variants.csv").replace(
            bundle_dir / "analysis_variants.csv"
        )
        (stage_dir / "fitness_scores.nc").replace(bundle_dir / "fitness_scores.nc")
        (stage_dir / "metadata.json").replace(bundle_dir / "metadata.json")
        metadata_switched = True

        converted = ScoreEnsemble.load(bundle_dir)
        converted.close()
        (bundle_dir / "fitness_proxies.nc").unlink()
        if _file_names(bundle_dir) != SCHEMA_V2_FILES:
            raise ValueError(f"Converted bundle inventory is incomplete: {bundle_dir}")
    except Exception:
        if metadata_switched and (bundle_dir / "fitness_proxies.nc").is_file():
            _write_bytes_atomically(bundle_dir / "metadata.json", original_metadata)
        raise
    finally:
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)

    return {
        "analysis_id": after["analysis_id"],
        "status": "converted",
        "semantic_sha256": after["semantic_sha256"],
        "files": after["files"],
    }
