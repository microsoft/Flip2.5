"""Archive failed HMC validations before removing local artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RetirementAction:
    """One diagnosed artifact approved for replacement by an exact raw result."""

    job_id: int
    job_name: str
    path: Path
    validation: dict[str, object]
    replacement: dict[str, object]


@dataclass(frozen=True)
class RawRetirementAction:
    """One unusable raw artifact verified against its exact remote source."""

    job_id: int
    job_name: str
    path: Path
    validation: dict[str, object]
    source: dict[str, object]


@dataclass(frozen=True)
class ProductFitRetirementAction:
    """One validated fit whose downstream products have immutable baselines."""

    job_id: int
    dataset: str
    path: Path
    size: int
    mtime_ns: int
    sha256: str
    validation: dict[str, object]
    source: dict[str, object]
    recovery_command: tuple[str, ...]
    downstream: dict[str, object]


def build_product_fit_retirement_action(
    audit_records: list[dict[str, object]],
    preflight_records: list[dict[str, object]],
    downstream_baseline: dict[str, object],
    *,
    amlt_root: str | Path,
    amlt_executable: str | Path,
    job_id: int,
    sha256: str,
) -> ProductFitRetirementAction:
    """Validate one exact sampled-MLE fit and its retained product baseline."""
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters.")
    audit_matches = [record for record in audit_records if record.get("job_id") == job_id]
    if len(audit_matches) != 1:
        raise ValueError(f"Expected one fit audit record for job {job_id}.")
    audit = audit_matches[0]
    if audit.get("mle_status") != "validated":
        raise ValueError(f"Sampled-MLE fit for job {job_id} is not validated.")
    validations = audit.get("sampled_mle")
    if not isinstance(validations, list) or len(validations) != 1:
        raise ValueError(f"Expected one sampled-MLE validation for job {job_id}.")
    validation = validations[0]
    if (
        not isinstance(validation, dict)
        or validation.get("kind") != "sampled_mle"
        or validation.get("status") != "validated"
    ):
        raise ValueError(f"Sampled-MLE validation is not usable for job {job_id}.")
    root = Path(amlt_root).resolve()
    path = Path(str(validation["path"])).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Refusing to retire path outside {root}: {path}")
    stat = path.stat()
    if stat.st_size != validation.get("bytes"):
        raise ValueError(f"Sampled-MLE byte size changed for job {job_id}.")

    preflight_matches = [
        record for record in preflight_records if record.get("job_id") == job_id
    ]
    if len(preflight_matches) != 1 or preflight_matches[0].get("issues"):
        raise ValueError(f"Fit source is not safely preflighted for job {job_id}.")
    sources = [
        artifact
        for artifact in preflight_matches[0].get("artifacts", [])
        if artifact.get("kind") == "sampled_mle"
    ]
    if len(sources) != 1:
        raise ValueError(f"Expected one sampled-MLE source for job {job_id}.")
    source = dict(sources[0])
    expected_path = (
        root
        / str(source["experiment"])
        / str(source["job_name"])
        / str(source["name"])
    ).resolve()
    if path != expected_path:
        raise ValueError(
            f"Sampled-MLE path does not match preflight for job {job_id}: {path}"
        )

    analyses = downstream_baseline.get("analyses")
    if not isinstance(analyses, list):
        raise ValueError("Downstream baseline has no analyses list.")
    baseline_matches = [
        record
        for record in analyses
        if isinstance(record, dict)
        and isinstance(record.get("manifest"), dict)
        and record["manifest"].get("job_id") == job_id
    ]
    if len(baseline_matches) != 1:
        raise ValueError(f"Expected one downstream baseline for job {job_id}.")
    baseline = baseline_matches[0]
    baseline_source = baseline.get("source_artifact")
    if not isinstance(baseline_source, dict) or (
        Path(str(baseline_source.get("path"))).resolve() != path
        or baseline_source.get("size") != stat.st_size
        or baseline_source.get("mtime_ns") != stat.st_mtime_ns
    ):
        raise ValueError(f"Downstream baseline source changed for job {job_id}.")
    score_bundle = baseline.get("score_bundle")
    result = baseline.get("result")
    cache_ids = baseline.get("mechanism_cache_ids")
    if (
        not isinstance(score_bundle, dict)
        or not isinstance(score_bundle.get("semantic_sha256"), str)
        or not isinstance(result, dict)
        or not isinstance(result.get("metadata_sha256"), str)
        or not isinstance(cache_ids, list)
        or not cache_ids
    ):
        raise ValueError(f"Downstream baseline is incomplete for job {job_id}.")
    recovery_command = (
        str(Path(amlt_executable).expanduser()),
        "results",
        "download",
        str(source["experiment"]),
        f":{source['job_name']}",
        "--include",
        str(source["name"]),
        "--output",
        str(root),
        "--n-workers",
        "1",
    )
    return ProductFitRetirementAction(
        job_id=job_id,
        dataset=str(audit["dataset"]),
        path=path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256,
        validation=dict(validation),
        source=source,
        recovery_command=recovery_command,
        downstream={
            "baseline_kind": downstream_baseline.get("baseline_kind"),
            "baseline_repository_commit": downstream_baseline.get(
                "repository_commit"
            ),
            "analysis_id": score_bundle["analysis_id"],
            "score_semantic_sha256": score_bundle["semantic_sha256"],
            "result_metadata_sha256": result["metadata_sha256"],
            "mechanism_cache_ids": list(cache_ids),
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_product_fit_retirement(
    action: ProductFitRetirementAction,
    *,
    archive_path: str | Path,
    reason: str,
) -> None:
    """Archive exact recovery evidence without removing the fit."""
    if not reason.strip():
        raise ValueError("Retirement reason must be non-empty.")
    archive = Path(archive_path)
    records = json.loads(archive.read_text(encoding="utf-8")) if archive.is_file() else []
    if any(record.get("job_id") == action.job_id for record in records):
        raise ValueError(f"Product fit job {action.job_id} is already archived.")
    records.append(
        {
            "job_id": action.job_id,
            "dataset": action.dataset,
            "status": "approved_pending_removal",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "artifact": {
                "path": str(action.path),
                "bytes": action.size,
                "mtime_ns": action.mtime_ns,
                "sha256": action.sha256,
                "validation": action.validation,
            },
            "source": action.source,
            "recovery_command": list(action.recovery_command),
            "downstream": action.downstream,
        }
    )
    records.sort(key=lambda record: int(record["job_id"]))
    _write_json(archive, records)


def remove_prepared_product_fit(
    action: ProductFitRetirementAction,
    *,
    archive_path: str | Path,
) -> None:
    """Remove an exact fit only after its prepared evidence is on disk."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise ValueError("Product-fit retirement evidence has not been prepared.")
    records = json.loads(archive.read_text(encoding="utf-8"))
    matches = [record for record in records if record.get("job_id") == action.job_id]
    if len(matches) != 1 or matches[0].get("status") != "approved_pending_removal":
        raise ValueError(f"Product fit job {action.job_id} is not pending removal.")
    record = matches[0]
    artifact = record.get("artifact")
    expected = {
        "path": str(action.path),
        "bytes": action.size,
        "mtime_ns": action.mtime_ns,
        "sha256": action.sha256,
    }
    if not isinstance(artifact, dict) or any(
        artifact.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(f"Archived artifact evidence changed for job {action.job_id}.")
    stat = action.path.stat()
    if stat.st_size != action.size or stat.st_mtime_ns != action.mtime_ns:
        raise ValueError(f"Product fit changed after approval for job {action.job_id}.")

    record["status"] = "removal_started"
    record["removal_started_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(archive, records)
    action.path.unlink()
    if action.path.exists():
        raise OSError(f"Product fit still exists after removal: {action.path}")
    record["status"] = "removed"
    record["removed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(archive, records)


def build_retirement_actions(
    audit_records: list[dict[str, object]],
    raw_preflight_records: list[dict[str, object]],
    *,
    amlt_root: str | Path,
    job_ids: set[int],
) -> tuple[RetirementAction, ...]:
    """Validate and return exact diagnosed files that may be retired."""
    root = Path(amlt_root).resolve()
    raw_by_job = {str(record["job_name"]): record for record in raw_preflight_records}
    actions = []
    found_ids = set()
    for record in audit_records:
        job_id = int(record["job_id"])
        if job_id not in job_ids:
            continue
        found_ids.add(job_id)
        job_name = str(record["job_name"])
        status = str(record["hmc_status"])
        if status not in {"limited", "unusable"}:
            raise ValueError(f"Diagnosed HMC is not problematic for {job_name}: {status}")
        diagnosed = record.get("diagnosed_hmc", [])
        if not isinstance(diagnosed, list) or len(diagnosed) != 1:
            raise ValueError(f"Expected one diagnosed HMC validation for {job_name}")
        validation = diagnosed[0]
        if not isinstance(validation, dict):
            raise ValueError(f"Invalid diagnosed HMC validation for {job_name}")
        path = Path(str(validation["path"])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Refusing to retire path outside {root}: {path}") from error

        raw_record = raw_by_job.get(job_name)
        if raw_record is None or raw_record.get("issues"):
            raise ValueError(f"Raw fallback is not safely preflighted for {job_name}")
        replacements = [
            artifact
            for artifact in raw_record.get("artifacts", [])
            if artifact.get("kind") == "raw_hmc"
        ]
        if len(replacements) != 1:
            raise ValueError(f"Expected one raw fallback for {job_name}")
        actions.append(
            RetirementAction(
                job_id=job_id,
                job_name=job_name,
                path=path,
                validation=validation,
                replacement=dict(replacements[0]),
            )
        )

    unknown_ids = job_ids - found_ids
    if unknown_ids:
        raise ValueError(f"Requested jobs are absent from the audit: {sorted(unknown_ids)}")
    return tuple(sorted(actions, key=lambda action: action.job_id))


def archive_and_remove(
    actions: tuple[RetirementAction, ...],
    *,
    archive_path: str | Path,
) -> None:
    """Atomically archive validation evidence, then remove exact diagnosed files."""
    archive = Path(archive_path)
    existing = []
    if archive.is_file():
        existing = json.loads(archive.read_text(encoding="utf-8"))
    existing_names = {str(record["job_name"]) for record in existing}
    duplicates = existing_names & {action.job_name for action in actions}
    if duplicates:
        raise ValueError(f"Diagnosed HMC already retired for: {sorted(duplicates)}")

    retired_at = datetime.now(timezone.utc).isoformat()
    records = [
        *existing,
        *(
            {
                "job_id": action.job_id,
                "job_name": action.job_name,
                "retired_at": retired_at,
                "reason": "replaced_by_raw_hmc_after_failed_validation",
                "artifacts": [action.validation],
                "replacement": action.replacement,
            }
            for action in actions
        ),
    ]
    records.sort(key=lambda record: int(record["job_id"]))
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(archive)

    for action in actions:
        action.path.unlink()
        if action.path.exists():
            raise OSError(f"Diagnosed HMC still exists after removal: {action.path}")


def build_raw_retirement_actions(
    audit_records: list[dict[str, object]],
    raw_preflight_records: list[dict[str, object]],
    *,
    amlt_root: str | Path,
    job_ids: set[int],
) -> tuple[RawRetirementAction, ...]:
    """Return exact unusable raw files that may be archived and removed."""
    root = Path(amlt_root).resolve()
    raw_by_job = {str(record["job_name"]): record for record in raw_preflight_records}
    actions = []
    found_ids = set()
    for record in audit_records:
        job_id = int(record["job_id"])
        if job_id not in job_ids:
            continue
        found_ids.add(job_id)
        job_name = str(record["job_name"])
        if record.get("raw_status") != "unusable":
            raise ValueError(f"Raw HMC is not unusable for {job_name}")
        validations = record.get("raw_hmc", [])
        if not isinstance(validations, list) or len(validations) != 1:
            raise ValueError(f"Expected one raw HMC validation for {job_name}")
        validation = validations[0]
        if not isinstance(validation, dict):
            raise ValueError(f"Invalid raw HMC validation for {job_name}")
        if validation.get("kind") != "raw_hmc" or validation.get("status") != "unusable":
            raise ValueError(f"Raw HMC validation is not unusable for {job_name}")
        path = Path(str(validation["path"])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Refusing to retire path outside {root}: {path}") from error

        raw_record = raw_by_job.get(job_name)
        if raw_record is None or raw_record.get("issues"):
            raise ValueError(f"Raw source is not safely preflighted for {job_name}")
        sources = [
            artifact
            for artifact in raw_record.get("artifacts", [])
            if artifact.get("kind") == "raw_hmc"
        ]
        if len(sources) != 1:
            raise ValueError(f"Expected one raw source for {job_name}")
        source = dict(sources[0])
        expected_path = (
            root
            / str(source["experiment"])
            / str(source["job_name"])
            / str(source["name"])
        ).resolve()
        if path != expected_path:
            raise ValueError(
                f"Raw HMC path does not match preflight for {job_name}: {path}"
            )
        actions.append(
            RawRetirementAction(
                job_id=job_id,
                job_name=job_name,
                path=path,
                validation=validation,
                source=source,
            )
        )

    unknown_ids = job_ids - found_ids
    if unknown_ids:
        raise ValueError(f"Requested jobs are absent from the audit: {sorted(unknown_ids)}")
    return tuple(sorted(actions, key=lambda action: action.job_id))


def archive_raw_and_remove(
    actions: tuple[RawRetirementAction, ...],
    *,
    archive_path: str | Path,
) -> None:
    """Atomically archive unusable raw validation evidence, then remove files."""
    archive = Path(archive_path)
    existing = []
    if archive.is_file():
        existing = json.loads(archive.read_text(encoding="utf-8"))
    existing_names = {str(record["job_name"]) for record in existing}
    duplicates = existing_names & {action.job_name for action in actions}
    if duplicates:
        raise ValueError(f"Raw HMC already retired for: {sorted(duplicates)}")

    retired_at = datetime.now(timezone.utc).isoformat()
    records = [
        *existing,
        *(
            {
                "job_id": action.job_id,
                "job_name": action.job_name,
                "retired_at": retired_at,
                "reason": "removed_after_failed_raw_validation",
                "artifacts": [action.validation],
                "source": action.source,
            }
            for action in actions
        ),
    ]
    records.sort(key=lambda record: int(record["job_id"]))
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(archive)

    for action in actions:
        action.path.unlink()
        if action.path.exists():
            raise OSError(f"Raw HMC still exists after removal: {action.path}")
