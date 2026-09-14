"""Audit selected local fit artifacts against source-grounded contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .fit_contracts import FitContractResolver
from .fit_download import parse_human_size
from .fit_inventory import ArtifactInventory, SelectedJob
from .fit_validation import (
    ArtifactValidation,
    FitExpectation,
    ValidationStatus,
    validate_hmc_artifact,
    validate_mle_artifact,
)


@dataclass(frozen=True)
class JobAudit:
    """Validation results and fallback decision for one selected job."""

    job: SelectedJob
    score_variable: str
    variant_count: int
    diagnosed_hmc: tuple[ArtifactValidation, ...]
    sampled_mle: tuple[ArtifactValidation, ...]
    raw_hmc: tuple[ArtifactValidation, ...]
    inventory_issues: tuple[str, ...]
    retired_diagnosed_hmc: tuple[ArtifactValidation, ...] = ()
    retired_raw_hmc: tuple[ArtifactValidation, ...] = ()

    @property
    def hmc_status(self) -> ValidationStatus | LiteralAmbiguous:
        status = _aggregate_status(self.diagnosed_hmc, required=self.job.require_hmc)
        if status == "missing" and self.retired_diagnosed_hmc:
            return "removed_after_failed_validation"
        return status

    @property
    def mle_status(self) -> ValidationStatus | LiteralAmbiguous:
        return _aggregate_status(self.sampled_mle, required=self.job.require_mle)

    @property
    def raw_status(self) -> str:
        if not self.raw_hmc:
            if self.retired_raw_hmc:
                return "removed_after_failed_validation"
            return "not_checked"
        statuses = {item.status for item in self.raw_hmc}
        if statuses == {"validated"}:
            return "usable_for_analysis"
        if "unusable" in statuses:
            return "unusable"
        return "usable_with_qualification"

    @property
    def fallback_required(self) -> bool:
        diagnosed_failed = self.hmc_status in {
            "limited",
            "unusable",
            "removed_after_failed_validation",
        }
        return (
            self.job.require_hmc
            and diagnosed_failed
            and self.raw_status != "usable_for_analysis"
            and not self.retired_raw_hmc
        )

    @property
    def preferred_hmc_download_required(self) -> bool:
        return (
            self.job.require_hmc
            and self.hmc_status == "missing"
            and not self.retired_diagnosed_hmc
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record."""
        return {
            **asdict(self.job),
            "job_name": self.job.job_name,
            "hmc_experiment": self.job.hmc_experiment,
            "hmc_job_name": self.job.hmc_job_name,
            "mle_experiment": self.job.mle_experiment,
            "mle_job_name": self.job.mle_job_name,
            "score_variable": self.score_variable,
            "variant_count": self.variant_count,
            "hmc_status": self.hmc_status,
            "mle_status": self.mle_status,
            "raw_status": self.raw_status,
            "fallback_required": self.fallback_required,
            "preferred_hmc_download_required": self.preferred_hmc_download_required,
            "inventory_issues": list(self.inventory_issues),
            "diagnosed_hmc": [item.to_dict() for item in self.diagnosed_hmc],
            "retired_diagnosed_hmc": [
                item.to_dict() for item in self.retired_diagnosed_hmc
            ],
            "sampled_mle": [item.to_dict() for item in self.sampled_mle],
            "raw_hmc": [item.to_dict() for item in self.raw_hmc],
            "retired_raw_hmc": [item.to_dict() for item in self.retired_raw_hmc],
        }


LiteralAmbiguous = str


def _aggregate_status(
    validations: tuple[ArtifactValidation, ...],
    *,
    required: bool,
) -> ValidationStatus | LiteralAmbiguous:
    if not required:
        return "not_required"
    if not validations:
        return "missing"
    if len(validations) > 1:
        return "ambiguous"
    return validations[0].status


def audit_job(
    inventory: ArtifactInventory,
    *,
    hmc_expectation: FitExpectation | None,
    mle_expectation: FitExpectation,
    retired_diagnosed_hmc: tuple[ArtifactValidation, ...] = (),
    retired_raw_hmc: tuple[ArtifactValidation, ...] = (),
) -> JobAudit:
    """Validate every local artifact associated with one selected job."""
    job = inventory.job
    inventory_issues = []
    if len(inventory.diagnosed_hmc) > 1:
        inventory_issues.append("multiple_diagnosed_hmc_artifacts")
    if len(inventory.sampled_mle) > 1:
        inventory_issues.append("multiple_sampled_mle_artifacts")
    if len(inventory.raw_hmc) > 1:
        inventory_issues.append("multiple_raw_hmc_artifacts")

    diagnosed_hmc = ()
    if hmc_expectation is not None:
        diagnosed_hmc = tuple(
            validate_hmc_artifact(path, hmc_expectation, diagnosed=True)
            for path in inventory.diagnosed_hmc
        )
    sampled_mle = tuple(
        validate_mle_artifact(path, mle_expectation)
        for path in inventory.sampled_mle
    )
    raw_hmc = ()
    if hmc_expectation is not None:
        raw_hmc = tuple(
            validate_hmc_artifact(path, hmc_expectation, diagnosed=False)
            for path in inventory.raw_hmc
        )
    return JobAudit(
        job=job,
        score_variable=mle_expectation.score_variable,
        variant_count=mle_expectation.variant_count,
        diagnosed_hmc=diagnosed_hmc,
        sampled_mle=sampled_mle,
        raw_hmc=raw_hmc,
        inventory_issues=tuple(inventory_issues),
        retired_diagnosed_hmc=retired_diagnosed_hmc,
        retired_raw_hmc=retired_raw_hmc,
    )


def audit_inventory(
    inventory: tuple[ArtifactInventory, ...],
    *,
    resolver: FitContractResolver,
    retired_diagnosed_hmc: Mapping[
        str, tuple[ArtifactValidation, ...]
    ] | None = None,
    retired_raw_hmc: Mapping[str, tuple[ArtifactValidation, ...]] | None = None,
) -> tuple[JobAudit, ...]:
    """Validate the complete selected-job inventory."""
    audits = []
    for item in inventory:
        hmc_expectation = (
            resolver.expectation(item.job, method="hmc")
            if item.job.require_hmc
            else None
        )
        mle_expectation = resolver.expectation(item.job, method="mle")
        audits.append(
            audit_job(
                item,
                hmc_expectation=hmc_expectation,
                mle_expectation=mle_expectation,
                retired_diagnosed_hmc=(retired_diagnosed_hmc or {}).get(
                    item.job.job_name,
                    (),
                ),
                retired_raw_hmc=(retired_raw_hmc or {}).get(item.job.job_name, ()),
            )
        )
    return tuple(audits)


def retired_diagnosed_from_records(
    records: tuple[dict[str, object], ...],
) -> dict[str, tuple[ArtifactValidation, ...]]:
    """Deserialize validation snapshots for deliberately removed artifacts."""
    retired = {}
    for record in records:
        job_name = str(record["job_name"])
        artifacts = tuple(
            _artifact_validation_from_dict(artifact)
            for artifact in record.get("artifacts", [])
        )
        if not artifacts:
            raise ValueError(f"Retired diagnosed record has no artifacts: {job_name}")
        if any(artifact.kind != "diagnosed_hmc" for artifact in artifacts):
            raise ValueError(f"Retired record contains a non-diagnosed artifact: {job_name}")
        if any(artifact.status not in {"limited", "unusable"} for artifact in artifacts):
            raise ValueError(f"Retired diagnosed artifact was not problematic: {job_name}")
        retired[job_name] = artifacts
    return retired


def retired_raw_from_records(
    records: tuple[dict[str, object], ...],
) -> dict[str, tuple[ArtifactValidation, ...]]:
    """Deserialize validation snapshots for deliberately removed raw artifacts."""
    retired = {}
    for record in records:
        job_name = str(record["job_name"])
        artifacts = tuple(
            _artifact_validation_from_dict(artifact)
            for artifact in record.get("artifacts", [])
        )
        if not artifacts:
            raise ValueError(f"Retired raw record has no artifacts: {job_name}")
        if any(artifact.kind != "raw_hmc" for artifact in artifacts):
            raise ValueError(f"Retired raw record contains another kind: {job_name}")
        if any(artifact.status not in {"limited", "unusable"} for artifact in artifacts):
            raise ValueError(f"Retired raw artifact was not problematic: {job_name}")
        retired[job_name] = artifacts
    return retired


def _artifact_validation_from_dict(payload: object) -> ArtifactValidation:
    if not isinstance(payload, dict):
        raise ValueError("Retired artifact validation must be a JSON object.")
    score_shape = payload.get("score_shape")
    predictive_shapes = payload.get("predictive_shapes", {})
    observed_shapes = payload.get("observed_shapes", {})
    h5py_compatible = payload.get("h5py_compatible")
    h5netcdf_compatible = payload.get("h5netcdf_compatible")
    if not isinstance(predictive_shapes, dict) or not isinstance(observed_shapes, dict):
        raise ValueError("Retired artifact shapes must be JSON objects.")
    return ArtifactValidation(
        path=str(payload["path"]),
        kind=str(payload["kind"]),
        status=str(payload["status"]),
        bytes=int(payload["bytes"]) if payload.get("bytes") is not None else None,
        issues=tuple(str(issue) for issue in payload.get("issues", [])),
        errors=tuple(str(error) for error in payload.get("errors", [])),
        groups=tuple(str(group) for group in payload.get("groups", [])),
        score_variable=str(payload["score_variable"]),
        score_shape=(
            tuple(int(size) for size in score_shape)
            if isinstance(score_shape, list)
            else None
        ),
        predictive_shapes=tuple(
            (str(name), tuple(int(size) for size in shape))
            for name, shape in sorted(predictive_shapes.items())
        ),
        observed_shapes=tuple(
            (str(name), tuple(int(size) for size in shape))
            for name, shape in sorted(observed_shapes.items())
        ),
        h5py_compatible=(
            bool(h5py_compatible) if h5py_compatible is not None else None
        ),
        h5netcdf_compatible=(
            bool(h5netcdf_compatible) if h5netcdf_compatible is not None else None
        ),
    )


def _status_counts(audits: tuple[JobAudit, ...], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for audit in audits:
        value = str(getattr(audit, field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def render_known_issues(
    audits: tuple[JobAudit, ...],
    *,
    blank_rows: tuple[dict[str, str], ...],
    preferred_preflight: tuple[dict[str, object], ...] = (),
    raw_preflight: tuple[dict[str, object], ...] = (),
) -> str:
    """Render the authoritative human-readable artifact audit."""
    hmc_counts = _status_counts(audits, "hmc_status")
    mle_counts = _status_counts(audits, "mle_status")
    fallback_count = sum(audit.fallback_required for audit in audits)
    missing_hmc_count = sum(audit.preferred_hmc_download_required for audit in audits)
    preferred_artifacts = [
        artifact
        for record in preferred_preflight
        for artifact in record.get("artifacts", [])
    ]
    raw_artifacts = [
        artifact
        for record in raw_preflight
        for artifact in record.get("artifacts", [])
    ]
    preferred_bytes = sum(
        parse_human_size(str(artifact["size"])) for artifact in preferred_artifacts
    )
    raw_bytes = sum(parse_human_size(str(artifact["size"])) for artifact in raw_artifacts)
    lines = [
        "# Fit Artifact Inventory and Known Issues",
        "",
        "This file is generated by `examples/fits/audit.py`. It is the central record",
        "for the selected `small-clean` and `large-clean` fit-download batch. The older",
        "PDZ and TrpB issue files remain historical evidence and are not replaced.",
        "",
        "## Summary",
        "",
        f"- Selected jobs: {len(audits)} (54 exact `Perfect!` rows plus MLE-only jobs 63-69).",
        f"- Diagnosed HMC status counts: `{hmc_counts}`.",
        f"- Sampled-MLE status counts: `{mle_counts}`.",
        f"- Jobs awaiting a preferred diagnosed-HMC download: {missing_hmc_count}.",
        f"- Jobs requiring raw-HMC fallback review: {fallback_count}.",
        f"- Preferred preflight recorded {len(preferred_artifacts)} artifact(s), about "
        f"{preferred_bytes / 1e9:.1f} GB, before accounting for later downloads.",
        f"- Raw fallback preflight recorded {len(raw_artifacts)} artifact(s), about "
        f"{raw_bytes / 1e9:.1f} GB.",
        "- `limited` means the fitted score remains readable but provenance, diagnostics,",
        "  reader compatibility, or observed-data evidence is incomplete.",
        "- `unusable` means a score or predictive array required by the current contract",
        "  is unreadable, non-finite, missing, or dimensionally inconsistent.",
        "",
        "## Complete Inventory",
        "",
        "| Family ID | Remote HMC | Remote MLE | Dataset | Model | Variants | "
        "Diagnosed HMC | Sampled MLE | Raw HMC | Next HMC action |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for audit in audits:
        model = audit.job.fitness_distribution
        if audit.job.growth_curve:
            model = f"{model}-{audit.job.growth_curve}"
        landscape = audit.job.dataset
        if audit.job.subset:
            landscape = f"{landscape}/{audit.job.subset}"
        lines.append(
            f"| `{audit.job.job_name}` | `{audit.job.hmc_experiment}:{audit.job.hmc_job_name}` | "
            f"`{audit.job.mle_experiment}:{audit.job.mle_job_name}` | {landscape} | {model} | "
            f"{audit.variant_count:,} | {audit.hmc_status} | {audit.mle_status} | "
            f"{audit.raw_status} | {_next_hmc_action(audit)} |"
        )

    affected = [
        audit
        for audit in audits
        if audit.hmc_status not in {"validated", "not_required"}
        or audit.mle_status != "validated"
        or audit.inventory_issues
        or audit.retired_diagnosed_hmc
        or audit.retired_raw_hmc
        or audit.raw_hmc
    ]
    preferred_by_job = {
        str(record["job_name"]): record for record in preferred_preflight
    }
    raw_by_job = {str(record["job_name"]): record for record in raw_preflight}
    lines.extend(["", "## Findings Requiring Follow-up", ""])
    for audit in affected:
        lines.append(f"### `{audit.job.job_name}`")
        lines.append("")
        lines.append(
            f"Contract: `{audit.score_variable}` over {audit.variant_count:,} fitted variants."
        )
        lines.append(
            f"Remote references: HMC `{audit.job.hmc_experiment}:{audit.job.hmc_job_name}`; "
            f"MLE `{audit.job.mle_experiment}:{audit.job.mle_job_name}`."
        )
        if audit.inventory_issues:
            lines.append(f"Inventory issues: `{', '.join(audit.inventory_issues)}`.")
        if (
            audit.job.require_hmc
            and not audit.diagnosed_hmc
            and not audit.retired_diagnosed_hmc
        ):
            lines.append(
                "- Diagnosed HMC: missing locally; download the preflighted "
                "preferred artifact first."
            )
            lines.extend(
                _render_remote_artifacts(
                    preferred_by_job.get(audit.job.job_name),
                    kind="diagnosed_hmc",
                    label="Preferred diagnosed HMC",
                )
            )
        for result in audit.diagnosed_hmc:
            lines.extend(_render_artifact(result))
        for result in audit.retired_diagnosed_hmc:
            lines.extend(_render_retired_diagnosed_artifact(result))
        if audit.job.require_mle and not audit.sampled_mle:
            lines.append("- Sampled MLE: missing.")
            lines.extend(
                _render_remote_artifacts(
                    preferred_by_job.get(audit.job.job_name),
                    kind="sampled_mle",
                    label="Preferred sampled MLE",
                )
            )
        for result in audit.sampled_mle:
            lines.extend(_render_artifact(result))
        for result in audit.raw_hmc:
            lines.extend(_render_artifact(result))
        for result in audit.retired_raw_hmc:
            lines.extend(_render_retired_artifact(result))
        if audit.fallback_required:
            lines.extend(
                _render_remote_artifacts(
                    raw_by_job.get(audit.job.job_name),
                    kind="raw_hmc",
                    label="Raw HMC fallback",
                )
            )
            lines.append(
                "- Disposition: download and validate the preflighted timestamped raw HMC; "
                "do not overwrite the diagnosed artifact."
            )
        lines.append("")

    lines.extend(
        [
            "## Blank Results Awaiting Bruce",
            "",
            "Jobs 63-69 were explicitly approved as MLE-only exceptions. The following blank",
            "rows remain excluded from downloads:",
            "",
            "| ID | Dataset | Subset | Fitness distribution | Growth curve |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in blank_rows:
        lines.append(
            f"| {row['ID']} | {row['Dataset'] or '(blank)'} | {row['Subset']} | "
            f"{row['Fitness Distribution']} | {row['Growth Curve']} |"
        )
    lines.extend(
        [
            "",
            "## Historical Evidence",
            "",
            "- `examples/pdz/KNOWN_ISSUES.md`: repaired/incomplete jobs 72, 76, and 78.",
            "- `examples/trpb/KNOWN_ISSUES.md`: limited jobs 15, 25, 37, and 43.",
            "",
        ]
    )
    return "\n".join(lines)


def _next_hmc_action(audit: JobAudit) -> str:
    if not audit.job.require_hmc:
        return "not required"
    if audit.preferred_hmc_download_required:
        return "download diagnosed"
    if audit.retired_raw_hmc:
        return "no usable HMC"
    if audit.raw_status == "usable_for_analysis" and (
        audit.retired_diagnosed_hmc
        or audit.hmc_status in {"limited", "unusable"}
    ):
        return "use raw HMC"
    if audit.fallback_required:
        return "download raw fallback"
    return "none"


def _render_remote_artifacts(
    record: dict[str, object] | None,
    *,
    kind: str,
    label: str,
) -> list[str]:
    if record is None:
        return [f"- {label} preflight: not recorded."]
    artifacts = [
        artifact for artifact in record.get("artifacts", []) if artifact["kind"] == kind
    ]
    issues = [str(issue) for issue in record.get("issues", []) if kind in str(issue)]
    lines = []
    for artifact in artifacts:
        lines.append(
            f"- {label}: `{artifact['experiment']}:{artifact['job_name']}` / "
            f"`{artifact['name']}` ({artifact['size']})."
        )
    if not artifacts:
        lines.append(f"- {label} preflight: no matching remote artifact recorded.")
    if issues:
        lines.append(f"- {label} preflight issues: `{', '.join(issues)}`.")
    return lines


def _render_artifact(result: ArtifactValidation) -> list[str]:
    lines = [
        f"- {result.kind}: `{result.path}` ({result.status}; "
        f"{result.bytes if result.bytes is not None else 'unknown'} bytes)."
    ]
    if result.issues:
        lines.append(f"  Issues: `{', '.join(result.issues)}`.")
    if result.errors:
        lines.append(f"  Errors: `{' | '.join(result.errors)}`.")
    if result.score_shape is not None:
        lines.append(f"  Score shape: `{result.score_shape}`.")
    if result.predictive_shapes:
        lines.append(f"  Predictive shapes: `{dict(result.predictive_shapes)}`.")
    if result.observed_shapes:
        lines.append(f"  Observed shapes: `{dict(result.observed_shapes)}`.")
    return lines


def _render_retired_diagnosed_artifact(result: ArtifactValidation) -> list[str]:
    return _render_retired_artifact(result)


def _render_retired_artifact(result: ArtifactValidation) -> list[str]:
    lines = [
        f"- {result.kind}: `{result.path}` ({result.status}; "
        f"{result.bytes if result.bytes is not None else 'unknown'} bytes at audit; "
        "removed locally after failed validation)."
    ]
    if result.issues:
        lines.append(f"  Issues: `{', '.join(result.issues)}`.")
    if result.errors:
        lines.append(f"  Errors: `{' | '.join(result.errors)}`.")
    if result.score_shape is not None:
        lines.append(f"  Score shape: `{result.score_shape}`.")
    if result.predictive_shapes:
        lines.append(f"  Predictive shapes: `{dict(result.predictive_shapes)}`.")
    if result.observed_shapes:
        lines.append(f"  Observed shapes: `{dict(result.observed_shapes)}`.")
    return lines
