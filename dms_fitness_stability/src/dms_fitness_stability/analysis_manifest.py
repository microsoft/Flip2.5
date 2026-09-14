"""Curated analysis manifests and exact fit-artifact resolution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .identity import AnalysisIdentity

ArtifactKind = Literal["diagnosed_hmc", "raw_hmc", "sampled_mle"]

IDENTITY_FIELDS = {
    "protein_family",
    "landscape",
    "inference_method",
    "model_spec",
    "fit_id",
}
ANALYSIS_SPEC_FIELDS = IDENTITY_FIELDS | {
    "job_id",
    "artifact_kind",
    "primary_score",
}
OPTIONAL_ANALYSIS_SPEC_FIELDS = {"scores"}


@dataclass(frozen=True)
class AnalysisSpec:
    """One curated analysis and the fit artifact it is allowed to use."""

    identity: AnalysisIdentity
    job_id: int
    artifact_kind: ArtifactKind
    primary_score: str
    scores: tuple[str, ...]

    @property
    def analysis_id(self) -> str:
        return self.identity.analysis_id

    def score_identity(self, score: str) -> AnalysisIdentity:
        """Return the derived analysis identity for one declared score."""
        if score not in self.scores:
            raise ValueError(f"Score {score!r} is not declared for {self.analysis_id}.")
        if len(self.scores) == 1:
            return self.identity
        view_name = score
        if score.startswith("mean_log_") and score.endswith("_enrichment"):
            view_name = score.removeprefix("mean_log_").removesuffix("_enrichment")
        return AnalysisIdentity(
            protein_family=self.identity.protein_family,
            landscape=f"{self.identity.landscape}-{view_name}",
            inference_method=self.identity.inference_method,
            model_spec=self.identity.model_spec,
            fit_id=self.identity.fit_id,
        )

    @classmethod
    def from_dict(cls, payload: object) -> AnalysisSpec:
        if not isinstance(payload, dict):
            raise ValueError("Analysis manifest entries must be JSON objects.")
        missing = ANALYSIS_SPEC_FIELDS - set(payload)
        extra = set(payload) - ANALYSIS_SPEC_FIELDS - OPTIONAL_ANALYSIS_SPEC_FIELDS
        if missing or extra:
            raise ValueError(
                f"Analysis manifest entry has missing={sorted(missing)} "
                f"and extra={sorted(extra)} fields."
            )
        job_id = payload["job_id"]
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            raise ValueError("job_id must be a positive integer.")
        artifact_kind = payload["artifact_kind"]
        if artifact_kind not in {"diagnosed_hmc", "raw_hmc", "sampled_mle"}:
            raise ValueError(f"Unsupported artifact_kind: {artifact_kind!r}.")
        primary_score = payload["primary_score"]
        if not isinstance(primary_score, str) or not primary_score.strip():
            raise ValueError("primary_score must be a non-empty string.")
        score_values = payload.get("scores", [primary_score])
        if (
            not isinstance(score_values, list)
            or not score_values
            or any(
                not isinstance(score, str) or not score.strip()
                for score in score_values
            )
            or len(set(score_values)) != len(score_values)
        ):
            raise ValueError("scores must be a non-empty list of unique strings.")
        if score_values[0] != primary_score:
            raise ValueError("primary_score must be the first declared score.")
        identity = AnalysisIdentity(
            **{field: payload[field] for field in IDENTITY_FIELDS}
        )
        if identity.inference_method in {"hmc", "hmc_count_based"} and artifact_kind not in {
            "diagnosed_hmc",
            "raw_hmc",
        }:
            raise ValueError(
                "HMC analyses require a diagnosed_hmc or raw_hmc artifact."
            )
        if identity.inference_method in {"mle", "mle_predictive"} and (
            artifact_kind != "sampled_mle"
        ):
            raise ValueError("MLE analyses require a sampled_mle artifact.")
        return cls(
            identity=identity,
            job_id=job_id,
            artifact_kind=artifact_kind,
            primary_score=primary_score,
            scores=tuple(score_values),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **{
                field: getattr(self.identity, field)
                for field in (
                    "protein_family",
                    "landscape",
                    "inference_method",
                    "model_spec",
                    "fit_id",
                )
            },
            "job_id": self.job_id,
            "artifact_kind": self.artifact_kind,
            "primary_score": self.primary_score,
        }
        if self.scores != (self.primary_score,):
            payload["scores"] = list(self.scores)
        return payload


@dataclass(frozen=True)
class ResolvedArtifact:
    """A manifest-approved local artifact and its audited fit contract."""

    job_id: int
    path: Path
    kind: ArtifactKind
    status: str
    score_variable: str
    bytes: int


def load_analysis_manifest(path: str | Path) -> tuple[AnalysisSpec, ...]:
    """Load a strict, ordered analysis manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Analysis manifest schema_version must be 2.")
    if set(payload) != {"schema_version", "analyses"}:
        raise ValueError("Analysis manifest must contain only schema_version and analyses.")
    entries = payload["analyses"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Analysis manifest must contain a non-empty analyses list.")
    specs = tuple(AnalysisSpec.from_dict(entry) for entry in entries)
    analysis_ids = [spec.analysis_id for spec in specs]
    if len(set(analysis_ids)) != len(analysis_ids):
        raise ValueError("Analysis manifest contains duplicate analysis IDs.")
    job_ids = [spec.job_id for spec in specs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Analysis manifest contains duplicate job IDs.")
    return specs


class ArtifactResolver:
    """Resolve curated analyses against the authoritative fit audit."""

    def __init__(
        self,
        audit_records: list[dict[str, object]],
        *,
        artifact_roots: Sequence[str | Path],
    ) -> None:
        if not artifact_roots:
            raise ValueError("At least one artifact root is required.")
        self.artifact_roots = tuple(Path(root).resolve() for root in artifact_roots)
        self._by_job_id: dict[int, dict[str, object]] = {}
        for record in audit_records:
            job_id = int(record["job_id"])
            if job_id in self._by_job_id:
                raise ValueError(f"Fit audit contains duplicate job ID {job_id}.")
            self._by_job_id[job_id] = record

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        artifact_roots: Sequence[str | Path],
    ) -> ArtifactResolver:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Fit audit must be a JSON list.")
        return cls(records, artifact_roots=artifact_roots)

    def resolve(self, spec: AnalysisSpec) -> ResolvedArtifact:
        """Resolve one spec, rejecting metadata drift and unusable artifacts."""
        record = self._by_job_id.get(spec.job_id)
        if record is None:
            raise ValueError(f"Job {spec.job_id} is absent from the fit audit.")
        self._validate_identity(spec, record)

        if spec.artifact_kind == "diagnosed_hmc":
            if record.get("hmc_status") != "validated":
                raise ValueError(
                    f"Job {spec.job_id} diagnosed HMC is not validated: "
                    f"{record.get('hmc_status')}."
                )
            artifacts = record.get("diagnosed_hmc", [])
        elif spec.artifact_kind == "raw_hmc":
            if record.get("raw_status") != "usable_for_analysis":
                raise ValueError(
                    f"Job {spec.job_id} raw HMC is not usable for analysis: "
                    f"{record.get('raw_status')}."
                )
            artifacts = record.get("raw_hmc", [])
        else:
            if spec.identity.inference_method not in {"mle", "mle_predictive"}:
                raise ValueError(
                    "sampled_mle artifacts require inference_method="
                    "'mle' or 'mle_predictive'."
                )
            if record.get("mle_status") != "validated":
                raise ValueError(
                    f"Job {spec.job_id} sampled MLE is not validated: "
                    f"{record.get('mle_status')}."
                )
            artifacts = record.get("sampled_mle", [])

        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise ValueError(
                f"Job {spec.job_id} must have exactly one {spec.artifact_kind} artifact."
            )
        artifact = artifacts[0]
        if not isinstance(artifact, dict):
            raise ValueError(f"Job {spec.job_id} has a malformed artifact record.")
        if artifact.get("kind") != spec.artifact_kind or artifact.get("status") != "validated":
            raise ValueError(
                f"Job {spec.job_id} {spec.artifact_kind} validation is not validated."
            )
        path = Path(str(artifact["path"])).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.artifact_roots):
            roots = ", ".join(str(root) for root in self.artifact_roots)
            raise ValueError(
                f"Job {spec.job_id} artifact lies outside approved roots "
                f"({roots}): {path}"
            )
        artifact_bytes = artifact.get("bytes")
        if not isinstance(artifact_bytes, int) or artifact_bytes <= 0:
            raise ValueError(f"Job {spec.job_id} artifact has no positive byte count.")
        if path.stat().st_size != artifact_bytes:
            raise ValueError(
                f"Job {spec.job_id} artifact size changed after audit: "
                f"{path.stat().st_size} != {artifact_bytes}."
            )
        return ResolvedArtifact(
            job_id=spec.job_id,
            path=path,
            kind=spec.artifact_kind,
            status=str(artifact["status"]),
            score_variable=str(record["score_variable"]),
            bytes=artifact_bytes,
        )

    def resolve_all(
        self,
        specs: tuple[AnalysisSpec, ...],
    ) -> tuple[ResolvedArtifact, ...]:
        """Resolve an ordered manifest without silently dropping failures."""
        return tuple(self.resolve(spec) for spec in specs)

    @staticmethod
    def _validate_identity(
        spec: AnalysisSpec,
        record: dict[str, object],
    ) -> None:
        identity = spec.identity
        if identity.inference_method not in {
            "hmc",
            "hmc_count_based",
            "mle",
            "mle_predictive",
        }:
            raise ValueError(
                f"Unsupported inference method: {identity.inference_method!r}."
            )
        if identity.protein_family != str(record["dataset"]):
            raise ValueError(
                f"Job {spec.job_id} dataset does not match protein_family."
            )
        expected_landscape = identity.protein_family
        subset = str(record.get("subset") or "")
        if subset:
            expected_landscape = f"{expected_landscape}-{subset}"
        if identity.landscape != expected_landscape:
            raise ValueError(
                f"Job {spec.job_id} subset does not match landscape: "
                f"{expected_landscape!r}."
            )
        expected_model = str(record["fitness_distribution"])
        growth_curve = str(record.get("growth_curve") or "")
        if growth_curve:
            expected_model = f"{expected_model}-{growth_curve}"
        if identity.model_spec != expected_model:
            raise ValueError(
                f"Job {spec.job_id} model does not match audit: {expected_model!r}."
            )
        expected_fit_id = (
            str(record["hmc_job_name"])
            if identity.inference_method in {"hmc", "hmc_count_based"}
            else str(record["mle_job_name"])
        )
        if identity.fit_id != expected_fit_id:
            raise ValueError(
                f"Job {spec.job_id} fit_id does not match audit: {expected_fit_id!r}."
            )
