"""Validated researcher-facing configuration for matrix stability analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

ANALYSIS_CONFIG_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = {
    "schema_version",
    "score_bundle",
    "score_name",
    "direction",
    "top_k",
    "seed",
}
_OPTIONAL_FIELDS = {
    "external_reference",
    "feature_bundle",
    "features",
    "mechanisms",
    "model_reference_min_finite_samples",
    "n_samples",
    "output",
    "reporting",
    "retain_k",
    "tracked_model_top_n",
}


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _optional_positive_integer(
    value: object,
    *,
    field: str,
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


def _string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates.")
    return list(value)


def _mechanism_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("mechanisms must be a JSON object when provided.")
    allowed = {
        "challenger_probability",
        "decision_k",
        "interval",
        "reference_scale_k",
    }
    extra = set(value) - allowed
    if extra:
        raise ValueError(
            f"mechanisms contains unknown fields: {sorted(extra)}."
        )
    if "decision_k" not in value:
        raise ValueError("mechanisms.decision_k is required.")
    normalized = dict(value)
    for field in ("decision_k", "reference_scale_k"):
        if field in value:
            normalized[field] = _optional_positive_integer(
                value[field],
                field=f"mechanisms.{field}",
            )
    probability = value.get("challenger_probability", 0.01)
    if (
        not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not 0 <= probability <= 1
    ):
        raise ValueError(
            "mechanisms.challenger_probability must be between zero and one."
        )
    normalized["challenger_probability"] = float(probability)
    interval = value.get("interval", [0.05, 0.95])
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in interval
        )
        or not 0 <= interval[0] < 0.5 < interval[1] <= 1
    ):
        raise ValueError(
            "mechanisms.interval must contain two probabilities that straddle 0.5."
        )
    normalized["interval"] = [float(item) for item in interval]
    return normalized


def validate_analysis_config(payload: object) -> dict[str, object]:
    """Validate and normalize one shallow analysis configuration object."""
    if not isinstance(payload, Mapping):
        raise ValueError("Analysis configuration must be a JSON object.")
    missing = _REQUIRED_FIELDS - set(payload)
    extra = set(payload) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing or extra:
        raise ValueError(
            "Analysis configuration fields differ from schema v1: "
            f"missing={sorted(missing)}, unknown={sorted(extra)}."
        )
    if payload["schema_version"] != ANALYSIS_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Analysis configuration schema_version must be "
            f"{ANALYSIS_CONFIG_SCHEMA_VERSION}."
        )

    direction = payload["direction"]
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    top_k = payload["top_k"]
    if (
        not isinstance(top_k, list)
        or not top_k
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in top_k
        )
        or top_k != sorted(set(top_k))
    ):
        raise ValueError("top_k must be a non-empty, increasing list of unique integers.")
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    normalized = dict(payload)
    normalized["score_bundle"] = _nonempty_string(
        payload["score_bundle"],
        field="score_bundle",
    )
    normalized["score_name"] = _nonempty_string(
        payload["score_name"],
        field="score_name",
    )
    normalized["top_k"] = list(top_k)
    for field in ("feature_bundle", "output"):
        if payload.get(field) is not None:
            normalized[field] = _nonempty_string(payload[field], field=field)
    for field in (
        "model_reference_min_finite_samples",
        "n_samples",
        "retain_k",
        "tracked_model_top_n",
    ):
        if field in payload:
            normalized[field] = _optional_positive_integer(
                payload[field],
                field=field,
            )

    reference = payload.get("external_reference")
    if reference is not None:
        if not isinstance(reference, Mapping):
            raise ValueError("external_reference must be null or a JSON object.")
        required_reference = {"value_column", "direction", "source"}
        optional_reference = {
            "identifiable_top_k",
            "rank_column",
            "tracked_top_n",
        }
        missing_reference = required_reference - set(reference)
        extra_reference = set(reference) - required_reference - optional_reference
        if missing_reference or extra_reference:
            raise ValueError(
                "external_reference fields differ from schema v1: "
                f"missing={sorted(missing_reference)}, "
                f"unknown={sorted(extra_reference)}."
            )
        if reference["direction"] not in {"higher", "lower"}:
            raise ValueError(
                "external_reference.direction must be 'higher' or 'lower'."
            )
        resolved_reference = dict(reference)
        for field in ("value_column", "source"):
            resolved_reference[field] = _nonempty_string(
                reference[field],
                field=f"external_reference.{field}",
            )
        if reference.get("rank_column") is not None:
            resolved_reference["rank_column"] = _nonempty_string(
                reference["rank_column"],
                field="external_reference.rank_column",
            )
        identifiable = reference.get("identifiable_top_k", True)
        if not isinstance(identifiable, bool):
            raise ValueError("external_reference.identifiable_top_k must be boolean.")
        resolved_reference["identifiable_top_k"] = identifiable
        resolved_reference["tracked_top_n"] = _optional_positive_integer(
            reference.get("tracked_top_n"),
            field="external_reference.tracked_top_n",
        )
        normalized["external_reference"] = resolved_reference

    features = payload.get("features")
    if features is not None:
        if not isinstance(features, Mapping) or set(features) - {
            "required",
            "permitted_missing",
        }:
            raise ValueError(
                "features may contain only required and permitted_missing lists."
            )
        required = _string_list(features.get("required", []), field="features.required")
        permitted = _string_list(
            features.get("permitted_missing", []),
            field="features.permitted_missing",
        )
        overlap = set(required) & set(permitted)
        if overlap:
            raise ValueError(
                "Feature names cannot be both required and permitted missing: "
                f"{sorted(overlap)}."
            )
        normalized["features"] = {
            "required": required,
            "permitted_missing": permitted,
        }

    mechanisms = payload.get("mechanisms")
    if mechanisms is not None:
        normalized["mechanisms"] = _mechanism_config(mechanisms)
    reporting = payload.get("reporting")
    if reporting is not None and not isinstance(reporting, Mapping):
        raise ValueError("reporting must be a JSON object when provided.")
    if reporting is not None:
        normalized["reporting"] = dict(reporting)
    return normalized


def load_analysis_config(path: str | Path) -> dict[str, object]:
    """Load one analysis configuration from JSON and validate it."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Analysis configuration is not valid JSON at {config_path}: {error.msg}."
        ) from error
    return validate_analysis_config(payload)


def analysis_config_sha256(config: Mapping[str, object]) -> str:
    """Return a deterministic digest of a validated configuration."""
    normalized = validate_analysis_config(config)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
