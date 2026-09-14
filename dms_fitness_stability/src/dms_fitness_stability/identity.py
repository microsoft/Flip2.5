"""Stable identity for one landscape and fitted analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _slug(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    slug = _SLUG_SEPARATOR.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{field} must contain an ASCII letter or number.")
    return slug


@dataclass(frozen=True)
class AnalysisIdentity:
    """Identify one landscape, inference procedure, model, and fit artifact."""

    protein_family: str
    landscape: str
    inference_method: str
    model_spec: str
    fit_id: str

    def __post_init__(self) -> None:
        for field in (
            "protein_family",
            "landscape",
            "inference_method",
            "model_spec",
            "fit_id",
        ):
            _slug(getattr(self, field), field=field)

    @property
    def analysis_id(self) -> str:
        """Return a readable identifier suitable for tables and directories."""
        return "__".join(
            _slug(getattr(self, field), field=field)
            for field in (
                "protein_family",
                "landscape",
                "inference_method",
                "model_spec",
                "fit_id",
            )
        )

    def metadata(self) -> dict[str, str]:
        """Return normalized run-level metadata fields."""
        return {
            "analysis_id": self.analysis_id,
            "protein_family": self.protein_family,
            "landscape": self.landscape,
            "inference_method": self.inference_method,
            "model_spec": self.model_spec,
            "fit_id": self.fit_id,
        }
