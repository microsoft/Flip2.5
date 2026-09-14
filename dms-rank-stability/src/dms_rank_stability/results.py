"""Persistence for compact rank-stability results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import _read_variants, _validate_coordinate, _validate_variants

METADATA_FILE = "metadata.json"
STABILITY_FILE = "stability.nc"
VARIANT_FILE = "variants.csv"
IDENTITY_FIELDS = ("run_id", "family", "landscape", "sampling", "model", "score")

_VARIABLES = {
    "selection_variant_index": (("sample", "selection_rank"), np.dtype("int32")),
    "selection_score": (("sample", "selection_rank"), np.dtype("float32")),
    "reference_score": (("variant",), np.dtype("float32")),
    "reference_n_finite_samples": (("variant",), np.dtype("int32")),
    "reference_variant_index": (("reference_rank",), np.dtype("int32")),
    "top_k_count": (("variant",), np.dtype("int32")),
    "top_k_probability": (("variant",), np.dtype("float64")),
    "cohort_score": (("sample", "cohort"), np.dtype("float32")),
    "cohort_rank": (("sample", "cohort"), np.dtype("float32")),
    "is_external_reference": (("cohort",), np.dtype("bool")),
    "is_model_reference": (("cohort",), np.dtype("bool")),
    "is_challenger": (("cohort",), np.dtype("bool")),
}


def _validate_rank_coordinate(dataset: xr.Dataset, name: str, size: int) -> None:
    if name not in dataset.coords:
        raise ValueError(f"Stability data are missing the {name!r} coordinate.")
    if dataset[name].dims != (name,) or not np.array_equal(
        dataset[name].values,
        np.arange(1, size + 1),
    ):
        raise ValueError(f"{name} coordinates must be contiguous from one.")


@dataclass
class RankResult:
    """Selections and exact rank trajectories from one sampled score matrix."""

    metadata: dict[str, object]
    variants: pd.DataFrame
    stability: xr.Dataset

    def validate(self) -> None:
        missing_identity = [name for name in IDENTITY_FIELDS if name not in self.metadata]
        if missing_identity:
            raise ValueError(f"Result metadata are missing fields: {missing_identity}")
        for name in IDENTITY_FIELDS:
            value = self.metadata[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Result metadata field {name!r} must be a non-empty string.")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Result metadata must contain finite JSON values.") from error

        if set(self.stability.data_vars) != set(_VARIABLES):
            raise ValueError("Stability data do not contain the required variable set.")
        for name, (dimensions, dtype) in _VARIABLES.items():
            values = self.stability[name]
            if values.dims != dimensions or values.dtype != dtype:
                raise ValueError(f"{name} must be a {dtype} array over {dimensions}.")

        n_samples = self.stability.sizes.get("sample", 0)
        n_variants = self.stability.sizes.get("variant", 0)
        retain_k = self.stability.sizes.get("selection_rank", 0)
        reference_k = self.stability.sizes.get("reference_rank", 0)
        cohort_size = self.stability.sizes.get("cohort", 0)
        if min(n_samples, n_variants, retain_k, reference_k, cohort_size) < 1:
            raise ValueError("All stability dimensions must be non-empty.")
        if retain_k != reference_k:
            raise ValueError("Selection and reference ranks must retain the same count.")

        _validate_coordinate(self.stability, "sample", n_samples)
        _validate_coordinate(self.stability, "variant", n_variants)
        _validate_coordinate(self.stability, "cohort", cohort_size)
        _validate_rank_coordinate(self.stability, "selection_rank", retain_k)
        _validate_rank_coordinate(self.stability, "reference_rank", reference_k)
        _validate_variants(self.variants, expected_rows=n_variants)

        if "source_sample_id" not in self.stability.coords:
            raise ValueError("Stability data must preserve source_sample_id.")
        for name in ("source_sample_id", "source_chain", "source_draw"):
            if name in self.stability.coords and self.stability[name].dims != ("sample",):
                raise ValueError(f"{name} must be aligned to the sample axis.")
        source_ids = np.asarray(self.stability["source_sample_id"].values)
        if len(np.unique(source_ids)) != n_samples:
            raise ValueError("source_sample_id values must be unique.")

        candidate = self.variants["is_candidate"].to_numpy(dtype=bool)
        selected = np.asarray(self.stability["selection_variant_index"].values)
        reference = np.asarray(self.stability["reference_variant_index"].values)
        cohort = np.asarray(self.stability["cohort_variant_index"].values)
        for name, indices in (
            ("selection_variant_index", selected),
            ("reference_variant_index", reference),
            ("cohort_variant_index", cohort),
        ):
            if np.any(indices < 0) or np.any(indices >= n_variants):
                raise ValueError(f"{name} contains indices outside the variant axis.")
            if not candidate[indices].all():
                raise ValueError(f"{name} contains a variant excluded from ranking.")
        if any(len(np.unique(row)) != retain_k for row in selected):
            raise ValueError("Each sampled selection must contain unique variants.")
        if len(np.unique(reference)) != reference_k or len(np.unique(cohort)) != cohort_size:
            raise ValueError("Reference and cohort variant indices must be unique.")

        selected_scores = np.asarray(self.stability["selection_score"].values)
        if not np.isfinite(selected_scores).all():
            raise ValueError("Every retained sampled selection must have a finite score.")
        cohort_scores = np.asarray(self.stability["cohort_score"].values)
        cohort_ranks = np.asarray(self.stability["cohort_rank"].values)
        if not np.array_equal(np.isfinite(cohort_scores), np.isfinite(cohort_ranks)):
            raise ValueError("Cohort scores and ranks must have matching finite values.")
        if np.any(cohort_ranks[np.isfinite(cohort_ranks)] < 1):
            raise ValueError("Finite cohort ranks must be positive.")

        roles = np.column_stack(
            [
                self.stability["is_external_reference"].values,
                self.stability["is_model_reference"].values,
                self.stability["is_challenger"].values,
            ]
        )
        if not roles.any(axis=1).all():
            raise ValueError("Every retained cohort variant must have at least one role.")

        counts = np.asarray(self.stability["top_k_count"].values)
        probabilities = np.asarray(self.stability["top_k_probability"].values)
        if np.any(counts < 0) or np.any(counts > n_samples):
            raise ValueError("top_k_count values must lie between zero and n_samples.")
        decision_k = self.metadata.get("decision_k")
        if not isinstance(decision_k, int) or not 1 <= decision_k <= retain_k:
            raise ValueError("Result metadata decision_k must lie on the retained rank axis.")
        expected_counts = np.bincount(
            selected[:, :decision_k].ravel(),
            minlength=n_variants,
        ).astype(np.int32)
        if not np.array_equal(counts, expected_counts):
            raise ValueError("top_k_count must be derived from the retained selections.")
        if not np.array_equal(probabilities, counts / n_samples):
            raise ValueError("top_k_probability must equal top_k_count divided by n_samples.")
        excluded_have_references = np.isfinite(
            self.stability["reference_score"].values
        )[~candidate]
        reference_counts = self.stability["reference_n_finite_samples"].values
        if np.any(reference_counts < 0) or np.any(reference_counts > n_samples):
            raise ValueError(
                "reference_n_finite_samples must lie between zero and n_samples."
            )
        if (
            np.any(counts[~candidate])
            or np.any(excluded_have_references)
            or np.any(reference_counts[~candidate])
        ):
            raise ValueError("Variants excluded from ranking cannot have ranking statistics.")

        expected_metadata = {
            "n_samples": n_samples,
            "n_variants": n_variants,
            "n_candidates": int(candidate.sum()),
            "retained_k": retain_k,
            "cohort_size": cohort_size,
        }
        for name, expected in expected_metadata.items():
            if self.metadata.get(name) != expected:
                raise ValueError(f"Result metadata field {name!r} must equal {expected}.")
        source_samples = self.metadata.get("n_source_samples")
        excluded_samples = self.metadata.get("n_excluded_samples")
        if (
            not isinstance(source_samples, int)
            or source_samples < n_samples
            or excluded_samples != source_samples - n_samples
        ):
            raise ValueError(
                "Result source and excluded sample counts are inconsistent."
            )

    def save(self, path: str | Path) -> Path:
        """Write metadata, variants, and stability arrays to a directory."""
        self.validate()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.variants.to_csv(root / VARIANT_FILE, index=False)
        encoding = {
            name: {
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
            }
            for name in self.stability.data_vars
        }
        self.stability.to_netcdf(
            root / STABILITY_FILE,
            engine="h5netcdf",
            encoding=encoding,
        )
        return root

    @classmethod
    def load(cls, path: str | Path) -> RankResult:
        """Load and validate a three-file rank result."""
        root = Path(path)
        required = (METADATA_FILE, VARIANT_FILE, STABILITY_FILE)
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"Result data are missing files: {missing}")
        stability = xr.open_dataset(root / STABILITY_FILE, engine="h5netcdf")
        result = cls(
            metadata=json.loads((root / METADATA_FILE).read_text(encoding="utf-8")),
            variants=_read_variants(root / VARIANT_FILE),
            stability=stability,
        )
        try:
            result.validate()
        except Exception:
            stability.close()
            raise
        return result

    def close(self) -> None:
        self.stability.close()
