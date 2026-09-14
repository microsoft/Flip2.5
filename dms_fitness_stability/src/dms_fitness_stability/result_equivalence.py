"""Semantic equivalence checks for persisted stability result bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .migration_audit import audit_dataset

RESULT_METADATA_FIELDS = (
    "analysis_id",
    "protein_family",
    "landscape",
    "inference_method",
    "model_spec",
    "fit_id",
    "job_id",
    "artifact_kind",
    "primary_score",
    "fit_method",
    "scores",
    "n_samples",
    "n_variants",
    "retained_k",
    "tracked_reported_top_n",
    "tracked_model_top_n",
    "top_k",
    "interval",
    "seed",
    "reference_fitness_source",
    "reported_fitness_available",
    "reported_fitness_coverage",
    "ranking_candidate_column",
    "ranking_candidate_definition",
)


def _table_comparison(
    observed_path: Path,
    expected_path: Path,
    *,
    atol: float,
    ignored_columns: tuple[str, ...] = (),
    exact: bool = False,
) -> dict[str, object]:
    observed = pd.read_csv(observed_path, low_memory=False)
    expected = pd.read_csv(expected_path, low_memory=False)
    ignored = [
        name
        for name in ignored_columns
        if name in observed.columns or name in expected.columns
    ]
    observed = observed.drop(columns=ignored, errors="ignore")
    expected = expected.drop(columns=ignored, errors="ignore")
    if observed.columns.tolist() != expected.columns.tolist():
        raise ValueError(
            f"Table columns differ for {observed_path.name}: "
            f"observed={observed.columns.tolist()}, expected={expected.columns.tolist()}."
        )
    if observed.shape != expected.shape:
        raise ValueError(
            f"Table shape differs for {observed_path.name}: "
            f"observed={observed.shape}, expected={expected.shape}."
        )

    maximum_absolute_difference = 0.0
    for name in observed.columns:
        left = observed[name]
        right = expected[name]
        if pd.api.types.is_float_dtype(left.dtype) and pd.api.types.is_float_dtype(
            right.dtype
        ):
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            if not np.allclose(
                left_values,
                right_values,
                rtol=0,
                atol=0 if exact else atol,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Table float values differ for {observed_path.name}:{name}."
                )
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            if finite.any():
                maximum_absolute_difference = max(
                    maximum_absolute_difference,
                    float(
                        np.max(
                            np.abs(left_values[finite] - right_values[finite])
                        )
                    ),
                )
        elif not left.equals(right):
            raise ValueError(
                f"Table values differ for {observed_path.name}:{name}."
            )
    return {
        "rows": len(observed),
        "columns": observed.columns.tolist(),
        "ignored_columns": ignored,
        "maximum_absolute_difference": maximum_absolute_difference,
        "status": "exact" if maximum_absolute_difference == 0 else "within_tolerance",
    }


def compare_stability_result_bundles(
    observed_dir: str | Path,
    expected_dir: str | Path,
    *,
    table_atol: float = 5e-12,
    ignored_columns: tuple[str, ...] = (),
    sample_chunk_size: int = 4_096,
) -> dict[str, object]:
    """Compare matrix-driven output with a fit-driven result bundle."""
    if table_atol < 0:
        raise ValueError("table_atol must be nonnegative.")
    observed_dir = Path(observed_dir)
    expected_dir = Path(expected_dir)
    required = {
        "metadata.json",
        "variants.csv",
        "stability.nc",
        "summary.csv",
        "overlaps.csv",
    }
    for result_dir in (observed_dir, expected_dir):
        missing = sorted(name for name in required if not (result_dir / name).is_file())
        if missing:
            raise ValueError(f"Result directory is missing files: {missing}")

    observed_metadata = json.loads(
        (observed_dir / "metadata.json").read_text(encoding="utf-8")
    )
    expected_metadata = json.loads(
        (expected_dir / "metadata.json").read_text(encoding="utf-8")
    )
    metadata_mismatches = {
        name: (observed_metadata.get(name), expected_metadata.get(name))
        for name in RESULT_METADATA_FIELDS
        if observed_metadata.get(name) != expected_metadata.get(name)
    }
    if metadata_mismatches:
        raise ValueError(f"Result metadata differs: {metadata_mismatches}.")

    observed_selection = audit_dataset(
        observed_dir / "stability.nc",
        sample_chunk_size=sample_chunk_size,
    )
    expected_selection = audit_dataset(
        expected_dir / "stability.nc",
        sample_chunk_size=sample_chunk_size,
    )
    for surface in ("sizes", "coordinates", "variables"):
        if observed_selection[surface] != expected_selection[surface]:
            raise ValueError(f"Result selection {surface} differ.")

    return {
        "analysis_id": observed_metadata["analysis_id"],
        "observed_dir": str(observed_dir.resolve()),
        "expected_dir": str(expected_dir.resolve()),
        "selection": {
            "sizes": observed_selection["sizes"],
            "coordinate_count": len(observed_selection["coordinates"]),
            "variable_count": len(observed_selection["variables"]),
            "status": "exact",
        },
        "variants": _table_comparison(
            observed_dir / "variants.csv",
            expected_dir / "variants.csv",
            atol=table_atol,
            ignored_columns=ignored_columns,
        ),
        "summary": _table_comparison(
            observed_dir / "summary.csv",
            expected_dir / "summary.csv",
            atol=table_atol,
            ignored_columns=ignored_columns,
        ),
        "overlaps": _table_comparison(
            observed_dir / "overlaps.csv",
            expected_dir / "overlaps.csv",
            atol=0,
            exact=True,
        ),
        "table_atol": table_atol,
        "status": "equivalent",
    }
