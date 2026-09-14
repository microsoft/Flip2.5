"""Equivalence checks for matrix-backed mechanism caches and tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .migration_audit import audit_dataset


def compare_mechanism_cache(
    observed_path: str | Path,
    expected_path: str | Path,
    *,
    sample_chunk_size: int = 4_096,
    allowed_observed_variables: tuple[str, ...] = (),
) -> dict[str, object]:
    """Require exact equality for legacy arrays, allowing named additions."""
    observed = audit_dataset(observed_path, sample_chunk_size=sample_chunk_size)
    expected = audit_dataset(expected_path, sample_chunk_size=sample_chunk_size)
    for surface in ("sizes", "coordinates"):
        if observed[surface] != expected[surface]:
            raise ValueError(f"Mechanism cache {surface} differ.")
    observed_variables = observed["variables"]
    expected_variables = expected["variables"]
    added_variables = sorted(set(observed_variables) - set(expected_variables))
    missing_variables = sorted(set(expected_variables) - set(observed_variables))
    unexpected_variables = sorted(
        set(added_variables) - set(allowed_observed_variables)
    )
    if missing_variables or unexpected_variables:
        raise ValueError(
            "Mechanism cache variables differ: "
            f"missing={missing_variables}, unexpected={unexpected_variables}."
        )
    changed_variables = sorted(
        name
        for name, record in expected_variables.items()
        if observed_variables[name] != record
    )
    if changed_variables:
        raise ValueError(
            f"Mechanism cache variables differ: changed={changed_variables}."
        )
    return {
        "sizes": observed["sizes"],
        "coordinate_count": len(observed["coordinates"]),
        "variable_count": len(observed["variables"]),
        "legacy_variable_count": len(expected_variables),
        "added_variables": added_variables,
        "observed_source_kind": observed["attrs"].get("source_kind"),
        "expected_source_kind": expected["attrs"].get("source_kind"),
        "status": "exact_with_additions" if added_variables else "exact",
    }


def compare_mechanism_table(
    observed_path: str | Path,
    expected_path: str | Path,
    *,
    atol: float = 5e-12,
    normalize_current_trpb_candidate_annotation: bool = False,
    allowed_observed_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    """Compare legacy columns exactly, allowing named observed additions."""
    if atol < 0:
        raise ValueError("atol must be nonnegative.")
    observed_path = Path(observed_path)
    expected_path = Path(expected_path)
    observed = pd.read_csv(observed_path, low_memory=False)
    expected = pd.read_csv(expected_path, low_memory=False)
    normalized_rows = 0
    if (
        normalize_current_trpb_candidate_annotation
        and "is_ranking_candidate" in expected
        and "protein_family" in expected
    ):
        mask = expected["protein_family"].eq("trpb")
        normalized_rows = int(mask.sum())
        expected["is_ranking_candidate"] = expected[
            "is_ranking_candidate"
        ].astype(object)
        expected.loc[mask, "is_ranking_candidate"] = np.nan
    added_columns = sorted(set(observed.columns) - set(expected.columns))
    missing_columns = sorted(set(expected.columns) - set(observed.columns))
    unexpected_columns = sorted(set(added_columns) - set(allowed_observed_columns))
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Mechanism table columns differ for {observed_path.name}: "
            f"missing={missing_columns}, unexpected={unexpected_columns}."
        )
    legacy_columns = [name for name in observed.columns if name in expected.columns]
    observed_legacy = observed[legacy_columns]
    expected = expected[legacy_columns]
    if len(observed) != len(expected):
        raise ValueError(
            f"Mechanism table shape differs for {observed_path.name}: "
            f"{len(observed)} rows != {len(expected)} rows."
        )

    maximum_absolute_difference = 0.0
    for name in legacy_columns:
        left = observed_legacy[name]
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
                atol=atol,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Mechanism table float values differ for "
                    f"{observed_path.name}:{name}."
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
                f"Mechanism table values differ for {observed_path.name}:{name}."
            )
    return {
        "rows": len(observed),
        "columns": observed.columns.tolist(),
        "legacy_column_count": len(legacy_columns),
        "added_columns": added_columns,
        "normalized_trpb_rows": normalized_rows,
        "maximum_absolute_difference": maximum_absolute_difference,
        "status": (
            "exact_with_additions"
            if added_columns and maximum_absolute_difference == 0
            else "within_tolerance_with_additions"
            if added_columns
            else "exact"
            if maximum_absolute_difference == 0
            else "within_tolerance"
        ),
    }
