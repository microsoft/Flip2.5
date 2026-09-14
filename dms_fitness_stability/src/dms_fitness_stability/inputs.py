"""Validation and normalization of raw DMS count tables."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class ReportedFitnessFallbackWarning(UserWarning):
    """Reported fitness is absent and count ratio is used as the reference."""


def _validated_counts(values: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{column!r} must contain only finite values.")
    if (numeric < 0).any():
        raise ValueError(f"{column!r} cannot contain negative counts.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column!r} must contain integer counts.")
    return numeric.astype(np.int64)


def prepare_counts(
    counts: pd.DataFrame,
    *,
    variant_id: str = "variant_id",
    input_count: str = "input_count",
    output_count: str = "output_count",
    reported_fitness: str = "reported_fitness",
) -> pd.DataFrame:
    """Pool replicate counts and choose a transparent ranking reference.

    Reported fitness is used only when it is present for every variant. If it is
    absent, pooled output/input count ratio is used and a warning is emitted.
    Partially observed reported fitness is rejected because the two scales must
    not be combined in one ranking.
    """
    required = {variant_id, input_count, output_count}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"Count table is missing required columns: {sorted(missing)}")
    if counts.empty:
        raise ValueError("Count table must contain at least one row.")
    if counts[variant_id].isna().any():
        raise ValueError(f"{variant_id!r} cannot contain missing values.")

    normalized = counts.copy()
    normalized[input_count] = _validated_counts(
        normalized[input_count],
        column=input_count,
    )
    normalized[output_count] = _validated_counts(
        normalized[output_count],
        column=output_count,
    )

    grouped = normalized.groupby(variant_id, sort=False, dropna=False)
    variants = grouped[[input_count, output_count]].sum().reset_index()
    variants = variants.rename(
        columns={
            input_count: "input_count",
            output_count: "output_count",
        }
    )
    variants["count_ratio"] = np.divide(
        variants["output_count"],
        variants["input_count"],
        out=np.full(len(variants), np.nan, dtype=float),
        where=variants["input_count"].to_numpy() > 0,
    )

    if reported_fitness in normalized:
        unique_reported = grouped[reported_fitness].nunique(dropna=True)
        inconsistent = unique_reported[unique_reported > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Reported fitness must be constant across replicate rows for each "
                f"variant; inconsistent IDs: {inconsistent.index.tolist()}"
            )
        reported = grouped[reported_fitness].first().reset_index(drop=True)
        reported = pd.to_numeric(reported, errors="raise").astype(float)
        if np.isinf(reported).any():
            raise ValueError("Reported fitness cannot contain infinite values.")
    else:
        reported = pd.Series(np.nan, index=variants.index, dtype=float)

    variants["reported_fitness"] = reported
    n_reported = int(reported.notna().sum())
    if n_reported == len(variants):
        variants["reference_fitness"] = reported
        variants["reference_fitness_source"] = "reported"
    elif n_reported == 0:
        warnings.warn(
            "Reported fitness is unavailable; using pooled output/input count "
            "ratio as the temporary reference.",
            ReportedFitnessFallbackWarning,
            stacklevel=2,
        )
        variants["reference_fitness"] = variants["count_ratio"]
        variants["reference_fitness_source"] = "count_ratio_fallback"
    else:
        raise ValueError(
            f"Reported fitness is available for {n_reported} of {len(variants)} "
            "variants. Choose a complete reported cohort or use count ratio for "
            "the full landscape explicitly."
        )

    if variant_id != "variant_id":
        variants = variants.rename(columns={variant_id: "variant_id"})
    return variants
