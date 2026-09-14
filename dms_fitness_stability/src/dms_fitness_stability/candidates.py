"""Shared ranking-candidate selection for aligned variant tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

RANKING_CANDIDATE_COLUMN = "is_ranking_candidate"


def ranking_candidate_mask(variants: pd.DataFrame) -> np.ndarray:
    """Return a validated full-table mask of rows eligible for ranking."""
    if RANKING_CANDIDATE_COLUMN not in variants:
        return np.ones(len(variants), dtype=bool)
    candidate = variants[RANKING_CANDIDATE_COLUMN]
    if candidate.isna().any() or not is_bool_dtype(candidate.dtype):
        raise ValueError(
            f"{RANKING_CANDIDATE_COLUMN} must contain only non-null booleans."
        )
    mask = candidate.to_numpy(dtype=bool)
    if not mask.any():
        raise ValueError("At least one variant must be eligible for ranking.")
    return mask


def ranking_candidate_indices(variants: pd.DataFrame) -> np.ndarray:
    """Return full-table row positions eligible for ranking."""
    indices = np.flatnonzero(ranking_candidate_mask(variants)).astype(np.int32)
    if not len(indices):
        raise ValueError("At least one variant must be eligible for ranking.")
    return indices
