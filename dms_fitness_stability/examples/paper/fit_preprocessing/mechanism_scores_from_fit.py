"""Legacy mechanism-cohort scanning from a direct-HMC fit artifact."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dms_fitness_stability.mechanisms import scan_fixed_cohort


def scan_fit_mechanism_scores(
    metadata: dict[str, object],
    variants: pd.DataFrame,
    *,
    sample_ids: np.ndarray,
    candidate_indices: np.ndarray,
    cohort_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Scan one fixed cohort from the fit variable recorded by legacy results."""
    hmc_file = metadata.get("hmc_file")
    if not isinstance(hmc_file, dict):
        raise ValueError("Legacy fit-backed mechanism scan requires hmc_file.")
    with h5py.File(Path(str(hmc_file["path"])), "r") as handle:
        target = handle["posterior"][str(metadata["posterior_variable"])]
        return scan_fixed_cohort(
            target,
            sample_ids=sample_ids,
            hmc_variant_indices=variants["hmc_variant_index"].to_numpy(
                dtype=np.int64
            ),
            candidate_variant_indices=candidate_indices,
            cohort_variant_indices=cohort_indices,
            transform=str(metadata["posterior_transform"]),
        )
