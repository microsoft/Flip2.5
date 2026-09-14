"""Regenerate the deterministic tiny example data."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dms_rank_stability import LandscapeFeatures, SampledScores
from dms_rank_stability.data import FEATURE_DTYPES

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Write one sampled-score bundle and one aligned feature bundle."""
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "A3D", "A4E", "A5F", "A6G", "A7H", "A8I", "control"],
            "is_candidate": [True] * 8 + [False],
            "variant_label": ["WT", "A2C", "A3D", "A4E", "A5F", "A6G", "A7H", "A8I", "control"],
            "reported_fitness": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 20.0],
            "reported_rank": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan],
            "reported_activity_level": ["high"] * 3 + ["medium"] * 3 + ["low"] * 2 + [np.nan],
            "is_wild_type": [True] + [False] * 8,
            "is_source_control": [False] * 8 + [True],
        }
    )
    random = np.random.default_rng(23)
    candidate_scores = (
        np.arange(8, 0, -1, dtype=np.float32)[None, :]
        + random.normal(0, 0.8, size=(12, 8)).astype(np.float32)
    )
    scores = np.column_stack(
        [candidate_scores, np.full(12, np.nan, dtype=np.float32)]
    )
    SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), scores)},
            coords={
                "sample": np.arange(12, dtype=np.int32),
                "variant": np.arange(len(variants), dtype=np.int32),
                "source_sample_id": ("sample", np.arange(100, 112, dtype=np.int64)),
                "source_chain": ("sample", np.repeat(np.arange(2, dtype=np.int32), 6)),
                "source_draw": ("sample", np.tile(np.arange(6, dtype=np.int32), 2)),
            },
        ),
    ).save(ROOT / "scores/toy-parameter-posterior-example")

    input_count = np.array([800, 600, 450, 300, 180, 90, 35, 10, 0], dtype=float)
    output_count = np.array([900, 620, 420, 280, 150, 70, 20, 4, 0], dtype=float)
    arrays = {
        "observation_count": ("variant", np.full(9, 3, dtype=np.int32)),
        "input_count": ("variant", input_count),
        "output_count": ("variant", output_count),
        "log_input_support": ("variant", np.log1p(input_count)),
        "zero_input_fraction": ("variant", np.array([0.0] * 7 + [1 / 3, 1.0])),
        "zero_output_fraction": ("variant", np.array([0.0] * 6 + [1 / 3, 2 / 3, 1.0])),
        "reduced_heterogeneity": ("variant", np.linspace(0.1, 1.2, 9)),
        "enrichment_range": ("variant", np.linspace(0.05, 0.9, 9)),
        "enrichment_mad": ("variant", np.linspace(0.02, 0.4, 9)),
    }
    feature_dataset = xr.Dataset(
        arrays,
        coords={"variant": np.arange(len(variants), dtype=np.int32)},
    )
    for name, dtype in FEATURE_DTYPES.items():
        feature_dataset[name] = feature_dataset[name].astype(dtype)
    LandscapeFeatures(
        variants[["variant_id", "is_candidate"]],
        feature_dataset,
    ).save(ROOT / "features/toy")


if __name__ == "__main__":
    main()
