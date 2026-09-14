import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability.result_equivalence import compare_stability_result_bundles
from dms_fitness_stability.results import StabilityResult


def _result() -> StabilityResult:
    analysis_id = "nuc__nuc-g1__hmc__exponential__fit"
    return StabilityResult(
        metadata={
            "analysis_id": analysis_id,
            "protein_family": "nuc",
            "landscape": "nuc-g1",
            "inference_method": "hmc",
            "model_spec": "exponential",
            "fit_id": "fit",
            "job_id": 1,
            "artifact_kind": "diagnosed_hmc",
            "primary_score": "model_fluorescence",
            "fit_method": "hmc_posterior",
            "scores": ["model_fluorescence"],
            "n_samples": 2,
            "n_variants": 2,
            "retained_k": 1,
            "tracked_reported_top_n": 1,
            "tracked_model_top_n": 0,
            "top_k": [1],
            "interval": [0.05, 0.95],
            "seed": 23,
        },
        variants=pd.DataFrame(
            {"variant_id": ["A", "B"], "count_ratio": [0.1, 0.2]}
        ),
        selection=xr.Dataset(
            {
                "variant_index": (
                    ("score_name", "sample", "selection_rank"),
                    np.array([[[0], [1]]], dtype=np.int32),
                )
            },
            coords={
                "score_name": ["model_fluorescence"],
                "sample": np.arange(2, dtype=np.int32),
                "sample_id": ("sample", np.arange(2, dtype=np.int64)),
                "selection_rank": np.array([1], dtype=np.int32),
            },
        ),
        summary=pd.DataFrame(
            {"variant_id": ["A", "B"], "posterior_value_mean": [0.1, 0.2]}
        ),
        overlaps=pd.DataFrame({"k": [1], "intersection": [1]}),
    )


def test_result_equivalence_requires_exact_selection_and_tolerates_csv_roundoff(
    tmp_path,
):
    expected = _result()
    observed = _result()
    expected_dir = expected.save(tmp_path / "expected")
    observed.variants.loc[0, "count_ratio"] += 1e-15
    observed.summary.loc[0, "posterior_value_mean"] += 1e-15
    observed_dir = observed.save(tmp_path / "observed")

    record = compare_stability_result_bundles(observed_dir, expected_dir)

    assert record["status"] == "equivalent"
    assert record["selection"]["status"] == "exact"

    observed.selection["variant_index"][0, 0, 0] = 1
    observed.save(observed_dir)
    with pytest.raises(ValueError, match="selection variables differ"):
        compare_stability_result_bundles(observed_dir, expected_dir)
