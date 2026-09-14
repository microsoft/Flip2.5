import numpy as np
import pandas as pd
import pytest

from dms_fitness_stability import ReportedFitnessFallbackWarning, prepare_counts


def test_prepare_counts_uses_complete_reported_fitness():
    counts = pd.DataFrame(
        {
            "variant_id": ["A", "A", "B", "B"],
            "replicate": [1, 2, 1, 2],
            "input_count": [10, 30, 20, 20],
            "output_count": [20, 20, 10, 10],
            "reported_fitness": [0.8, 0.8, 0.2, 0.2],
        }
    )

    result = prepare_counts(counts)

    assert result["variant_id"].tolist() == ["A", "B"]
    np.testing.assert_array_equal(result["input_count"], [40, 40])
    np.testing.assert_allclose(result["count_ratio"], [1.0, 0.5])
    np.testing.assert_allclose(result["reference_fitness"], [0.8, 0.2])
    assert result["reference_fitness_source"].unique().tolist() == ["reported"]


def test_prepare_counts_warns_and_uses_count_ratio_without_reported_fitness():
    counts = pd.DataFrame(
        {
            "variant_id": ["A", "A", "B"],
            "input_count": [10, 30, 0],
            "output_count": [20, 20, 4],
        }
    )

    with pytest.warns(ReportedFitnessFallbackWarning):
        result = prepare_counts(counts)

    np.testing.assert_allclose(result.loc[0, "reference_fitness"], 1.0)
    assert np.isnan(result.loc[1, "reference_fitness"])
    assert result["reported_fitness"].isna().all()
    assert result["reference_fitness_source"].unique().tolist() == ["count_ratio_fallback"]


def test_prepare_counts_rejects_partial_reported_fitness():
    counts = pd.DataFrame(
        {
            "variant_id": ["A", "B"],
            "input_count": [10, 20],
            "output_count": [20, 10],
            "reported_fitness": [0.8, np.nan],
        }
    )

    with pytest.raises(ValueError, match="available for 1 of 2 variants"):
        prepare_counts(counts)


def test_prepare_counts_rejects_inconsistent_reported_fitness():
    counts = pd.DataFrame(
        {
            "variant_id": ["A", "A"],
            "input_count": [10, 20],
            "output_count": [20, 30],
            "reported_fitness": [0.8, 0.7],
        }
    )

    with pytest.raises(ValueError, match="constant across replicate rows"):
        prepare_counts(counts)
