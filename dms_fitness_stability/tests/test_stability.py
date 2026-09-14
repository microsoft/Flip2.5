import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import (
    pairwise_overlap,
    reference_overlap,
    select_top_k,
    summarize_inclusion,
)


@pytest.fixture
def scores():
    return xr.DataArray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 2.0, 3.0, 1.0],
        ],
        dims=("sample", "variant"),
        coords={"sample": [10, 11, 12], "variant": ["A", "B", "C", "D"]},
    )


@pytest.fixture
def variants():
    return pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D"],
            "reference_fitness": [0.1, 0.4, 0.3, 0.2],
        }
    )


def test_select_top_k_and_summarize_inclusion(scores, variants):
    selection = select_top_k(scores, k=2)

    np.testing.assert_array_equal(
        selection["variant_index"],
        [[0, 1], [1, 2], [0, 2]],
    )
    np.testing.assert_allclose(selection["score"], [[4, 3], [4, 3], [4, 3]])

    summary = summarize_inclusion(selection, variants).set_index("variant_id")
    np.testing.assert_allclose(summary.loc[["A", "B", "C"], "top_k_probability"], 2 / 3)
    assert "D" not in summary.index


def test_select_top_k_supports_lower_scores_as_better(scores):
    selection = select_top_k(scores, k=1, higher_is_better=False)

    np.testing.assert_array_equal(selection["variant_index"].values.ravel(), [3, 0, 3])


def test_selection_requires_enough_finite_values(scores):
    scores[0, :3] = np.nan

    with pytest.raises(ValueError, match="Sample 0 has fewer than k=2"):
        select_top_k(scores, k=2)


def test_pairwise_overlap_is_reproducible_and_has_no_self_pairs(scores):
    selection = select_top_k(scores, k=2)

    first = pairwise_overlap(selection, n_pairs=20, seed=7)
    second = pairwise_overlap(selection, n_pairs=20, seed=7)

    assert first.equals(second)
    assert (first["left_sample"] != first["right_sample"]).all()
    assert first["intersection"].between(0, 2).all()


def test_reference_overlap_matches_fixed_reference_top_k(scores, variants):
    selection = select_top_k(scores, k=2)

    overlap = reference_overlap(selection, variants["reference_fitness"])

    np.testing.assert_array_equal(overlap["intersection"], [1, 2, 1])
    np.testing.assert_allclose(overlap["jaccard"], [1 / 3, 1.0, 1 / 3])
