import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import rankdata

from dms_rank_stability import RankResult, SampledScores, analyze_scores


def _metadata() -> dict[str, str]:
    return {
        "run_id": "synthetic-higher",
        "family": "synthetic",
        "landscape": "small",
        "sampling": "parameter_posterior",
        "model": "example",
        "score": "fitness",
    }


def _sampled_scores() -> SampledScores:
    variants = pd.DataFrame(
        {
            "variant_id": ["", "control", "C", "D", "E", "F"],
            "is_candidate": [True, False, True, True, True, True],
            "reported_rank": [2.0, np.nan, 1.0, 3.0, np.nan, np.nan],
        }
    )
    values = np.array(
        [
            [10.0, 1000.0, 9.0, 8.0, 7.0, 6.0],
            [10.0, 1000.0, 9.0, 8.0, 11.0, 6.0],
            [10.0, 1000.0, 9.0, 8.0, 7.0, np.nan],
            [10.0, 1000.0, 9.0, 8.0, 8.0, 6.0],
        ],
        dtype=np.float32,
    )
    dataset = xr.Dataset(
        {"score": (("sample", "variant"), values)},
        coords={
            "sample": np.arange(4),
            "variant": np.arange(6),
            "source_sample_id": ("sample", np.array([10, 20, 30, 40])),
            "source_chain": ("sample", np.array([0, 0, 1, 1], dtype=np.int32)),
            "source_draw": ("sample", np.array([0, 1, 0, 1], dtype=np.int32)),
        },
    )
    return SampledScores(variants, dataset)


def test_matrix_scan_matches_dense_ranks_and_retains_decision_cohort():
    scores = _sampled_scores()
    result = analyze_scores(
        scores,
        metadata=_metadata(),
        retain_k=2,
        decision_k=2,
        external_top_n=3,
        model_top_n=2,
        challenger_probability=0.01,
        variant_chunk_size=2,
    )

    candidate_indices = np.array([0, 2, 3, 4, 5], dtype=np.int32)
    candidate_scores = scores.score.values[:, candidate_indices]
    dense_order = np.argsort(-np.where(np.isfinite(candidate_scores), candidate_scores, -np.inf))
    expected_selection = candidate_indices[dense_order[:, :2]]
    np.testing.assert_array_equal(
        result.stability["selection_variant_index"],
        expected_selection,
    )
    np.testing.assert_array_equal(result.stability["reference_variant_index"], [0, 2])
    np.testing.assert_array_equal(
        result.stability["reference_n_finite_samples"],
        [4, 0, 4, 4, 4, 3],
    )
    assert 1 not in result.stability["selection_variant_index"].values

    cohort = result.stability["cohort_variant_index"].values
    np.testing.assert_array_equal(cohort, [0, 2, 3, 4])
    dense_ranks = rankdata(
        -candidate_scores,
        axis=1,
        method="average",
        nan_policy="omit",
    )
    positions = np.searchsorted(candidate_indices, cohort)
    np.testing.assert_allclose(
        result.stability["cohort_rank"],
        dense_ranks[:, positions],
        equal_nan=True,
    )
    challenger = result.stability["is_challenger"].values
    np.testing.assert_array_equal(cohort[challenger], [4])
    np.testing.assert_array_equal(result.stability["source_sample_id"], [10, 20, 30, 40])


def test_matrix_scan_supports_lower_scores():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "excluded", "C"],
            "is_candidate": [True, False, True],
        }
    )
    values = np.array([[3.0, -100.0, 1.0], [1.0, -100.0, 2.0]], dtype=np.float32)
    scores = SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), values)},
            coords={
                "sample": np.arange(2),
                "variant": np.arange(3),
                "source_sample_id": ("sample", np.array([5, 9])),
            },
        ),
    )
    metadata = {**_metadata(), "run_id": "synthetic-lower"}

    result = analyze_scores(
        scores,
        metadata=metadata,
        retain_k=1,
        decision_k=1,
        external_top_n=0,
        model_top_n=1,
        higher_is_better=False,
        variant_chunk_size=1,
    )

    np.testing.assert_array_equal(
        result.stability["selection_variant_index"],
        [[2], [0]],
    )
    np.testing.assert_array_equal(result.stability["reference_variant_index"], [2])
    assert result.metadata["higher_is_better"] is False


def test_matrix_scan_excludes_samples_without_enough_finite_candidates():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "is_candidate": [True, True, True],
            "reported_rank": [1.0, 2.0, 3.0],
        }
    )
    scores = SampledScores(
        variants,
        xr.Dataset(
            {
                "score": (
                    ("sample", "variant"),
                    np.array(
                        [
                            [3.0, 2.0, 1.0],
                            [3.0, np.nan, np.nan],
                            [1.0, 3.0, 2.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            },
            coords={
                "sample": np.arange(3),
                "variant": np.arange(3),
                "source_sample_id": ("sample", np.array([10, 20, 30])),
            },
        ),
    )

    result = analyze_scores(
        scores,
        metadata=_metadata(),
        retain_k=2,
        decision_k=1,
        external_top_n=2,
        model_top_n=1,
        variant_chunk_size=1,
    )

    assert result.metadata["n_source_samples"] == 3
    assert result.metadata["n_samples"] == 2
    assert result.metadata["n_excluded_samples"] == 1
    np.testing.assert_array_equal(result.stability["source_sample_id"], [10, 30])
    np.testing.assert_array_equal(
        result.stability["selection_variant_index"],
        [[0, 1], [1, 2]],
    )


def test_matrix_scan_replays_declared_block_order_for_exact_ties():
    variants = pd.DataFrame(
        {
            "variant_id": list("ABCDEF"),
            "is_candidate": True,
            "reported_rank": np.arange(1, 7, dtype=float),
        }
    )
    scores = SampledScores(
        variants,
        xr.Dataset(
            {
                "score": (
                    ("sample", "variant"),
                    np.array(
                        [
                            [0.0, 2.0, 1.0, 2.0, 1.0, 0.0],
                            [2.0, 0.0, 0.0, 2.0, 2.0, 3.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            },
            coords={
                "sample": np.arange(2),
                "variant": np.arange(6),
                "source_sample_id": ("sample", np.array([10, 20])),
                "selection_block": (
                    "variant",
                    np.array([1, 1, 0, 0, 2, 2], dtype=np.int32),
                ),
            },
        ),
    )

    result = analyze_scores(
        scores,
        metadata=_metadata(),
        retain_k=3,
        decision_k=1,
        external_top_n=3,
        model_top_n=1,
        variant_chunk_size=2,
    )

    np.testing.assert_array_equal(
        result.stability["selection_variant_index"],
        [[1, 3, 2], [5, 4, 3]],
    )


def test_rank_result_round_trip_writes_only_three_files(tmp_path):
    result = analyze_scores(
        _sampled_scores(),
        metadata=_metadata(),
        retain_k=2,
        decision_k=2,
        external_top_n=3,
        model_top_n=2,
        variant_chunk_size=2,
    )

    result.save(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "metadata.json",
        "variants.csv",
        "stability.nc",
    }
    loaded = RankResult.load(tmp_path)
    try:
        assert loaded.variants.loc[0, "variant_id"] == ""
        xr.testing.assert_equal(loaded.stability, result.stability)
        assert loaded.metadata == result.metadata
    finally:
        loaded.close()


def test_rank_result_rejects_counts_not_derived_from_selections():
    result = analyze_scores(
        _sampled_scores(),
        metadata=_metadata(),
        retain_k=2,
        decision_k=2,
        external_top_n=3,
        model_top_n=2,
        variant_chunk_size=2,
    )
    result.stability["top_k_count"].values[0] -= 1
    result.stability["top_k_probability"].values[0] = (
        result.stability["top_k_count"].values[0] / result.metadata["n_samples"]
    )

    with pytest.raises(ValueError, match="derived from the retained selections"):
        result.validate()
