import numpy as np
import pytest

from dms_rank_stability.mechanisms import (
    boundary_difficulty,
    count_within_score_window,
    local_score_spacing,
    partial_spearman,
    score_uncertainty,
    standardized_regression,
)


def test_score_uncertainty_uses_fixed_reference_scale():
    samples = np.array(
        [[8.0, 5.0], [10.0, 6.0], [12.0, 7.0], [14.0, 8.0]]
    )
    reference_scores = np.array([12.0, 10.0, 8.0, 6.0, 4.0])

    features = score_uncertainty(
        samples,
        reference_scores=reference_scores,
        interval=(0.25, 0.75),
    )

    np.testing.assert_allclose(features["score_lower"], [9.5, 5.75])
    np.testing.assert_allclose(features["score_median"], [11.0, 6.5])
    np.testing.assert_allclose(features["score_upper"], [12.5, 7.25])
    np.testing.assert_allclose(features["score_interval_width"], [3.0, 1.5])
    np.testing.assert_allclose(
        features["normalized_score_interval_width"],
        [0.75, 0.375],
    )
    assert features["reference_score_iqr"] == 4.0


def test_score_uncertainty_accepts_lower_is_better_reference_order():
    samples = np.array(
        [[8.0, 5.0], [10.0, 6.0], [12.0, 7.0], [14.0, 8.0]]
    )

    features = score_uncertainty(
        samples,
        reference_scores=np.array([4.0, 6.0, 8.0, 10.0, 12.0]),
        interval=(0.25, 0.75),
        higher_is_better=False,
    )

    np.testing.assert_allclose(features["score_interval_width"], [3.0, 1.5])
    np.testing.assert_allclose(
        features["normalized_score_interval_width"],
        [0.75, 0.375],
    )


def test_spacing_and_window_crowding_use_only_reference_scores():
    reference_scores = np.array([10.0, 9.0, 8.0, 6.0, 2.0])

    spacing = local_score_spacing(reference_scores, neighborhood=1)
    within_window = count_within_score_window(reference_scores, window_fraction=0.25)

    np.testing.assert_allclose(spacing["score_spacing"], [1.0, 1.0, 1.5, 3.0, 4.0])
    np.testing.assert_array_equal(within_window["competitor_count"], 0)
    assert within_window["score_window"] == pytest.approx(0.75)


def test_boundary_difficulty_compares_uncertainty_with_cutoff_spacing():
    features = boundary_difficulty(
        np.array([10.0, 8.0, 5.0, 4.0]),
        np.array([[10.0, 8.0], [12.0, 9.0], [14.0, 10.0], [16.0, 11.0]]),
        sample_self_consistency=0.7,
        k=2,
        interval=(0.25, 0.75),
        boundary_window=2,
    )

    assert features["boundary_score_gap"] == 3.0
    assert features["boundary_local_score_spacing"] == 2.0
    assert features["boundary_score_uncertainty"] == pytest.approx(2.25)
    assert features["uncertainty_to_spacing_ratio"] == pytest.approx(1.125)
    assert features["top_k_instability"] == pytest.approx(0.3)


def test_standardized_regression_recovers_known_effect_signs():
    first = np.arange(8, dtype=float)
    second = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

    result = standardized_regression(
        3 * first - 2 * second,
        {"first": first, "second": second},
    )

    assert result["n_observations"] == 8
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["coefficient_first"] > 0
    assert result["coefficient_second"] < 0


def test_partial_spearman_matches_explicit_rank_residualization():
    first = np.array([1, 3, 2, 5, 4, 8, 6, 7], dtype=float)
    second = np.array([8, 6, 7, 3, 5, 1, 4, 2], dtype=float)
    control = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    design = np.column_stack(
        [np.ones(len(first)), np.array([1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5])]
    )
    first_rank = np.array([1, 3, 2, 5, 4, 8, 6, 7], dtype=float)
    second_rank = np.array([8, 6, 7, 3, 5, 1, 4, 2], dtype=float)
    left = first_rank - design @ np.linalg.lstsq(design, first_rank, rcond=None)[0]
    right = second_rank - design @ np.linalg.lstsq(design, second_rank, rcond=None)[0]

    observed = partial_spearman(first, second, {"control": control})

    assert observed == pytest.approx(np.corrcoef(left, right)[0, 1])
