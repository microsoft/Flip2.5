import numpy as np
from scipy.stats import rankdata

from dms_fitness_stability.streaming import (
    accumulate_tracked_ranks,
    empty_top_k,
    empty_tracked_ranks,
    merge_top_k,
    select_reference_top_k,
)


def test_streamed_top_k_and_tracked_ranks_match_dense_calculation():
    scores = np.array(
        [
            [5.0, 2.0, 4.0, 1.0, 3.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [3.0, 3.0, 2.0, np.nan, 1.0],
        ],
        dtype=np.float32,
    )
    tracked_indices = np.array([0, 1, 4])
    tracked_scores = scores[:, tracked_indices]
    top_indices, top_scores = empty_top_k(len(scores), k=2)
    tracked_ranks = empty_tracked_ranks(len(scores), len(tracked_indices))

    for start, stop in ((0, 2), (2, 5)):
        top_indices, top_scores = merge_top_k(
            top_indices,
            top_scores,
            np.arange(start, stop),
            scores[:, start:stop],
        )
        accumulate_tracked_ranks(
            tracked_ranks,
            scores[:, start:stop],
            tracked_scores,
        )

    expected_scores = np.sort(
        np.where(np.isfinite(scores), scores, -np.inf),
        axis=1,
    )[:, -2:][:, ::-1]
    expected_ranks = rankdata(-scores, axis=1, method="average", nan_policy="omit")[
        :, tracked_indices
    ]
    np.testing.assert_allclose(top_scores, expected_scores)
    np.testing.assert_allclose(np.take_along_axis(scores, top_indices, axis=1), top_scores)
    np.testing.assert_allclose(tracked_ranks, expected_ranks)


def test_reference_top_k_uses_sample_medians_and_deterministic_ties():
    scores = np.array(
        [
            [5.0, 4.0, 1.0, np.nan],
            [1.0, 2.0, 4.0, np.nan],
            [3.0, 3.0, 3.0, np.nan],
        ],
        dtype=np.float32,
    )

    indices, reference_scores = select_reference_top_k(scores, k=2)

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(reference_scores, [3.0, 3.0])


def test_lower_scores_stream_in_ascending_order_without_changing_values():
    scores = np.array(
        [
            [5.0, 2.0, 4.0, 1.0, 3.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [3.0, 3.0, 2.0, np.nan, 1.0],
        ],
        dtype=np.float32,
    )
    tracked_indices = np.array([0, 1, 4])
    tracked_scores = scores[:, tracked_indices]
    top_indices, top_scores = empty_top_k(
        len(scores),
        k=2,
        higher_is_better=False,
    )
    tracked_ranks = empty_tracked_ranks(len(scores), len(tracked_indices))

    for start, stop in ((0, 2), (2, 5)):
        top_indices, top_scores = merge_top_k(
            top_indices,
            top_scores,
            np.arange(start, stop),
            scores[:, start:stop],
            higher_is_better=False,
        )
        accumulate_tracked_ranks(
            tracked_ranks,
            scores[:, start:stop],
            tracked_scores,
            higher_is_better=False,
        )

    expected_ranks = rankdata(
        scores,
        axis=1,
        method="average",
        nan_policy="omit",
    )[:, tracked_indices]
    expected_indices = np.argsort(
        np.where(np.isfinite(scores), scores, np.inf),
        axis=1,
    )[:, :2]
    np.testing.assert_array_equal(top_indices, expected_indices)
    np.testing.assert_allclose(
        top_scores,
        np.take_along_axis(scores, expected_indices, axis=1),
    )
    np.testing.assert_allclose(tracked_ranks, expected_ranks)

    reference_indices, reference_scores = select_reference_top_k(
        scores,
        k=2,
        higher_is_better=False,
    )
    np.testing.assert_array_equal(reference_indices, [4, 3])
    np.testing.assert_allclose(reference_scores, [2.0, 2.5])


def test_zero_width_tracked_rank_accumulator_is_a_no_op():
    ranks = empty_tracked_ranks(3, 0)
    tracked_scores = np.empty((3, 0), dtype=np.float32)

    accumulate_tracked_ranks(
        ranks,
        np.ones((3, 2), dtype=np.float32),
        tracked_scores,
    )

    assert ranks.shape == (3, 0)
