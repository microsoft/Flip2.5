import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dms_fitness_stability.hmc_scanning import (
    _tracked_summary,
    result_tables,
    scan_hmc_scores,
)
from dms_fitness_stability.identity import AnalysisIdentity


def test_generic_hmc_scan_supports_multiple_scores_and_candidate_subset():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "C", "E"],
            "matrix_variant_index": np.array([0, 2, 4], dtype=np.int32),
            "reported_fitness": [0.1, 0.9, 0.5],
            "reported_rank": [3.0, 1.0, 2.0],
            "reference_fitness": [0.1, 0.9, 0.5],
            "reference_fitness_source": ["reported"] * 3,
        }
    )
    fitted_scores = {
        "latent": np.array(
            [
                [5.0, -1.0, 3.0, -1.0, 1.0],
                [1.0, -1.0, 5.0, -1.0, 3.0],
                [3.0, -1.0, 1.0, -1.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "proxy": np.array(
            [
                [1.0, -1.0, 3.0, -1.0, 5.0],
                [3.0, -1.0, 5.0, -1.0, 1.0],
                [5.0, -1.0, 1.0, -1.0, 3.0],
            ],
            dtype=np.float32,
        ),
    }

    def read_scores(*, variant_start, variant_stop, local_indices, **_):
        return {
            name: values[:, variant_start:variant_stop][:, local_indices]
            for name, values in fitted_scores.items()
        }

    identity = AnalysisIdentity(
        protein_family="synthetic",
        landscape="subset",
        inference_method="hmc",
        model_spec="test-model",
        fit_id="fit-1",
    )
    scan = scan_hmc_scores(
        variants,
        identity=identity,
        score_names=("latent", "proxy"),
        n_hmc_variants=5,
        n_chains=1,
        draws_per_chain=3,
        variant_chunk_size=2,
        read_scores=read_scores,
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=2,
        tracked_model_top_n=1,
        n_samples=None,
        seed=23,
        label="synthetic-subset",
        show_progress=False,
    )

    candidate_columns = np.array([0, 2, 4])
    tracked_indices = np.array([1, 2])
    for score_name, fitted in fitted_scores.items():
        candidates = fitted[:, candidate_columns]
        expected_order = np.argsort(-candidates, axis=1)[:, :2]
        expected_scores = np.take_along_axis(candidates, expected_order, axis=1)
        np.testing.assert_array_equal(
            scan.selection["variant_index"].sel(score_name=score_name),
            expected_order,
        )
        np.testing.assert_allclose(
            scan.selection["score"].sel(score_name=score_name),
            expected_scores,
        )
        np.testing.assert_allclose(
            scan.selection["tracked_rank"].sel(score_name=score_name),
            rankdata(-candidates, axis=1, method="average")[:, tracked_indices],
        )

    summary, overlaps = result_tables(
        scan.selection,
        variants,
        identity=identity,
        top_k=(1, 2),
        interval=(0.05, 0.95),
        n_pairs=5,
        seed=23,
    )
    assert set(summary["score"]) == {"latent", "proxy"}
    assert set(summary["k"]) == {1, 2}
    assert set(overlaps["comparison"]) == {
        "sample_vs_sample",
        "sample_vs_reference",
    }
    assert "higher_is_better_latent" not in scan.selection.attrs


def test_generic_hmc_scan_excludes_ineligible_fitted_rows_from_all_ranks():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "stop-control", "C", "D"],
            "hmc_variant_index": np.array([0, 1, 3, 4], dtype=np.int32),
            "is_ranking_candidate": [True, False, True, True],
            "reported_fitness": [0.9, 99.0, 0.5, 0.1],
            "reported_rank": [1.0, 1.0, 2.0, 3.0],
            "reference_fitness": [0.9, 99.0, 0.5, 0.1],
            "reference_fitness_source": ["reported"] * 4,
        }
    )
    fitted = np.array(
        [
            [5.0, 100.0, -1.0, 3.0, 1.0],
            [1.0, 100.0, -1.0, 5.0, 3.0],
            [3.0, 100.0, -1.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )

    def read_scores(*, variant_start, variant_stop, local_indices, **_):
        return {
            "latent": fitted[:, variant_start:variant_stop][:, local_indices]
        }

    scan = scan_hmc_scores(
        variants,
        identity=AnalysisIdentity(
            protein_family="synthetic",
            landscape="excluded-control",
            inference_method="hmc",
            model_spec="test-model",
            fit_id="fit-1",
        ),
        score_names=("latent",),
        n_hmc_variants=5,
        n_chains=1,
        draws_per_chain=3,
        variant_chunk_size=2,
        read_scores=read_scores,
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=2,
        tracked_model_top_n=2,
        n_samples=None,
        seed=23,
        label="synthetic-control",
        show_progress=False,
    )

    candidate_rows = np.array([0, 2, 3], dtype=np.int32)
    candidate_scores = fitted[:, [0, 3, 4]]
    candidate_order = np.argsort(-candidate_scores, axis=1)[:, :2]
    expected_rows = candidate_rows[candidate_order]
    np.testing.assert_array_equal(scan.candidate_indices, candidate_rows)
    np.testing.assert_array_equal(
        scan.selection["variant_index"].sel(score_name="latent"),
        expected_rows,
    )
    np.testing.assert_allclose(
        scan.selection["tracked_rank"].sel(score_name="latent"),
        rankdata(-candidate_scores, axis=1, method="average")[:, :2],
    )
    assert 1 not in scan.selection["variant_index"].values
    assert 1 not in scan.selection["model_reference_variant_index"].values
    assert 1 not in scan.reported_indices


def test_generic_scan_supports_lower_scores_without_external_reference():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "reported_fitness": [np.nan, np.nan, np.nan],
            "reported_rank": [np.nan, np.nan, np.nan],
        }
    )
    scores = np.array(
        [
            [3.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )

    def read_scores(*, variant_start, variant_stop, local_indices, **_):
        return {
            "measurement_error": scores[
                :, variant_start:variant_stop
            ][:, local_indices]
        }

    scan = scan_hmc_scores(
        variants,
        identity=AnalysisIdentity(
            protein_family="unknown",
            landscape="synthetic-errors",
            inference_method="simulation",
            model_spec="benchmark",
            fit_id="seed-23",
        ),
        score_names=("measurement_error",),
        higher_is_better={"measurement_error": False},
        n_hmc_variants=3,
        n_chains=1,
        draws_per_chain=3,
        variant_chunk_size=2,
        read_scores=read_scores,
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=0,
        tracked_model_top_n=1,
        n_samples=None,
        seed=23,
        label="synthetic-errors",
        require_reported_top_k=False,
        show_progress=False,
    )

    np.testing.assert_array_equal(
        scan.selection["variant_index"].sel(score_name="measurement_error"),
        [[1, 2], [0, 1], [1, 0]],
    )
    np.testing.assert_allclose(
        scan.selection["score"].sel(score_name="measurement_error"),
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
    )
    np.testing.assert_array_equal(
        scan.selection["model_reference_variant_index"].sel(
            score_name="measurement_error"
        ),
        [1, 0],
    )
    assert scan.selection.attrs["higher_is_better_measurement_error"] == "false"
    assert scan.reported_indices.size == 0


def test_generic_scan_supports_lower_external_reference():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "reported_fitness": [3.0, 1.0, 2.0],
            "reported_rank": [3.0, 1.0, 2.0],
        }
    )
    scores = np.array(
        [
            [3.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )

    def read_scores(*, variant_start, variant_stop, local_indices, **_):
        return {"error": scores[:, variant_start:variant_stop][:, local_indices]}

    identity = AnalysisIdentity(
        protein_family="unknown",
        landscape="synthetic-errors",
        inference_method="simulation",
        model_spec="benchmark",
        fit_id="seed-23",
    )
    scan = scan_hmc_scores(
        variants,
        identity=identity,
        score_names=("error",),
        higher_is_better={"error": False},
        n_hmc_variants=3,
        n_chains=1,
        draws_per_chain=3,
        variant_chunk_size=2,
        read_scores=read_scores,
        top_k=(1,),
        retain_k=1,
        tracked_top_n=3,
        tracked_model_top_n=1,
        n_samples=None,
        seed=23,
        label="synthetic-errors",
        reported_higher_is_better=False,
        show_progress=False,
    )
    summary, overlaps = result_tables(
        scan.selection,
        variants,
        identity=identity,
        top_k=(1,),
        interval=(0.05, 0.95),
        n_pairs=2,
        seed=23,
        reference_higher_is_better=False,
    )

    np.testing.assert_array_equal(
        scan.selection["tracked_variant_index"],
        [1, 2, 0],
    )
    reference = overlaps[overlaps["comparison"].eq("sample_vs_reference")]
    np.testing.assert_array_equal(reference["intersection"], [1, 0, 1])
    assert set(summary["score"]) == {"error"}


def test_tracked_summary_handles_sparse_columns_without_warnings():
    variants = pd.DataFrame({"reported_rank": [1.0, 2.0, 3.0]})
    values = np.array(
        [[np.nan, 1.0, 1.0], [np.nan, np.nan, 2.0], [np.nan, np.nan, np.nan]]
    )
    ranks = np.array(
        [[np.nan, 2.0, 2.0], [np.nan, np.nan, 1.0], [np.nan, np.nan, np.nan]]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = _tracked_summary(
            variants,
            np.arange(3, dtype=np.int32),
            values,
            ranks,
            k=1,
            interval=(0.05, 0.95),
        )

    assert summary["n_finite_samples"].tolist() == [0, 1, 2]
    assert np.isnan(summary.loc[0, "posterior_value_mean"])
    assert np.isnan(summary.loc[1, "posterior_value_sd"])
    assert summary.loc[2, "posterior_value_mean"] == 1.5
    assert summary.loc[2, "posterior_top_k_probability"] == 0.5
