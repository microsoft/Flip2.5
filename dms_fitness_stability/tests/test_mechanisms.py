import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import rankdata

from dms_fitness_stability.adapters.k50 import K50Landscape, K50Library
from dms_fitness_stability.adapters.nuclease import NucleaseLandscape
from dms_fitness_stability.adapters.trpb import TrpBLandscape
from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.mechanism_reporting import (
    mechanism_candidate_indices,
    score_sample_positions,
)
from dms_fitness_stability.mechanisms import (
    boundary_difficulty,
    count_within_score_window,
    grouped_output_disagreement,
    local_score_spacing,
    paired_replicate_features,
    partial_spearman,
    scan_fixed_cohort,
    scan_proxy_fixed_cohort,
    score_uncertainty,
    standardized_regression,
)
from dms_fitness_stability.score_ensembles import ScoreEnsemble


def test_paired_replicate_features_are_depth_normalized_and_count_aware():
    input_counts = np.array([[100, 100], [200, 200]], dtype=float)
    concordant_output = np.array([[200, 100], [400, 200]], dtype=float)
    discordant_output = np.array([[200, 100], [200, 400]], dtype=float)

    concordant = paired_replicate_features(
        input_counts,
        concordant_output,
        pseudocount=0,
    )
    discordant = paired_replicate_features(
        input_counts,
        discordant_output,
        pseudocount=0,
    )

    np.testing.assert_allclose(
        concordant["log_enrichment"][0],
        concordant["log_enrichment"][1],
    )
    np.testing.assert_allclose(concordant["reduced_heterogeneity"], 0, atol=1e-12)
    assert np.all(
        discordant["reduced_heterogeneity"]
        > concordant["reduced_heterogeneity"]
    )
    np.testing.assert_allclose(concordant["log_input_support"], np.log1p([300, 300]))


def test_grouped_output_disagreement_removes_timepoint_means():
    # Two timepoints, two output replicates, two variants. Replicates agree at
    # each timepoint even though the timepoint means differ strongly.
    concordant = np.array(
        [
            [[100, 50], [200, 100]],
            [[400, 100], [800, 200]],
        ],
        dtype=float,
    )
    discordant = concordant.copy()
    discordant[1, 1] = [100, 800]

    first = grouped_output_disagreement(concordant, pseudocount=0)
    second = grouped_output_disagreement(discordant, pseudocount=0)

    np.testing.assert_allclose(first["reduced_heterogeneity"], 0, atol=1e-12)
    assert np.all(second["reduced_heterogeneity"] > first["reduced_heterogeneity"])
    assert first["observation_count"].tolist() == [4, 4]


def test_score_uncertainty_uses_fixed_reference_scale():
    samples = np.array(
        [
            [8.0, 5.0],
            [10.0, 6.0],
            [12.0, 7.0],
            [14.0, 8.0],
        ]
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
    # The IQR of [12, 10, 8, 6, 4] is 4.
    np.testing.assert_allclose(
        features["normalized_score_interval_width"],
        [0.75, 0.375],
    )
    assert features["reference_score_iqr"] == 4.0


def test_score_uncertainty_accepts_lower_is_better_reference_order():
    samples = np.array(
        [
            [8.0, 5.0],
            [10.0, 6.0],
            [12.0, 7.0],
            [14.0, 8.0],
        ]
    )
    reference_scores = np.array([4.0, 6.0, 8.0, 10.0, 12.0])

    features = score_uncertainty(
        samples,
        reference_scores=reference_scores,
        interval=(0.25, 0.75),
        higher_is_better=False,
    )

    np.testing.assert_allclose(features["score_interval_width"], [3.0, 1.5])
    np.testing.assert_allclose(
        features["normalized_score_interval_width"],
        [0.75, 0.375],
    )
    assert features["reference_score_iqr"] == 4.0


def test_local_spacing_and_window_crowding_use_only_median_scores():
    reference_scores = np.array([10.0, 9.0, 8.0, 6.0, 2.0])

    spacing = local_score_spacing(reference_scores, neighborhood=1)
    within_window = count_within_score_window(reference_scores, window_fraction=0.25)

    np.testing.assert_allclose(spacing["score_spacing"], [1.0, 1.0, 1.5, 3.0, 4.0])
    # Reference-score IQR is 3, so the window is 0.75 and no other score enters it.
    np.testing.assert_array_equal(within_window["competitor_count"], 0)
    assert within_window["score_window"] == pytest.approx(0.75)


def test_boundary_difficulty_compares_uncertainty_with_cutoff_spacing():
    reference_scores = np.array([10.0, 8.0, 5.0, 4.0])
    top_score_samples = np.array(
        [
            [10.0, 8.0],
            [12.0, 9.0],
            [14.0, 10.0],
            [16.0, 11.0],
        ]
    )

    features = boundary_difficulty(
        reference_scores,
        top_score_samples,
        sample_self_consistency=0.7,
        k=2,
        interval=(0.25, 0.75),
        boundary_window=2,
    )

    assert features["boundary_score_gap"] == 3.0
    assert features["boundary_local_score_spacing"] == 2.0
    assert features["top_k_instability"] == pytest.approx(0.3)
    assert features["boundary_score_uncertainty"] == pytest.approx(2.25)
    assert features["uncertainty_to_spacing_ratio"] == pytest.approx(1.125)


def test_mechanism_features_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="same two-dimensional shape"):
        paired_replicate_features(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="at least two replicates"):
        grouped_output_disagreement(np.ones((2, 1, 3)))


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_fixed_cohort_scan_matches_dense_sample_and_reference_ranks(transform):
    scores = np.array(
        [
            [5.0, 2.0, 4.0, 1.0, 3.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [3.0, 3.0, 2.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    posterior = scores[None, :, :]
    candidate_hmc_indices = np.arange(scores.shape[1], dtype=np.int64)
    cohort_indices = np.array([3, 0, 4], dtype=np.int32)

    tracked = scan_fixed_cohort(
        posterior,
        sample_ids=np.arange(len(scores)),
        hmc_variant_indices=candidate_hmc_indices,
        candidate_variant_indices=np.arange(scores.shape[1], dtype=np.int32),
        cohort_variant_indices=cohort_indices,
        transform=transform,
        variant_chunk_size=2,
    )

    np.testing.assert_allclose(tracked["score"], scores[:, cohort_indices])
    np.testing.assert_allclose(
        tracked["rank"],
        rankdata(-scores, axis=1, method="average")[:, cohort_indices],
    )
    median_scores = np.median(scores, axis=0)
    np.testing.assert_allclose(
        tracked["reference_score"],
        median_scores[cohort_indices],
    )
    expected_order = np.lexsort((np.arange(scores.shape[1]), -median_scores))
    expected_rank = np.empty(scores.shape[1], dtype=np.int32)
    expected_rank[expected_order] = np.arange(1, scores.shape[1] + 1)
    np.testing.assert_array_equal(
        tracked["reference_rank"],
        expected_rank[cohort_indices],
    )
    np.testing.assert_allclose(tracked["candidate_reference_score"], median_scores)
    np.testing.assert_array_equal(
        tracked["candidate_variant_index"],
        np.arange(scores.shape[1]),
    )
    np.testing.assert_array_equal(tracked["reference_order"], expected_order)


def test_fixed_cohort_scan_excludes_high_scoring_control_from_reference_and_ranks():
    scores = np.array(
        [
            [5.0, 100.0, 3.0, 1.0],
            [1.0, 100.0, 5.0, 3.0],
            [3.0, 100.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )
    candidates = np.array([0, 2, 3], dtype=np.int32)
    cohort = np.array([0, 3], dtype=np.int32)

    tracked = scan_fixed_cohort(
        scores[None, :, :],
        sample_ids=np.arange(len(scores)),
        hmc_variant_indices=np.arange(scores.shape[1], dtype=np.int64),
        candidate_variant_indices=candidates,
        cohort_variant_indices=cohort,
        transform="identity",
        variant_chunk_size=2,
    )

    candidate_scores = scores[:, candidates]
    medians = np.median(candidate_scores, axis=0)
    order_positions = np.lexsort((candidates, -medians))
    np.testing.assert_allclose(
        tracked["rank"],
        rankdata(-candidate_scores, axis=1, method="average")[:, [0, 2]],
    )
    np.testing.assert_array_equal(tracked["candidate_variant_index"], candidates)
    np.testing.assert_allclose(tracked["candidate_reference_score"], medians)
    np.testing.assert_array_equal(tracked["reference_order"], candidates[order_positions])
    assert 1 not in tracked["reference_order"]


def test_proxy_fixed_cohort_scan_matches_hmc_reader():
    scores = np.array(
        [
            [5.0, 2.0, 4.0, 1.0, 3.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [3.0, 3.0, 2.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidates = np.arange(scores.shape[1], dtype=np.int32)
    cohort = np.array([3, 0, 4], dtype=np.int32)
    common = {
        "sample_ids": np.arange(len(scores)),
        "candidate_variant_indices": candidates,
        "cohort_variant_indices": cohort,
        "variant_chunk_size": 2,
    }

    hmc = scan_fixed_cohort(
        scores[None],
        hmc_variant_indices=np.arange(scores.shape[1]),
        transform="identity",
        **common,
    )
    proxy = scan_proxy_fixed_cohort(
        scores,
        proxy_variant_indices=np.arange(scores.shape[1]),
        **common,
    )

    assert set(proxy) == set(hmc)
    for name in hmc:
        np.testing.assert_array_equal(proxy[name], hmc[name])
    np.testing.assert_array_equal(proxy["n_finite_samples"], len(scores))


def test_proxy_fixed_cohort_scan_supports_lower_scores():
    scores = np.array(
        [
            [5.0, 2.0, 4.0, 1.0, 3.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [3.0, 3.0, 2.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    cohort = np.array([3, 0, 4], dtype=np.int32)

    tracked = scan_proxy_fixed_cohort(
        scores,
        sample_ids=np.arange(len(scores)),
        proxy_variant_indices=np.arange(scores.shape[1]),
        candidate_variant_indices=np.arange(scores.shape[1], dtype=np.int32),
        cohort_variant_indices=cohort,
        variant_chunk_size=2,
        higher_is_better=False,
    )

    np.testing.assert_allclose(tracked["score"], scores[:, cohort])
    np.testing.assert_allclose(
        tracked["rank"],
        rankdata(scores, axis=1, method="average")[:, cohort],
    )
    median_scores = np.median(scores, axis=0)
    expected_order = np.lexsort((np.arange(scores.shape[1]), median_scores))
    expected_rank = np.empty(scores.shape[1], dtype=np.int32)
    expected_rank[expected_order] = np.arange(1, scores.shape[1] + 1)
    np.testing.assert_array_equal(
        tracked["reference_order"],
        expected_order,
    )
    np.testing.assert_array_equal(
        tracked["reference_rank"],
        expected_rank[cohort],
    )


def test_proxy_fixed_cohort_scan_tracks_partially_undefined_scores():
    scores = np.array(
        [
            [5.0, 4.0, 3.0, np.nan],
            [4.0, np.nan, 5.0, np.nan],
            [3.0, 5.0, 4.0, np.nan],
        ],
        dtype=np.float32,
    )
    tracked = scan_proxy_fixed_cohort(
        scores,
        sample_ids=np.arange(len(scores)),
        proxy_variant_indices=np.arange(scores.shape[1]),
        candidate_variant_indices=np.arange(scores.shape[1], dtype=np.int32),
        cohort_variant_indices=np.array([1, 2], dtype=np.int32),
        variant_chunk_size=2,
    )

    np.testing.assert_array_equal(tracked["n_finite_samples"], [2, 3])
    np.testing.assert_array_equal(
        tracked["candidate_n_finite_samples"],
        [3, 2, 3, 0],
    )
    assert np.isnan(tracked["rank"][1, 0])
    assert np.isnan(tracked["candidate_reference_score"][-1])
    assert tracked["reference_order"][-1] == 3
    np.testing.assert_allclose(tracked["reference_score"], [4.5, 4.0])
    uncertainty = score_uncertainty(
        tracked["score"],
        reference_scores=np.array([5.0, 4.0, 3.0]),
    )
    np.testing.assert_array_equal(uncertainty["n_finite_samples"], [2, 3])
    assert np.isfinite(uncertainty["score_interval_width"]).all()


def test_proxy_fixed_cohort_excludes_singleton_from_reference_order():
    scores = np.array(
        [
            [3.0, 2.0, np.nan],
            [2.0, 3.0, np.nan],
            [3.0, 2.0, 100.0],
        ],
        dtype=np.float32,
    )

    tracked = scan_proxy_fixed_cohort(
        scores,
        sample_ids=np.arange(3, dtype=np.int64),
        proxy_variant_indices=np.arange(3, dtype=np.int64),
        candidate_variant_indices=np.arange(3, dtype=np.int32),
        cohort_variant_indices=np.array([0, 1], dtype=np.int32),
        variant_chunk_size=2,
    )

    np.testing.assert_array_equal(tracked["reference_order"], [0, 1, 2])
    assert np.isnan(tracked["candidate_reference_score"][2])
    assert tracked["candidate_n_finite_samples"][2] == 1

    direct_hmc = scan_proxy_fixed_cohort(
        scores,
        sample_ids=np.arange(3, dtype=np.int64),
        proxy_variant_indices=np.arange(3, dtype=np.int64),
        candidate_variant_indices=np.arange(3, dtype=np.int32),
        cohort_variant_indices=np.array([0, 1], dtype=np.int32),
        variant_chunk_size=2,
        min_reference_samples=1,
    )

    np.testing.assert_array_equal(direct_hmc["reference_order"], [2, 0, 1])
    assert direct_hmc["candidate_reference_score"][2] == 100


def test_mechanism_candidates_follow_result_candidate_count():
    variants = pd.DataFrame(
        {
            "variant_id": ["AAA", "AAC", "AAD"],
            "is_ranking_candidate": [True, False, True],
        }
    )

    full_axis = mechanism_candidate_indices(
        {"n_variants": 3},
        variants,
    )
    ranking_axis = mechanism_candidate_indices(
        {"n_variants": 2},
        variants,
    )

    np.testing.assert_array_equal(full_axis, [0, 1, 2])
    np.testing.assert_array_equal(ranking_axis, [0, 2])


def test_supporting_audits_exclude_controls_and_report_reference_semantics():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary_audits", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G1",
        inference_method="hmc",
        model_spec="exponential",
        fit_id="fit-1",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["", "stop-control", "A", "B", "C"],
            "is_wild_type": [True, False, False, False, False],
            "is_source_control": [False, True, False, False, False],
            "is_ranking_candidate": [True, False, True, True, True],
            "reported_fitness": [4.0, 99.0, 3.0, 2.0, 1.0],
            "reported_rank": [1.0, 0.0, 2.0, 3.0, 4.0],
            "reported_activity_level": [
                "activity_greater_than_WT",
                np.nan,
                "activity_greater_than_WT",
                "activity_greater_than_0",
                "non-functional",
            ],
        }
    )
    cache = xr.Dataset(
        {
            "candidate_variant_index": ("candidate", [0, 2, 3, 4]),
            "candidate_reference_score": ("candidate", [4.0, 3.0, 2.0, 1.0]),
            "reference_order": ("reference_position", [0, 2, 3, 4]),
        }
    )

    diagnostic, model_top = summary._reference_diagnostic(
        identity,
        {"reference_fitness_source": "highest-gate EF"},
        variants,
        cache,
        top_k=2,
    )
    categories = pd.DataFrame(
        summary._reference_category_rows(
            identity,
            variants,
            cache,
            top_k=2,
        )
    )

    assert diagnostic["n_fitted_variants"] == 5
    assert diagnostic["n_ranking_candidates"] == 4
    assert diagnostic["n_excluded_from_ranking"] == 1
    assert diagnostic["all_finite_spearman"] == pytest.approx(1.0)
    assert diagnostic["model_reported_top_k_intersection"] == 2
    assert diagnostic["model_reversed_reported_top_k_intersection"] == 0
    assert diagnostic["wild_type_model_rank"] == 1
    assert model_top == {0, 2}
    assert categories.groupby("cohort")["count"].sum().eq(2).all()
    assert "unavailable" not in set(categories["category"])

    margin = np.array([1, 3, 2, 5, 4, 8, 6, 7], dtype=float)
    uncertainty = np.array([8, 6, 7, 3, 5, 1, 4, 2], dtype=float)
    mechanism_rows = pd.DataFrame(
        {
            "analysis_id": identity.analysis_id,
            "protein_family": identity.protein_family,
            "landscape": identity.landscape,
            "model_spec": identity.model_spec,
            "is_primary_cohort": True,
            "is_model_top_100": [True] * 5 + [False] * 3,
            "is_challenger_q_ge_001": [False] * 5 + [True] * 3,
            "mechanism_input_count": np.arange(1, 9),
            "mechanism_output_count": np.arange(2, 10),
            "normalized_score_interval_width": uncertainty,
            "score_margin_to_rank_100": margin,
            "top_100_inclusion_probability": margin / 10,
            "is_source_control": False,
        }
    )
    mechanism_diagnostics = pd.DataFrame(
        summary._mechanism_diagnostic_rows(mechanism_rows)
    ).set_index("stratum")

    assert set(mechanism_diagnostics.index) == {
        "primary",
        "model_top_k",
        "challengers",
    }
    assert mechanism_diagnostics.loc["primary", "n_variants"] == 8
    assert np.isfinite(
        mechanism_diagnostics.loc[
            "primary",
            "decision_margin_vs_score_uncertainty_partial_spearman_input_adjusted",
        ]
    )
    assert mechanism_diagnostics["source_control_rows"].eq(0).all()


def test_standardized_regression_recovers_known_effects():
    first = np.arange(8, dtype=float)
    second = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    response = 3 * first - 2 * second

    result = standardized_regression(
        response,
        {"first": first, "second": second},
    )

    assert result["n_observations"] == 8
    assert result["r_squared"] == pytest.approx(1.0)
    # Standardized coefficients retain the signs of the generating effects.
    assert result["coefficient_first"] > 0
    assert result["coefficient_second"] < 0


def test_partial_spearman_matches_explicit_rank_residualization():
    first = np.array([1, 3, 2, 5, 4, 8, 6, 7], dtype=float)
    second = np.array([8, 6, 7, 3, 5, 1, 4, 2], dtype=float)
    control = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    design = np.column_stack(
        [np.ones(len(first)), rankdata(control, method="average")]
    )
    first_rank = rankdata(first, method="average")
    second_rank = rankdata(second, method="average")
    first_residual = first_rank - design @ np.linalg.lstsq(
        design,
        first_rank,
        rcond=None,
    )[0]
    second_residual = second_rank - design @ np.linalg.lstsq(
        design,
        second_rank,
        rcond=None,
    )[0]
    expected = np.corrcoef(first_residual, second_residual)[0, 1]

    observed = partial_spearman(first, second, {"control": control})

    assert observed == pytest.approx(expected)


def test_mechanism_cohort_unions_model_challenger_and_reported_sets():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    variants = pd.DataFrame(
        {
            "reported_rank": [4.0, 1.0, 5.0, 2.0, 3.0],
        }
    )
    sample_top_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 4],
        ],
        dtype=np.int32,
    )
    model_reference = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    cohort, probability, strongest, omitted = summary._cohort_indices(
        variants,
        sample_top_indices,
        model_reference,
        k=2,
        challenger_probability=0.5,
        reported_top_n=3,
        reported_supported_indices=np.array([0, 1, 2, 3]),
    )

    np.testing.assert_allclose(probability, [0.75, 0.75, 0.5, 0.0, 0.0])
    # Model top 2: {0,1}; challenger: {2}; supported reported context: {1,3}.
    np.testing.assert_array_equal(cohort, [0, 1, 2, 3])
    np.testing.assert_array_equal(strongest, [2])
    assert omitted == 1


def test_nuclease_mechanism_features_pool_counts_without_replicate_discordance():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary_nuc", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "A3D"],
            "hmc_variant_index": [0, 1, 2],
            "input_count": [10, 20, 0],
            "output_count": [30, 0, 5],
        }
    )
    landscape = NucleaseLandscape(
        generation="G3",
        variants=variants,
        counts={
            "ic1": np.array([10, 20, 0]),
            "oc1": np.array([[10, 0, 1], [8, 0, 1], [7, 0, 1], [5, 0, 2]]),
        },
        thresholds={"ft": np.array([0.1, 0.2, 0.3, 0.4])},
        wild_type_index=0,
    )
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G3",
        inference_method="hmc",
        model_spec="exponential",
        fit_id="small-clean_59",
    )

    features, inventory = summary._landscape_count_features(identity, landscape)

    np.testing.assert_array_equal(features["input_count"], [10, 20, 0])
    np.testing.assert_array_equal(features["output_count"], [30, 0, 5])
    np.testing.assert_allclose(features["log_input_support"], np.log1p([10, 20, 0]))
    assert np.isnan(features["reduced_heterogeneity"]).all()
    assert np.isnan(features["enrichment_range"]).all()
    assert inventory["discordance_available"] is False
    assert inventory["replicate_count"] == 1
    assert inventory["observation_structure"] == (
        "pooled_generation_counts_no_replicate_contrast"
    )


def test_k50_mechanism_features_use_matching_protease_counts():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary_k50", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    starting = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[100, 200, 300], [400, 500, 600]],
        ]
    )
    timepoint = np.zeros((2, 2, 11, 3), dtype=np.int64)
    timepoint[0, 0] = np.array([1, 2, 3])
    timepoint[0, 1] = np.array([0, 1, 0])
    timepoint[1] = 1_000
    library = K50Library(
        name="v1",
        global_sequence_indices=np.arange(3),
        variants=("AAA", "AAC", "AAD"),
        counts_c0=starting,
        counts_cg0=timepoint,
    )
    landscape = K50Landscape(
        variants=pd.DataFrame({"variant_id": library.variants}),
        libraries={"v1": library},
        ordinal_sequences=np.zeros((3, 86), dtype=np.int64),
        scrambled_indices=np.empty(0, dtype=np.int64),
        qpcr_log_protease_concentration=np.zeros(11),
        qpcr_log2_survival=np.zeros((2, 11, 8)),
        log_expected_protease_concentration=np.zeros((2, 11)),
    )
    identity = AnalysisIdentity(
        protein_family="k50",
        landscape="k50-chymotrypsin",
        inference_method="mle_predictive",
        model_spec="(MASSIVE, only MLE)",
        fit_id="k50_mle_lr0-01",
    )

    features, inventory = summary._landscape_count_features(
        identity,
        landscape,
        score_name="mean_log_chymotrypsin_enrichment",
    )

    np.testing.assert_array_equal(features["input_count"], [5, 7, 9])
    np.testing.assert_array_equal(features["output_count"], [11, 33, 33])
    np.testing.assert_array_equal(features["observation_count"], [22, 22, 22])
    assert inventory["protease"] == "chymotrypsin"
    assert inventory["landscape"] == "k50-chymotrypsin"
    assert inventory["replicate_count"] == 2
    assert inventory["group_count"] == 11
    assert inventory["discordance_available"] is False


def test_four_site_mechanism_features_pool_inputs_and_compare_outputs():
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary_trpb", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    landscape = TrpBLandscape(
        library="four-site",
        variants=pd.DataFrame(
            {"variant_id": ["AAA", "AAC", "AAD"], "hmc_variant_index": [0, 1, 2]}
        ),
        starting_counts=np.array([[10, 20, 30], [20, 30, 40]]),
        timepoint_counts=np.array(
            [
                [[10, 20, 30], [20, 40, 60]],
                [[20, 30, 40], [40, 60, 80]],
            ]
        ),
        times=np.array([16, 20]),
    )
    identity = AnalysisIdentity(
        protein_family="trpb",
        landscape="trpb-four-site",
        inference_method="mle_predictive",
        model_spec="exponential-exponential",
        fit_id="mle_64",
    )

    features, inventory = summary._landscape_count_features(identity, landscape)

    np.testing.assert_array_equal(features["input_count"], [30, 50, 70])
    np.testing.assert_allclose(features["log_input_support"], np.log1p([30, 50, 70]))
    assert len(features["reduced_heterogeneity"]) == 3
    assert inventory["observation_structure"] == (
        "time_series_output_replicates_two_inputs"
    )
    assert inventory["input_replicate_count"] == 2


def test_hmc_count_mechanisms_map_source_sample_ids(tmp_path):
    script_path = (
        Path(__file__).parents[1] / "examples" / "paper" / "summarize_mechanisms.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_summary_hmc_count", script_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    manifest_path = tmp_path / "analyses.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "analyses": [
                    {
                        "protein_family": "nuc",
                        "landscape": "nuc-G3",
                        "inference_method": "hmc_count_based",
                        "model_spec": "exponential",
                        "fit_id": "small-clean_59",
                        "job_id": 59,
                        "artifact_kind": "diagnosed_hmc",
                        "primary_score": "highest_gate_enrichment",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = summary._manifest(manifest_path)

    assert len(manifest) == 1
    assert manifest[0].identity.inference_method == "hmc_count_based"
    ensemble = ScoreEnsemble(
        metadata={},
        variants=pd.DataFrame(),
        analysis_variants=pd.DataFrame(),
        scores=xr.Dataset(
            coords={
                "sample": np.arange(3),
                "source_sample_id": ("sample", [1, 3, 7]),
            }
        ),
    )
    np.testing.assert_array_equal(
        score_sample_positions(ensemble, np.array([7, 1])),
        [2, 0],
    )
    with pytest.raises(ValueError, match="absent from the score bundle"):
        score_sample_positions(ensemble, np.array([2]))
