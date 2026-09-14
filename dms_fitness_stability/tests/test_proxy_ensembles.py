import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import (
    AnalysisIdentity,
    ScoreEnsemble,
    analyze_score_ensemble,
)
from dms_fitness_stability.score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    HMC_COUNT_BASED_TARGET,
    LEGACY_PROXY_SCHEMA_VERSION,
)


def _legacy_bundle_parts() -> tuple[dict[str, object], pd.DataFrame, xr.Dataset]:
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-g3",
        inference_method="mle_predictive",
        model_spec="exponential",
        fit_id="mle-59",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "stop-control"],
            "is_ranking_candidate": [True, True, False],
            "reported_fitness": [0.6, 2.9, np.nan],
            "reported_rank": [2.0, 1.0, np.nan],
            "reference_fitness": [0.6, 2.9, np.nan],
            "reference_fitness_source": ["reported", "reported", "unavailable"],
        }
    )
    fitness_proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.array(
                    [
                        [
                            [0.5, 3.0, 1.0],
                            [0.7, 2.5, 1.2],
                            [0.6, 2.8, 0.9],
                        ]
                    ],
                    dtype=np.float32,
                ),
            ),
            "observed_proxy": (
                ("score_name", "variant"),
                np.array([[0.6, 2.9, 1.1]], dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["highest_gate_enrichment"],
            "sample": np.arange(3),
            "variant": np.arange(3),
        },
    )
    metadata = {
        **identity.metadata(),
        "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "score_contracts": {
            "highest_gate_enrichment": "nuc-g3-highest-gate-enrichment-v1"
        },
        "scores": ["highest_gate_enrichment"],
        "n_samples": 3,
        "n_variants": 3,
    }
    return metadata, variants, fitness_proxies


def _ensemble() -> ScoreEnsemble:
    metadata, variants, fitness_proxies = _legacy_bundle_parts()
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=fitness_proxies.rename(
            {
                "fitness_proxy": "fitness_score",
                "observed_proxy": "observed_score",
            }
        ),
    )


def _hmc_count_based_ensemble() -> ScoreEnsemble:
    ensemble = _ensemble()
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-g3",
        inference_method="hmc_count_based",
        model_spec="exponential",
        fit_id="small-clean_59",
    )
    ensemble.metadata.update(
        {
            **identity.metadata(),
            "fit_method": "hmc_count_based",
            "uncertainty_target": HMC_COUNT_BASED_TARGET,
            "n_chains": 2,
            "draws_per_chain": 2,
            "n_source_samples": 4,
        }
    )
    ensemble.scores = ensemble.scores.assign_coords(
        source_sample_id=("sample", np.array([0, 2, 3], dtype=np.int64)),
        source_chain=("sample", np.array([0, 1, 1], dtype=np.int32)),
        source_draw=("sample", np.array([0, 0, 1], dtype=np.int32)),
    )
    ensemble.validate()
    return ensemble


def _legacy_hmc_count_bundle_parts() -> tuple[
    dict[str, object],
    pd.DataFrame,
    xr.Dataset,
]:
    ensemble = _hmc_count_based_ensemble()
    metadata = {
        key: value
        for key, value in ensemble.metadata.items()
        if key not in {"fit_method", "n_analysis_variants"}
    }
    metadata["schema_version"] = LEGACY_PROXY_SCHEMA_VERSION
    fitness_proxies = ensemble.scores.rename(
        {
            "fitness_score": "fitness_proxy",
            "observed_score": "observed_proxy",
        }
    )
    return metadata, ensemble.variants, fitness_proxies


def test_score_ensemble_round_trips_standard_four_file_bundle(tmp_path):
    ensemble = _ensemble()

    output_dir = ensemble.save(tmp_path / "ensemble")
    loaded = ScoreEnsemble.load(output_dir)
    try:
        assert {path.name for path in output_dir.iterdir()} == {
            "metadata.json",
            "variants.csv",
            "analysis_variants.csv",
            "fitness_scores.nc",
        }
        assert loaded.metadata == ensemble.metadata
        pd.testing.assert_frame_equal(loaded.variants, ensemble.variants)
        pd.testing.assert_frame_equal(
            loaded.analysis_variants,
            ensemble.analysis_variants,
        )
        xr.testing.assert_equal(loaded.scores, ensemble.scores)
        assert loaded.fitness_score.dtype == np.float32
    finally:
        loaded.close()


def test_hmc_count_based_score_round_trip_preserves_chain_draw_coordinates(tmp_path):
    ensemble = _hmc_count_based_ensemble()

    output_dir = ensemble.save(tmp_path / "hmc-ensemble")
    loaded = ScoreEnsemble.load(output_dir)
    try:
        np.testing.assert_array_equal(loaded.scores["source_sample_id"], [0, 2, 3])
        np.testing.assert_array_equal(loaded.scores["source_chain"], [0, 1, 1])
        np.testing.assert_array_equal(loaded.scores["source_draw"], [0, 0, 1])
    finally:
        loaded.close()


def test_schema_v1_bundle_loads_through_standard_score_contract(
    tmp_path,
    write_schema_v1_bundle,
):
    metadata, variants, fitness_proxies = _legacy_bundle_parts()
    output_dir = write_schema_v1_bundle(
        tmp_path / "legacy",
        metadata=metadata,
        variants=variants,
        legacy_dataset=fitness_proxies,
    )

    loaded = ScoreEnsemble.load(output_dir)
    try:
        assert {path.name for path in output_dir.iterdir()} == {
            "metadata.json",
            "variants.csv",
            "fitness_proxies.nc",
        }
        assert loaded.source_schema_version == LEGACY_PROXY_SCHEMA_VERSION
        assert set(loaded.scores.data_vars) == {"fitness_score", "observed_score"}
        np.testing.assert_array_equal(
            loaded.analysis_variants["matrix_variant_index"],
            [0, 1, 2],
        )
        xr.testing.assert_equal(
            loaded.fitness_score.rename("fitness_proxy"),
            fitness_proxies["fitness_proxy"],
        )
    finally:
        loaded.close()


def test_schema_v1_hmc_count_bundle_preserves_chain_draw_coordinates(
    tmp_path,
    write_schema_v1_bundle,
):
    metadata, variants, fitness_proxies = _legacy_hmc_count_bundle_parts()
    output_dir = write_schema_v1_bundle(
        tmp_path / "legacy-hmc",
        metadata=metadata,
        variants=variants,
        legacy_dataset=fitness_proxies,
    )

    loaded = ScoreEnsemble.load(output_dir)
    try:
        np.testing.assert_array_equal(loaded.scores["source_sample_id"], [0, 2, 3])
        np.testing.assert_array_equal(loaded.scores["source_chain"], [0, 1, 1])
        np.testing.assert_array_equal(loaded.scores["source_draw"], [0, 0, 1])
    finally:
        loaded.close()


def test_score_ensemble_close_removes_builder_owned_temporary_files(tmp_path):
    ensemble = _ensemble()
    temporary_path = tmp_path / "score.dat"
    temporary_path.write_bytes(b"temporary")
    ensemble.temporary_paths = (temporary_path,)

    ensemble.close()

    assert not temporary_path.exists()


def test_score_ensemble_requires_contract_for_every_score():
    ensemble = _ensemble()
    ensemble.metadata["score_contracts"] = {}

    with pytest.raises(ValueError, match="map every score_name"):
        ensemble.validate()


def test_score_ensemble_rejects_variant_order_drift():
    ensemble = _ensemble()
    ensemble.scores = ensemble.scores.assign_coords(variant=[1, 0, 2])

    with pytest.raises(ValueError, match="variant coordinate"):
        ensemble.validate()


def test_analyze_schema_v1_score_uses_predictive_labels_and_candidate_subset(
    tmp_path,
    write_schema_v1_bundle,
):
    metadata, variants, fitness_proxies = _legacy_bundle_parts()
    output_dir = write_schema_v1_bundle(
        tmp_path / "legacy",
        metadata=metadata,
        variants=variants,
        legacy_dataset=fitness_proxies,
    )
    ensemble = ScoreEnsemble.load(output_dir)
    try:
        result = analyze_score_ensemble(
            ensemble,
            top_k=(1,),
            retain_k=1,
            tracked_top_n=2,
            tracked_model_top_n=1,
            variant_chunk_size=2,
            n_pairs=5,
            show_progress=False,
        )
    finally:
        ensemble.close()

    assert result.selection.attrs["fit_method"] == "mle_predictive"
    assert set(result.summary["fit_method"]) == {"mle_predictive"}
    assert set(result.overlaps["fit_method"]) == {"mle_predictive"}
    assert result.metadata["fit_method"] == "mle_predictive"
    assert result.metadata["proxy_ensemble_schema_version"] == 1
    assert result.metadata["schema_version"] == 2
    assert result.metadata["n_variants"] == 2
    assert result.metadata["n_fitted_variants"] == 3
    assert result.metadata["n_excluded_from_ranking"] == 1
    np.testing.assert_array_equal(
        result.selection["variant_index"].sel(
            score_name="highest_gate_enrichment"
        ),
        np.ones((3, 1), dtype=np.int32),
    )


def test_analyze_hmc_count_based_score_uses_source_sample_coordinates():
    result = analyze_score_ensemble(
        _hmc_count_based_ensemble(),
        top_k=(1,),
        retain_k=1,
        tracked_top_n=2,
        tracked_model_top_n=1,
        variant_chunk_size=2,
        n_pairs=5,
        show_progress=False,
    )

    assert result.selection.attrs["fit_method"] == "hmc_count_based"
    assert set(result.summary["fit_method"]) == {"hmc_count_based"}
    assert set(result.overlaps["fit_method"]) == {"hmc_count_based"}
    assert result.metadata["fit_method"] == "hmc_count_based"
    assert result.metadata["n_source_samples"] == 4
    assert result.metadata["n_score_samples"] == 3
    np.testing.assert_array_equal(result.selection["sample_id"], [0, 2, 3])
    np.testing.assert_array_equal(result.selection["source_chain"], [0, 1, 1])
    np.testing.assert_array_equal(result.selection["source_draw"], [0, 0, 1])


def test_analyze_score_ensemble_omits_non_identifiable_reference_top_k():
    ensemble = _ensemble()
    ensemble.metadata["reference_top_k_identifiable"] = False

    result = analyze_score_ensemble(
        ensemble,
        top_k=(1,),
        retain_k=1,
        tracked_top_n=0,
        tracked_model_top_n=1,
        variant_chunk_size=2,
        n_pairs=5,
        show_progress=False,
    )

    assert set(result.overlaps["comparison"]) == {"sample_vs_sample"}
    assert result.metadata["tracked_reported_top_n"] == 0
    assert result.metadata["reference_top_k_identifiable"] is False
    assert not result.summary["is_tracked_reported_cohort"].any()


def test_analyze_score_ensemble_excludes_draws_without_enough_finite_candidates():
    ensemble = _ensemble()
    ensemble.scores["fitness_score"][0, 1, :2] = np.nan
    ensemble.metadata["finite_ranking_candidates_per_sample"] = [2, 0, 2]

    result = analyze_score_ensemble(
        ensemble,
        top_k=(1,),
        retain_k=1,
        tracked_top_n=2,
        tracked_model_top_n=1,
        variant_chunk_size=2,
        n_pairs=2,
        show_progress=False,
    )

    np.testing.assert_array_equal(result.selection["sample_id"], [0, 2])
    assert result.metadata["n_source_samples"] == 3
    assert result.metadata["n_eligible_samples"] == 2
    assert result.metadata["n_excluded_samples"] == 1


def test_score_model_reference_requires_two_finite_samples():
    ensemble = _ensemble()
    ensemble.scores["fitness_score"][0, :, 2] = np.array(
        [np.nan, np.nan, 100.0],
        dtype=np.float32,
    )
    for variants in (ensemble.variants, ensemble.analysis_variants):
        variants.loc[2, "is_ranking_candidate"] = True
        variants.loc[2, "reported_fitness"] = 0.1
        variants.loc[2, "reported_rank"] = 3.0
    ensemble.metadata["finite_ranking_candidates_per_sample"] = [2, 2, 3]

    result = analyze_score_ensemble(
        ensemble,
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=3,
        tracked_model_top_n=1,
        variant_chunk_size=2,
        n_pairs=2,
        show_progress=False,
    )

    np.testing.assert_array_equal(
        result.selection["model_tracked_variant_index"].sel(
            score_name="highest_gate_enrichment"
        ),
        [1],
    )
    assert result.metadata["model_reference_min_finite_samples"] == 2
