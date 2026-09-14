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
    LEGACY_PROXY_SCHEMA_VERSION,
    POSTERIOR_PARAMETER_TARGET,
    SCORE_ENSEMBLE_SCHEMA_VERSION,
    direct_hmc_reference_metadata,
)
from dms_fitness_stability.score_paths import score_bundle_root, score_collection_root


def _direct_hmc_ensemble() -> ScoreEnsemble:
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape="pdz-cis",
        inference_method="hmc",
        model_spec="exponential-exponential",
        fit_id="small-clean_1",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["AA", "AB", "AC"],
            "hmc_variant_index": [0, 1, 2],
            "is_ranking_candidate": [True, True, True],
            "reported_fitness": [0.1, 0.2, 0.3],
            "reported_rank": [3.0, 2.0, 1.0],
        }
    )
    analysis_variants = variants.iloc[[0, 2]].copy().reset_index(drop=True)
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        np.array([0, 2], dtype=np.int64),
    )
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                np.array(
                    [
                        [
                            [0.1, 0.2, 0.3],
                            [0.2, 0.3, 0.4],
                            [0.3, 0.4, 0.5],
                            [0.4, 0.5, 0.6],
                        ]
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "score_name": ["log_mean_growth"],
            "sample": np.arange(4, dtype=np.int32),
            "variant": np.arange(3, dtype=np.int32),
            "source_sample_id": (
                "sample",
                np.array([0, 1, 2, 3], dtype=np.int64),
            ),
            "source_chain": ("sample", np.array([0, 0, 1, 1], dtype=np.int32)),
            "source_draw": ("sample", np.array([0, 1, 0, 1], dtype=np.int32)),
        },
    )
    metadata = {
        **identity.metadata(),
        "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
        "fit_method": "hmc_posterior",
        "uncertainty_target": POSTERIOR_PARAMETER_TARGET,
        "score_contracts": {"log_mean_growth": "pdz-log-r-identity-v1"},
        "scores": ["log_mean_growth"],
        "n_samples": 4,
        "n_variants": 3,
        "n_analysis_variants": 2,
        "n_chains": 2,
        "draws_per_chain": 2,
        "sample_view": "full",
            "variant_chunk_size": 2,
    }
    return ScoreEnsemble(
        metadata=metadata,
        variants=variants,
        analysis_variants=analysis_variants,
        scores=scores,
    )


def _legacy_bundle_parts() -> tuple[dict[str, object], pd.DataFrame, xr.Dataset]:
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape="pdz-cis",
        inference_method="mle_predictive",
        model_spec="exponential-exponential",
        fit_id="mle_1",
    )
    variants = pd.DataFrame({"variant_id": ["AA", "AB"]})
    proxies = xr.Dataset(
        {
            "fitness_proxy": (
                ("score_name", "sample", "variant"),
                np.array([[[0.1, 0.2], [0.2, 0.3]]], dtype=np.float32),
            ),
            "observed_proxy": (
                ("score_name", "variant"),
                np.array([[0.15, 0.25]], dtype=np.float32),
            ),
        },
        coords={
            "score_name": ["dimsum_fitness"],
            "sample": np.arange(2),
            "variant": np.arange(2),
        },
    )
    metadata = {
        **identity.metadata(),
        "schema_version": LEGACY_PROXY_SCHEMA_VERSION,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "score_contracts": {"dimsum_fitness": "pdz-dimsum-v1"},
        "scores": ["dimsum_fitness"],
        "n_samples": 2,
        "n_variants": 2,
    }
    return metadata, variants, proxies


def test_direct_hmc_score_ensemble_round_trip_does_not_require_observed_score(
    tmp_path,
):
    ensemble = _direct_hmc_ensemble()

    output_dir = ensemble.save(tmp_path / "scores")
    loaded = ScoreEnsemble.load(output_dir)
    try:
        assert {path.name for path in output_dir.iterdir()} == {
            "metadata.json",
            "variants.csv",
            "analysis_variants.csv",
            "fitness_scores.nc",
        }
        assert loaded.observed_score is None
        assert loaded.source_schema_version == 2
        pd.testing.assert_frame_equal(loaded.variants, ensemble.variants)
        pd.testing.assert_frame_equal(
            loaded.analysis_variants,
            ensemble.analysis_variants,
        )
        xr.testing.assert_equal(loaded.scores, ensemble.scores)
    finally:
        loaded.close()


def test_score_ensemble_from_analysis_view_builds_identity_mapping():
    metadata, variants, fitness_proxies = _legacy_bundle_parts()
    scores = fitness_proxies.rename(
        {
            "fitness_proxy": "fitness_score",
            "observed_proxy": "observed_score",
        }
    )

    ensemble = ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
    )

    assert ensemble.source_schema_version == 2
    assert ensemble.metadata["fit_method"] == "mle_predictive"
    assert ensemble.metadata["n_analysis_variants"] == 2
    np.testing.assert_array_equal(
        ensemble.analysis_variants["matrix_variant_index"],
        [0, 1],
    )
    xr.testing.assert_equal(ensemble.scores, scores)


def test_score_bundle_root_uses_one_method_and_view_hierarchy(tmp_path):
    base = tmp_path / "paper/analysis_data/scores"

    assert score_bundle_root(tmp_path, "mle_predictive") == (
        base / "mle_predictive/canonical"
    )
    assert score_bundle_root(tmp_path, "hmc_count_based") == (
        base / "hmc_count_based/canonical"
    )
    assert score_bundle_root(tmp_path, "hmc", sample_view="release_1000") == (
        base / "hmc/release_1000"
    )
    assert score_collection_root(base, "mle_predictive") == (
        base / "mle_predictive/canonical"
    )

    with pytest.raises(ValueError, match="sample_view='canonical'"):
        score_bundle_root(tmp_path, "mle_predictive", sample_view="full")


def test_score_ensemble_lazily_normalizes_legacy_proxy_bundle(
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
        assert loaded.source_schema_version == 1
        assert set(loaded.scores.data_vars) == {"fitness_score", "observed_score"}
        np.testing.assert_array_equal(
            loaded.analysis_variants["matrix_variant_index"],
            [0, 1],
        )
        xr.testing.assert_equal(
            loaded.fitness_score.rename("fitness_proxy"),
            fitness_proxies["fitness_proxy"],
        )
    finally:
        loaded.close()


def test_score_ensemble_rejects_analysis_variant_identity_drift():
    ensemble = _direct_hmc_ensemble()
    ensemble.analysis_variants.loc[1, "variant_id"] = "AB"

    with pytest.raises(ValueError, match="do not match matrix_variant_index"):
        ensemble.validate()


def test_score_ensemble_accepts_new_method_with_explicit_sample_semantics():
    ensemble = _direct_hmc_ensemble()
    identity = AnalysisIdentity(
        protein_family="unknown",
        landscape="synthetic-errors",
        inference_method="simulation",
        model_spec="benchmark",
        fit_id="seed-23",
    )
    ensemble.metadata.update(
        {
            **identity.metadata(),
            "fit_method": "simulation",
            "uncertainty_target": "synthetic_measurement_repeatability",
            "sample_kind": "simulated experiment",
        }
    )

    ensemble.validate()

    del ensemble.metadata["sample_kind"]
    with pytest.raises(ValueError, match="sample_kind"):
        ensemble.validate()


def test_analysis_config_selects_lower_score_without_external_reference():
    ensemble = _direct_hmc_ensemble()
    identity = AnalysisIdentity(
        protein_family="unknown",
        landscape="synthetic-errors",
        inference_method="simulation",
        model_spec="benchmark",
        fit_id="seed-23",
    )
    ensemble.metadata.update(
        {
            **identity.metadata(),
            "fit_method": "simulation",
            "uncertainty_target": "synthetic_measurement_repeatability",
            "sample_kind": "simulated experiment",
        }
    )
    config = {
        "schema_version": 1,
        "score_bundle": "scores",
        "score_name": "log_mean_growth",
        "direction": "lower",
        "top_k": [1],
        "seed": 23,
        "external_reference": None,
        "model_reference_min_finite_samples": 2,
    }

    result = analyze_score_ensemble(
        ensemble,
        config=config,
        variant_chunk_size=2,
        n_pairs=2,
        show_progress=False,
    )

    np.testing.assert_array_equal(
        result.selection["variant_index"].sel(score_name="log_mean_growth"),
        np.zeros((4, 1), dtype=np.int32),
    )
    np.testing.assert_allclose(
        result.selection["score"].sel(score_name="log_mean_growth")[:, 0],
        [0.1, 0.2, 0.3, 0.4],
    )
    np.testing.assert_array_equal(
        result.selection["model_reference_variant_index"].sel(
            score_name="log_mean_growth"
        ),
        [0],
    )
    assert set(result.overlaps["comparison"]) == {"sample_vs_sample"}
    assert result.metadata["reference_top_k_identifiable"] is False
    assert result.metadata["tracked_reported_top_n"] == 0
    assert result.metadata["model_reference_min_finite_samples"] == 2
    assert result.metadata["higher_is_better"] == {"log_mean_growth": False}
    assert result.metadata["analysis_config"] == config
    assert len(result.metadata["analysis_config_sha256"]) == 64


def test_analyze_direct_hmc_score_ensemble_uses_exact_analysis_view_mapping():
    result = analyze_score_ensemble(
        _direct_hmc_ensemble(),
        top_k=(1,),
        retain_k=1,
        tracked_top_n=1,
        tracked_model_top_n=1,
        variant_chunk_size=2,
        n_pairs=2,
        show_progress=False,
    )

    assert result.variants["variant_id"].tolist() == ["AA", "AC"]
    assert "matrix_variant_index" not in result.variants
    assert result.metadata["fit_method"] == "hmc_posterior"
    assert result.metadata["model_reference_min_finite_samples"] == 1
    assert result.metadata["n_fitted_variants"] == 3
    assert result.metadata["n_variants"] == 2
    np.testing.assert_array_equal(result.selection["sample_id"], [0, 1, 2, 3])
    assert "source_chain" not in result.selection.coords
    np.testing.assert_array_equal(
        result.selection["variant_index"].sel(score_name="log_mean_growth"),
        np.ones((4, 1), dtype=np.int32),
    )


def test_direct_hmc_reference_metadata_preserves_family_contract():
    ensemble = _direct_hmc_ensemble()

    metadata = direct_hmc_reference_metadata(
        ensemble.metadata,
        ensemble.analysis_variants,
    )

    assert metadata == {
        "reference_fitness_source": "reported",
        "reported_fitness_available": True,
        "reference_metadata_contract": "direct-hmc-reference-v1",
    }
