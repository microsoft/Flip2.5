from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import rankdata

from dms_fitness_stability import analyze_score_ensemble
from dms_fitness_stability.adapters import trpb as trpb_adapter
from dms_fitness_stability.adapters.trpb import (
    TRPB_PREDICTIVE_SCORE_CONTRACT,
    TRPB_PREDICTIVE_SCORE_NAME,
    TrpBLandscape,
    build_trpb_four_site_mle_score_ensemble,
    build_trpb_hmc_count_score_ensemble,
    build_trpb_hmc_score_ensemble,
    build_trpb_mle_score_ensemble,
    load_trpb_landscape,
    scan_trpb_hmc,
    trpb_johnston_fitness,
    trpb_three_site_fitness,
)


def _write_flat_counts(path: Path) -> None:
    rows = []
    for timepoint, time in (("T1", 1), ("T2", 2)):
        for index, variant in enumerate(("AAA", "AAC", "AAD", "AAE"), start=1):
            rows.append(
                {
                    "Timepoint": timepoint,
                    "Time (h)": time,
                    "AAs": variant,
                    "InputCount_1": index * 10,
                    "OutputCount_1": index * (10 + time),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_reported(path: Path, variants: tuple[str, ...]) -> None:
    pd.DataFrame(
        {
            "AAs": variants,
            "fitness": np.linspace(0.1, 0.9, len(variants)),
            "active": [True] * len(variants),
        }
    ).to_csv(path, index=False)


def test_trpb_johnston_fitness_reproduces_source_filtering_and_aggregation():
    variants = ["*AAA", "A*AA", "AAAA", "AAAC", "AAAD"]
    starting = np.tile(
        np.array([[0, 0, 0, 0, 0], [20, 20, 20, 20, 20]]),
        (2, 1, 1),
    )
    timepoint = np.tile(
        np.array([10, 10, 40, 20, 10]),
        (2, 6, 2, 1),
    )
    timepoint[:, :, 1, 4] = 0
    timepoint[0, 0, :, 3] = 40
    timepoint[0, 1, 0, 3] = 0
    timepoint[1, 1:, :, 3] = 40

    fitness = trpb_johnston_fitness(
        starting,
        timepoint,
        variants,
        np.array([12, 16, 20, 24, 28, 36], dtype=float),
    )

    assert TRPB_PREDICTIVE_SCORE_NAME == "johnston_fitness"
    assert TRPB_PREDICTIVE_SCORE_CONTRACT == "trpb-johnston-fitness-v1"
    np.testing.assert_allclose(
        fitness[:, :4],
        np.array([[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 1.0]]),
        atol=1e-12,
    )
    assert np.isnan(fitness[:, 4]).all()


def test_trpb_three_site_fitness_supports_flat_and_hierarchical_counts():
    variants = ["*AA", "A*A", "AAA", "AAC", "AAD"]
    starting = np.array([10, 10, 10, 10, 10], dtype=np.int64)
    flat = np.tile(np.array([10, 10, 40, 20, 10]), (2, 1))
    hierarchical = np.tile(
        np.array([10, 10, 40, 20, 10]),
        (2, 2, 1),
    )
    hierarchical[0, 0, 3] = 0

    flat_fitness = trpb_three_site_fitness(
        starting,
        flat,
        variants,
        np.array([18, 44], dtype=float),
    )
    hierarchical_fitness = trpb_three_site_fitness(
        starting,
        hierarchical,
        variants,
        np.array([20, 36], dtype=float),
    )

    np.testing.assert_allclose(flat_fitness, [0.0, 0.0, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(
        hierarchical_fitness,
        [0.0, 0.0, 1.0, 0.5, 0.0],
    )


def test_build_trpb_four_site_mle_score_ensemble_preserves_count_draws(tmp_path):
    variants = np.array(["*AAA", "A*AA", "AAAA", "AAAC"])
    starting = np.array([[0, 0, 0, 0], [20, 20, 20, 20]], dtype=np.int64)
    observed_timepoint = np.tile(
        np.array([10, 10, 40, 20], dtype=np.int64),
        (6, 2, 1),
    )
    times = np.array([12, 16, 20, 24, 28, 36], dtype=float)
    observed_fitness = trpb_johnston_fitness(
        starting,
        observed_timepoint,
        variants,
        times,
    )
    variant_table = pd.DataFrame(
        {
            "variant_id": variants,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "reported_fitness": observed_fitness,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 4,
        }
    )
    landscape = TrpBLandscape(
        library="four-site",
        variants=variant_table,
        starting_counts=starting,
        timepoint_counts=observed_timepoint,
        times=times,
    )
    predictive_starting = np.tile(starting, (2, 1, 1))
    predictive_timepoint = np.stack(
        [
            observed_timepoint,
            np.tile(np.array([10, 10, 20, 40]), (6, 2, 1)),
        ]
    )
    fit_path = tmp_path / "four_site_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=starting)
        observed.create_dataset("timepoint_counts", data=observed_timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset(
            "starting_counts",
            data=predictive_starting[None],
        )
        predictive.create_dataset(
            "timepoint_counts",
            data=predictive_timepoint[None],
        )

    ensemble = build_trpb_four_site_mle_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        fit_id="mle-64",
        job_id=64,
        variant_chunk_size=2,
        show_progress=False,
    )

    expected = trpb_johnston_fitness(
        predictive_starting,
        predictive_timepoint,
        variants,
        times,
    )
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(
            score_name=TRPB_PREDICTIVE_SCORE_NAME
        ),
        expected,
    )
    np.testing.assert_allclose(
        ensemble.observed_score.sel(
            score_name=TRPB_PREDICTIVE_SCORE_NAME
        ),
        observed_fitness,
    )
    assert ensemble.metadata["analysis_id"] == (
        "trpb__trpb-four-site__mle-predictive__"
        "exponential-exponential__mle-64"
    )
    assert ensemble.metadata["observed_oracle"]["max_abs_error"] <= 1e-12
    assert ensemble.metadata["score_status"] == {"johnston_fitness": "verified"}


def test_build_trpb_mle_score_ensemble_supports_flat_three_site_counts(tmp_path):
    variants = np.array(["*AA", "A*A", "AAA", "AAC"])
    starting = np.array([10, 10, 10, 10], dtype=np.int64)
    observed_timepoint = np.array(
        [[10, 10, 40, 20], [10, 10, 40, 20]],
        dtype=np.int64,
    )
    times = np.array([18, 44], dtype=float)
    observed_fitness = trpb_three_site_fitness(
        starting,
        observed_timepoint,
        variants,
        times,
    )
    variant_table = pd.DataFrame(
        {
            "variant_id": variants,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "reported_fitness": observed_fitness,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 4,
        }
    )
    landscape = TrpBLandscape(
        library="libA",
        variants=variant_table,
        starting_counts=starting,
        timepoint_counts=observed_timepoint,
        times=times,
    )
    predictive_starting = np.stack([starting, starting])
    predictive_timepoint = np.stack(
        [
            observed_timepoint,
            np.array([[10, 10, 20, 40], [10, 10, 20, 40]], dtype=np.int64),
        ]
    )
    fit_path = tmp_path / "liba_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=starting)
        observed.create_dataset("timepoint_counts", data=observed_timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("starting_counts", data=predictive_starting[None])
        predictive.create_dataset("timepoint_counts", data=predictive_timepoint[None])

    ensemble = build_trpb_mle_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        fit_id="mle-7",
        job_id=7,
        variant_chunk_size=2,
        show_progress=False,
    )

    expected = trpb_three_site_fitness(
        predictive_starting,
        predictive_timepoint,
        variants,
        times,
    )
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(
            score_name=TRPB_PREDICTIVE_SCORE_NAME
        ),
        expected,
    )
    assert ensemble.metadata["analysis_id"] == (
        "trpb__trpb-liba__mle-predictive__"
        "exponential-exponential__mle-7"
    )
    assert ensemble.metadata["observed_oracle"]["max_abs_error"] <= 1e-12


def test_build_trpb_mle_score_ensemble_supports_hierarchical_counts(tmp_path):
    variants = np.array(["*AA", "A*A", "AAA", "AAC"])
    starting = np.array([10, 10, 10, 10], dtype=np.int64)
    observed_timepoint = np.tile(
        np.array([10, 10, 40, 20], dtype=np.int64),
        (2, 2, 1),
    )
    times = np.array([20, 36], dtype=float)
    observed_fitness = trpb_three_site_fitness(
        starting,
        observed_timepoint,
        variants,
        times,
    )
    variant_table = pd.DataFrame(
        {
            "variant_id": variants,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "reported_fitness": observed_fitness,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 4,
        }
    )
    landscape = TrpBLandscape(
        library="libD",
        variants=variant_table,
        starting_counts=starting,
        timepoint_counts=observed_timepoint,
        times=times,
    )
    predictive_starting = np.stack([starting, starting])
    alternative = np.tile(
        np.array([10, 10, 20, 40], dtype=np.int64),
        (2, 2, 1),
    )
    predictive_timepoint = np.stack([observed_timepoint, alternative])
    fit_path = tmp_path / "libd_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=starting)
        observed.create_dataset("timepoint_counts", data=observed_timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("starting_counts", data=predictive_starting[None])
        predictive.create_dataset("timepoint_counts", data=predictive_timepoint[None])

    ensemble = build_trpb_mle_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="gamma",
        growth_curve="exponential",
        fit_id="mle-21",
        job_id=21,
        variant_chunk_size=2,
        show_progress=False,
    )

    expected = trpb_three_site_fitness(
        predictive_starting,
        predictive_timepoint,
        variants,
        times,
    )
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(
            score_name=TRPB_PREDICTIVE_SCORE_NAME
        ),
        expected,
    )
    assert ensemble.metadata["analysis_id"] == (
        "trpb__trpb-libd__mle-predictive__gamma-exponential__mle-21"
    )
    assert ensemble.metadata["finite_ranking_candidates_per_sample"] == [4, 4]


def test_build_trpb_hmc_count_score_ensemble_matches_johnston_formula(
    tmp_path,
):
    variants = np.array(["*AA", "A*A", "AAA", "AAC"])
    starting = np.array([10, 10, 10, 10], dtype=np.int64)
    observed_timepoint = np.tile(
        np.array([10, 10, 40, 20], dtype=np.int64),
        (2, 2, 1),
    )
    times = np.array([20, 36], dtype=float)
    observed_fitness = trpb_three_site_fitness(
        starting,
        observed_timepoint,
        variants,
        times,
    )
    variant_table = pd.DataFrame(
        {
            "variant_id": variants,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "reported_fitness": observed_fitness,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 4,
        }
    )
    landscape = TrpBLandscape(
        library="libD",
        variants=variant_table,
        starting_counts=starting,
        timepoint_counts=observed_timepoint,
        times=times,
    )
    predictive_starting = np.broadcast_to(starting, (2, 2, 4)).copy()
    alternative = np.tile(
        np.array([10, 10, 20, 40], dtype=np.int64),
        (2, 2, 1),
    )
    predictive_timepoint = np.stack(
        [observed_timepoint, alternative, alternative, observed_timepoint]
    ).reshape(2, 2, *observed_timepoint.shape)
    fit_path = tmp_path / "libd_hmc.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=starting)
        observed.create_dataset("timepoint_counts", data=observed_timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("starting_counts", data=predictive_starting)
        predictive.create_dataset("timepoint_counts", data=predictive_timepoint)

    ensemble = build_trpb_hmc_count_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="gamma",
        growth_curve="exponential",
        fit_id="small-clean_21",
        job_id=21,
        artifact_kind="diagnosed_hmc",
        sample_chunk_size=1,
        variant_chunk_size=2,
        scratch_dir=tmp_path / "scratch",
        show_progress=False,
    )
    temporary_path = ensemble.temporary_paths[0]
    try:
        expected = trpb_three_site_fitness(
            predictive_starting,
            predictive_timepoint,
            variants,
            times,
        ).reshape(4, 4)
        np.testing.assert_allclose(
            ensemble.fitness_score.sel(
                score_name="johnston_fitness"
            ),
            expected,
        )
        np.testing.assert_array_equal(ensemble.scores["source_sample_id"], [0, 1, 2, 3])
        np.testing.assert_array_equal(ensemble.scores["source_chain"], [0, 0, 1, 1])
        np.testing.assert_array_equal(ensemble.scores["source_draw"], [0, 1, 0, 1])
        assert ensemble.source_schema_version == 2
        assert ensemble.metadata["inference_method"] == "hmc_count_based"
        assert ensemble.metadata["finite_ranking_candidates_per_sample"] == [4] * 4
        assert temporary_path.is_file()
    finally:
        ensemble.close()
    assert not temporary_path.exists()


def test_load_trpb_landscape_preserves_full_fit_order_and_aligns_reported_subset(
    tmp_path,
):
    count_path = tmp_path / "libA_counts.csv"
    reported_path = tmp_path / "libA_fitness.csv"
    _write_flat_counts(count_path)
    _write_reported(reported_path, ("AAA", "AAD", "AAE"))

    landscape = load_trpb_landscape(
        "libA",
        count_path=count_path,
        reported_fitness_path=reported_path,
    )

    assert landscape.variants["variant_id"].tolist() == ["AAA", "AAC", "AAD", "AAE"]
    assert landscape.variants["hmc_variant_index"].tolist() == [0, 1, 2, 3]
    assert np.isnan(landscape.variants.loc[1, "reported_fitness"])
    assert landscape.variants["reference_fitness_source"].tolist() == [
        "reported",
        "unavailable",
        "reported",
        "reported",
    ]
    np.testing.assert_array_equal(landscape.starting_counts, [10, 20, 30, 40])
    np.testing.assert_array_equal(landscape.timepoint_counts.shape, [2, 4])


def test_load_trpb_landscape_rejects_reported_rows_out_of_hmc_order(tmp_path):
    count_path = tmp_path / "libA_counts.csv"
    reported_path = tmp_path / "libA_fitness.csv"
    _write_flat_counts(count_path)
    _write_reported(reported_path, ("AAD", "AAA"))

    with pytest.raises(ValueError, match="not in fitted HMC column order"):
        load_trpb_landscape(
            "libA",
            count_path=count_path,
            reported_fitness_path=reported_path,
        )


def _synthetic_landscape(library: str) -> TrpBLandscape:
    variants = pd.DataFrame(
        {
            "variant_id": ["AAA", "AAC", "AAD", "AAE"],
            "library": [library] * 4,
            "variant_label": ["AAA", "AAC", "AAD", "AAE"],
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "input_count": [10, 20, 30, 40],
            "output_count": [20, 40, 60, 80],
            "count_ratio": [2.0] * 4,
            "reported_fitness": [0.1, np.nan, 0.9, 0.5],
            "reported_rank": [3.0, np.nan, 1.0, 2.0],
            "reference_fitness": [0.1, np.nan, 0.9, 0.5],
            "reference_fitness_source": [
                "reported",
                "unavailable",
                "reported",
                "reported",
            ],
        }
    )
    starting = np.array([10, 20, 30, 40], dtype=np.int64)
    timepoint = np.array([[20, 40, 60, 80], [30, 50, 70, 90]], dtype=np.int64)
    return TrpBLandscape(
        library=library,
        variants=variants,
        starting_counts=starting,
        timepoint_counts=timepoint,
        times=np.array([1.0, 2.0]),
    )


def _write_synthetic_hmc(
    path: Path,
    *,
    variable: str,
    values: np.ndarray,
    landscape: TrpBLandscape,
    embed_observed_counts: bool = True,
    embed_posterior_predictive: bool = True,
) -> None:
    xr.Dataset({variable: (("chain", "draw", "a"), values[None])}).to_netcdf(
        path,
        group="posterior",
        engine="h5netcdf",
        encoding={variable: {"chunksizes": (1, len(values), 2)}},
    )
    if embed_posterior_predictive:
        draws = len(values)
        starting_counts = np.broadcast_to(
            landscape.starting_counts,
            (draws, landscape.n_hmc_variants),
        )
        timepoint_counts = np.broadcast_to(
            landscape.timepoint_counts,
            (draws, *landscape.timepoint_counts.shape),
        )
        xr.Dataset(
            {
                "starting_counts": (
                    ("chain", "draw", "a"),
                    starting_counts[None],
                ),
                "timepoint_counts": (
                    ("chain", "draw", "b", "a"),
                    timepoint_counts[None],
                ),
            }
        ).to_netcdf(
            path,
            mode="a",
            group="posterior_predictive",
            engine="h5netcdf",
        )
    if not embed_observed_counts:
        return
    xr.Dataset(
        {
            "starting_counts": (("a",), landscape.starting_counts),
            "timepoint_counts": (
                ("b", "a"),
                landscape.timepoint_counts,
            ),
        }
    ).to_netcdf(
        path,
        mode="a",
        group="observed_data",
        engine="h5netcdf",
    )


@pytest.mark.parametrize(
    ("library", "growth_rate_prior", "variable", "native_values"),
    [
        ("libA", "exponential", "r", np.exp),
        ("libA", "lomax", "r", np.exp),
        ("libD", "exponential", "log_r_mean", lambda values: values),
        ("libD", "lomax", "log_r_mean", lambda values: values),
        ("libD", "gamma", "inv_r_mean", lambda values: np.exp(-values)),
    ],
)
def test_trpb_hmc_scan_uses_common_log_growth_ranking(
    tmp_path,
    library,
    growth_rate_prior,
    variable,
    native_values,
):
    log_growth = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 2.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    landscape = _synthetic_landscape(library)
    fit_path = tmp_path / "fit.nc"
    _write_synthetic_hmc(
        fit_path,
        variable=variable,
        values=native_values(log_growth),
        landscape=landscape,
    )

    result = scan_trpb_hmc(
        landscape,
        fit_path,
        growth_rate_prior=growth_rate_prior,
        growth_curve="exponential",
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=2,
        tracked_model_top_n=1,
        fit_id="synthetic-fit",
        n_pairs=5,
        show_progress=False,
    )
    ensemble = build_trpb_hmc_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior=growth_rate_prior,
        growth_curve="exponential",
        fit_id="synthetic-fit",
        sample_chunk_size=2,
        variant_chunk_size=2,
        scratch_dir=tmp_path / "scratch",
        show_progress=False,
    )
    try:
        matrix_result = analyze_score_ensemble(
            ensemble,
            top_k=(1, 2),
            retain_k=2,
            tracked_top_n=2,
            tracked_model_top_n=1,
            variant_chunk_size=2,
            n_pairs=5,
            show_progress=False,
        )
    finally:
        ensemble.close()
    for name in result.selection.data_vars:
        np.testing.assert_array_equal(
            matrix_result.selection[name].sel(score_name="log_mean_growth"),
            result.selection[name].sel(score_name="log_mean_growth"),
        )
    pd.testing.assert_frame_equal(
        matrix_result.summary,
        result.summary[result.summary["score"] == "log_mean_growth"].reset_index(
            drop=True
        ),
        check_exact=True,
    )

    expected_top_scores = np.sort(log_growth, axis=1)[:, -2:][:, ::-1]
    tracked_indices = np.array([2, 3])
    expected_ranks = rankdata(-log_growth, axis=1, method="average")[
        :, tracked_indices
    ]
    np.testing.assert_allclose(
        result.selection["score"].sel(score_name="log_mean_growth"),
        expected_top_scores,
    )
    np.testing.assert_allclose(
        result.selection["tracked_rank"].sel(score_name="log_mean_growth"),
        expected_ranks,
    )
    np.testing.assert_array_equal(
        result.selection["model_reference_variant_index"].sel(
            score_name="log_mean_growth"
        ),
        [0, 1],
    )
    np.testing.assert_allclose(
        result.selection["model_tracked_rank"].sel(score_name="log_mean_growth"),
        rankdata(-log_growth, axis=1, method="average")[:, [0]],
    )
    model_row = result.summary[
        (result.summary["score"] == "log_mean_growth")
        & (result.summary["k"] == 1)
        & (result.summary["variant_index"] == 0)
    ].iloc[0]
    assert model_row["is_tracked_model_cohort"]
    assert model_row["model_reference_rank"] == 1
    assert result.metadata["analysis_id"] == (
        f"trpb__trpb-{library.lower()}__hmc__"
        f"{growth_rate_prior}-exponential__synthetic-fit"
    )
    assert result.metadata["tracked_model_top_n"] == 1
    expected_count_ratio = np.array([5.0, 4.5, 13 / 3, 4.25])
    np.testing.assert_allclose(
        result.selection["score"].sel(score_name="count_ratio"),
        np.tile(expected_count_ratio[:2], (len(log_growth), 1)),
    )
    assert result.metadata["scores"] == ["log_mean_growth", "count_ratio"]
    assert result.metadata["observed_counts_status"] == "matched"
    assert result.metadata["artifact_status"] == "validated"
    assert result.metadata["artifact_issues"] == []
    assert result.metadata["h5netcdf_compatible"] is True
    assert result.metadata["warnings"] == []
    assert result.metadata["n_variants"] == 4
    assert result.metadata["n_reported_variants"] == 3


def test_trpb_hmc_scan_records_missing_observed_counts_as_limited(tmp_path):
    log_growth = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    landscape = _synthetic_landscape("libD")
    fit_path = tmp_path / "fit.nc"
    _write_synthetic_hmc(
        fit_path,
        variable="log_r_mean",
        values=log_growth,
        landscape=landscape,
        embed_observed_counts=False,
    )

    result = scan_trpb_hmc(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        top_k=(1,),
        retain_k=1,
        tracked_top_n=1,
        n_pairs=2,
        show_progress=False,
    )

    assert result.metadata["observed_counts_status"] == "not_embedded"
    assert result.metadata["count_alignment_evidence"] == "source_count_table_order"
    assert result.metadata["artifact_status"] == "limited"
    assert result.metadata["artifact_issues"] == ["missing_observed_data"]
    assert "does not embed observed counts" in result.metadata["warnings"][0]


def test_trpb_hmc_scan_records_h5netcdf_reader_fallback(tmp_path, monkeypatch):
    log_growth = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    landscape = _synthetic_landscape("libD")
    fit_path = tmp_path / "fit.nc"
    _write_synthetic_hmc(
        fit_path,
        variable="log_r_mean",
        values=log_growth,
        landscape=landscape,
    )
    monkeypatch.setattr(
        trpb_adapter,
        "_h5netcdf_compatibility",
        lambda path: (False, "KeyError: malformed metadata"),
    )

    result = scan_trpb_hmc(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        top_k=(1,),
        retain_k=1,
        tracked_top_n=1,
        n_pairs=2,
        show_progress=False,
    )

    assert result.metadata["observed_counts_status"] == "matched"
    assert result.metadata["artifact_status"] == "limited"
    assert result.metadata["artifact_issues"] == ["h5netcdf_open_failure"]
    assert result.metadata["h5netcdf_compatible"] is False
    assert result.metadata["h5netcdf_open_error"] == "KeyError: malformed metadata"
    assert "read directly with h5py" in result.metadata["warnings"][0]


def test_trpb_hmc_scan_omits_count_ratio_when_predictive_counts_are_missing(tmp_path):
    log_growth = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    landscape = _synthetic_landscape("libD")
    fit_path = tmp_path / "fit.nc"
    _write_synthetic_hmc(
        fit_path,
        variable="log_r_mean",
        values=log_growth,
        landscape=landscape,
        embed_posterior_predictive=False,
    )

    result = scan_trpb_hmc(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        top_k=(1,),
        retain_k=1,
        tracked_top_n=1,
        n_pairs=2,
        show_progress=False,
    )

    assert result.metadata["scores"] == ["log_mean_growth"]
    assert result.metadata["artifact_status"] == "limited"
    assert result.metadata["artifact_issues"] == [
        "missing_posterior_predictive_counts"
    ]
    assert "count_ratio could not be analyzed" in result.metadata["warnings"][0]
