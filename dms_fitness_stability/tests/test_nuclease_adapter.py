from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import rankdata

from dms_fitness_stability import analyze_score_ensemble
from dms_fitness_stability.adapters import nuclease as nuclease_adapter
from dms_fitness_stability.adapters.nuclease import (
    NUCB_WT_SEQUENCE,
    NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS,
    NUCLEASE_PREDICTIVE_SCORE_CONTRACT,
    NUCLEASE_PREDICTIVE_SCORE_CONTRACTS,
    NUCLEASE_PREDICTIVE_SCORE_NAME,
    NUCLEASE_REPORTED_FITNESS_COLUMN,
    NUCLEASE_SCORE_NAME,
    NucleaseLandscape,
    build_nuclease_g3_mle_score_ensemble,
    build_nuclease_hmc_count_score_ensemble,
    build_nuclease_hmc_score_ensemble,
    build_nuclease_mle_score_ensemble,
    load_nuclease_landscape,
    mutation_signature,
    normalized_count_enrichment,
    nuclease_g3_highest_gate_enrichment,
    nuclease_highest_gate_enrichment,
    scan_nuclease_hmc,
)


def _reported_activity(path: Path) -> None:
    pd.DataFrame(
        {
            "mutations": [
                "[]",
                "[['A', 3, 'D'], ['A', 2, 'C']]",
                "[['A', 2, 'C']]",
            ],
            "activity_level": [
                "activity_greater_than_0",
                "activity_greater_than_WT",
                "non-functional",
            ],
            "is_functional": [True, True, False],
            "generations": ["['g1']", "['g1']", "['g1']"],
        }
    ).to_csv(path, index=False)


def _g1_dataset() -> dict[str, object]:
    return {
        "variants": ["A2C", "", "A2C_A3D", "A4*"],
        "ic1": np.array([[1, 2, 3, 4], [10, 20, 30, 40]]),
        "ic2": np.array([[2, 3, 4, 5], [20, 30, 40, 50]]),
        "ic3": np.array([3, 4, 5, 6]),
        "lc": np.array([[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]),
        "hc1": np.array([7, 8, 9, 10]),
        "hc2": np.array([[8, 9, 10, 11], [9, 10, 11, 12]]),
        "hc3": np.array([10, 11, 12, 13]),
        "lt": np.array([0.1, 0.115, 0.1]),
        "ht": np.array([0.501, 0.6, 0.37]),
    }


def _reported_fitness(
    path: Path,
    generation: str,
    variants: tuple[str, ...],
) -> None:
    pd.DataFrame(
        {
            "mutations": variants,
            NUCLEASE_REPORTED_FITNESS_COLUMN[generation]: np.arange(
                1,
                len(variants) + 1,
                dtype=float,
            ),
        }
    ).to_csv(path, index=False)


def _negative_controls(path: Path, variants: tuple[str, ...]) -> None:
    pd.DataFrame({"mutations": variants}).to_csv(path, index=False)


def test_mutation_signature_matches_flipv3_ordering():
    assert mutation_signature("[['A', 73, 'R'], ['D', 64, 'S']]") == "D64S_A73R"
    assert mutation_signature("A73R_D64S") == "D64S_A73R"
    assert mutation_signature([]) == ""
    with pytest.raises(ValueError, match="positions must be unique"):
        mutation_signature("[['A', 2, 'C'], ['A', 2, 'D']]")


def test_normalized_count_enrichment_uses_full_variant_universe():
    input_counts = np.array([[80, 20], [40, 40]], dtype=np.int64)
    output_counts = np.array([[20, 30], [10, 30]], dtype=np.int64)

    enrichment = normalized_count_enrichment(input_counts, output_counts)

    np.testing.assert_allclose(
        enrichment,
        np.array([[0.5, 3.0], [0.5, 1.5]]),
    )


def test_nuclease_g3_highest_gate_enrichment_selects_fourth_gate():
    counts = {
        "ic1": np.array([[80, 20], [40, 40]], dtype=np.int64),
        "oc1": np.array(
            [
                [[1, 1], [2, 2], [3, 3], [20, 30]],
                [[4, 4], [5, 5], [6, 6], [10, 30]],
            ],
            dtype=np.int64,
        ),
    }

    enrichment = nuclease_g3_highest_gate_enrichment(counts)

    assert NUCLEASE_PREDICTIVE_SCORE_NAME == "highest_gate_enrichment"
    assert NUCLEASE_PREDICTIVE_SCORE_CONTRACT == "nuc-g3-highest-gate-enrichment-v1"
    np.testing.assert_allclose(
        enrichment,
        np.array([[0.5, 3.0], [0.5, 1.5]]),
    )


def test_nuclease_g4_highest_gate_enrichment_uses_the_g4_contract():
    counts = {
        "ic1": np.array([80, 20], dtype=np.int64),
        "oc1": np.array(
            [[1, 1], [2, 2], [3, 3], [20, 30]],
            dtype=np.int64,
        ),
    }

    enrichment = nuclease_highest_gate_enrichment("G4", counts)

    assert NUCLEASE_PREDICTIVE_SCORE_CONTRACTS["G4"] == (
        "nuc-g4-highest-gate-enrichment-v1"
    )
    np.testing.assert_allclose(enrichment, [0.5, 3.0])


def test_nuclease_g2_highest_gate_enrichment_pools_all_inputs():
    counts = {
        "ic1": np.array([[40, 10], [30, 20]], dtype=np.int64),
        "ic2": np.array([10, 10], dtype=np.int64),
        "c975": np.array([20, 20], dtype=np.int64),
    }

    enrichment = nuclease_highest_gate_enrichment("G2", counts)

    np.testing.assert_allclose(enrichment, [0.75, 1.5])


def test_nuclease_g1_highest_gate_enrichment_uses_source_normalization():
    counts = {
        "ic1": np.array([[10, 20], [30, 40]], dtype=np.int64),
        "ic2": np.array([[50, 60], [70, 80]], dtype=np.int64),
        "ic3": np.array([90, 100], dtype=np.int64),
        "hc2": np.array([[1, 1], [20, 30]], dtype=np.int64),
    }
    pooled_input = np.array([250, 300])
    source_factor = (
        NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS["input"]
        / NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS["output"]
    )

    enrichment = nuclease_highest_gate_enrichment("G1", counts)

    np.testing.assert_allclose(
        enrichment,
        np.array([20 / pooled_input[0], 30 / pooled_input[1]]) * source_factor,
    )


@pytest.mark.parametrize("generation", ["G3", "G4"])
def test_build_nuclease_mle_score_ensemble_preserves_draws_and_controls(
    tmp_path,
    generation,
):
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "stop-control"],
            "is_source_control": [False, False, True],
            "is_ranking_candidate": [True, True, False],
            "reported_fitness": [0.5, 3.0, np.nan],
            "reported_rank": [2.0, 1.0, np.nan],
        }
    )
    observed_counts = {
        "ic1": np.array([80, 20, 100], dtype=np.int64),
        "oc1": np.array(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [20, 30, 50]],
            dtype=np.int64,
        ),
    }
    landscape = NucleaseLandscape(
        generation=generation,
        variants=variants,
        counts=observed_counts,
        thresholds={"ft": np.arange(4, dtype=float)},
        wild_type_index=0,
    )
    predictive_input = np.array(
        [[[80, 20, 100], [40, 40, 120]]],
        dtype=np.int64,
    )
    predictive_output = np.array(
        [
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [20, 30, 50]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 30, 60]],
            ]
        ],
        dtype=np.int64,
    )
    fit_path = tmp_path / "g3_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("ic1", data=observed_counts["ic1"])
        observed.create_dataset("oc1", data=observed_counts["oc1"])
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("ic1", data=predictive_input)
        predictive.create_dataset("oc1", data=predictive_output)

    builder = (
        build_nuclease_g3_mle_score_ensemble
        if generation == "G3"
        else build_nuclease_mle_score_ensemble
    )
    fit_id = "mle-59" if generation == "G3" else "mle-61"
    job_id = 59 if generation == "G3" else 61
    ensemble = builder(
        landscape,
        fit_path,
        fitness_distribution="exponential",
        fit_id=fit_id,
        job_id=job_id,
        sample_chunk_size=1,
    )

    np.testing.assert_allclose(
        ensemble.fitness_score.sel(
            score_name=NUCLEASE_PREDICTIVE_SCORE_NAME
        ),
        np.array([[0.5, 3.0, 1.0], [0.5, 1.5, 1.0]]),
    )
    np.testing.assert_allclose(
        ensemble.observed_score.sel(
            score_name=NUCLEASE_PREDICTIVE_SCORE_NAME
        ),
        [0.5, 3.0, 1.0],
    )
    assert ensemble.metadata["analysis_id"] == (
        f"nuc__nuc-{generation.lower()}__mle-predictive__exponential__{fit_id}"
    )
    assert ensemble.source_schema_version == 2
    assert ensemble.metadata["score_contracts"] == {
        NUCLEASE_PREDICTIVE_SCORE_NAME: NUCLEASE_PREDICTIVE_SCORE_CONTRACTS[
            generation
        ]
    }
    assert ensemble.metadata["n_ranking_candidates"] == 2
    assert ensemble.metadata["n_source_controls"] == 1
    assert ensemble.metadata["finite_ranking_candidates_per_sample"] == [2, 2]
    assert ensemble.metadata["normalization_universe"] == (
        "all fitted variants including source controls"
    )
    assert ensemble.metadata["observed_oracle"]["max_abs_error"] <= 1e-12


def test_build_nuclease_hmc_count_score_ensemble_preserves_normalization_and_draws(
    tmp_path,
):
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "stop-control"],
            "is_source_control": [False, False, True],
            "is_ranking_candidate": [True, True, False],
            "reported_fitness": [0.5, 3.0, np.nan],
            "reported_rank": [2.0, 1.0, np.nan],
        }
    )
    observed_counts = {
        "ic1": np.array([80, 20, 100], dtype=np.int64),
        "oc1": np.array(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [20, 30, 50]],
            dtype=np.int64,
        ),
    }
    landscape = NucleaseLandscape(
        generation="G3",
        variants=variants,
        counts=observed_counts,
        thresholds={"ft": np.arange(4, dtype=float)},
        wild_type_index=0,
    )
    predictive_input = np.array(
        [
            [[80, 20, 100], [40, 40, 120]],
            [[100, 20, 80], [60, 40, 100]],
        ],
        dtype=np.int64,
    )
    predictive_output = np.array(
        [
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [20, 30, 50]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 30, 60]],
            ],
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [50, 30, 20]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [30, 20, 50]],
            ],
        ],
        dtype=np.int64,
    )
    fit_path = tmp_path / "g3_hmc.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("ic1", data=observed_counts["ic1"])
        observed.create_dataset("oc1", data=observed_counts["oc1"])
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("ic1", data=predictive_input)
        predictive.create_dataset("oc1", data=predictive_output)

    ensemble = build_nuclease_hmc_count_score_ensemble(
        landscape,
        fit_path,
        fitness_distribution="exponential",
        fit_id="small-clean_59",
        job_id=59,
        artifact_kind="diagnosed_hmc",
        sample_chunk_size=1,
        scratch_dir=tmp_path / "scratch",
        show_progress=False,
    )
    temporary_path = ensemble.temporary_paths[0]
    try:
        expected = nuclease_highest_gate_enrichment(
            "G3",
            {"ic1": predictive_input, "oc1": predictive_output},
        ).reshape(4, 3)
        np.testing.assert_allclose(
                ensemble.fitness_score.sel(
                score_name="highest_gate_enrichment"
            ),
            expected,
        )
        np.testing.assert_array_equal(ensemble.scores["source_sample_id"], [0, 1, 2, 3])
        np.testing.assert_array_equal(ensemble.scores["source_chain"], [0, 0, 1, 1])
        np.testing.assert_array_equal(ensemble.scores["source_draw"], [0, 1, 0, 1])
        assert ensemble.source_schema_version == 2
        assert ensemble.metadata["inference_method"] == "hmc_count_based"
        assert ensemble.metadata["n_ranking_candidates"] == 2
        assert temporary_path.is_file()
    finally:
        ensemble.close()
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("generation", "observed_counts"),
    [
        (
            "G1",
            {
                "ic1": np.array([[10, 20, 30], [30, 40, 50]]),
                "ic2": np.array([[50, 60, 70], [70, 80, 90]]),
                "ic3": np.array([90, 100, 110]),
                "hc2": np.array([[1, 1, 1], [20, 30, 50]]),
            },
        ),
        (
            "G2",
            {
                "ic1": np.array([[40, 10, 20], [30, 20, 30]]),
                "ic2": np.array([10, 10, 20]),
                "c975": np.array([20, 20, 40]),
            },
        ),
    ],
)
def test_build_nuclease_mle_score_ensemble_pools_g1_g2_inputs(
    tmp_path,
    generation,
    observed_counts,
):
    observed_proxy = nuclease_highest_gate_enrichment(generation, observed_counts)
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C", "stop-control"],
            "is_source_control": [False, False, True],
            "is_ranking_candidate": [True, True, False],
            "reported_fitness": [*observed_proxy[:2], np.nan],
            "reported_rank": [2.0, 1.0, np.nan],
        }
    )
    landscape = NucleaseLandscape(
        generation=generation,
        variants=variants,
        counts=observed_counts,
        thresholds={},
        wild_type_index=0,
    )
    fit_path = tmp_path / f"{generation.lower()}_samples.nc"
    predictive_counts = {
        name: np.stack([values, values[..., ::-1]])
        for name, values in observed_counts.items()
    }
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        predictive = handle.create_group("posterior_predictive")
        for name, values in observed_counts.items():
            observed.create_dataset(name, data=values)
            predictive.create_dataset(name, data=predictive_counts[name][None])

    ensemble = build_nuclease_mle_score_ensemble(
        landscape,
        fit_path,
        fitness_distribution="exponential",
        fit_id=f"mle-{generation.lower()}",
        sample_chunk_size=1,
    )

    expected = nuclease_highest_gate_enrichment(generation, predictive_counts)
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(
            score_name=NUCLEASE_PREDICTIVE_SCORE_NAME
        ),
        expected,
    )
    assert ensemble.metadata["score_status"] == {
        NUCLEASE_PREDICTIVE_SCORE_NAME: "verified"
    }


def test_load_nuclease_landscape_preserves_fitted_order_and_aligns_activity(
    tmp_path,
    monkeypatch,
):
    activity_path = tmp_path / "landscape.csv"
    _reported_activity(activity_path)
    _reported_fitness(
        tmp_path / "g1.csv",
        "G1",
        ("[['A', 2, 'C']]", "[]", "[['A', 2, 'C'], ['A', 3, 'D']]"),
    )
    _negative_controls(tmp_path / "g1_neg_control.csv", ("[['A', 4, '*']]",))
    monkeypatch.setattr(
        nuclease_adapter,
        "load_nuclease_data",
        lambda count_dir, gen: _g1_dataset(),
    )

    landscape = load_nuclease_landscape(
        "G1",
        count_dir=tmp_path,
        reported_activity_path=activity_path,
    )

    assert len(NUCB_WT_SEQUENCE) == 142
    assert landscape.variants["variant_id"].tolist() == [
        "A2C",
        "",
        "A2C_A3D",
        "A4*",
    ]
    assert landscape.wild_type_index == 1
    assert landscape.variants["is_wild_type"].tolist() == [False, True, False, False]
    assert landscape.variants["hmc_variant_index"].tolist() == [0, 1, 2, 3]
    assert landscape.variants["is_source_control"].tolist() == [False, False, False, True]
    assert landscape.variants["is_ranking_candidate"].tolist() == [True, True, True, False]
    assert landscape.variants["reported_activity_level"].tolist()[:3] == [
        "non-functional",
        "activity_greater_than_0",
        "activity_greater_than_WT",
    ]
    assert pd.isna(landscape.variants.loc[3, "reported_activity_level"])
    assert landscape.variants["reported_fitness"].tolist()[:3] == [1.0, 2.0, 3.0]
    assert pd.isna(landscape.variants.loc[3, "reported_fitness"])
    assert landscape.variants["reported_rank"].tolist()[:3] == [3.0, 2.0, 1.0]
    assert landscape.variants["reference_fitness_source"].tolist() == [
        "Thomas et al. 2025 highest-gate enrichment factor",
        "Thomas et al. 2025 highest-gate enrichment factor",
        "Thomas et al. 2025 highest-gate enrichment factor",
        "unavailable",
    ]
    assert landscape.counts["ic1"].shape == (2, 4)
    assert landscape.counts["lc"].shape == (3, 4)
    assert landscape.counts["hc2"].shape == (2, 4)
    np.testing.assert_array_equal(
        landscape.variants["input_count"],
        np.array([36, 59, 82, 105]),
    )


@pytest.mark.parametrize(
    ("generation", "dataset", "expected_shapes"),
    [
        (
            "G2",
            {
                "variants": ["", "A2C"],
                "ic1": np.ones((2, 2), dtype=int),
                "ic2": np.ones(2, dtype=int),
                "c86": np.ones(2, dtype=int),
                "c93": np.ones(2, dtype=int),
                "c975": np.ones(2, dtype=int),
                "ft": np.array([0.201, 0.318, 0.385]),
            },
            {"ic1": (2, 2), "ic2": (2,), "c86": (2,), "c93": (2,), "c975": (2,)},
        ),
        (
            "G3",
            {
                "variants": ["", "A2C"],
                "ic1": np.ones(2, dtype=int),
                "oc1": np.ones((4, 2), dtype=int),
                "ft": np.array([0.118, 0.217, 0.285, 0.346]),
            },
            {"ic1": (2,), "oc1": (4, 2)},
        ),
        (
            "G4",
            {
                "variants": ["", "A2C"],
                "ic1": np.ones(2, dtype=int),
                "oc1": np.ones((4, 2), dtype=int),
                "ft": np.array([0.124, 0.199, 0.306, 0.384]),
            },
            {"ic1": (2,), "oc1": (4, 2)},
        ),
    ],
)
def test_load_nuclease_landscape_preserves_generation_count_axes(
    tmp_path,
    monkeypatch,
    generation,
    dataset,
    expected_shapes,
):
    activity_path = tmp_path / "landscape.csv"
    _reported_activity(activity_path)
    _reported_fitness(
        tmp_path / f"{generation.lower()}.csv",
        generation,
        ("[]", "[['A', 2, 'C']]"),
    )
    _negative_controls(tmp_path / f"{generation.lower()}_neg_control.csv", ())
    monkeypatch.setattr(
        nuclease_adapter,
        "load_nuclease_data",
        lambda count_dir, gen: dataset,
    )

    landscape = load_nuclease_landscape(
        generation,
        count_dir=tmp_path,
        reported_activity_path=activity_path,
    )

    assert {name: array.shape for name, array in landscape.counts.items()} == expected_shapes


def test_scan_nuclease_hmc_uses_generation_target_and_tracks_ranks(tmp_path):
    variants = pd.DataFrame(
        {
            "variant_id": ["A2C", "", "A2C_A3D", "A4*"],
            "generation": ["G1"] * 4,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "is_wild_type": [False, True, False, False],
            "is_source_control": [False, False, False, True],
            "is_ranking_candidate": [True, True, True, False],
            "input_count": [10, 20, 30, 40],
            "output_count": [20, 40, 60, 80],
            "reported_fitness": [0.1, 0.5, 0.9, np.nan],
            "reported_rank": [3.0, 2.0, 1.0, np.nan],
            "reference_fitness": [0.1, 0.5, 0.9, np.nan],
            "reference_fitness_source": [
                "Thomas et al. 2025 highest-gate enrichment factor",
                "Thomas et al. 2025 highest-gate enrichment factor",
                "Thomas et al. 2025 highest-gate enrichment factor",
                "unavailable",
            ],
        }
    )
    counts = {"ic1": np.array([10, 20, 30, 40], dtype=np.int64)}
    landscape = NucleaseLandscape(
        generation="G1",
        variants=variants,
        counts=counts,
        thresholds={},
        wild_type_index=1,
    )
    fluorescence = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 2.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    fit_path = tmp_path / "fit.nc"
    xr.Dataset(
        {
            "absolute_mean_log_fluorescence": (
                ("chain", "draw", "a"),
                fluorescence[None],
            )
        }
    ).to_netcdf(
        fit_path,
        group="posterior",
        engine="h5netcdf",
        encoding={
            "absolute_mean_log_fluorescence": {
                "chunksizes": (1, len(fluorescence), 2)
            }
        },
    )
    xr.Dataset({"ic1": (("a",), counts["ic1"])}).to_netcdf(
        fit_path,
        mode="a",
        group="observed_data",
        engine="h5netcdf",
    )

    result = scan_nuclease_hmc(
        landscape,
        fit_path,
        fitness_distribution="exponential",
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=2,
        tracked_model_top_n=1,
        fit_id="synthetic-fit",
        job_id=55,
        artifact_kind="diagnosed_hmc",
        n_pairs=5,
        show_progress=False,
    )
    ensemble = build_nuclease_hmc_score_ensemble(
        landscape,
        fit_path,
        fitness_distribution="exponential",
        fit_id="synthetic-fit",
        job_id=55,
        artifact_kind="diagnosed_hmc",
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
            matrix_result.selection[name],
            result.selection[name],
        )

    candidate_fluorescence = fluorescence[:, :3]
    expected_top_scores = np.sort(candidate_fluorescence, axis=1)[:, -2:][:, ::-1]
    np.testing.assert_allclose(
        result.selection["score"].sel(score_name=NUCLEASE_SCORE_NAME),
        expected_top_scores,
    )
    np.testing.assert_allclose(
        result.selection["tracked_rank"].sel(score_name=NUCLEASE_SCORE_NAME),
        rankdata(-candidate_fluorescence, axis=1, method="average")[:, [2, 1]],
    )
    np.testing.assert_array_equal(
        result.selection["model_reference_variant_index"].sel(
            score_name=NUCLEASE_SCORE_NAME
        ),
        [0, 1],
    )
    assert result.metadata["analysis_id"] == (
        "nuc__nuc-g1__hmc__exponential__synthetic-fit"
    )
    assert result.metadata["posterior_variable"] == (
        "absolute_mean_log_fluorescence"
    )
    assert result.metadata["scores"] == [NUCLEASE_SCORE_NAME]
    assert result.metadata["n_variants"] == 3
    assert result.metadata["n_fitted_variants"] == 4
    assert result.metadata["n_excluded_from_ranking"] == 1
    assert result.metadata["observed_counts_status"] == "matched"
    assert result.metadata["artifact_status"] == "validated"
    assert result.metadata["job_id"] == 55
    assert result.metadata["artifact_kind"] == "diagnosed_hmc"
