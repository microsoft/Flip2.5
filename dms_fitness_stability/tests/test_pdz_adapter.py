from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import rankdata

from dms_fitness_stability import analyze_score_ensemble
from dms_fitness_stability.adapters import pdz as pdz_adapter
from dms_fitness_stability.adapters.pdz import (
    CRIPT_WT_SEQUENCE,
    PDZ3_WT_SEQUENCE,
    PDZ_CRIPT_LANDSCAPES,
    PDZ_LANDSCAPES,
    PDZ_LIBRARY_PROVENANCE,
    PDZ_LIBRARY_SPLIT,
    PDZ_PREDICTIVE_SCORE_CONTRACT,
    PDZ_PREDICTIVE_SCORE_NAME,
    PDZ_SCORE_NAMES,
    PDZCisLandscape,
    PDZLandscape,
    PDZTransLandscape,
    _canonicalize_sequence,
    _format_full_sequence,
    _reconstruct_cript_sequences,
    _reconstruct_sequences,
    build_pdz_cis_mle_score_ensemble,
    build_pdz_hmc_count_score_ensemble,
    build_pdz_hmc_score_ensemble,
    build_pdz_mle_score_ensemble,
    load_pdz_cis,
    load_pdz_cript,
    pdz_cis_variant_label,
    pdz_dimsum_fitness,
    pdz_variant_label,
    scan_pdz_trans_hmc,
)


def test_pdz_uses_one_type_for_all_landscapes():
    assert PDZ_CRIPT_LANDSCAPES == ("cript-c", "cript-n")
    assert PDZ_LANDSCAPES == ("trans-1", "trans-2", "cis", "cript-c", "cript-n")
    assert PDZ_LIBRARY_PROVENANCE["cis"] == {
        "geo_library": "cript2xa3_cis",
        "variant_molecule": "CRIPT",
        "reported_fitness_file": "GSE265816_cript2xa3_sub.csv.gz",
    }
    assert PDZ_LIBRARY_PROVENANCE["cript-c"]["geo_library"] == "cript_c"
    assert PDZ_LIBRARY_PROVENANCE["cript-n"]["geo_library"] == "cript_n"
    assert PDZTransLandscape is PDZLandscape
    assert PDZCisLandscape is PDZLandscape


def test_pdz_trans_halves_reconstruct_the_same_wild_type():
    trans_1_variant = PDZ3_WT_SEQUENCE[:PDZ_LIBRARY_SPLIT] + CRIPT_WT_SEQUENCE
    trans_2_variant = PDZ3_WT_SEQUENCE[PDZ_LIBRARY_SPLIT:] + CRIPT_WT_SEQUENCE
    expected = f"{PDZ3_WT_SEQUENCE}:{CRIPT_WT_SEQUENCE}"

    assert _reconstruct_sequences([trans_1_variant], "trans-1") == [expected]
    assert _reconstruct_sequences([trans_2_variant], "trans-2") == [expected]


def test_pdz_cript_halves_reconstruct_the_same_wild_type():
    assert _reconstruct_cript_sequences([CRIPT_WT_SEQUENCE[:4]], "cript-n") == [
        CRIPT_WT_SEQUENCE
    ]
    assert _reconstruct_cript_sequences([CRIPT_WT_SEQUENCE[4:]], "cript-c") == [
        CRIPT_WT_SEQUENCE
    ]


def test_pdz_reported_sequence_format_and_stop_canonicalization():
    full = PDZ3_WT_SEQUENCE + CRIPT_WT_SEQUENCE

    assert _format_full_sequence(full) == f"{PDZ3_WT_SEQUENCE}:{CRIPT_WT_SEQUENCE}"
    assert _canonicalize_sequence(f"{PDZ3_WT_SEQUENCE[:-1]}*:{CRIPT_WT_SEQUENCE}") == (
        f"{PDZ3_WT_SEQUENCE[:-1]}:{CRIPT_WT_SEQUENCE}"
    )
    assert _canonicalize_sequence(f"{PDZ3_WT_SEQUENCE}:KN*") == f"{PDZ3_WT_SEQUENCE}:KN"


def test_pdz_variant_label_formats_substitutions_and_stops():
    assert pdz_variant_label(f"{PDZ3_WT_SEQUENCE}:{CRIPT_WT_SEQUENCE}") == "WT"
    assert pdz_variant_label(f"{PDZ3_WT_SEQUENCE[:-1]}C:{CRIPT_WT_SEQUENCE}") == (
        f"PDZ {PDZ3_WT_SEQUENCE[-1]}{len(PDZ3_WT_SEQUENCE)}C"
    )
    assert pdz_variant_label(f"{PDZ3_WT_SEQUENCE}:{CRIPT_WT_SEQUENCE[:-1]}") == (
        f"CRIPT stop@{len(CRIPT_WT_SEQUENCE)}"
    )


def test_pdz_cis_variant_label_formats_substitutions():
    assert pdz_cis_variant_label(CRIPT_WT_SEQUENCE) == "WT"
    assert pdz_cis_variant_label(f"A{CRIPT_WT_SEQUENCE[1:]}") == "CRIPT K1A"


def test_pdz_sequence_helpers_reject_wrong_lengths():
    with pytest.raises(ValueError, match="108-residue PDZ3-CRIPT"):
        _format_full_sequence("TOO-SHORT")
    with pytest.raises(ValueError, match="fitted PDZ variant"):
        _reconstruct_sequences(["TOO-SHORT"], "trans-1")


def test_load_pdz_cis_aligns_reported_fitness_to_full_fit(monkeypatch, tmp_path):
    raw_variants = [
        CRIPT_WT_SEQUENCE,
        f"A{CRIPT_WT_SEQUENCE[1:]}",
        f"{CRIPT_WT_SEQUENCE[:-1]}A",
        f"*{CRIPT_WT_SEQUENCE[1:]}",
    ]
    starting = np.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=np.int64)
    timepoint = np.array([[20, 60, 60, 40], [2, 6, 6, 4]], dtype=np.int64)
    monkeypatch.setattr(
        pdz_adapter,
        "load_pdz_dataset",
        lambda _: {
            "variants": raw_variants,
            "starting_counts": starting,
            "timepoint_counts": timepoint,
        },
    )
    reported_path = tmp_path / "reported.csv"
    pd.DataFrame(
        {
            "aa_seq": raw_variants[:3],
            "WT": ["TRUE", pd.NA, pd.NA],
            "fitness": [0.0, 2.0, 1.0],
            "sigma": [0.1, 0.2, 0.3],
        }
    ).to_csv(reported_path, index=False)

    landscape = load_pdz_cis(tmp_path / "counts.tsv", reported_fitness_path=reported_path)

    assert isinstance(landscape, PDZCisLandscape)
    assert landscape.source == "cis"
    assert landscape.n_hmc_variants == 4
    assert landscape.wild_type_index == 0
    assert landscape.variants["variant_id"].tolist() == raw_variants[:3]
    assert landscape.variants["hmc_variant_index"].tolist() == [0, 1, 2]
    assert landscape.fitted_variant_ids == tuple(raw_variants)
    assert landscape.variants["input_count"].tolist() == [11, 22, 33]
    assert landscape.variants["output_count"].tolist() == [22, 66, 66]
    assert landscape.variants["reported_rank"].tolist() == [3.0, 1.0, 2.0]


def test_build_pdz_cis_mle_score_ensemble_preserves_axis_and_candidate_subset(
    tmp_path,
):
    variant_ids = (
        CRIPT_WT_SEQUENCE,
        f"A{CRIPT_WT_SEQUENCE[1:]}",
        f"C{CRIPT_WT_SEQUENCE[1:]}",
        f"D{CRIPT_WT_SEQUENCE[1:]}",
    )
    observed_starting = np.array(
        [[100, 80, 60, 40], [90, 70, 50, 30]],
        dtype=np.int64,
    )
    observed_timepoint = np.array(
        [[100, 160, 60, 20], [90, 140, 100, 15]],
        dtype=np.int64,
    )
    observed_fitness = pdz_dimsum_fitness(
        observed_starting,
        observed_timepoint,
        0,
    )
    variants = pd.DataFrame(
        {
            "variant_id": variant_ids[:3],
            "source": ["cis"] * 3,
            "variant_label": ["WT", "K1A", "K1C"],
            "input_count": observed_starting[:, :3].sum(axis=0),
            "output_count": observed_timepoint[:, :3].sum(axis=0),
            "count_ratio": [1.0, 2.0, 1.6],
            "reported_fitness": observed_fitness[:3],
            "reference_fitness": observed_fitness[:3],
            "reference_fitness_source": ["reported"] * 3,
            "hmc_variant_index": np.arange(3, dtype=np.int32),
            "reported_fitness_sigma": [0.1] * 3,
            "reported_rank": pd.Series(observed_fitness[:3]).rank(
                ascending=False
            ),
        }
    )
    landscape = PDZLandscape(
        source="cis",
        variants=variants,
        starting_counts=observed_starting,
        timepoint_counts=observed_timepoint,
        wild_type_index=0,
        fitted_variant_ids=variant_ids,
    )
    predictive_starting = np.stack(
        [observed_starting, observed_starting],
    )
    predictive_timepoint = np.stack(
        [
            observed_timepoint,
            np.array([[100, 80, 120, 40], [90, 70, 150, 35]]),
        ]
    )
    fit_path = tmp_path / "pdz_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=observed_starting)
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

    ensemble = build_pdz_cis_mle_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="lomax",
        growth_curve="exponential",
        fit_id="mle-3",
        job_id=3,
        variant_chunk_size=2,
    )

    expected = pdz_dimsum_fitness(
        predictive_starting,
        predictive_timepoint,
        0,
    )
    assert PDZ_PREDICTIVE_SCORE_NAME == "dimsum_fitness"
    assert PDZ_PREDICTIVE_SCORE_CONTRACT == "pdz-dimsum-fitness-v1"
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(score_name="dimsum_fitness"),
        expected,
    )
    assert ensemble.source_schema_version == 2
    assert ensemble.variants["variant_id"].tolist() == list(variant_ids)
    assert ensemble.variants["is_ranking_candidate"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert ensemble.metadata["score_status"] == {"dimsum_fitness": "provisional"}
    assert ensemble.metadata["n_ranking_candidates"] == 3
    assert ensemble.metadata["variant_chunk_size"] == 2


def test_build_pdz_mle_score_ensemble_accepts_cript_landscape(tmp_path):
    variant_ids = (
        CRIPT_WT_SEQUENCE,
        f"A{CRIPT_WT_SEQUENCE[1:]}",
        f"C{CRIPT_WT_SEQUENCE[1:]}",
    )
    starting = np.array([[100, 80, 60], [90, 70, 50]], dtype=np.int64)
    timepoint = np.array([[100, 160, 60], [90, 140, 100]], dtype=np.int64)
    observed_fitness = pdz_dimsum_fitness(starting, timepoint, 0)
    variants = pd.DataFrame(
        {
            "variant_id": variant_ids,
            "source": ["cript-n"] * 3,
            "variant_label": ["WT", "K1A", "K1C"],
            "input_count": starting.sum(axis=0),
            "output_count": timepoint.sum(axis=0),
            "count_ratio": timepoint.sum(axis=0) / starting.sum(axis=0),
            "reported_fitness": observed_fitness,
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 3,
            "hmc_variant_index": np.arange(3, dtype=np.int32),
            "reported_fitness_sigma": [0.1] * 3,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
        }
    )
    landscape = PDZLandscape(
        source="cript-n",
        variants=variants,
        starting_counts=starting,
        timepoint_counts=timepoint,
        wild_type_index=0,
        fitted_variant_ids=variant_ids,
    )
    fit_path = tmp_path / "pdz_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=starting)
        observed.create_dataset("timepoint_counts", data=timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset(
            "starting_counts",
            data=np.stack([starting, starting])[None],
        )
        predictive.create_dataset(
            "timepoint_counts",
            data=np.stack([timepoint, timepoint])[None],
        )

    ensemble = build_pdz_mle_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="exponential",
        growth_curve="exponential",
        fit_id="mle-76",
        job_id=76,
    )

    assert ensemble.metadata["analysis_id"] == (
        "pdz__pdz-cript-n__mle-predictive__exponential-exponential__mle-76"
    )
    assert ensemble.metadata["observed_agreement"]["n_compared"] == 3


def test_build_pdz_hmc_count_score_ensemble_preserves_chain_draws_and_formula(
    tmp_path,
):
    variant_ids = (
        CRIPT_WT_SEQUENCE,
        f"A{CRIPT_WT_SEQUENCE[1:]}",
        f"C{CRIPT_WT_SEQUENCE[1:]}",
    )
    observed_starting = np.array([[100, 80, 60], [90, 70, 50]], dtype=np.int64)
    observed_timepoint = np.array([[100, 160, 60], [90, 140, 100]], dtype=np.int64)
    observed_fitness = pdz_dimsum_fitness(
        observed_starting,
        observed_timepoint,
        0,
    )
    variants = pd.DataFrame(
        {
            "variant_id": variant_ids,
            "source": ["cis"] * 3,
            "variant_label": ["WT", "K1A", "K1C"],
            "input_count": observed_starting.sum(axis=0),
            "output_count": observed_timepoint.sum(axis=0),
            "count_ratio": observed_timepoint.sum(axis=0)
            / observed_starting.sum(axis=0),
            "reported_fitness": observed_fitness,
            "reference_fitness": observed_fitness,
            "reference_fitness_source": ["reported"] * 3,
            "hmc_variant_index": np.arange(3, dtype=np.int32),
            "reported_fitness_sigma": [0.1] * 3,
            "reported_rank": pd.Series(observed_fitness).rank(ascending=False),
        }
    )
    landscape = PDZLandscape(
        source="cis",
        variants=variants,
        starting_counts=observed_starting,
        timepoint_counts=observed_timepoint,
        wild_type_index=0,
        fitted_variant_ids=variant_ids,
    )
    predictive_starting = np.broadcast_to(
        observed_starting,
        (2, 2, *observed_starting.shape),
    ).copy()
    predictive_timepoint = np.stack(
        [
            observed_timepoint,
            np.array([[100, 80, 140], [90, 70, 170]]),
            np.array([[100, 120, 100], [90, 105, 135]]),
            np.array([[100, 200, 20], [90, 175, 65]]),
        ]
    ).reshape(2, 2, *observed_timepoint.shape)
    fit_path = tmp_path / "pdz_hmc.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        observed.create_dataset("starting_counts", data=observed_starting)
        observed.create_dataset("timepoint_counts", data=observed_timepoint)
        predictive = handle.create_group("posterior_predictive")
        predictive.create_dataset("starting_counts", data=predictive_starting)
        predictive.create_dataset("timepoint_counts", data=predictive_timepoint)

    ensemble = build_pdz_hmc_count_score_ensemble(
        landscape,
        fit_path,
        growth_rate_prior="lomax",
        growth_curve="exponential",
        fit_id="small-clean_3",
        job_id=3,
        artifact_kind="diagnosed_hmc",
        sample_chunk_size=1,
        variant_chunk_size=2,
        scratch_dir=tmp_path / "scratch",
        show_progress=False,
    )
    temporary_path = ensemble.temporary_paths[0]
    try:
        expected = pdz_dimsum_fitness(
            predictive_starting,
            predictive_timepoint,
            0,
        ).reshape(4, 3)
        np.testing.assert_allclose(
            ensemble.fitness_score.sel(score_name="dimsum_fitness"),
            expected,
        )
        np.testing.assert_array_equal(ensemble.scores["source_sample_id"], [0, 1, 2, 3])
        np.testing.assert_array_equal(ensemble.scores["source_chain"], [0, 0, 1, 1])
        np.testing.assert_array_equal(ensemble.scores["source_draw"], [0, 1, 0, 1])
        assert ensemble.source_schema_version == 2
        assert ensemble.metadata["inference_method"] == "hmc_count_based"
        assert ensemble.metadata["n_chains"] == 2
        assert ensemble.metadata["draws_per_chain"] == 2
        assert temporary_path.is_file()
    finally:
        ensemble.close()
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("source", "raw_variants", "full_variants"),
    [
        ("cript-n", ["KNYK", "ANYK", "CNYK"], ["KNYKQTSV", "ANYKQTSV", "CNYKQTSV"]),
        ("cript-c", ["QTSV", "ATSV", "CTSV"], ["KNYKQTSV", "KNYKATSV", "KNYKCTSV"]),
    ],
)
def test_load_pdz_cript_aligns_full_reported_sequences(
    monkeypatch,
    tmp_path,
    source,
    raw_variants,
    full_variants,
):
    starting = np.array([[10, 20, 30], [1, 2, 3]], dtype=np.int64)
    timepoint = np.array([[20, 60, 60], [2, 6, 6]], dtype=np.int64)
    monkeypatch.setattr(
        pdz_adapter,
        "load_pdz_dataset",
        lambda _: {
            "variants": raw_variants,
            "starting_counts": starting,
            "timepoint_counts": timepoint,
        },
    )
    reported_path = tmp_path / "reported.csv"
    pd.DataFrame(
        {
            "aa_seq": full_variants,
            "WT": ["TRUE", pd.NA, pd.NA],
            "fitness": [0.0, 2.0, 1.0],
            "sigma": [0.1, 0.2, 0.3],
        }
    ).to_csv(reported_path, index=False)

    landscape = load_pdz_cript(
        source,
        tmp_path / "counts.tsv",
        reported_fitness_path=reported_path,
    )

    assert landscape.source == source
    assert landscape.n_hmc_variants == 3
    assert landscape.wild_type_index == 0
    assert landscape.variants["variant_id"].tolist() == full_variants
    assert landscape.variants["hmc_variant_index"].tolist() == [0, 1, 2]
    assert landscape.variants["reported_rank"].tolist() == [3.0, 1.0, 2.0]


def _write_synthetic_hmc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    growth = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 2.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    starting = np.array(
        [
            [[10, 10, 10, 10], [10, 10, 10, 10]],
            [[10, 10, 10, 10], [10, 10, 10, 10]],
            [[10, 10, 10, 10], [10, 10, 10, 10]],
        ],
        dtype=np.int32,
    )
    timepoint = np.array(
        [
            [[40, 30, 20, 10], [40, 30, 20, 10]],
            [[10, 40, 30, 20], [10, 40, 30, 20]],
            [[40, 20, 30, 10], [40, 20, 30, 10]],
        ],
        dtype=np.int32,
    )
    xr.Dataset({"log_r_mean": (("chain", "draw", "b"), growth[None])}).to_netcdf(
        path,
        group="posterior",
        engine="h5netcdf",
        encoding={"log_r_mean": {"chunksizes": (1, 3, 2)}},
    )
    xr.Dataset(
        {
            "starting_counts": (("chain", "draw", "a", "b"), starting[None]),
            "timepoint_counts": (("chain", "draw", "a", "b"), timepoint[None]),
        }
    ).to_netcdf(
        path,
        mode="a",
        group="posterior_predictive",
        engine="h5netcdf",
        encoding={
            "starting_counts": {"chunksizes": (1, 3, 1, 2)},
            "timepoint_counts": {"chunksizes": (1, 3, 1, 2)},
        },
    )
    return growth, starting, timepoint


def _synthetic_landscape() -> PDZTransLandscape:
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D"],
            "source": ["trans-1"] * 4,
            "input_count": [20] * 4,
            "output_count": [80, 60, 40, 20],
            "count_ratio": [4.0, 3.0, 2.0, 1.0],
            "reported_fitness": [0.1, 0.4, 0.3, 0.2],
            "reference_fitness": [0.1, 0.4, 0.3, 0.2],
            "reference_fitness_source": ["reported"] * 4,
            "hmc_variant_index": np.arange(4, dtype=np.int32),
            "reported_fitness_sigma": [0.1] * 4,
            "reported_rank": [4.0, 1.0, 2.0, 3.0],
        }
    )
    observed_starting = np.full((2, 4), 10, dtype=np.int64)
    observed_timepoint = np.array([[40, 30, 20, 10], [40, 30, 20, 10]])
    return PDZTransLandscape(
        source="trans-1",
        variants=variants,
        starting_counts=observed_starting,
        timepoint_counts=observed_timepoint,
        wild_type_index=3,
        fitted_variant_ids=("A", "B", "C", "D"),
    )


def test_full_pdz_hmc_scan_matches_dense_scores_and_ranks(tmp_path):
    hmc_path = tmp_path / "fit.nc"
    growth, starting, timepoint = _write_synthetic_hmc(hmc_path)
    landscape = _synthetic_landscape()
    result = scan_pdz_trans_hmc(
        landscape,
        hmc_path,
        top_k=(1, 2),
        retain_k=2,
        tracked_top_n=3,
        tracked_model_top_n=1,
        fit_id="synthetic-fit",
        n_pairs=5,
        show_progress=False,
    )
    ensemble = build_pdz_hmc_score_ensemble(
        landscape,
        hmc_path,
        growth_rate_prior="exponential",
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
            tracked_top_n=3,
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

    count_ratio = timepoint.sum(axis=1) / starting.sum(axis=1)
    wild_type_starting = starting[:, :, 3, None]
    wild_type_timepoint = timepoint[:, :, 3, None]
    replicate_fitness = np.log(timepoint / starting) - np.log(
        wild_type_timepoint / wild_type_starting
    )
    variance = 1 / starting + 1 / timepoint + 1 / wild_type_starting + 1 / wild_type_timepoint
    dimsum = (replicate_fitness / variance).sum(axis=1) / (1 / variance).sum(axis=1)
    dense_scores = {
        "log_mean_growth": growth,
        "count_ratio": count_ratio,
        "dimsum_fitness": dimsum,
    }
    tracked_indices = np.array([1, 2, 3])
    for score_name in PDZ_SCORE_NAMES:
        scores = dense_scores[score_name]
        expected_top_scores = np.sort(scores, axis=1)[:, -2:][:, ::-1]
        expected_ranks = rankdata(-scores, axis=1, method="average")[:, tracked_indices]
        np.testing.assert_allclose(
            result.selection["score"].sel(score_name=score_name),
            expected_top_scores,
        )
        np.testing.assert_allclose(
            result.selection["tracked_score"].sel(score_name=score_name),
            scores[:, tracked_indices],
        )
        np.testing.assert_allclose(
            result.selection["tracked_rank"].sel(score_name=score_name),
            expected_ranks,
        )
    np.testing.assert_array_equal(
        result.selection["model_reference_variant_index"].sel(
            score_name="log_mean_growth"
        ),
        [0, 1],
    )
    np.testing.assert_allclose(
        result.selection["model_reference_score"].sel(
            score_name="log_mean_growth"
        ),
        [4.0, 3.0],
    )
    np.testing.assert_array_equal(
        result.selection["model_tracked_variant_index"].sel(
            score_name="log_mean_growth"
        ),
        [0],
    )
    np.testing.assert_allclose(
        result.selection["model_tracked_rank"].sel(score_name="log_mean_growth"),
        rankdata(-growth, axis=1, method="average")[:, [0]],
    )
    model_row = result.summary[
        (result.summary["score"] == "log_mean_growth")
        & (result.summary["k"] == 1)
        & (result.summary["variant_index"] == 0)
    ].iloc[0]
    assert model_row["is_tracked_model_cohort"]
    assert model_row["model_reference_rank"] == 1
    assert model_row["posterior_median_rank"] == 1
    assert result.metadata["n_samples"] == 3
    assert result.metadata["analysis_id"] == (
        "pdz__pdz-trans-1__hmc__exponential-exponential__synthetic-fit"
    )
    assert result.metadata["tracked_model_top_n"] == 1
    assert result.metadata["geo_library"] == "pdz3cript_b1"
    assert result.metadata["variant_molecule"] == "PDZ3-CRIPT"
    assert result.metadata["observed_counts_status"] == "not_embedded"
    assert set(result.summary["k"]) == {1, 2}
