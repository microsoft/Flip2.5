import h5py
import numpy as np
import pandas as pd
import pytest

from dms_fitness_stability.adapters import k50 as k50_adapter
from dms_fitness_stability.adapters.k50 import (
    K50_PREDICTIVE_SCORE_CONTRACT,
    K50_PREDICTIVE_SCORE_NAME,
    K50_PREDICTIVE_SCORE_NAMES,
    build_k50_mle_score_ensemble,
    k50_mean_log_enrichment_by_protease,
    k50_mean_log_protease_enrichment,
    load_k50_landscape,
)


def _library_counts(n_variants: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    c0 = np.arange(2 * 2 * n_variants).reshape(2, 2, n_variants) + offset
    cg0 = np.arange(2 * 2 * 11 * n_variants).reshape(2, 2, 11, n_variants) + offset
    return c0, cg0


def _dataset() -> dict[str, object]:
    dataset: dict[str, object] = {
        "seq_ids": {"BBB": 0, "AAA": 1, "CCC": 2},
        "ordinal_seqs": np.zeros((3, 86), dtype=int),
        "scrambled_inds": np.array([1], dtype=int),
        "qpcr_log_protease_conc": np.arange(11, dtype=float),
        "qpcr_log2_survival": np.zeros((2, 11, 8), dtype=float),
        "log_expected_protease_conc": np.zeros((2, 11), dtype=float),
    }
    for library, indices, offset in (
        ("v1", [0, 1], 0),
        ("v2", [1, 2], 1_000),
        ("v3", [0], 2_000),
        ("v4", [2], 3_000),
    ):
        c0, cg0 = _library_counts(len(indices), offset)
        dataset[f"{library}_seqids"] = np.asarray(indices)
        dataset[f"{library}_counts_c0"] = c0
        dataset[f"{library}_counts_cg0"] = cg0
    return dataset


def _reported_fitness(path) -> None:
    pd.DataFrame(
        {
            "variant_id": ["AAA", "AAA", "BBB", "CCC"],
            "source_aa_seq_full": ["AAA", "AAA", "BBB", "CCC"],
            "reported_fitness_raw": [4.0, 6.0, -2.0, 1.0],
            "reported_fitness_95ci_width": [0.1, 0.3, 0.2, 0.4],
            "reported_fitness_ml": ["4.0", ">5", "<-1", "-"],
            "reported_wt_name": ["domain-a", "domain-a", "domain-b", "domain-c"],
            "reported_wt_cluster": ["a", "a", "b", "c"],
        }
    ).to_csv(path, index=False)


def test_k50_mean_log_protease_enrichment_preserves_draws():
    untreated = np.ones((2, 2, 2, 2), dtype=np.int64)
    treated = np.ones((2, 2, 2, 11, 2), dtype=np.int64)
    treated[0, ..., 0] = 3
    treated[1, ..., 1] = 3

    score = k50_mean_log_protease_enrichment(untreated, treated)

    expected_high = np.log(1.4)
    expected_low = np.log(0.6)
    assert K50_PREDICTIVE_SCORE_NAME == "mean_log_protease_enrichment"
    assert K50_PREDICTIVE_SCORE_CONTRACT == (
        "k50-mean-log-protease-enrichment-v1"
    )
    np.testing.assert_allclose(
        score,
        [[expected_high, expected_low], [expected_low, expected_high]],
    )


def test_k50_mean_log_enrichment_by_protease_preserves_protease_axis():
    untreated = np.ones((2, 2, 2), dtype=np.int64)
    treated = np.ones((2, 2, 11, 2), dtype=np.int64)
    treated[0, ..., 0] = 3
    treated[1, ..., 1] = 3

    by_protease = k50_mean_log_enrichment_by_protease(untreated, treated)

    expected_high = np.log(1.4)
    expected_low = np.log(0.6)
    np.testing.assert_allclose(
        by_protease,
        [[expected_high, expected_low], [expected_low, expected_high]],
    )
    np.testing.assert_allclose(
        by_protease.mean(axis=-2),
        k50_mean_log_protease_enrichment(untreated, treated),
    )


def test_load_k50_landscape_preserves_global_and_library_order(tmp_path, monkeypatch):
    source = _dataset()
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: source)
    reported_path = tmp_path / "reported.csv"
    _reported_fitness(reported_path)

    landscape = load_k50_landscape(
        tmp_path,
        reported_fitness_path=reported_path,
        expected_scrambled_count=None,
    )

    assert landscape.variants["variant_id"].tolist() == ["BBB", "AAA", "CCC"]
    assert landscape.variants["global_sequence_index"].tolist() == [0, 1, 2]
    assert landscape.variants["is_scrambled"].tolist() == [False, True, False]
    assert landscape.variants["is_source_control"].tolist() == [False, True, False]
    assert landscape.variants["is_ranking_candidate"].tolist() == [True, False, True]
    assert landscape.libraries["v1"].variants == ("BBB", "AAA")
    assert landscape.libraries["v2"].variants == ("AAA", "CCC")
    np.testing.assert_array_equal(
        landscape.libraries["v1"].counts_c0,
        source["v1_counts_c0"],
    )
    np.testing.assert_array_equal(
        landscape.libraries["v1"].counts_cg0,
        source["v1_counts_cg0"],
    )
    assert landscape.libraries["v1"].counts_c0.shape == (2, 2, 2)
    assert landscape.libraries["v1"].counts_cg0.shape == (2, 2, 11, 2)
    assert set(landscape.variants["reference_identity"]) == {
        "domain-specific; no global wild type"
    }
    assert landscape.variants["reported_fitness"].tolist()[:2] == [-1.0, 4.5]
    assert np.isnan(landscape.variants.loc[2, "reported_fitness"])
    assert landscape.variants["reported_fitness_raw"].tolist()[:2] == [-2.0, 5.0]
    assert landscape.variants["reported_fitness_status"].tolist()[:2] == [
        "below_lower_limit",
        "mixed",
    ]
    assert landscape.variants["reported_measurements"].tolist()[:2] == [1, 2]
    assert landscape.variants["reported_rank"].tolist()[:2] == [2.0, 1.0]


def test_build_k50_mle_score_ensemble_aggregates_library_occurrences(
    tmp_path, monkeypatch
):
    source = _dataset()
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: source)
    reported_path = tmp_path / "reported.csv"
    _reported_fitness(reported_path)
    landscape = load_k50_landscape(
        tmp_path,
        reported_fitness_path=reported_path,
        expected_scrambled_count=None,
    )
    observed_sum = np.zeros((2, landscape.n_fitted_variants))
    occurrences = np.zeros(landscape.n_fitted_variants, dtype=np.int8)
    for library in landscape.libraries.values():
        observed_sum[:, library.global_sequence_indices] += (
            k50_mean_log_enrichment_by_protease(
                library.counts_c0,
                library.counts_cg0,
            )
        )
        occurrences[library.global_sequence_indices] += 1
    observed_global = observed_sum / occurrences[None]
    candidate = landscape.variants["is_ranking_candidate"].to_numpy(dtype=bool)
    landscape.variants.loc[candidate, "reported_fitness"] = observed_global[
        :, candidate
    ].mean(axis=0)
    landscape.variants.loc[candidate, "reference_fitness"] = landscape.variants.loc[
        candidate, "reported_fitness"
    ]

    predictive_counts = {}
    fit_path = tmp_path / "k50_samples.nc"
    with h5py.File(fit_path, "w") as handle:
        observed = handle.create_group("observed_data")
        predictive = handle.create_group("posterior_predictive")
        for library_name, library in landscape.libraries.items():
            observed.create_dataset(
                f"{library_name}_counts_c0",
                data=library.counts_c0,
            )
            observed.create_dataset(
                f"{library_name}_counts_cg0",
                data=library.counts_cg0,
            )
            predictive_c0 = np.stack(
                [library.counts_c0, library.counts_c0[..., ::-1]]
            )
            predictive_cg0 = np.stack(
                [library.counts_cg0, library.counts_cg0[..., ::-1]]
            )
            predictive_counts[library_name] = (predictive_c0, predictive_cg0)
            predictive.create_dataset(
                f"{library_name}_counts_c0",
                data=predictive_c0[None],
            )
            predictive.create_dataset(
                f"{library_name}_counts_cg0",
                data=predictive_cg0[None],
            )

    ensemble = build_k50_mle_score_ensemble(
        landscape,
        fit_path,
        fit_id="k50-mle",
        job_id=63,
        scratch_dir=tmp_path,
        variant_chunk_size=1,
        show_progress=False,
    )
    temporary_path = ensemble.temporary_paths[0]
    try:
        expected_sum = np.zeros((2, 2, landscape.n_fitted_variants))
        for library_name, library in landscape.libraries.items():
            c0, cg0 = predictive_counts[library_name]
            by_protease = k50_mean_log_enrichment_by_protease(c0, cg0)
            expected_sum[:, :, library.global_sequence_indices] += np.moveaxis(
                by_protease,
                -2,
                0,
            )
        expected = expected_sum / occurrences[None, None]

        np.testing.assert_allclose(
            ensemble.fitness_score,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        assert ensemble.scores["score_name"].values.tolist() == list(
            K50_PREDICTIVE_SCORE_NAMES
        )
        assert ensemble.source_schema_version == 2
        assert temporary_path.exists()
        assert ensemble.variants["is_ranking_candidate"].tolist() == [
            True,
            False,
            True,
        ]
        assert ensemble.metadata["score_status"] == {
            score_name: "sensitivity" for score_name in K50_PREDICTIVE_SCORE_NAMES
        }
        assert ensemble.metadata["reference_top_k_identifiable"] is False
        assert ensemble.metadata["min_finite_variants_per_sample"] == 3
        assert ensemble.metadata["max_finite_variants_per_sample"] == 3
    finally:
        ensemble.close()
    assert not temporary_path.exists()


def test_load_k50_landscape_preserves_calibration_axes(tmp_path, monkeypatch):
    source = _dataset()
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: source)
    reported_path = tmp_path / "reported.csv"
    _reported_fitness(reported_path)

    landscape = load_k50_landscape(
        tmp_path,
        reported_fitness_path=reported_path,
        expected_scrambled_count=None,
    )

    assert landscape.ordinal_sequences.shape == (3, 86)
    assert landscape.qpcr_log_protease_concentration.shape == (11,)
    assert landscape.qpcr_log2_survival.shape == (2, 11, 8)
    assert landscape.log_expected_protease_concentration.shape == (2, 11)


def test_load_k50_landscape_exposes_censored_rank_bounds(tmp_path, monkeypatch):
    source = _dataset()
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: source)
    reported_path = tmp_path / "reported.csv"
    pd.DataFrame(
        {
            "variant_id": ["AAA", "BBB", "CCC"],
            "source_aa_seq_full": ["AAA", "BBB", "CCC"],
            "reported_fitness_raw": [6.2, 5.8, 1.0],
            "reported_fitness_95ci_width": [0.2, 0.3, 0.4],
            "reported_fitness_ml": [">5", ">5", "1.0"],
            "reported_wt_name": ["domain-a", "domain-b", "domain-c"],
            "reported_wt_cluster": ["a", "b", "c"],
        }
    ).to_csv(reported_path, index=False)

    landscape = load_k50_landscape(
        tmp_path,
        reported_fitness_path=reported_path,
        expected_scrambled_count=None,
    )
    variants = landscape.variants.set_index("variant_id")

    assert variants.loc[["AAA", "BBB"], "reported_rank"].tolist() == [1.5, 1.5]
    assert variants.loc[["AAA", "BBB"], "reported_rank_min"].tolist() == [1.0, 1.0]
    assert variants.loc[["AAA", "BBB"], "reported_rank_max"].tolist() == [2.0, 2.0]
    assert variants.loc["CCC", "reported_rank_min"] == 3.0
    assert variants.loc["CCC", "reported_rank_max"] == 3.0


def test_load_k50_landscape_validates_scramble_cardinality(tmp_path, monkeypatch):
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: _dataset())
    reported_path = tmp_path / "reported.csv"
    _reported_fitness(reported_path)

    with pytest.raises(ValueError, match="Expected 64,238 scrambled sequences"):
        load_k50_landscape(tmp_path, reported_fitness_path=reported_path)


def test_load_k50_landscape_rejects_transposed_count_axes(tmp_path, monkeypatch):
    source = _dataset()
    source["v1_counts_cg0"] = np.zeros((2, 11, 2, 2), dtype=int)
    monkeypatch.setattr(k50_adapter, "load_k50_data", lambda data_dir: source)
    reported_path = tmp_path / "reported.csv"
    _reported_fitness(reported_path)

    with pytest.raises(ValueError, match="treated counts must have shape"):
        load_k50_landscape(
            tmp_path,
            reported_fitness_path=reported_path,
            expected_scrambled_count=None,
        )
