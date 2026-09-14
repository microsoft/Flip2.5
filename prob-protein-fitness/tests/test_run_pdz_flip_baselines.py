import numpy as np
import pandas as pd
import pytest

from scripts.run_pdz_flip_baselines import (
    benchmark_replicates,
    flip2_random_splits,
    load_zero_shot_scores,
    one_hot_flatten,
    proxy_correlations,
    rank_space_compatibility,
    random_split_learning_curves,
    summarize,
    summarize_learning_curves,
    summarize_learning_curves_pooled,
)


def _dataset(replicate: int) -> pd.DataFrame:
    target = np.arange(6, dtype=float) + replicate * 0.1
    published_target = np.array([0.0, 3.0, 1.0, 5.0, 2.0, 4.0])
    return pd.DataFrame(
        {
            "replicate": replicate,
            "sequence": ["AA:CC", "AC:CC", "CA:CC", "CC:CC", "DA:CC", "DC:CC"],
            "target": target,
            "proxy_log_r_mean": target,
            "proxy_count_ratio": np.exp(target),
            "proxy_dimsum_fitness": -target,
            "hmc_source": "trans-1",
            "set": ["train", "train", "train", "train", "test", "test"],
            "validation": [False, False, False, True, False, False],
            "published_target": published_target,
        }
    )


def test_one_hot_removes_separator():
    encoded = one_hot_flatten(pd.Series(["AA:CC", "AC:CC"]))
    assert encoded.shape == (2, 4 * 20)


def test_proxy_correlations_use_sequence_aligned_flip_reported_target(tmp_path):
    datasets = [_dataset(0), _dataset(1)]
    flip_path = tmp_path / "flip.csv"
    datasets[0][["sequence", "published_target"]].rename(
        columns={"published_target": "target"}
    ).to_csv(flip_path, index=False)

    correlations = proxy_correlations(datasets, flip_path=flip_path)
    expected = datasets[0]["published_target"].corr(
        datasets[0]["proxy_log_r_mean"], method="spearman"
    )

    assert len(correlations) == 6
    assert correlations["n_variants"].eq(6).all()
    assert correlations["reference"].eq("FLIP reported target").all()
    assert correlations.loc[
        correlations.proxy.isin(["proxy_log_r_mean", "proxy_count_ratio"]), "spearman"
    ].eq(expected).all()
    assert correlations.loc[
        correlations.proxy.eq("proxy_dimsum_fitness"), "spearman"
    ].eq(-expected).all()
    assert not np.isclose(expected, 1.0)

    datasets[1].loc[0, "published_target"] = 99.0
    with pytest.raises(ValueError, match="does not match the FLIP dataset"):
        proxy_correlations(datasets, flip_path=flip_path)


def test_rank_space_compatibility_uses_fixed_complete_cases(tmp_path):
    datasets = [_dataset(replicate) for replicate in range(8)]
    datasets[3].loc[0, "proxy_count_ratio"] = np.nan
    flip_path = tmp_path / "flip.csv"
    datasets[0][["sequence", "published_target", "set", "validation"]].rename(
        columns={"published_target": "target"}
    ).to_csv(flip_path, index=False)

    summary, calibration, diagnostics = rank_space_compatibility(
        datasets,
        flip_path=flip_path,
        seed=7,
    )

    assert len(summary) == 3
    assert summary["n_reference"].eq(4).all()
    assert summary["n_calibration"].eq(4).all()
    assert summary.loc[summary.proxy == "proxy_count_ratio", "n_variants"].item() == 5
    assert summary.loc[summary.proxy != "proxy_count_ratio", "n_variants"].eq(6).all()
    assert summary["n_sd_floored"].eq(summary["n_variants"]).all()
    assert summary[
        [
            "consensus_spearman_percentile",
            "discrepancy_percentile",
            "upper_tail_fraction",
            "coverage_50",
            "coverage_80",
            "coverage_95",
        ]
    ].apply(lambda values: values.between(0, 1).all()).all()
    assert len(calibration) == 12
    assert calibration.groupby("proxy")["replicate"].nunique().eq(4).all()
    assert len(diagnostics) == 17
    assert "AA:CC" not in set(
        diagnostics.loc[diagnostics.proxy == "proxy_count_ratio", "sequence"]
    )


def test_load_zero_shot_scores_averages_components(tmp_path):
    sequences = pd.Series(["AA:CC", "AC:CC"])
    cache = pd.DataFrame(
        {
            "sequence": sequences,
            "esm2_650m": [np.nan, np.nan],
            "esm2_650m_pdz": [1.0, 3.0],
            "esm2_650m_cript": [3.0, 5.0],
        }
    )
    path = tmp_path / "scores.csv"
    cache.to_csv(path, index=False)

    loaded = load_zero_shot_scores(path, sequences)

    assert loaded.columns.tolist() == ["sequence", "esm2_650m"]
    assert loaded["esm2_650m"].tolist() == [2.0, 4.0]


def test_benchmark_replicates_with_optional_scores():
    datasets = [_dataset(0), _dataset(1)]
    scores = pd.DataFrame(
        {
            "sequence": datasets[0]["sequence"],
            "esm2_650m": np.arange(6, dtype=float),
            "carp_640m": np.arange(6, dtype=float)[::-1],
            "dayhoff": np.array([0.0, 1.0, 3.0, 2.0, 5.0, 4.0]),
        }
    )

    results = benchmark_replicates(datasets, zero_shot_scores=scores)
    summary = summarize(results)

    assert len(results) == 24
    assert set(results["target_source"]) == {"published", "hmc"}
    assert set(results.loc[results["target_source"] == "hmc", "replicate"]) == {0, 1}
    assert results.loc[results["target_source"] == "published", "replicate"].isna().all()
    assert "Ridge (one-hot)" in set(results["model"])
    assert "Ridge (one-hot + likelihoods)" in set(results["model"])
    assert set(results.loc[results["model"].str.contains("zero-shot"), "evaluation_set"]) == {
        "full",
        "test",
    }
    assert summary.loc[summary["target_source"] == "published", "n"].eq(1).all()
    assert summary.loc[summary["target_source"] == "hmc", "n"].eq(2).all()
    published_ridge = summary[
        (summary["target_source"] == "published")
        & (summary["model"] == "Ridge (one-hot)")
    ].iloc[0]
    assert published_ridge["paper_spearman"] == 0.476
    assert np.isclose(
        published_ridge["difference_from_paper"],
        published_ridge["mean"] - 0.476,
    )


def test_benchmark_replicates_accepts_proxy_target_and_missing_train_value():
    datasets = [_dataset(0), _dataset(1)]
    datasets[0].loc[0, "proxy_dimsum_fitness"] = np.nan

    results = benchmark_replicates(datasets, target_column="proxy_dimsum_fitness")

    assert results["target_proxy"].eq("proxy_dimsum_fitness").all()
    assert len(results) == 3
    assert results.loc[results.target_source == "published", "n_train_finite"].item() == 3
    assert results.loc[
        (results.target_source == "hmc") & (results.replicate == 0),
        "n_train_finite",
    ].item() == 2
    assert results["n_test_finite"].eq(2).all()


def test_random_split_learning_curves_use_fixed_schedule():
    datasets = [_dataset(0), _dataset(1)]
    splits = flip2_random_splits(
        len(datasets[0]),
        min_train=2,
        max_fraction=0.8,
        n_training_sizes=2,
        n_repeats=2,
        seed=7,
    )
    curves = random_split_learning_curves(
        datasets,
        min_train=2,
        max_fraction=0.8,
        n_training_sizes=2,
        n_split_repeats=2,
        split_seed=7,
    )
    curve_summary = summarize_learning_curves(curves)
    pooled_summary = summarize_learning_curves_pooled(curves)

    assert len(splits) == 4
    assert len(curves) == 12
    assert set(curves["target_source"]) == {"published", "hmc"}
    assert curve_summary.loc[curve_summary.target_source == "published", "n"].eq(2).all()
    assert curve_summary.loc[curve_summary.target_source == "hmc", "n"].eq(2).all()
    assert pooled_summary.loc[pooled_summary.target_source == "published", "n"].eq(2).all()
    assert pooled_summary.loc[pooled_summary.target_source == "hmc", "n"].eq(4).all()

    proxy_curves = random_split_learning_curves(
        datasets,
        target_column="proxy_count_ratio",
        min_train=2,
        max_fraction=0.8,
        n_training_sizes=2,
        n_split_repeats=2,
        split_seed=7,
    )
    proxy_summary = summarize_learning_curves(proxy_curves)
    assert proxy_curves["target_proxy"].eq("proxy_count_ratio").all()
    assert proxy_summary["target_proxy"].eq("proxy_count_ratio").all()