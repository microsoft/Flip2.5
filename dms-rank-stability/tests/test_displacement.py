from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import manuscript


def _collection(sampling: str, medians: list[float]) -> dict[str, pd.DataFrame]:
    run_id = f"pdz-synthetic-{sampling}"
    runs = pd.DataFrame({
        "run_id": [run_id], "family": ["pdz"], "landscape": ["pdz-synthetic"],
        "model": ["exponential-exponential"],
    })
    percentiles = pd.DataFrame({
        "run_id": [run_id] * 4, "variant_index": [0, 1, 2, 3],
        "cohort_top_n": [100] * 4, "reported_rank": [1.0, 2.0, 3.0, 4.0],
        "posterior_median_sampled_rank": medians,
    })
    other_cohort = percentiles.assign(cohort_top_n=1000, posterior_median_sampled_rank=10000)
    return {"runs": runs, "rank_percentiles": pd.concat([percentiles, other_cohort])}


def test_rank_displacement_averages_absolute_differences_without_reranking():
    collection = _collection("parameter_posterior", [3.0, 1.0, 3.0, 404.0])
    original = collection["rank_percentiles"].copy(deep=True)
    rows = manuscript.rank_displacement_rows({"parameter_posterior": collection})
    assert len(rows) == 1
    assert rows.iloc[0].mean_absolute_rank_displacement == 100.75
    assert rows.iloc[0].supported_candidates == 4
    assert rows.iloc[0].cohort_top_n == 100
    assert rows.iloc[0].sampling == "parameter_posterior"
    pd.testing.assert_frame_equal(collection["rank_percentiles"], original)


def test_rank_displacement_preserves_zero_and_counts_only_finite_pairs():
    tables = {
        "parameter_posterior": _collection("parameter_posterior", [1.0, 2.0, 3.0, 4.0]),
        "mle_predictive": _collection("mle_predictive", [3.0, np.nan, np.inf, 4.5]),
    }
    rows = manuscript.rank_displacement_rows(tables).set_index("sampling")
    assert rows.loc["parameter_posterior", "mean_absolute_rank_displacement"] == 0
    assert rows.loc["parameter_posterior", "supported_candidates"] == 4
    assert rows.loc["mle_predictive", "mean_absolute_rank_displacement"] == 1.25
    assert rows.loc["mle_predictive", "supported_candidates"] == 2


def test_rank_displacement_table_aligns_schemes_and_retains_unmatched_models(tmp_path: Path):
    parameter = _collection("parameter_posterior", [3.0, 1.0, 3.0, 404.0])
    optimized = _collection("mle_predictive", [1.0, 2.0, 3.0, 4.0])
    extra_run = optimized["runs"].assign(run_id="optimized-only", model="gamma-exponential")
    extra_values = optimized["rank_percentiles"].assign(
        run_id="optimized-only", posterior_median_sampled_rank=5.0
    )
    optimized["runs"] = pd.concat([optimized["runs"], extra_run], ignore_index=True)
    optimized["rank_percentiles"] = pd.concat(
        [optimized["rank_percentiles"], extra_values], ignore_index=True
    )
    tables = {"mle_predictive": optimized, "parameter_posterior": parameter}
    manuscript.export_rank_displacement(tables, tmp_path)
    rows = pd.read_csv(tmp_path / "rank_displacement.csv")
    assert len(rows) == 3 and rows.run_id.is_unique
    assert set(rows.model) == {"exponential-exponential", "gamma-exponential"}
    optimized_only = rows.loc[rows.run_id.eq("optimized-only")].iloc[0]
    assert optimized_only.mean_absolute_rank_displacement == 2.5
    tex = (tmp_path / "rank_displacement.tex").read_text()
    assert r"\begin{longtable}[c]{llrrr}" in tex
    assert r"\endfirsthead" in tex and r"\endhead" in tex
    assert "Parameter posterior & Optimized-fit counts & Posterior counts" in tex
    assert "PDZ synthetic & E/E & 100.8 & 0.0 & --" in tex
    assert "PDZ synthetic & G/E & -- & 2.5 & --" in tex
    assert "tab:rank-displacement" in tex


def test_rank_displacement_rejects_duplicate_candidates_and_missing_runs():
    collection = _collection("parameter_posterior", [1.0, 2.0, 3.0, 4.0])
    values = collection["rank_percentiles"]
    collection["rank_percentiles"] = pd.concat([values, values.iloc[[0]]])
    with pytest.raises(ValueError, match="one row per candidate"):
        manuscript.rank_displacement_rows({"parameter_posterior": collection})
    collection["rank_percentiles"] = values[values.cohort_top_n.eq(1000)]
    with pytest.raises(ValueError, match="Every declared run"):
        manuscript.rank_displacement_rows({"parameter_posterior": collection})
