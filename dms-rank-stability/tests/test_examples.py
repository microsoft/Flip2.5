import numpy as np
import pandas as pd
import pytest

from analysis.examples import representative_examples, run_summaries


def _example_data(values=(0.1, 0.5, 0.9)):
    records = []
    for family in ("pdz", "trpb", "nuc"):
        for index, value in enumerate(values):
            records.append({
                "run_id": f"{family}-{index}", "family": family,
                "landscape": f"{family}-{index}", "model": "exponential",
                "area": value / 2, "spearman": value, "rank_width": value * 100,
                "shared_fraction": value,
            })
    records = pd.DataFrame(records)
    tables = {}
    for sampling in ("parameter_posterior", "mle_predictive", "posterior_predictive"):
        percentiles = records[["run_id", "area", "spearman"]].rename(columns={
            "area": "ecdf_to_uniform_area", "spearman": "external_vs_posterior_median_rank_spearman"
        }).assign(cohort_top_n=100)
        tables[sampling] = {
            "runs": records[["run_id", "family", "landscape", "model"]],
            "rank_percentiles": pd.concat([percentiles] * 4, ignore_index=True),
            "model_rank_intervals": records[["run_id", "rank_width"]].rename(
                columns={"rank_width": "rank_interval_width"}
            ),
            "top_k_curves": records[["run_id", "shared_fraction"]].rename(
                columns={"shared_fraction": "sample_self_consistency"}
            ).assign(k=100),
        }
    fits = records[["family", "landscape", "model"]].assign(
        kind="hmc", native_score_diagnostics_available=True, divergences=0, tree_depth_hits=0,
        r_hat_nonfinite=0, r_hat_extreme=1.001, ess_bulk_nonfinite=0, ess_tail_nonfinite=0,
    )
    return tables, fits


def test_representative_selection_uses_each_run_once_and_is_order_invariant():
    tables, fits = _example_data()
    summaries = run_summaries(tables)
    assert len(summaries) == 27
    examples, audit, candidates = representative_examples(tables, fits)
    assert [example[0] for example in examples] == ["pdz-1", "trpb-1", "nuc-1"]
    assert audit.selected.sum() == 9
    assert candidates.selected.sum() == 3
    shuffled = {name: {key: frame.sample(frac=1, random_state=8) for key, frame in group.items()}
                for name, group in tables.items()}
    assert representative_examples(shuffled, fits.sample(frac=1))[0] == examples
    np.testing.assert_allclose(candidates.loc[candidates.selected, "center_distance"], 0)
    np.testing.assert_allclose(candidates.loc[candidates.selected, "max_mean_distance"], 0)


def test_diagnostic_exclusions_do_not_remove_overview_records():
    tables, fits = _example_data()
    fits.loc[fits.landscape.eq("pdz-1"), "tree_depth_hits"] = 10
    examples, audit, candidates = representative_examples(tables, fits)
    assert examples[0][0] != "pdz-1"
    assert len(audit) == 27 and len(candidates) == 9
    assert not candidates.loc[candidates.landscape.eq("pdz-1"), "eligible"].any()


def test_missing_diagnostic_output_does_not_exclude_a_checked_fit():
    tables, fits = _example_data()
    expected = representative_examples(tables, fits)[0]
    selected = fits.landscape.eq("pdz-1")
    fits.loc[selected, "native_score_diagnostics_available"] = False
    for column in ("r_hat_nonfinite", "r_hat_extreme", "ess_bulk_nonfinite",
                   "ess_tail_nonfinite"):
        fits[column] = fits[column].astype(float)
        fits.loc[selected, column] = np.nan
    examples, _, candidates = representative_examples(tables, fits)
    assert examples == expected
    assert candidates.loc[candidates.landscape.eq("pdz-1"), "eligible"].all()


def test_explicit_examples_are_preserved_and_audited_without_stored_diagnostics():
    tables, fits = _example_data()
    examples = (("pdz-0", "exponential", "PDZ 0"),
                ("trpb-2", "exponential", "TrpB 2"),
                ("nuc-1", "exponential", "NucB 1"))
    fits["native_score_diagnostics_available"] = False
    for column in ("r_hat_nonfinite", "r_hat_extreme", "ess_bulk_nonfinite",
                   "ess_tail_nonfinite"):
        fits[column] = np.nan
    selected, audit, candidates = representative_examples(tables, fits, examples=examples)
    assert selected == examples
    assert audit.selected.sum() == 9 and candidates.selected.sum() == 3
    with pytest.raises(ValueError, match="three distinct"):
        representative_examples(tables, fits, examples=(examples[0],) * 3)
    with pytest.raises(ValueError, match="explicit example"):
        representative_examples(tables, fits, examples=(*examples[:2], ("nuc-absent", "E", "NucB")))


def test_balanced_selection_prioritizes_worst_metric_over_mean_distance():
    tables, fits = _example_data((0.1, 0.3, 0.5, 0.7, 0.9))
    for group in tables.values():
        for suffix, value in (("-0", 0.5), ("-2", 0.1)):
            for name, metric, scale in (
                ("rank_percentiles", "external_vs_posterior_median_rank_spearman", 1),
                ("model_rank_intervals", "rank_interval_width", 100),
                ("top_k_curves", "sample_self_consistency", 1),
            ):
                frame = group[name]
                frame.loc[frame.run_id.str.endswith(suffix), metric] = value * scale
    examples, _, candidates = representative_examples(tables, fits)
    assert [example[0] for example in examples] == ["pdz-1", "trpb-1", "nuc-1"]
    candidates = candidates.set_index("landscape")
    tail_case = candidates.loc["pdz-0"]
    selected = candidates.loc["pdz-1"]
    assert tail_case.center_distance < selected.center_distance
    assert tail_case.max_mean_distance > selected.max_mean_distance
