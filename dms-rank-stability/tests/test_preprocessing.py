import h5py
import numpy as np
import pandas as pd

from dms_rank_stability import SampledScores
from preprocessing.fitness import (
    nuc_g3_highest_gate_enrichment,
    pdz_dimsum_fitness,
    trpb_four_site_fitness,
    trpb_three_site_fitness,
)
from preprocessing.predictive import (
    extract_nuc_g3_predictive_scores,
    extract_pdz_predictive_scores,
    extract_trpb_predictive_scores,
)
from preprocessing.scores import (
    compare_score_data,
    extract_parameter_scores,
    publish_scores,
    read_native_samples,
    select_sample_ids,
    selection_blocks,
    transform_scores,
    variant_positions,
)


def test_fit_reader_projects_chain_major_scores_and_preserves_source_blocks(tmp_path):
    fit_path = tmp_path / "fit.nc"
    native = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6) + 1
    native[1, 1, 3] = np.inf
    with h5py.File(fit_path, "w") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset("r", data=native, chunks=(1, 2, 2))

    source = pd.DataFrame(
        {
            "variant_id": list("ABCDEF"),
            "is_candidate": [True] * 6,
        }
    )
    target = source.iloc[[0, 2, 3, 5]].reset_index(drop=True)
    positions = variant_positions(source, target)
    blocks = selection_blocks(
        positions,
        source_size=len(source),
        block_size=2,
        priority_position=3,
    )
    selected = select_sample_ids(6, 4, seed=23)
    with h5py.File(fit_path) as handle:
        fitted = read_native_samples(
            handle["posterior/r"],
            selected,
            draws_per_chain=3,
            variant_start=0,
            variant_stop=6,
        )
    scores = transform_scores(fitted[:, positions], "positive")
    output = publish_scores(
        target,
        scores,
        selected,
        tmp_path / "scores",
        draws_per_chain=3,
        blocks=blocks,
    )

    loaded = SampledScores.load(output)
    try:
        expected = np.log(native.reshape(6, 6)[selected][:, positions].astype(np.float32))
        expected[~np.isfinite(expected)] = np.nan
        np.testing.assert_equal(loaded.score.values, expected)
        np.testing.assert_array_equal(loaded.dataset["source_sample_id"], selected)
        np.testing.assert_array_equal(
            loaded.dataset["source_chain"] * 3 + loaded.dataset["source_draw"],
            selected,
        )
        np.testing.assert_array_equal(loaded.dataset["selection_block"], [1, 0, 0, 2])
    finally:
        loaded.close()


def test_parameter_extraction_matches_dense_projected_oracle(tmp_path):
    fit_path = tmp_path / "fit.nc"
    native = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6) + 1
    native[1, 1, 3] = np.inf
    with h5py.File(fit_path, "w") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset("r", data=native, chunks=(1, 2, 2))
    source = pd.DataFrame(
        {
            "variant_id": list("ABCDEF"),
            "is_candidate": [True] * 6,
        }
    )
    target = source.iloc[[0, 2, 3, 5]].reset_index(drop=True)

    output = extract_parameter_scores(
        fit_path,
        source,
        target,
        tmp_path / "parameter",
        posterior_variable="r",
        transform="positive",
        sample_count=4,
        seed=23,
        priority_variant_id="D",
        sample_chunk_size=2,
    )

    selected = select_sample_ids(6, 4, seed=23)
    loaded = SampledScores.load(output)
    try:
        expected = np.log(native.reshape(6, 6)[selected][:, [0, 2, 3, 5]].astype(np.float32))
        expected[~np.isfinite(expected)] = np.nan
        np.testing.assert_equal(loaded.score.values, expected)
        np.testing.assert_array_equal(loaded.dataset["selection_block"], [1, 0, 0, 2])
    finally:
        loaded.close()


def test_pdz_dimsum_fitness_matches_replicate_weighting_oracle():
    starting = np.array([[[20, 30, 40], [40, 30, 20]]], dtype=np.int64)
    timepoint = np.array([[[10, 60, 80], [20, 15, 40]]], dtype=np.int64)
    wild_type_starting = starting[:, :, 2, None]
    wild_type_timepoint = timepoint[:, :, 2, None]
    replicate_fitness = np.log(timepoint / starting) - np.log(
        wild_type_timepoint / wild_type_starting
    )
    variance = (
        1 / starting
        + 1 / timepoint
        + 1 / wild_type_starting
        + 1 / wild_type_timepoint
    )
    expected = (replicate_fitness / variance).sum(axis=1) / (1 / variance).sum(axis=1)

    np.testing.assert_allclose(
        pdz_dimsum_fitness(starting, timepoint, 2),
        expected,
    )


def test_trpb_fitness_functions_match_source_aggregation_fixtures():
    three_site_variants = np.array(["*AA", "A*A", "AAA", "AAC", "AAD"])
    three_site_starting = np.array([10, 10, 10, 10, 10], dtype=np.int64)
    three_site_output = np.tile(
        np.array([10, 10, 40, 20, 10]),
        (2, 1),
    )
    np.testing.assert_allclose(
        trpb_three_site_fitness(
            three_site_starting,
            three_site_output,
            three_site_variants,
            np.array([18, 44], dtype=float),
        ),
        [0.0, 0.0, 1.0, 0.5, 0.0],
    )

    four_site_variants = np.array(["*AAA", "A*AA", "AAAA", "AAAC", "AAAD"])
    four_site_starting = np.tile(
        np.array([[0, 0, 0, 0, 0], [20, 20, 20, 20, 20]]),
        (2, 1, 1),
    )
    four_site_output = np.tile(
        np.array([10, 10, 40, 20, 10]),
        (2, 6, 2, 1),
    )
    four_site_output[:, :, 1, 4] = 0
    four_site_output[0, 0, :, 3] = 40
    four_site_output[0, 1, 0, 3] = 0
    four_site_output[1, 1:, :, 3] = 40
    fitness = trpb_four_site_fitness(
        four_site_starting,
        four_site_output,
        four_site_variants,
        np.array([12, 16, 20, 24, 28, 36], dtype=float),
    )
    np.testing.assert_allclose(
        fitness[:, :4],
        [[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 1.0]],
        atol=1e-12,
    )
    assert np.isnan(fitness[:, 4]).all()


def test_nuc_g3_enrichment_uses_full_landscape_and_highest_gate():
    input_counts = np.array([[80, 20], [40, 40]], dtype=np.int64)
    output_counts = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [20, 30]],
            [[4, 4], [5, 5], [6, 6], [10, 30]],
        ],
        dtype=np.int64,
    )

    np.testing.assert_allclose(
        nuc_g3_highest_gate_enrichment(input_counts, output_counts),
        [[0.5, 3.0], [0.5, 1.5]],
    )


def _write_predictive_fit(path, observed):
    with h5py.File(path, "w") as handle:
        observed_group = handle.create_group("observed_data")
        predictive_group = handle.create_group("posterior_predictive")
        for name, values in observed.items():
            observed_group.create_dataset(name, data=values)
            predictive_group.create_dataset(
                name,
                data=np.broadcast_to(values, (2, 3, *values.shape)),
                chunks=(1, 2, *values.shape[:-1], min(2, values.shape[-1])),
            )


def test_pdz_predictive_extraction_matches_dense_formula(tmp_path):
    observed = {
        "starting_counts": np.array([[20, 30, 40], [40, 30, 20]], dtype=np.int32),
        "timepoint_counts": np.array([[10, 60, 80], [20, 15, 40]], dtype=np.int32),
    }
    fit_path = tmp_path / "pdz.nc"
    _write_predictive_fit(fit_path, observed)
    variants = pd.DataFrame(
        {"variant_id": ["A", "B", "WT"], "is_candidate": [True] * 3}
    )
    output = extract_pdz_predictive_scores(
        fit_path,
        variants,
        tmp_path / "pdz",
        wild_type_variant_id="WT",
        sample_count=4,
        sample_chunk_size=2,
        preserve_chain_draw=True,
    )

    loaded = SampledScores.load(output)
    try:
        expected_row = pdz_dimsum_fitness(
            observed["starting_counts"],
            observed["timepoint_counts"],
            2,
        ).astype(np.float32)
        np.testing.assert_equal(
            loaded.score.values,
            np.broadcast_to(expected_row, loaded.score.shape),
        )
    finally:
        loaded.close()


def test_trpb_predictive_extraction_matches_both_dense_formulas(tmp_path):
    variants = pd.DataFrame(
        {
            "variant_id": ["*AA", "A*A", "AAA", "AAC", "AAD"],
            "is_candidate": [True] * 5,
        }
    )
    three_site = {
        "starting_counts": np.array([10, 10, 10, 10, 10], dtype=np.int32),
        "timepoint_counts": np.tile(
            np.array([10, 10, 40, 20, 10], dtype=np.int32),
            (2, 1),
        ),
    }
    three_path = tmp_path / "three.nc"
    _write_predictive_fit(three_path, three_site)
    three_output = extract_trpb_predictive_scores(
        three_path,
        variants,
        tmp_path / "three",
        times=np.array([18, 44]),
        four_site=False,
        sample_count=4,
        sample_chunk_size=2,
        preserve_chain_draw=True,
    )
    loaded = SampledScores.load(three_output)
    try:
        expected = trpb_three_site_fitness(
            three_site["starting_counts"],
            three_site["timepoint_counts"],
            variants["variant_id"].to_numpy(),
            np.array([18, 44]),
        ).astype(np.float32)
        np.testing.assert_equal(
            loaded.score.values,
            np.broadcast_to(expected, loaded.score.shape),
        )
    finally:
        loaded.close()

    four_variants = variants.copy()
    four_variants["variant_id"] = ["*AAA", "A*AA", "AAAA", "AAAC", "AAAD"]
    four_site = {
        "starting_counts": np.tile(
            np.array([[0, 0, 0, 0, 0], [20, 20, 20, 20, 20]], dtype=np.int32),
            (1, 1),
        ),
        "timepoint_counts": np.tile(
            np.array([10, 10, 40, 20, 10], dtype=np.int32),
            (6, 2, 1),
        ),
    }
    four_path = tmp_path / "four.nc"
    _write_predictive_fit(four_path, four_site)
    four_output = extract_trpb_predictive_scores(
        four_path,
        four_variants,
        tmp_path / "four",
        times=np.array([12, 16, 20, 24, 28, 36]),
        four_site=True,
        sample_count=4,
        sample_chunk_size=2,
        preserve_chain_draw=False,
    )
    loaded = SampledScores.load(four_output)
    try:
        expected = trpb_four_site_fitness(
            four_site["starting_counts"],
            four_site["timepoint_counts"],
            four_variants["variant_id"].to_numpy(),
            np.array([12, 16, 20, 24, 28, 36]),
        ).astype(np.float32)
        np.testing.assert_equal(
            loaded.score.values,
            np.broadcast_to(expected, loaded.score.shape),
        )
        assert "source_chain" not in loaded.dataset.coords
    finally:
        loaded.close()


def test_nuc_predictive_extraction_matches_dense_formula(tmp_path):
    observed = {
        "ic1": np.array([80, 20], dtype=np.int32),
        "oc1": np.array(
            [[1, 1], [2, 2], [3, 3], [20, 30]],
            dtype=np.int32,
        ),
    }
    fit_path = tmp_path / "nuc.nc"
    _write_predictive_fit(fit_path, observed)
    variants = pd.DataFrame(
        {"variant_id": ["", "A2C"], "is_candidate": [True, True]}
    )
    output = extract_nuc_g3_predictive_scores(
        fit_path,
        variants,
        tmp_path / "nuc",
        sample_count=4,
        sample_chunk_size=2,
        preserve_chain_draw=True,
    )

    loaded = SampledScores.load(output)
    try:
        expected = nuc_g3_highest_gate_enrichment(
            observed["ic1"],
            observed["oc1"],
        ).astype(np.float32)
        np.testing.assert_equal(
            loaded.score.values,
            np.broadcast_to(expected, loaded.score.shape),
        )
    finally:
        loaded.close()


def test_score_comparison_selects_reference_samples_by_stable_id(tmp_path):
    variants = pd.DataFrame(
        {"variant_id": ["A", "B"], "is_candidate": [True, True]}
    )
    reference_values = np.arange(12, dtype=np.float32).reshape(6, 2)
    publish_scores(
        variants,
        reference_values,
        np.arange(6, dtype=np.int64),
        tmp_path / "reference",
        draws_per_chain=3,
    )
    selected = np.array([0, 2, 3, 5], dtype=np.int64)
    publish_scores(
        variants,
        reference_values[selected],
        selected,
        tmp_path / "generated",
        draws_per_chain=3,
    )

    compare_score_data(tmp_path / "generated", tmp_path / "reference")
