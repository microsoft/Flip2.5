import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_rank_stability import LandscapeFeatures, SampledScores, analyze_scores
from dms_rank_stability.data import FEATURE_DTYPES
from dms_rank_stability.tables import (
    TABLE_NAMES,
    build_run_tables,
    combine_run_tables,
    ecdf_to_uniform_area,
    write_tables,
)


@pytest.mark.parametrize("values,expected", [
    ([0.5] * 10, 0.25), ([0.0, 1.0] * 5, 0.25),
    ([0.0] * 10, 0.5), ([1.0] * 10, 0.5), ([0.25, 0.75], 0.125),
])
def test_percentile_area_is_distributional_not_rank_error(values, expected):
    assert ecdf_to_uniform_area(np.array(values)) == pytest.approx(expected)
    assert ecdf_to_uniform_area(np.array(values)[::-1]) == pytest.approx(expected)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-0.01], [1.01]])
def test_percentile_area_rejects_invalid_percentiles(values):
    with pytest.raises(ValueError):
        ecdf_to_uniform_area(np.array(values))


def _result_and_features():
    random = np.random.default_rng(23)
    candidate_count = 12
    variants = pd.DataFrame(
        {
            "variant_id": [f"V{index}" for index in range(candidate_count)] + ["control"],
            "is_candidate": [True] * candidate_count + [False],
            "reported_fitness": [float(candidate_count - index) for index in range(candidate_count)]
            + [1000.0],
            "reported_rank": [float(index + 1) for index in range(candidate_count)]
            + [np.nan],
            "reported_activity_level": [
                "high" if index < 4 else "medium" if index < 8 else "low"
                for index in range(candidate_count)
            ]
            + [np.nan],
            "is_wild_type": [True] + [False] * candidate_count,
            "is_source_control": [False] * candidate_count + [True],
        }
    )
    candidate_scores = (
        np.arange(candidate_count, 0, -1, dtype=np.float32)[None, :]
        + random.normal(0, 1.5, size=(10, candidate_count)).astype(np.float32)
    )
    score_values = np.column_stack(
        [candidate_scores, np.full(10, 1000.0, dtype=np.float32)]
    )
    scores = SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), score_values)},
            coords={
                "sample": np.arange(10),
                "variant": np.arange(len(variants)),
                "source_sample_id": ("sample", np.arange(100, 110)),
            },
        ),
    )
    metadata = {
        "run_id": "synthetic-parameter-example",
        "family": "synthetic",
        "landscape": "synthetic-landscape",
        "sampling": "parameter_posterior",
        "model": "example",
        "score": "fitness",
        "score_path": "data/downloads/scores/synthetic",
        "feature_path": "data/downloads/features/synthetic",
        "k_values": [1, 2, 4, 6],
        "interval": [0.05, 0.95],
        "seed": 23,
        "external_value_column": "reported_fitness",
        "external_higher_is_better": True,
    }
    result = analyze_scores(
        scores,
        metadata=metadata,
        retain_k=6,
        decision_k=4,
        external_top_n=8,
        model_top_n=4,
        challenger_probability=0.1,
        variant_chunk_size=3,
    )

    feature_variants = variants.iloc[::-1].reset_index(drop=True)[
        ["variant_id", "is_candidate"]
    ]
    variant_number = feature_variants["variant_id"].map(
        {f"V{index}": index for index in range(candidate_count)} | {"control": 12}
    ).to_numpy()
    arrays = {}
    for feature_name, dtype in FEATURE_DTYPES.items():
        if feature_name == "observation_count":
            values = (variant_number + 2).astype(dtype)
        elif feature_name in {"zero_input_fraction", "zero_output_fraction"}:
            values = (variant_number / 100).astype(dtype)
        else:
            values = (variant_number + 1.5).astype(dtype)
        arrays[feature_name] = ("variant", values)
    features = LandscapeFeatures(
        feature_variants,
        xr.Dataset(arrays, coords={"variant": np.arange(len(feature_variants))}),
    )
    return result, features


def test_build_run_tables_produces_all_paper_products_from_results_and_features():
    result, features = _result_and_features()

    tables = build_run_tables(result, features)

    assert tuple(tables) == TABLE_NAMES
    assert tables["runs"].iloc[0]["n_candidates"] == 12
    assert tables["runs"].iloc[0]["n_excluded_from_ranking"] == 1
    assert tables["top_k_curves"]["k"].tolist() == [1, 2, 4, 6]
    counts = np.bincount(
        result.stability["selection_variant_index"].values[:, :4].ravel(),
        minlength=len(result.variants),
    )
    expected_consistency = np.sum(counts * (counts - 1)) / (10 * 9 * 4)
    observed = tables["top_k_curves"].set_index("k").loc[
        4,
        "sample_self_consistency",
    ]
    assert observed == expected_consistency
    assert set(tables["rank_percentiles"]["cohort_top_n"]) == {4, 8}
    assert len(tables["model_rank_intervals"]) == 4
    assert len(tables["functionality_scores"]) == 12
    assert tables["functionality_scores"]["model_reference_rank"].tolist() == list(
        range(1, 13)
    )
    assert tables["functionality_scores"]["n_finite_samples"].eq(10).all()
    mechanism = tables["mechanism_variants"].set_index("variant_id")
    for variant_id, row in mechanism.iterrows():
        variant_number = int(variant_id.removeprefix("V"))
        assert row["mechanism_input_count"] == variant_number + 1.5
    assert tables["mechanism_runs"].iloc[0]["model_top_k_count"] == 4
    assert set(tables["mechanism_effects"]["stage"]) == {
        "score_precision",
        "rank_propagation",
    }
    assert tables["mechanism_inventory"].iloc[0]["feature_variant_count"] == 13
    assert set(tables["mechanism_diagnostics"]["stratum"]) == {
        "primary",
        "model_top_k",
        "challengers",
    }
    assert not tables["reference_category_sensitivity"].empty
    diagnostic = tables["reference_diagnostics"].iloc[0]
    assert diagnostic["n_ranking_candidates"] == 12
    assert diagnostic["n_source_controls"] == 1


def test_combined_tables_add_cross_run_summaries_and_publish_exact_files(tmp_path):
    first_result, features = _result_and_features()
    second_result, _ = _result_and_features()
    second_result.metadata["run_id"] = "synthetic-parameter-second"
    second_result.metadata["model"] = "second-example"

    combined = combine_run_tables(
        [
            build_run_tables(first_result, features),
            build_run_tables(second_result, features),
        ]
    )
    outputs = write_tables({"parameter_posterior": combined}, tmp_path)

    assert combined["mechanism_inventory"].iloc[0]["analysis_run_count"] == 2
    diagnostics = combined["reference_diagnostics"]
    assert diagnostics["within_landscape_model_top_k_pair_count"].eq(1).all()
    assert diagnostics["within_landscape_model_top_k_overlap_mean"].eq(1.0).all()
    assert set(combined["mechanism_effects"]["scope"]) == {
        "run",
        "landscape_median",
    }
    output = outputs[0]
    assert output == tmp_path / "parameter_posterior"
    assert {path.name for path in output.iterdir()} == {
        *{f"{name}.csv" for name in TABLE_NAMES if name != "functionality_scores"},
        "functionality_scores.csv.gz",
    }
    functionality = pd.read_csv(output / "functionality_scores.csv.gz")
    assert len(functionality) == 24


def test_lower_is_better_scores_flow_through_all_tables():
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D", "E"],
            "is_candidate": True,
            "reported_fitness": [1.0, 2.0, 3.0, 4.0, 5.0],
            "reported_rank": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.2, 2.2, 2.8, 4.2, 4.8],
            [0.8, 1.8, 3.2, 3.8, 5.2],
            [1.1, 1.9, 3.1, 4.1, 4.9],
        ],
        dtype=np.float32,
    )
    scores = SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), values)},
            coords={
                "sample": np.arange(4),
                "variant": np.arange(5),
                "source_sample_id": ("sample", np.arange(4)),
            },
        ),
    )
    result = analyze_scores(
        scores,
        metadata={
            "run_id": "synthetic-lower",
            "family": "synthetic",
            "landscape": "lower-errors",
            "sampling": "mle_predictive",
            "model": "example",
            "score": "error",
            "score_path": "data/downloads/scores/lower",
            "feature_path": "data/downloads/features/lower",
            "k_values": [1, 2, 3],
            "interval": [0.05, 0.95],
            "seed": 23,
            "external_value_column": "reported_fitness",
            "external_higher_is_better": False,
        },
        retain_k=3,
        decision_k=2,
        external_top_n=3,
        model_top_n=2,
        challenger_probability=0.01,
        higher_is_better=False,
        variant_chunk_size=2,
        min_reference_samples=2,
    )
    feature_arrays = {
        name: ("variant", np.arange(1, 6, dtype=dtype))
        for name, dtype in FEATURE_DTYPES.items()
    }
    features = LandscapeFeatures(
        variants[["variant_id", "is_candidate"]],
        xr.Dataset(feature_arrays, coords={"variant": np.arange(5)}),
    )

    tables = build_run_tables(result, features)

    assert set(tables) == set(TABLE_NAMES)
    assert tables["rank_agreement"]["variant_id"].tolist()[:3] == ["A", "B", "C"]
    mechanism = tables["mechanism_variants"].set_index("variant_id")
    assert mechanism.loc["A", "score_margin_to_top_k_boundary"] > 0
    assert mechanism.loc["B", "score_margin_to_top_k_boundary"] == 0
    diagnostic = tables["reference_diagnostics"].iloc[0]
    assert diagnostic["all_finite_spearman"] == pytest.approx(1.0)
    assert diagnostic["model_external_top_k_intersection"] == 2


def test_shared_features_may_cover_variants_excluded_by_one_score_run():
    result, features = _result_and_features()
    features.variants["is_candidate"] = True

    tables = build_run_tables(result, features)

    assert tables["runs"].iloc[0]["n_candidates"] == 12
    assert tables["mechanism_inventory"].iloc[0]["feature_candidate_count"] == 13
    assert "control" not in set(tables["mechanism_variants"]["variant_id"])
