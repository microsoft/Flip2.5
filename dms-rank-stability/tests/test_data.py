import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_rank_stability.data import FEATURE_DTYPES, LandscapeFeatures, SampledScores


def _variants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["", "A2C", "A3D"],
            "is_candidate": [True, True, False],
        }
    )


def test_sampled_scores_round_trip_preserves_empty_wild_type(tmp_path):
    dataset = xr.Dataset(
        {
            "score": (
                ("sample", "variant"),
                np.array([[1.0, 2.0, np.nan], [2.0, 1.0, np.nan]], dtype=np.float32),
            )
        },
        coords={
            "sample": np.arange(2),
            "variant": np.arange(3),
            "source_sample_id": ("sample", np.array([8, 11], dtype=np.int64)),
            "source_chain": ("sample", np.array([0, 1], dtype=np.int32)),
            "source_draw": ("sample", np.array([8, 3], dtype=np.int32)),
        },
    )

    SampledScores(_variants(), dataset).save(tmp_path)
    loaded = SampledScores.load(tmp_path)
    try:
        assert loaded.variants.loc[0, "variant_id"] == ""
        np.testing.assert_array_equal(loaded.score.values, dataset["score"].values)
    finally:
        loaded.close()


def test_sampled_scores_reject_noncontiguous_variant_coordinate():
    dataset = xr.Dataset(
        {
            "score": (
                ("sample", "variant"),
                np.ones((2, 3), dtype=np.float32),
            )
        },
        coords={"sample": [0, 1], "variant": [0, 2, 3]},
    )

    with pytest.raises(ValueError, match="variant coordinates"):
        SampledScores(_variants(), dataset).validate()


def test_sampled_scores_require_stable_source_sample_ids():
    dataset = xr.Dataset(
        {
            "score": (
                ("sample", "variant"),
                np.ones((2, 3), dtype=np.float32),
            )
        },
        coords={"sample": np.arange(2), "variant": np.arange(3)},
    )

    with pytest.raises(ValueError, match="source_sample_id"):
        SampledScores(_variants(), dataset).validate()


def test_sampled_scores_validate_and_preserve_selection_blocks(tmp_path):
    dataset = xr.Dataset(
        {
            "score": (
                ("sample", "variant"),
                np.ones((2, 3), dtype=np.float32),
            )
        },
        coords={
            "sample": np.arange(2),
            "variant": np.arange(3),
            "source_sample_id": ("sample", np.array([2, 5])),
            "selection_block": ("variant", np.array([1, 0, 0], dtype=np.int32)),
        },
    )

    SampledScores(_variants(), dataset).save(tmp_path)
    loaded = SampledScores.load(tmp_path)
    try:
        np.testing.assert_array_equal(loaded.dataset["selection_block"], [1, 0, 0])
    finally:
        loaded.close()

    dataset = dataset.assign_coords(selection_block=("variant", [0, 1, 0]))
    with pytest.raises(ValueError, match="occupy contiguous variants"):
        SampledScores(_variants(), dataset).validate()


def test_landscape_features_round_trip_and_align_by_identity(tmp_path):
    arrays = {}
    for name, dtype in FEATURE_DTYPES.items():
        values = np.arange(3, dtype=dtype)
        arrays[name] = ("variant", values)
    dataset = xr.Dataset(arrays, coords={"variant": np.arange(3)})

    LandscapeFeatures(_variants(), dataset).save(tmp_path)
    loaded = LandscapeFeatures.load(tmp_path)
    try:
        indices = loaded.indices_for(
            pd.DataFrame({"variant_id": ["A3D", "", "A2C"]})
        )
        np.testing.assert_array_equal(indices, [2, 0, 1])
        assert loaded.dataset["observation_count"].dtype == np.dtype("int32")
        assert loaded.dataset["input_count"].dtype == np.dtype("float64")
    finally:
        loaded.close()


def test_landscape_features_reject_missing_identity():
    arrays = {
        name: ("variant", np.zeros(3, dtype=dtype))
        for name, dtype in FEATURE_DTYPES.items()
    }
    features = LandscapeFeatures(
        _variants(),
        xr.Dataset(arrays, coords={"variant": np.arange(3)}),
    )

    with pytest.raises(ValueError, match="missing variant IDs"):
        features.indices_for(pd.DataFrame({"variant_id": ["not-present"]}))
