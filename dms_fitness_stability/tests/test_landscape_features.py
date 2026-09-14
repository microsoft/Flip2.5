import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import AnalysisIdentity, LandscapeFeatureBundle
from dms_fitness_stability.landscape_features import (
    LANDSCAPE_FEATURE_DTYPES,
    LANDSCAPE_FEATURE_SCHEMA_VERSION,
    audit_landscape_feature_source,
)


def _bundle() -> LandscapeFeatureBundle:
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G3",
        inference_method="hmc",
        model_spec="exponential",
        fit_id="fit",
    )
    variants = pd.DataFrame(
        {
            "source_variant_index": np.arange(3, dtype=np.int64),
            "variant_id": ["", "A2C", "A3D"],
        }
    )
    features = xr.Dataset(
        {
            name: (
                "variant",
                np.arange(3, dtype=dtype)
                if name == "observation_count"
                else np.arange(3, dtype=dtype) / 10,
            )
            for name, dtype in LANDSCAPE_FEATURE_DTYPES.items()
        },
        coords={"variant": np.arange(3, dtype=np.int32)},
    )
    return LandscapeFeatureBundle(
        metadata={
            "schema_version": LANDSCAPE_FEATURE_SCHEMA_VERSION,
            "feature_id": "nuc__nuc-g3",
            "protein_family": identity.protein_family,
            "landscape": identity.landscape,
            "score_name": None,
            "n_variants": 3,
            "derivation_contract": "landscape-count-features-v1",
            "inventory_fields": [],
        },
        variants=variants,
        features=features,
    )


def test_landscape_feature_bundle_round_trip_preserves_axis_and_dtypes(tmp_path):
    bundle = _bundle()

    output_dir = bundle.save(tmp_path / "features")
    loaded = LandscapeFeatureBundle.load(output_dir)
    try:
        assert {path.name for path in output_dir.iterdir()} == {
            "metadata.json",
            "variants.csv",
            "features.nc",
        }
        assert loaded.variants["variant_id"].tolist() == ["", "A2C", "A3D"]
        for name, dtype in LANDSCAPE_FEATURE_DTYPES.items():
            assert loaded.features[name].dtype == dtype
        xr.testing.assert_equal(loaded.features, bundle.features)
    finally:
        loaded.close()


def test_landscape_feature_bundle_rejects_reordered_axis():
    bundle = _bundle()
    bundle.variants["source_variant_index"] = [0, 2, 1]

    with pytest.raises(ValueError, match="contiguous and match row order"):
        bundle.validate()


def test_landscape_feature_bundle_aligns_result_variants_by_source_index():
    bundle = _bundle()
    result_variants = pd.DataFrame(
        {
            "variant_id": ["", "A3D"],
            "matrix_variant_index": [0, 2],
        }
    )

    np.testing.assert_array_equal(
        bundle.align_result_variants(result_variants),
        [0, 2],
    )
    result_variants.loc[1, "variant_id"] = "A2C"
    with pytest.raises(ValueError, match="do not match"):
        bundle.align_result_variants(result_variants)


def test_landscape_feature_bundle_accepts_matching_legacy_mapping():
    bundle = _bundle()
    result_variants = pd.DataFrame(
        {
            "variant_id": ["", "A3D"],
            "matrix_variant_index": [0, 2],
            "hmc_variant_index": [0, 2],
        }
    )

    np.testing.assert_array_equal(
        bundle.align_result_variants(result_variants),
        [0, 2],
    )
    result_variants.loc[1, "hmc_variant_index"] = 1
    with pytest.raises(ValueError, match="disagree"):
        bundle.align_result_variants(result_variants)


def test_landscape_feature_source_audit_requires_exact_values():
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape="nuc-G3",
        inference_method="hmc",
        model_spec="exponential",
        fit_id="fit",
    )
    variants = pd.DataFrame(
        {
            "variant_id": ["", "A2C"],
            "input_count": [10.0, 20.0],
            "output_count": [30.0, 40.0],
        }
    )
    landscape = type(
        "Landscape",
        (),
        {
            "variants": variants,
            "counts": {"ic1": np.ones(2)},
            "n_fitted_variants": 2,
        },
    )()
    bundle = LandscapeFeatureBundle.from_landscape(identity, landscape)

    assert audit_landscape_feature_source(bundle, identity, landscape)["status"] == "exact"
    changed = bundle.features["input_count"].values.copy()
    changed[0] += 1
    bundle.features["input_count"] = ("variant", changed)
    with pytest.raises(AssertionError, match="Feature source mismatch"):
        audit_landscape_feature_source(bundle, identity, landscape)
