from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import (
    AnalysisIdentity,
    StabilityResult,
    analyze_stability,
    extract_score_samples,
    load_fit,
    prepare_counts,
)


def _variants():
    counts = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D"],
            "input_count": [10, 10, 10, 10],
            "output_count": [10, 40, 30, 20],
            "reported_fitness": [0.1, 0.4, 0.3, 0.2],
        }
    )
    return prepare_counts(counts)


def _scores():
    return xr.DataArray(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0], [4.0, 2.0, 3.0, 1.0]],
        dims=("sample", "variant"),
        coords={"sample": [0, 1, 2], "variant": ["A", "B", "C", "D"]},
    )


def _identity():
    return AnalysisIdentity(
        protein_family="toy",
        landscape="toy",
        inference_method="mle_bootstrap",
        model_spec="ratio-model",
        fit_id="bootstrap-1",
    )


def test_extract_score_samples_stacks_scistanpy_inference_dimensions():
    posterior = xr.Dataset(
        {
            "inv_rate": (
                ("chain", "draw", "b"),
                np.array([[[4.0, 2.0], [2.0, 1.0]]]),
            )
        }
    )
    fit = SimpleNamespace(inference_obj=SimpleNamespace(posterior=posterior))

    result = extract_score_samples(
        fit,
        variable="inv_rate",
        variant_dim="b",
        variant_ids=["A", "B"],
        transform="negative_log",
    )

    assert result.dims == ("sample", "variant")
    assert result.coords["variant"].values.tolist() == ["A", "B"]
    np.testing.assert_allclose(result, -np.log([[4.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("method", ["hmc", "mle"])
def test_load_fit_uses_scistanpy_result_wrappers(tmp_path, method):
    fit_path = tmp_path / "fit.nc"
    inference_data = az.from_dict(
        posterior={"growth": np.array([[[1.0, 2.0], [2.0, 3.0]]])},
        posterior_predictive={
            "starting_counts": np.array([[[10, 20], [11, 19]]]),
            "timepoint_counts": np.array([[[20, 10], [21, 9]]]),
        },
    )
    inference_data.to_netcdf(fit_path)

    fit = load_fit(fit_path, method=method)

    assert fit.inference_obj.posterior["growth"].shape == (1, 2, 2)
    assert "starting_counts" in fit.inference_obj.posterior_predictive


def test_analyze_stability_round_trips_five_file_bundle(tmp_path):
    result = analyze_stability(
        _scores(),
        _variants(),
        identity=_identity(),
        score_name="growth",
        top_k=[1, 2],
        n_pairs=20,
        seed=7,
    )
    result.variants.loc[0, "variant_id"] = ""
    result.variants["optional_value"] = [np.nan, 1.0, 2.0, 3.0]

    output_dir = result.save(tmp_path / "result")
    loaded = StabilityResult.load(output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "metadata.json",
        "variants.csv",
        "stability.nc",
        "summary.csv",
        "overlaps.csv",
    }
    assert loaded.metadata == result.metadata
    assert loaded.metadata["analysis_id"] == (
        "toy__toy__mle-bootstrap__ratio-model__bootstrap-1"
    )
    assert loaded.metadata["inference_method"] == "mle_bootstrap"
    assert loaded.metadata["model_spec"] == "ratio-model"
    assert loaded.variants.loc[0, "variant_id"] == ""
    assert pd.isna(loaded.variants.loc[0, "optional_value"])
    xr.testing.assert_equal(loaded.selection, result.selection)
    assert set(loaded.summary["k"]) == {1, 2}
    assert set(loaded.overlaps["comparison"]) == {
        "sample_vs_sample",
        "sample_vs_reference",
    }


def test_analyze_stability_rejects_misaligned_variants():
    scores = _scores().assign_coords(variant=["B", "A", "C", "D"])

    with pytest.raises(ValueError, match="not aligned"):
        analyze_stability(
            scores,
            _variants(),
            identity=_identity(),
            score_name="growth",
            top_k=2,
        )
