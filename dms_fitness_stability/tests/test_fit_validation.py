from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from dms_fitness_stability.fit_validation import (
    DIAGNOSED_GROUPS,
    FitExpectation,
    validate_hmc_artifact,
    validate_mle_artifact,
)


def _write_hmc(
    path: Path,
    *,
    variant_count: int = 4,
    include_observed: bool = True,
    include_diagnostics: bool = True,
) -> None:
    values = np.arange(2 * variant_count, dtype=float).reshape(1, 2, variant_count)
    xr.Dataset({"score": (("chain", "draw", "a"), values)}).to_netcdf(
        path,
        group="posterior",
        engine="h5netcdf",
    )
    counts = np.ones((1, 2, 1, variant_count), dtype=np.int64)
    xr.Dataset(
        {
            "starting_counts": (("chain", "draw", "replicate", "a"), counts),
            "timepoint_counts": (("chain", "draw", "replicate", "a"), counts),
        }
    ).to_netcdf(path, mode="a", group="posterior_predictive", engine="h5netcdf")
    if include_observed:
        xr.Dataset(
            {
                "starting_counts": (("replicate", "a"), counts[0, 0]),
                "timepoint_counts": (("replicate", "a"), counts[0, 0]),
            }
        ).to_netcdf(path, mode="a", group="observed_data", engine="h5netcdf")
    if include_diagnostics:
        with h5py.File(path, "a") as handle:
            for group in DIAGNOSED_GROUPS:
                handle.require_group(group)


def _write_mle(path: Path, *, variant_count: int = 4) -> None:
    values = np.arange(2 * variant_count, dtype=float).reshape(1, 2, variant_count)
    xr.Dataset({"score": (("chain", "draw", "a"), values)}).to_netcdf(
        path,
        group="posterior",
        engine="h5netcdf",
    )
    xr.Dataset({"score": (("a",), values[0, 0])}).to_netcdf(
        path,
        mode="a",
        group="mle",
        engine="h5netcdf",
    )


def _expectation() -> FitExpectation:
    return FitExpectation(
        score_variable="score",
        variant_count=4,
        posterior_shape=(1, 2, 4),
        predictive_variables=("starting_counts", "timepoint_counts"),
        observed_arrays=(
            ("starting_counts", np.ones((1, 4), dtype=np.int64)),
            ("timepoint_counts", np.ones((1, 4), dtype=np.int64)),
        ),
    )


def test_validate_hmc_artifact_accepts_complete_diagnosed_file(tmp_path):
    path = tmp_path / "fit_diagnosed.nc"
    _write_hmc(path)

    result = validate_hmc_artifact(path, _expectation(), diagnosed=True)

    assert result.status == "validated"
    assert result.issues == ()
    assert result.score_shape == (1, 2, 4)
    assert dict(result.predictive_shapes) == {
        "starting_counts": (1, 2, 1, 4),
        "timepoint_counts": (1, 2, 1, 4),
    }


def test_validate_raw_hmc_does_not_require_diagnostic_groups(tmp_path):
    path = tmp_path / "fit-20260101010101.nc"
    _write_hmc(path, include_diagnostics=False)

    result = validate_hmc_artifact(path, _expectation(), diagnosed=False)

    assert result.status == "validated"
    assert result.issues == ()


def test_validate_hmc_artifact_marks_missing_provenance_as_limited(tmp_path):
    path = tmp_path / "fit_diagnosed.nc"
    _write_hmc(path, include_observed=False, include_diagnostics=False)

    result = validate_hmc_artifact(path, _expectation(), diagnosed=True)

    assert result.status == "limited"
    assert "observed_variable_missing:starting_counts" in result.issues
    assert "diagnostic_group_missing:sample_stats" in result.issues


def test_validate_hmc_artifact_marks_observed_value_mismatch_as_limited(tmp_path):
    path = tmp_path / "fit_diagnosed.nc"
    _write_hmc(path)
    with h5py.File(path, "a") as handle:
        handle["observed_data/starting_counts"][0, 0] = 2

    result = validate_hmc_artifact(path, _expectation(), diagnosed=True)

    assert result.status == "limited"
    assert "observed_value_mismatch:starting_counts" in result.issues


def test_validate_hmc_artifact_rejects_wrong_variant_width(tmp_path):
    path = tmp_path / "fit_diagnosed.nc"
    _write_hmc(path, variant_count=3)

    result = validate_hmc_artifact(path, _expectation(), diagnosed=True)

    assert result.status == "unusable"
    assert "score_variant_count_mismatch" in result.issues


def test_validate_mle_artifact_accepts_sampled_fit_and_point_estimate(tmp_path):
    path = tmp_path / "fit_samples.nc"
    _write_mle(path)

    result = validate_mle_artifact(path, _expectation())

    assert result.status == "validated"
    assert result.issues == ()
    assert result.score_shape == (1, 2, 4)
