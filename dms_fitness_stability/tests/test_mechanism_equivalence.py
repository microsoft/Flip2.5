import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability.mechanism_equivalence import (
    compare_mechanism_cache,
    compare_mechanism_table,
)


def test_mechanism_equivalence_requires_exact_cache_and_normalizes_annotation(
    tmp_path,
):
    matrix_cache = tmp_path / "matrix.nc"
    fit_cache = tmp_path / "fit.nc"
    dataset = xr.Dataset(
        {"score": (("sample", "variant"), np.ones((2, 2), dtype=np.float32))},
        coords={"sample": [0, 1], "variant": [0, 1]},
    )
    dataset.to_netcdf(matrix_cache, engine="h5netcdf")
    dataset.to_netcdf(fit_cache, engine="h5netcdf")

    record = compare_mechanism_cache(matrix_cache, fit_cache, sample_chunk_size=1)
    assert record["status"] == "exact"

    changed = dataset.copy(deep=True)
    changed["score"][0, 0] = 2
    changed.to_netcdf(fit_cache, engine="h5netcdf")
    with pytest.raises(ValueError, match="variables differ"):
        compare_mechanism_cache(matrix_cache, fit_cache)

    matrix_table = tmp_path / "matrix.csv"
    fit_table = tmp_path / "fit.csv"
    pd.DataFrame(
        {
            "protein_family": ["trpb", "nuc"],
            "is_ranking_candidate": [np.nan, True],
            "value": [0.1, 0.2],
        }
    ).to_csv(matrix_table, index=False)
    pd.DataFrame(
        {
            "value": [0.1 + 1e-15, 0.2],
            "is_ranking_candidate": [False, True],
            "protein_family": ["trpb", "nuc"],
        }
    ).to_csv(fit_table, index=False)
    table_record = compare_mechanism_table(
        matrix_table,
        fit_table,
        normalize_current_trpb_candidate_annotation=True,
    )
    assert table_record["normalized_trpb_rows"] == 1
    assert table_record["status"] == "within_tolerance"


def test_mechanism_equivalence_allows_only_named_additive_fields(tmp_path):
    matrix_cache = tmp_path / "matrix.nc"
    fit_cache = tmp_path / "fit.nc"
    legacy = xr.Dataset(
        {"score": (("sample", "variant"), np.ones((2, 2), dtype=np.float32))},
        coords={"sample": [0, 1], "variant": [0, 1]},
    )
    additive = legacy.assign(
        n_finite_samples=("variant", np.full(2, 2, dtype=np.int32))
    )
    additive.to_netcdf(matrix_cache, engine="h5netcdf")
    legacy.to_netcdf(fit_cache, engine="h5netcdf")

    with pytest.raises(ValueError, match="unexpected=.*n_finite_samples"):
        compare_mechanism_cache(matrix_cache, fit_cache)
    cache_record = compare_mechanism_cache(
        matrix_cache,
        fit_cache,
        allowed_observed_variables=("n_finite_samples",),
    )
    assert cache_record["added_variables"] == ["n_finite_samples"]
    assert cache_record["status"] == "exact_with_additions"

    matrix_table = tmp_path / "matrix.csv"
    fit_table = tmp_path / "fit.csv"
    pd.DataFrame({"value": [0.1], "n_finite_samples": [2]}).to_csv(
        matrix_table, index=False
    )
    pd.DataFrame({"value": [0.1]}).to_csv(fit_table, index=False)

    with pytest.raises(ValueError, match="unexpected=.*n_finite_samples"):
        compare_mechanism_table(matrix_table, fit_table)
    table_record = compare_mechanism_table(
        matrix_table,
        fit_table,
        allowed_observed_columns=("n_finite_samples",),
    )
    assert table_record["added_columns"] == ["n_finite_samples"]
    assert table_record["status"] == "exact_with_additions"
