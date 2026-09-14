from pathlib import Path

import h5py
import numpy as np
import pytest

from dms_fitness_stability.mle_preflight import (
    validate_posterior_predictive_counts,
    validate_sampled_mle_counts,
)


def _write_counts(
    path: Path,
    observed: np.ndarray,
    predictive: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        observed_group = handle.create_group("observed_data")
        observed_group.create_dataset("counts", data=observed)
        predictive_group = handle.create_group("posterior_predictive")
        predictive_group.create_dataset("counts", data=predictive)


def test_validate_sampled_mle_counts_accepts_fixed_total_draws(tmp_path):
    observed = np.array([[2, 3, 5], [1, 4, 5]], dtype=np.int64)
    draws = np.stack([observed, observed[..., ::-1]])
    path = tmp_path / "samples.nc"
    _write_counts(path, observed, draws[None])

    result = validate_sampled_mle_counts(
        path,
        {"counts": observed},
        expected_draws=2,
        sample_chunk_size=1,
        variant_chunk_size=2,
    )

    assert result["status"] == "ready"
    assert result["n_draws"] == 2
    assert result["count_variables"]["counts"]["predictive_shape"] == [
        1,
        2,
        2,
        3,
    ]


def test_validate_sampled_mle_counts_rejects_changed_totals(tmp_path):
    observed = np.array([2, 3, 5], dtype=np.int64)
    draws = np.array([[2, 3, 5], [2, 3, 6]], dtype=np.int64)
    path = tmp_path / "samples.nc"
    _write_counts(path, observed, draws[None])

    with pytest.raises(ValueError, match="totals differ"):
        validate_sampled_mle_counts(
            path,
            {"counts": observed},
            expected_draws=2,
            sample_chunk_size=1,
            variant_chunk_size=2,
        )


def test_validate_sampled_mle_counts_rejects_staged_count_mismatch(tmp_path):
    observed = np.array([2, 3, 5], dtype=np.int64)
    draws = np.stack([observed, observed])
    path = tmp_path / "samples.nc"
    _write_counts(path, observed, draws[None])

    with pytest.raises(ValueError, match="differs from the staged"):
        validate_sampled_mle_counts(
            path,
            {"counts": np.array([1, 4, 5], dtype=np.int64)},
            expected_draws=2,
        )


def test_validate_posterior_predictive_counts_accepts_multiple_chains(tmp_path):
    observed = np.array([2, 3, 5], dtype=np.int64)
    predictive = np.array(
        [
            [[2, 3, 5], [5, 3, 2]],
            [[3, 2, 5], [2, 5, 3]],
        ],
        dtype=np.int64,
    )
    path = tmp_path / "hmc.nc"
    _write_counts(path, observed, predictive)
    with h5py.File(path, "a") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset("score", data=np.zeros((2, 2, 3)))

    result = validate_posterior_predictive_counts(
        path,
        {"counts": observed},
        posterior_variable="score",
        sample_chunk_size=1,
        variant_chunk_size=2,
    )

    assert result["n_chains"] == 2
    assert result["n_draws"] == 2
    assert result["n_samples"] == 4
    assert result["posterior_variable"] == "score"


def test_validate_posterior_predictive_counts_rejects_target_sample_drift(tmp_path):
    observed = np.array([2, 3, 5], dtype=np.int64)
    predictive = np.broadcast_to(observed, (2, 2, 3)).copy()
    path = tmp_path / "hmc.nc"
    _write_counts(path, observed, predictive)
    with h5py.File(path, "a") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset("score", data=np.zeros((1, 2, 3)))

    with pytest.raises(ValueError, match="match chain/draw shape"):
        validate_posterior_predictive_counts(
            path,
            {"counts": observed},
            posterior_variable="score",
        )
