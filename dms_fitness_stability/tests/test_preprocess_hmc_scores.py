import h5py
import numpy as np

from dms_fitness_stability.hmc_score_extraction import hmc_score_geometry


def test_hmc_score_geometry_reads_only_posterior_shape(tmp_path):
    path = tmp_path / "fit.nc"
    with h5py.File(path, "w") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset(
            "log_r_mean",
            data=np.ones((2, 600, 3), dtype=np.float32),
            chunks=(1, 100, 3),
        )

    assert hmc_score_geometry(
        path,
        posterior_variable="log_r_mean",
    ) == (
        2,
        600,
        3,
    )
