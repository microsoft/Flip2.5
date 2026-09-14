from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis.fit_inventory import _longtable, artifact_summary, format_fit_inventory, parse_command


def test_fit_inventory_longtables_are_explicitly_centered():
    tex = _longtable("Fits", "tab:fits", ["Landscape", "Model"], [["trpb-libB", "EE"]])
    assert r"\begin{longtable}[c]{ll}" in tex
    assert "trpb-libB" in tex


def test_inventory_does_not_classify_diagnostics_by_output_format():
    fits = pd.DataFrame({
        "kind": ["hmc", "hmc"], "landscape": ["pdz-trans-1", "pdz-cis"],
        "model": ["exponential-exponential"] * 2, "job_id": [1, 3],
        "artifact_kind": ["hmc", "diagnosed_hmc"], "artifact_num_chains": [8, 8],
        "artifact_num_warmup": [2000, 2000], "artifact_num_samples": [2000, 2000],
        "divergences": [0, 0], "tree_depth_hits": [0, 0],
        "r_hat_extreme": [np.nan, 1.001], "min_bfmi": [0.7, 0.8],
    })
    original = fits.copy(deep=True)
    tex = format_fit_inventory(fits)
    assert "All fits underwent sampler diagnostics" in tex
    assert "not that diagnostics were omitted" in tex
    assert "File &" not in tex and "diagnosed artifact" not in tex and "raw artifact" not in tex
    assert " & -- & 0.7" in tex and " & 1.0010 & 0.8" in tex
    pd.testing.assert_frame_equal(fits, original)


def test_job_commands_preserve_overrides_and_defaults():
    options = parse_command(
        "--dataset pdz --subset cis --fitness_dist gamma --growth_curve sigmoid "
        "--alpha 5 --n_warmup 8000 --centered",
        {"alpha": 0.1, "exp_beta": 10.0},
    )
    assert options["alpha"] == 5.0
    assert options["exp_beta"] == 10.0
    assert options["n_warmup"] == 8000
    assert options["n_chains"] == 8
    assert options["noncentered"] is False


def test_inventory_reads_metadata_and_handles_missing_diagnostics(tmp_path: Path):
    path = tmp_path / "fit.nc"
    energy = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    with h5py.File(path, "w") as handle:
        handle.attrs["num_chains"] = [2]
        handle.attrs["max_depth"] = [10]
        handle.create_dataset("posterior/score", shape=(2, 3, 2), dtype="float32")
        handle.create_dataset("sample_stats/divergent__", data=[[0, 1, 0], [0, 0, 0]])
        handle.create_dataset("sample_stats/treedepth__", data=[[4, 10, 5], [4, 5, 6]])
        handle.create_dataset("sample_stats/energy__", data=energy)
        handle.create_dataset("variable_diagnostic_stats/metric", data=[b"r_hat", b"ess_bulk"])
        handle.create_dataset("variable_diagnostic_stats/score", data=[[1.0, 1.02], [200, 100]])
    summary = artifact_summary(path, "score")
    assert summary["divergences"] == 1
    assert summary["tree_depth_hits"] == 1
    assert summary["artifact_draw_count"] == 6
    assert summary["r_hat_extreme"] == 1.02
    assert summary["ess_bulk_extreme"] == 100
    expected = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(energy, axis=1)
    np.testing.assert_allclose(summary["bfmi_by_chain"], expected)
    assert artifact_summary(tmp_path / "absent.nc", "score") == {"artifact_available": False}
