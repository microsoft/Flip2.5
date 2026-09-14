import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _module():
    script = Path(__file__).parents[1] / "examples/paper/relocate_score_roots.py"
    spec = importlib.util.spec_from_file_location("relocate_score_roots", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relocate_roots_moves_both_method_roots_atomically(tmp_path):
    module = _module()
    methods = {}
    for method in ("mle_predictive", "hmc_count_based"):
        old_root = tmp_path / "old" / method
        new_root = tmp_path / "scores" / method / "canonical"
        old_root.mkdir(parents=True)
        (old_root / "marker.txt").write_text(method, encoding="utf-8")
        methods[method] = {
            "old_root": str(old_root),
            "new_root": str(new_root),
        }

    module._relocate_roots(tmp_path, methods)
    module._relocate_roots(tmp_path, methods)

    for method, record in methods.items():
        assert not Path(record["old_root"]).exists()
        assert (Path(record["new_root"]) / "marker.txt").read_text(
            encoding="utf-8"
        ) == method


def test_relocate_roots_resumes_after_one_method_root_moved(tmp_path):
    module = _module()
    methods = {}
    for position, method in enumerate(("mle_predictive", "hmc_count_based")):
        old_root = tmp_path / "old" / method
        new_root = tmp_path / "scores" / method / "canonical"
        active_root = new_root if position == 0 else old_root
        active_root.mkdir(parents=True)
        (active_root / "marker.txt").write_text(method, encoding="utf-8")
        methods[method] = {
            "old_root": str(old_root),
            "new_root": str(new_root),
        }

    module._relocate_roots(tmp_path, methods)

    for method, record in methods.items():
        assert not Path(record["old_root"]).exists()
        assert (Path(record["new_root"]) / "marker.txt").read_text(
            encoding="utf-8"
        ) == method


def test_update_cache_preserves_arrays_and_rewrites_provenance(tmp_path):
    module = _module()
    old_root = tmp_path / "old-scores"
    score_path = tmp_path / "scores/analysis/fitness_scores.nc"
    score_path.parent.mkdir(parents=True)
    score_path.write_bytes(b"score matrix")
    cache_path = tmp_path / "cache.nc"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("score", data=values)
        handle.attrs["source_kind"] = "mle_proxy"
        handle.attrs["source_path"] = str(
            old_root / "analysis/fitness_proxies.nc"
        )
        handle.attrs["source_size"] = np.array([100], dtype=np.int64)
        handle.attrs["source_mtime_ns"] = np.array([200], dtype=np.int64)
        handle.attrs["score_contract"] = "score-v1"

    updated = module._update_cache(
        cache_path,
        score_path=score_path,
        score_contract="score-v1",
        old_root=old_root,
    )

    assert updated is True
    with h5py.File(cache_path, "r") as handle:
        np.testing.assert_array_equal(handle["score"][:], values)
        assert handle.attrs["source_kind"] == "score_ensemble"
        assert handle.attrs["source_path"] == str(score_path.resolve())
        assert int(handle.attrs["source_size"]) == score_path.stat().st_size
        assert int(handle.attrs["source_mtime_ns"]) == score_path.stat().st_mtime_ns
        assert handle.attrs["score_contract"] == "score-v1"

    assert (
        module._update_cache(
            cache_path,
            score_path=score_path,
            score_contract="score-v1",
            old_root=old_root,
        )
        is False
    )
