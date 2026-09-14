import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_rank_stability import LandscapeFeatures, SampledScores
from dms_rank_stability.data import FEATURE_DTYPES
from dms_rank_stability.runs import execute_runs, load_run_config, main, select_runs

PAPER_CONFIG = Path(__file__).parents[1] / "analysis/runs.json"


def _settings() -> dict[str, object]:
    return {
        "k_values": [1, 2],
        "retain_k": 2,
        "decision_k": 2,
        "external_top_n": 2,
        "model_top_n": 2,
        "interval": [0.05, 0.95],
        "challenger_probability": 0.01,
        "seed": 23,
    }


def _run(run_id: str, sampling: str, *, higher_is_better: bool) -> dict[str, object]:
    return {
        "run_id": run_id,
        "family": "synthetic",
        "landscape": "three-variants",
        "sampling": sampling,
        "model": "example",
        "score": "fitness",
        "score_path": f"data/downloads/scores/{run_id}",
        "feature_path": "data/downloads/features/three-variants",
        "score_higher_is_better": higher_is_better,
        "external_value_column": "reported_fitness",
        "external_rank_column": "reported_rank",
        "external_higher_is_better": higher_is_better,
        "min_reference_samples": 1,
        "variant_chunk_size": 2,
    }


def _write_config(tmp_path, runs):
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    path = analysis / "runs.json"
    path.write_text(
        json.dumps({"settings": _settings(), "runs": runs}),
        encoding="utf-8",
    )
    return path


def _write_scores(root, run):
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "is_candidate": [True, True, True],
            "reported_fitness": [3.0, 2.0, 1.0],
            "reported_rank": [1.0, 2.0, 3.0],
        }
    )
    values = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
    SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), values)},
            coords={
                "sample": np.arange(2),
                "variant": np.arange(3),
                "source_sample_id": ("sample", np.array([4, 9], dtype=np.int64)),
            },
        ),
    ).save(root / run["score_path"])


def _write_features(root, run):
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "is_candidate": [True, True, True],
        }
    )
    arrays = {
        name: (
            "variant",
            np.arange(1, 4, dtype=dtype),
        )
        for name, dtype in FEATURE_DTYPES.items()
    }
    LandscapeFeatures(
        variants,
        xr.Dataset(arrays, coords={"variant": np.arange(3)}),
    ).save(root / run["feature_path"])


def test_run_config_is_strict_and_filters_all_public_sampling_methods(tmp_path):
    runs = [
        _run("synthetic-parameter", "parameter_posterior", higher_is_better=True),
        _run("synthetic-mle", "mle_predictive", higher_is_better=True),
        _run("synthetic-predictive", "posterior_predictive", higher_is_better=False),
    ]
    config = load_run_config(_write_config(tmp_path, runs))

    assert [run.run_id for run in config.runs] == [run["run_id"] for run in runs]
    assert [run.run_id for run in select_runs(
        config.runs,
        sampling_methods=("posterior_predictive",),
    )] == ["synthetic-predictive"]
    with pytest.raises(ValueError, match="Unknown requested run IDs"):
        select_runs(config.runs, run_ids=("misspelled",))

    runs[0]["unexpected"] = True
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        load_run_config(_write_config(tmp_path / "invalid", runs))


def test_paper_run_config_locks_complete_scope():
    config = load_run_config(PAPER_CONFIG)

    assert len(config.runs) == 158
    assert Counter(run.sampling for run in config.runs) == {
        "parameter_posterior": 49,
        "mle_predictive": 60,
        "posterior_predictive": 49,
    }
    assert Counter((run.sampling, run.family) for run in config.runs) == {
        ("parameter_posterior", "pdz"): 14,
        ("parameter_posterior", "trpb"): 28,
        ("parameter_posterior", "nuc"): 7,
        ("mle_predictive", "pdz"): 19,
        ("mle_predictive", "trpb"): 34,
        ("mle_predictive", "nuc"): 7,
        ("posterior_predictive", "pdz"): 14,
        ("posterior_predictive", "trpb"): 28,
        ("posterior_predictive", "nuc"): 7,
    }
    assert len({run.landscape for run in config.runs}) == 19
    assert len({run.feature_path for run in config.runs}) == 19
    assert len({run.score_path for run in config.runs}) == 158
    assert {
        (sampling, family): {
            run.score
            for run in config.runs
            if run.sampling == sampling and run.family == family
        }
        for sampling in (
            "parameter_posterior",
            "mle_predictive",
            "posterior_predictive",
        )
        for family in ("pdz", "trpb", "nuc")
    } == {
        ("parameter_posterior", "pdz"): {"log_mean_growth"},
        ("parameter_posterior", "trpb"): {"log_mean_growth"},
        ("parameter_posterior", "nuc"): {"model_fluorescence"},
        ("mle_predictive", "pdz"): {"dimsum_fitness"},
        ("mle_predictive", "trpb"): {"johnston_fitness"},
        ("mle_predictive", "nuc"): {"highest_gate_enrichment"},
        ("posterior_predictive", "pdz"): {"dimsum_fitness"},
        ("posterior_predictive", "trpb"): {"johnston_fitness"},
        ("posterior_predictive", "nuc"): {"highest_gate_enrichment"},
    }
    assert {
        run.min_reference_samples for run in config.runs
        if run.sampling == "parameter_posterior"
    } == {1}
    assert {
        run.min_reference_samples for run in config.runs
        if run.sampling != "parameter_posterior"
    } == {2}
    assert {
        run.variant_chunk_size
        for run in config.runs
        if run.sampling != "parameter_posterior"
    } == {4096}
    assert all(run.score_higher_is_better for run in config.runs)
    assert all(run.external_higher_is_better for run in config.runs)
    assert len({run.variant_chunk_size for run in config.runs}) == 20
    assert min(run.variant_chunk_size for run in config.runs) == 337
    assert max(run.variant_chunk_size for run in config.runs) == 4096
    tie_sensitive = next(
        run
        for run in config.runs
        if run.run_id == "trpb-libd-parameter-posterior-gamma-exponential"
    )
    assert tie_sensitive.variant_chunk_size == 567
    assert config.settings.k_values == (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
    assert config.settings.seed == 23


def test_method_neutral_runner_executes_reuses_and_rejects_mismatch(tmp_path):
    runs = [
        _run("synthetic-parameter", "parameter_posterior", higher_is_better=True),
        _run("synthetic-predictive", "posterior_predictive", higher_is_better=False),
    ]
    config = load_run_config(_write_config(tmp_path, runs))
    for run in runs:
        _write_scores(config.root, run)
    result_root = tmp_path / "analysis/results"

    first = execute_runs(config, config.runs, result_root=result_root)
    second = execute_runs(config, config.runs, result_root=result_root)

    assert [outcome.state for outcome in first] == ["run", "run"]
    assert [outcome.state for outcome in second] == ["reuse", "reuse"]
    metadata = json.loads((first[0].result_path / "metadata.json").read_text())
    assert metadata["n_source_samples"] == 2
    assert metadata["n_excluded_samples"] == 0
    metadata_path = first[0].result_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FileExistsError, match="does not match"):
        execute_runs(config, config.runs[:1], result_root=result_root)


@pytest.mark.parametrize("external_data", [False, True])
def test_command_entrypoint_filters_and_executes_one_run(tmp_path, capsys, external_data):
    run = _run("synthetic-parameter", "parameter_posterior", higher_is_better=True)
    config_path = _write_config(tmp_path, [run])
    _write_scores(tmp_path, run)
    _write_features(tmp_path, run)
    result_root = tmp_path / "results"
    table_root = tmp_path / "tables"
    arguments = [
        "--config",
        str(config_path),
        "--result-root",
        str(result_root),
        "--table-root",
        str(table_root),
        "--run-id",
        run["run_id"],
    ]
    if external_data:
        data_root = tmp_path / "analysis data"
        (tmp_path / "data/downloads").rename(data_root)
        arguments.extend(["--data-root", str(data_root)])

    main([*arguments, "--dry-run"])
    assert "(dry run)" in capsys.readouterr().out
    assert not result_root.exists()
    assert not table_root.exists()

    main(arguments)

    output = capsys.readouterr().out
    assert "Completed 1 run(s): 1 run, 0 reused." in output
    result_path = result_root / "parameter_posterior" / run["run_id"]
    metadata = json.loads((result_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["score_path"] == run["score_path"]
    assert metadata["feature_path"] == run["feature_path"]
    table_dir = table_root / "parameter_posterior"
    assert len(list(table_dir.glob("*.csv"))) == 13
    assert (table_dir / "functionality_scores.csv.gz").is_file()

    main(arguments)
    assert "Completed 1 run(s): 0 run, 1 reused." in capsys.readouterr().out


@pytest.mark.parametrize("prefix", ["data/downloads/", ""])
def test_custom_input_root_preserves_config_and_output_root(tmp_path, monkeypatch, prefix):
    run = _run("synthetic-parameter", "parameter_posterior", higher_is_better=True)
    run["score_path"] = f"{prefix}scores/{run['run_id']}"
    run["feature_path"] = f"{prefix}features/three-variants"
    repository_root = tmp_path / "checkout"
    config_path = _write_config(repository_root, [run])
    original = load_run_config(config_path)
    monkeypatch.chdir(tmp_path)

    relocated = load_run_config(config_path, data_root="analysis data")

    assert relocated.root == original.root == repository_root
    assert relocated.runs == original.runs
    assert relocated.settings == original.settings
    assert relocated.data_root == tmp_path / "analysis data"
    assert original.input_path(original.runs[0].score_path) == (
        repository_root / run["score_path"]
    )
    assert relocated.input_path(relocated.runs[0].score_path) == (
        tmp_path / "analysis data/scores" / run["run_id"]
    )
    assert relocated.input_path(relocated.runs[0].feature_path) == (
        tmp_path / "analysis data/features/three-variants"
    )
