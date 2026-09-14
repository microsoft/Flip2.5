import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from dms_fitness_stability import LandscapeFeatureBundle, StabilityResult
from dms_fitness_stability.mechanism_reporting import MECHANISM_TABLE_NAMES
from dms_fitness_stability.reporting import PAPER_TABLE_NAMES

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples/tabular"


def test_unknown_tabular_landscape_runs_from_csv_to_figure(tmp_path):
    for name in (
        "analysis.json",
        "counts.csv",
        "scores.csv",
        "source.json",
        "variants.csv",
    ):
        shutil.copy2(EXAMPLE / name, tmp_path / name)
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "preprocess.py"),
            "--source",
            str(tmp_path / "source.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "run.py"),
            "--config",
            str(tmp_path / "analysis.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = StabilityResult.load(output / "result")
    assert result.metadata["protein_family"] == "novel-enzyme"
    assert result.metadata["reference_top_k_identifiable"] is False
    assert result.metadata["higher_is_better"] == {"measurement_error": False}
    assert result.metadata["analysis_config"] == json.loads(
        (tmp_path / "analysis.json").read_text(encoding="utf-8")
    )
    assert "matrix_variant_index" in result.variants
    assert "hmc_variant_index" not in result.variants
    assert "control" not in set(
        result.variants.iloc[
            result.selection["variant_index"].values.ravel()
        ]["variant_id"]
    )

    features = LandscapeFeatureBundle.load(output / "features")
    try:
        np.testing.assert_array_equal(
            features.align_result_variants(result.variants),
            np.arange(5),
        )
    finally:
        features.close()
    assert {
        path.stem for path in (output / "tables").glob("*.csv")
    } == set(PAPER_TABLE_NAMES) | set(MECHANISM_TABLE_NAMES)
    mechanism_variants = pd.read_csv(
        output / "tables/mechanism_variants.csv",
        low_memory=False,
    )
    mechanism_run = pd.read_csv(output / "tables/mechanism_runs.csv").iloc[0]
    assert mechanism_variants["variant_id"].tolist() == ["A", "B", "C"]
    assert not mechanism_variants["is_model_top_k"].iloc[1:].any()
    assert mechanism_variants["is_challenger"].iloc[1:].all()
    assert np.isfinite(
        mechanism_variants[
            [
                "mechanism_input_count",
                "normalized_score_interval_width",
                "rank_interval_width",
                "top_k_inclusion_probability",
            ]
        ]
    ).all().all()
    assert mechanism_run["decision_k"] == 1
    assert mechanism_run["model_top_k_count"] == 1
    assert mechanism_run["challenger_count"] == 2
    pixels = mpimg.imread(output / "figures/inclusion.png")
    assert pixels.size > 0
    assert float(np.nanstd(pixels)) > 0.01
