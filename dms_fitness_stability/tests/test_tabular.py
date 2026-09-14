import json

import numpy as np
import pandas as pd

from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.tabular import (
    build_paired_feature_bundle_from_table,
    build_score_ensemble_from_table,
    load_tabular_source,
)


def _identity() -> AnalysisIdentity:
    return AnalysisIdentity(
        protein_family="unknown",
        landscape="synthetic-errors",
        inference_method="simulation",
        model_spec="benchmark",
        fit_id="seed-23",
    )


def _variants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["A", "B", "C"],
            "is_ranking_candidate": [True, True, False],
            "variant_label": ["alpha", "beta", "control"],
        }
    )


def test_tabular_score_builder_preserves_declared_axes_and_sample_ids(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "variant_id": variant_id,
                "score_name": "measurement_error",
                "fitness_score": value,
            }
            for sample_id, values in ((20, [3, 1, 100]), (10, [1, 2, 100]))
            for variant_id, value in zip(["A", "B", "C"], values, strict=True)
        ]
    )
    ensemble = build_score_ensemble_from_table(
        _variants(),
        rows,
        identity=_identity(),
        score_contracts={"measurement_error": "synthetic-error-v1"},
        uncertainty_target="synthetic_measurement_repeatability",
        fit_method="simulation",
        sample_kind="simulated experiment",
    )

    output = ensemble.save(tmp_path / "scores")

    np.testing.assert_array_equal(ensemble.scores["source_sample_id"], [10, 20])
    np.testing.assert_allclose(
        ensemble.fitness_score.sel(score_name="measurement_error"),
        [[1, 2, 100], [3, 1, 100]],
    )
    assert ensemble.analysis_variants["variant_label"].tolist() == [
        "alpha",
        "beta",
        "control",
    ]
    assert (output / "fitness_scores.nc").is_file()


def test_tabular_paired_counts_build_existing_feature_contract(tmp_path):
    counts = pd.DataFrame(
        [
            {
                "replicate_id": replicate,
                "variant_id": variant,
                "input_count": input_count,
                "output_count": output_count,
            }
            for replicate, inputs, outputs in (
                ("r2", [20, 10, 5], [40, 10, 5]),
                ("r1", [10, 20, 5], [20, 40, 5]),
            )
            for variant, input_count, output_count in zip(
                ["A", "B", "C"],
                inputs,
                outputs,
                strict=True,
            )
        ]
    )
    bundle = build_paired_feature_bundle_from_table(
        _variants(),
        counts,
        identity=_identity(),
    )

    output = bundle.save(tmp_path / "features")

    np.testing.assert_allclose(bundle.features["input_count"], [30, 30, 10])
    np.testing.assert_allclose(bundle.features["output_count"], [60, 50, 10])
    np.testing.assert_array_equal(bundle.features["observation_count"], [2, 2, 2])
    assert bundle.metadata["observation_structure"] == (
        "paired_input_output_replicates"
    )
    assert (output / "features.nc").is_file()


def test_tabular_source_descriptor_loads_both_standard_bundles(tmp_path):
    variants = _variants()
    scores = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "variant_id": variant_id,
                "score_name": "measurement_error",
                "fitness_score": value,
            }
            for sample_id, values in ((0, [3, 1, 100]), (1, [1, 2, 100]))
            for variant_id, value in zip(["A", "B", "C"], values, strict=True)
        ]
    )
    counts = pd.DataFrame(
        [
            {
                "replicate_id": replicate,
                "variant_id": variant,
                "input_count": 10,
                "output_count": output,
            }
            for replicate, outputs in (("r1", [20, 10, 5]), ("r2", [10, 20, 5]))
            for variant, output in zip(["A", "B", "C"], outputs, strict=True)
        ]
    )
    variants.to_csv(tmp_path / "variants.csv", index=False)
    scores.to_csv(tmp_path / "scores.csv", index=False)
    counts.to_csv(tmp_path / "counts.csv", index=False)
    descriptor = tmp_path / "source.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "protein_family": "unknown",
                    "landscape": "synthetic-errors",
                    "inference_method": "simulation",
                    "model_spec": "benchmark",
                    "fit_id": "seed-23",
                },
                "variants": "variants.csv",
                "scores": "scores.csv",
                "counts": "counts.csv",
                "score_contracts": {
                    "measurement_error": "synthetic-error-v1"
                },
                "uncertainty_target": "synthetic_measurement_repeatability",
                "fit_method": "simulation",
                "sample_kind": "simulated experiment",
            }
        ),
        encoding="utf-8",
    )

    ensemble, features = load_tabular_source(descriptor)

    assert features is not None
    assert ensemble.metadata["source_descriptor_sha256"] == (
        features.metadata["source_descriptor_sha256"]
    )
    assert {record["role"] for record in ensemble.metadata["source_files"]} == {
        "variants",
        "scores",
        "counts",
    }
