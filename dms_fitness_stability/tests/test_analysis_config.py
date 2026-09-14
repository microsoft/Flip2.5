import json

import pytest

from dms_fitness_stability.analysis_config import (
    analysis_config_sha256,
    load_analysis_config,
    validate_analysis_config,
)


def _minimal_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "score_bundle": "scores/example",
        "score_name": "measurement_error",
        "direction": "lower",
        "top_k": [1, 5, 10],
        "seed": 23,
    }


def test_analysis_config_loads_minimal_researcher_policy(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(_minimal_config()), encoding="utf-8")

    config = load_analysis_config(path)

    assert config == _minimal_config()
    assert analysis_config_sha256(config) == analysis_config_sha256(dict(config))


def test_analysis_config_validates_optional_reference_and_features():
    config = {
        **_minimal_config(),
        "external_reference": {
            "value_column": "published_error",
            "rank_column": "published_rank",
            "direction": "lower",
            "source": "synthetic benchmark",
            "tracked_top_n": 10,
        },
        "features": {
            "required": ["input_count"],
            "permitted_missing": ["reduced_heterogeneity"],
        },
        "model_reference_min_finite_samples": 2,
    }

    normalized = validate_analysis_config(config)

    assert normalized["external_reference"]["identifiable_top_k"] is True
    assert normalized["features"] == config["features"]
    assert normalized["model_reference_min_finite_samples"] == 2


def test_analysis_config_validates_mechanism_settings():
    config = {
        **_minimal_config(),
        "mechanisms": {
            "decision_k": 5,
            "reference_scale_k": 10,
        },
    }

    normalized = validate_analysis_config(config)

    assert normalized["mechanisms"] == {
        "decision_k": 5,
        "reference_scale_k": 10,
        "challenger_probability": 0.01,
        "interval": [0.05, 0.95],
    }


def test_analysis_config_rejects_unknown_mechanism_setting():
    config = {
        **_minimal_config(),
        "mechanisms": {"decision_k": 5, "family": "pdz"},
    }

    with pytest.raises(ValueError, match="unknown fields.*family"):
        validate_analysis_config(config)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"direction": "ascending"}, "higher.*lower"),
        ({"top_k": [5, 1]}, "increasing"),
        ({"seed": -1}, "non-negative"),
        ({"family": "synthetic"}, "unknown=\\['family'\\]"),
    ],
)
def test_analysis_config_rejects_ambiguous_or_unknown_choices(update, message):
    config = {**_minimal_config(), **update}

    with pytest.raises(ValueError, match=message):
        validate_analysis_config(config)
