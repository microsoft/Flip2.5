"""Run stability analysis, tidy reporting, and plotting for a tabular source."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dms_fitness_stability import (
    LandscapeFeatureBundle,
    ScoreEnsemble,
    analyze_score_ensemble,
    build_paper_tables,
    plot_inclusion,
)
from dms_fitness_stability.analysis_config import load_analysis_config
from dms_fitness_stability.mechanism_reporting import build_mechanism_tables

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "analysis.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output directory declared in the analysis JSON.",
    )
    return parser.parse_args()


def _relative_to_config(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config_path.parent / path


def _load_features(
    config: dict[str, object],
    config_path: Path,
    result_variants,
) -> LandscapeFeatureBundle | None:
    feature_path = config.get("feature_bundle")
    requirements = config.get("features")
    if feature_path is None:
        if requirements:
            raise ValueError(
                "analysis.json declares feature requirements but no feature_bundle."
            )
        return None
    bundle = LandscapeFeatureBundle.load(
        _relative_to_config(config_path, feature_path)
    )
    try:
        indices = bundle.align_result_variants(result_variants)
        requirements = dict(requirements or {})
        required = list(requirements.get("required", []))
        permitted_missing = list(requirements.get("permitted_missing", []))
        unknown = set(required + permitted_missing) - set(bundle.features.data_vars)
        if unknown:
            raise ValueError(
                f"Feature bundle is missing configured variables: {sorted(unknown)}."
            )
        for name in required:
            if not np.isfinite(bundle.features[name].values[indices]).all():
                raise ValueError(
                    f"Required feature {name!r} contains missing values for analyzed "
                    "variants; move it to features.permitted_missing or regenerate "
                    "the feature bundle."
                )
    except Exception:
        bundle.close()
        raise
    return bundle


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_analysis_config(config_path)
    output = args.output or _relative_to_config(
        config_path,
        config.get("output", "output"),
    )
    ensemble = ScoreEnsemble.load(
        _relative_to_config(config_path, config["score_bundle"])
    )
    feature_bundle = None
    try:
        result = analyze_score_ensemble(
            ensemble,
            config=config,
            show_progress=False,
        )
        feature_bundle = _load_features(config, config_path, result.variants)
        mechanism_config = config.get("mechanisms")
        mechanism_tables = {}
        if mechanism_config is not None:
            if feature_bundle is None:
                raise ValueError(
                    "analysis.json declares mechanisms but no feature_bundle."
                )
            settings = dict(mechanism_config)
            mechanism_tables = build_mechanism_tables(
                result,
                ensemble,
                feature_bundle,
                score_name=str(config["score_name"]),
                direction=str(config["direction"]),
                decision_k=int(settings["decision_k"]),
                challenger_probability=float(
                    settings["challenger_probability"]
                ),
                reference_scale_k=int(
                    settings.get("reference_scale_k") or 1000
                ),
                interval=tuple(settings["interval"]),
            )
    finally:
        if feature_bundle is not None:
            feature_bundle.close()
        ensemble.close()

    result_path = result.save(output / "result")
    reporting = dict(config.get("reporting") or {})
    main_k = int(reporting.get("main_k", max(config["top_k"])))
    k_values = tuple(reporting.get("k_values", config["top_k"]))
    tables = {
        **build_paper_tables(
            result,
            score=str(config["score_name"]),
            main_k=main_k,
            k_values=k_values,
            percentile_cohorts=tuple(
                reporting.get("percentile_cohorts", config["top_k"])
            ),
            candidate_probability=float(
                reporting.get("candidate_probability", 0.01)
            ),
        ),
        **mechanism_tables,
    }
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)

    figure, _, _ = plot_inclusion(
        result,
        score=str(config["score_name"]),
        k=main_k,
        n_variants=min(20, len(result.variants)),
    )
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "inclusion.png"
    figure.savefig(figure_path, dpi=150)
    print(f"Saved numerical result to {result_path}")
    print(f"Saved {len(tables)} tidy tables to {table_dir}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
