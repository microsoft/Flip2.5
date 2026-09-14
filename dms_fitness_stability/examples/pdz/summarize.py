"""Summarize all saved PDZ landscapes without reopening HMC fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from dms_fitness_stability.adapters.pdz import PDZ_LANDSCAPES

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _result_paths(result_dir: Path) -> list[tuple[str, Path]]:
    paths = []
    for landscape in PDZ_LANDSCAPES:
        landscape_dir = result_dir / landscape
        model_paths = sorted(landscape_dir.glob("*/metadata.json"))
        if model_paths:
            paths.extend((landscape, path.parent) for path in model_paths)
        elif (landscape_dir / "metadata.json").is_file():
            paths.append((landscape, landscape_dir))
    return paths


def summarize_results(result_dir: Path, *, k: int) -> pd.DataFrame:
    """Return one compact stability row per saved PDZ fit and score."""
    rows = []
    for landscape, fit_dir in _result_paths(result_dir):
        metadata = json.loads(
            (fit_dir / "metadata.json").read_text(encoding="utf-8")
        )
        model = metadata["model"]
        summary = pd.read_csv(fit_dir / "summary.csv")
        overlaps = pd.read_csv(fit_dir / "overlaps.csv")
        available_k = set(summary["k"].unique())
        if k not in available_k:
            raise ValueError(
                f"{landscape} {model}: K={k} is unavailable; "
                f"choose from {sorted(available_k)}."
            )
        for score in summary["score"].unique():
            selected = summary[(summary["score"] == score) & (summary["k"] == k)]
            tracked = selected[selected["is_tracked_reported_cohort"]]
            reported_top_k = tracked[tracked["reported_rank"] <= k]
            score_overlaps = overlaps[
                (overlaps["score"] == score) & (overlaps["k"] == k)
            ]
            pairwise = score_overlaps[
                score_overlaps["comparison"] == "sample_vs_sample"
            ]
            reference = score_overlaps[
                score_overlaps["comparison"] == "sample_vs_reference"
            ]
            rows.append(
                {
                    "landscape": landscape,
                    "model": model,
                    "score": score,
                    "k": k,
                    "variants_ever_selected": int((selected["top_k_count"] > 0).sum()),
                    "max_inclusion_probability": selected["top_k_probability"].max(),
                    "median_reported_top_k_persistence": reported_top_k[
                        "posterior_top_k_probability"
                    ].median(),
                    "median_sample_vs_sample_recall": pairwise["recall"].median(),
                    "median_sample_vs_reported_recall": reference["recall"].median(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "pdz" / "results",
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    overview = summarize_results(args.result_dir, k=args.k)
    output = args.output or args.result_dir / f"overview_top_{args.k}.csv"
    overview.to_csv(output, index=False)
    print(overview.to_string(index=False))
    print(f"Saved overview to {output}")


if __name__ == "__main__":
    main()
