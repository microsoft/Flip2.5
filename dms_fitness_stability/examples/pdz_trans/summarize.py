"""Summarize the saved PDZ trans stability analyses without reopening HMC fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def summarize_results(result_dir: Path, *, k: int) -> pd.DataFrame:
    """Return one compact stability row per source and score."""
    rows = []
    for source in ("trans-1", "trans-2"):
        summary = pd.read_csv(result_dir / source / "summary.csv")
        overlaps = pd.read_csv(result_dir / source / "overlaps.csv")
        available_k = set(summary["k"].unique())
        if k not in available_k:
            raise ValueError(
                f"{source}: K={k} is unavailable; choose from {sorted(available_k)}."
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
                    "source": source,
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
    parser.add_argument("--result-dir", type=Path, default=HERE / "results")
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
