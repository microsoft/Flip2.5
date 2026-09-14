"""Build standard score and feature bundles from the tabular example."""

from __future__ import annotations

import argparse
from pathlib import Path

from dms_fitness_stability.tabular import load_tabular_source

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "source.json")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores, features = load_tabular_source(args.source)
    score_path = scores.save(args.output / "scores")
    print(f"Saved score bundle to {score_path}")
    if features is not None:
        feature_path = features.save(args.output / "features")
        print(f"Saved feature bundle to {feature_path}")


if __name__ == "__main__":
    main()
