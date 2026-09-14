"""Legacy trans-only runner; use examples/pdz/run.py for all PDZ landscapes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dms_fitness_stability.adapters.pdz import load_pdz_trans, scan_pdz_hmc

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan full Exponential-Exponential PDZ trans HMC fits."
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=("trans-1", "trans-2"),
        default=("trans-1", "trans-2"),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data-root>/pdz/results.",
    )
    parser.add_argument("--trans-1-hmc", type=Path, default=None)
    parser.add_argument("--trans-2-hmc", type=Path, default=None)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Posterior samples per source. Default: all available samples.",
    )
    parser.add_argument("--retain-k", type=int, default=1000)
    parser.add_argument("--tracked-top-n", type=int, default=1000)
    parser.add_argument("--top-k", type=int, nargs="+", default=(50, 100, 1000))
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_root / "pdz" / "results"
    pdz_data = args.data_root / "pdz_trans"
    raw_data = pdz_data / "raw"
    count_paths = {
        source: raw_data / f"{source}.tsv" for source in ("trans-1", "trans-2")
    }
    fit_data = pdz_data / "fits"
    hmc_paths = {
        "trans-1": args.trans_1_hmc
        or fit_data / "trans-1_exponential_exponential.nc",
        "trans-2": args.trans_2_hmc
        or fit_data / "trans-2_exponential_exponential.nc",
    }
    reported_fitness_path = raw_data / "GSE265816_pdz3cript_sub.csv.gz"
    required_paths = [reported_fitness_path, *count_paths.values()]
    required_paths.extend(hmc_paths[source] for source in args.source)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PDZ inputs: {missing}")

    for source in args.source:
        print(f"Loading raw {source} counts and reported fitness", flush=True)
        landscape = load_pdz_trans(
            source,
            count_paths=count_paths,
            reported_fitness_path=reported_fitness_path,
        )
        result = scan_pdz_hmc(
            landscape,
            hmc_paths[source],
            top_k=tuple(args.top_k),
            retain_k=args.retain_k,
            tracked_top_n=args.tracked_top_n,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        output = result.save(output_dir / source)
        print(
            f"Saved {source} result to {output} "
            f"({result.metadata['elapsed_seconds'] / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
