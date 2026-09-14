"""Run one of the six optional fit-to-score examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing.predictive import (  # noqa: E402
    extract_nuc_g3_predictive_scores,
    extract_pdz_predictive_scores,
    extract_trpb_predictive_scores,
)
from preprocessing.scores import (  # noqa: E402
    compare_score_data,
    extract_parameter_scores,
)

EXAMPLES = (
    "pdz-hmc",
    "pdz-mle",
    "trpb-hmc",
    "trpb-mle",
    "nuc-hmc",
    "nuc-mle",
)

RUN_IDS = {
    "pdz-parameter": "pdz-cis-parameter-posterior-lomax-exponential",
    "pdz-mle": "pdz-cis-mle-predictive-lomax-exponential",
    "pdz-posterior": "pdz-cis-posterior-predictive-lomax-exponential",
    "trpb-parameter": "trpb-liba-parameter-posterior-exponential-exponential",
    "trpb-mle": "trpb-four-site-mle-predictive-exponential-exponential",
    "trpb-posterior": "trpb-liba-posterior-predictive-exponential-exponential",
    "nuc-parameter": "nuc-g3-parameter-posterior-exponential",
    "nuc-mle": "nuc-g3-mle-predictive-exponential",
    "nuc-posterior": "nuc-g3-posterior-predictive-exponential",
}


def _read_variants(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    na_values = {column: [""] for column in columns if column != "variant_id"}
    return pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
        na_values=na_values,
    )


def _score_path(data_root: Path, sampling: str, run_id: str) -> Path:
    path = data_root / sampling / run_id
    if not (path / "variants.csv").is_file() or not (path / "scores.nc").is_file():
        raise FileNotFoundError(
            f"Downloaded score data are required for variant alignment: {path}"
        )
    return path


def _output_path(output_root: Path, sampling: str, run_id: str) -> Path:
    return output_root / sampling / run_id


def _check(output: Path, reference: Path, *, enabled: bool) -> None:
    if enabled:
        compare_score_data(output, reference)
        print(f"Exact reference match: {output}", flush=True)


def _run_hmc(
    example: str,
    fit_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    sample_count: int | None,
    seed: int,
    sample_chunk_size: int,
    overwrite: bool,
    check: bool,
) -> None:
    family = example.removesuffix("-hmc")
    parameter_id = RUN_IDS[f"{family}-parameter"]
    posterior_id = RUN_IDS[f"{family}-posterior"]
    parameter_reference = _score_path(data_root, "parameter_posterior", parameter_id)
    posterior_reference = _score_path(
        data_root,
        "posterior_predictive",
        posterior_id,
    )
    source_variants = _read_variants(posterior_reference / "variants.csv")
    parameter_variants = _read_variants(parameter_reference / "variants.csv")
    parameter_output = _output_path(
        output_root,
        "parameter_posterior",
        parameter_id,
    )
    posterior_output = _output_path(
        output_root,
        "posterior_predictive",
        posterior_id,
    )

    parameter_options = {
        "pdz": ("log_r_mean", "identity", "KNYKQTSV"),
        "trpb": ("r", "positive", None),
        "nuc": ("log_fluorescence", "identity", None),
    }
    posterior_variable, transform, priority_variant = parameter_options[family]
    extract_parameter_scores(
        fit_path,
        source_variants,
        parameter_variants,
        parameter_output,
        posterior_variable=posterior_variable,
        transform=transform,
        sample_count=sample_count,
        seed=seed,
        priority_variant_id=priority_variant,
        sample_chunk_size=sample_chunk_size,
        overwrite=overwrite,
    )
    _check(parameter_output, parameter_reference, enabled=check)

    if family == "pdz":
        extract_pdz_predictive_scores(
            fit_path,
            source_variants,
            posterior_output,
            wild_type_variant_id="KNYKQTSV",
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=True,
            overwrite=overwrite,
        )
    elif family == "trpb":
        extract_trpb_predictive_scores(
            fit_path,
            source_variants,
            posterior_output,
            times=np.array([18, 44], dtype=float),
            four_site=False,
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=True,
            overwrite=overwrite,
        )
    else:
        extract_nuc_g3_predictive_scores(
            fit_path,
            source_variants,
            posterior_output,
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=True,
            overwrite=overwrite,
        )
    _check(posterior_output, posterior_reference, enabled=check)


def _run_mle(
    example: str,
    fit_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    sample_count: int | None,
    seed: int,
    sample_chunk_size: int,
    overwrite: bool,
    check: bool,
) -> None:
    family = example.removesuffix("-mle")
    run_id = RUN_IDS[f"{family}-mle"]
    reference = _score_path(data_root, "mle_predictive", run_id)
    variants = _read_variants(reference / "variants.csv")
    output = _output_path(output_root, "mle_predictive", run_id)
    if family == "pdz":
        extract_pdz_predictive_scores(
            fit_path,
            variants,
            output,
            wild_type_variant_id="KNYKQTSV",
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=False,
            overwrite=overwrite,
        )
    elif family == "trpb":
        extract_trpb_predictive_scores(
            fit_path,
            variants,
            output,
            times=np.array([12, 16, 20, 24, 28, 36], dtype=float),
            four_site=True,
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=False,
            overwrite=overwrite,
        )
    else:
        extract_nuc_g3_predictive_scores(
            fit_path,
            variants,
            output,
            sample_count=sample_count,
            seed=seed,
            sample_chunk_size=sample_chunk_size,
            preserve_chain_draw=False,
            overwrite=overwrite,
        )
    _check(output, reference, enabled=check)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=EXAMPLES)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/downloads/scores",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "preprocessing/output",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--samples", type=int, default=128)
    sample_group.add_argument("--all-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sample-chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.fit.is_file():
        raise FileNotFoundError(f"Fit file does not exist: {args.fit}")
    sample_count = None if args.all_samples else args.samples
    options = {
        "sample_count": sample_count,
        "seed": args.seed,
        "sample_chunk_size": args.sample_chunk_size,
        "overwrite": args.overwrite,
        "check": not args.no_check,
    }
    if args.example.endswith("-hmc"):
        _run_hmc(
            args.example,
            args.fit,
            args.data_root,
            args.output_root,
            **options,
        )
    else:
        _run_mle(
            args.example,
            args.fit,
            args.data_root,
            args.output_root,
            **options,
        )


if __name__ == "__main__":
    main()
