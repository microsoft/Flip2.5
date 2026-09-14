"""Run FLIP-style Ridge one-hot baselines for the PDZ cis model.

The PDZ cis NetCDF variant axis is recovered by grouping Bruce's
TSV file by ``aa_seq`` with sorted amino-acid sequences and summing the observed count
columns. The summed counts are checked against the MLE NetCDF observed-data group
before fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Same AA-index as in Kevin's FLIP code
AAINDEX_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
INPUT_COUNT_COLUMNS = [
    "input1_e1_s0_bNA_count",
    "input2_e2_s0_bNA_count",
    "input3_e3_s0_bNA_count",
]
OUTPUT_COUNT_COLUMNS = [
    "output1A_e1_s1_b1_count",
    "output2A_e2_s1_b1_count",
    "output3A_e3_s1_b1_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Ridge one-hot baselines to PDZ cis observed and pseudo-dataset targets."
    )
    parser.add_argument(
        "--metadata-tsv",
        type=Path,
        default=Path("data/cis_from_bruce_azure.tsv"),
        help="Bruce TSV with nt_seq, aa_seq, and replicate input/output counts.",
    )
    parser.add_argument(
        "--mle-nc",
        type=Path,
        default=Path("data/pdz_cis_exponential_exponential_samples.nc"),
    )
    parser.add_argument(
        "--hmc-nc",
        type=Path,
        default=Path("data/pdz_cis_exponential_exponential_diagnosed.nc"),
    )
    parser.add_argument("--no-hmc", action="store_true", help="Skip HMC pseudo-datasets.")
    parser.add_argument(
        "--ground-truth-source",
        choices=("observed_counts", "mle_log_r_mean"),
        default="observed_counts",
    )
    parser.add_argument(
        "--pseudo-target-source",
        choices=("posterior_predictive_counts", "posterior_log_r_mean"),
        default="posterior_log_r_mean",
    )
    parser.add_argument("--n-pseudodatasets", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--ridge-tol", type=float, default=1e-5)
    parser.add_argument("--ridge-max-iter", type=int, default=1_000_000)
    parser.add_argument("--interval", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pdz_cis_linear_baselines"),
    )
    return parser.parse_args()

def load_cis_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"aa_seq", *INPUT_COUNT_COLUMNS, *OUTPUT_COUNT_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns from {path}: {missing}")

    count_columns = INPUT_COUNT_COLUMNS + OUTPUT_COUNT_COLUMNS
    grouped = df.groupby("aa_seq", sort=True, as_index=False)[count_columns].sum()
    return grouped


def validate_observed_counts(metadata: pd.DataFrame, idata: az.InferenceData) -> None:
    """
    We have this function to validate the sequence-to-observed data mapping.
    """
    expected = idata.observed_data.sizes["b"]
    if len(metadata) != expected:
        raise ValueError(f"Expected {expected} unique aa_seq rows, found {len(metadata)}")

    starting_counts = metadata[INPUT_COUNT_COLUMNS].to_numpy().T
    timepoint_counts = metadata[OUTPUT_COUNT_COLUMNS].to_numpy().T
    if not np.array_equal(starting_counts, idata.observed_data["starting_counts"].values):
        raise ValueError("Aggregated metadata starting counts do not match MLE NetCDF observed data.")
    if not np.array_equal(timepoint_counts, idata.observed_data["timepoint_counts"].values):
        raise ValueError("Aggregated metadata timepoint counts do not match MLE NetCDF observed data.")


def one_hot_flatten(sequences: pd.Series) -> np.ndarray:
    clean_sequences = [str(sequence).replace(":", "").upper() for sequence in sequences]
    max_len = max(len(sequence) for sequence in clean_sequences)
    aa_to_idx = {aa: idx for idx, aa in enumerate(AAINDEX_ALPHABET)}
    encoded = np.zeros((len(clean_sequences), max_len, len(AAINDEX_ALPHABET)), dtype=np.float32)

    for row, sequence in enumerate(clean_sequences):
        for col, aa in enumerate(sequence):
            aa_idx = aa_to_idx.get(aa)
            if aa_idx is not None:
                encoded[row, col, aa_idx] = 1.0

    return encoded.reshape(len(clean_sequences), -1)


def make_split(n_variants: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(n_variants)
    train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
    return np.asarray(train_idx), np.asarray(test_idx)


def observed_count_target(idata: az.InferenceData) -> np.ndarray:
    c0 = idata.observed_data["starting_counts"]
    cf = idata.observed_data["timepoint_counts"]
    target = np.log((cf + 1) / (c0 + 1)).where(lambda values: np.isfinite(values))
    return target.mean(dim=[dim for dim in target.dims if dim != "b"], skipna=True).values


def mle_log_r_target(idata: az.InferenceData) -> np.ndarray:
    target = idata.mle["log_r_mean"]
    if target.dims != ("b",):
        target = target.mean(dim=[dim for dim in target.dims if dim != "b"], skipna=True)
    return target.values


def posterior_predictive_count_targets(
    idata: az.InferenceData, n_draws: int, seed: int
) -> xr.DataArray:
    c0 = idata.posterior_predictive["starting_counts"]
    cf = idata.posterior_predictive["timepoint_counts"]
    target = np.log((cf + 1) / (c0 + 1)).where(lambda values: np.isfinite(values))
    target = target.mean(
        dim=[dim for dim in target.dims if dim not in {"chain", "draw", "b"}],
        skipna=True,
    )
    return sample_draws(target.stack(sample=("chain", "draw")), n_draws, seed)


def posterior_log_r_targets(idata: az.InferenceData, n_draws: int, seed: int) -> xr.DataArray:
    target = idata.posterior["log_r_mean"]
    target = target.mean(
        dim=[dim for dim in target.dims if dim not in {"chain", "draw", "b"}],
        skipna=True,
    )
    return sample_draws(target.stack(sample=("chain", "draw")), n_draws, seed)


def sample_draws(targets: xr.DataArray, n_draws: int, seed: int) -> xr.DataArray:
    targets = targets.transpose("sample", "b")
    available = targets.sizes["sample"]
    if n_draws > available:
        raise ValueError(f"Requested {n_draws} pseudo-datasets, but only {available} draws exist.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(available, size=n_draws, replace=False)
    return targets.isel(sample=selected).reset_index("sample", drop=True).assign_coords(
        sample=np.arange(n_draws)
    )


def fit_ridge_one_hot(
    x_one_hot: np.ndarray,
    train_target: np.ndarray,
    eval_target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int,
) -> float:
    train_idx = train_idx[np.isfinite(train_target[train_idx])]
    test_idx = test_idx[np.isfinite(eval_target[test_idx])]
    scaler = StandardScaler()
    y_train = scaler.fit_transform(train_target[train_idx, None])
    y_test = scaler.transform(eval_target[test_idx, None])

    model = Ridge(solver="auto", tol=tol, max_iter=max_iter, alpha=alpha)
    model.fit(x_one_hot[train_idx], y_train)
    predictions = model.predict(x_one_hot[test_idx]).reshape(-1, 1)
    return float(spearmanr(y_test, predictions).correlation)


def summarize(results: pd.DataFrame, interval: float) -> pd.DataFrame:
    lower_q = (1.0 - interval) / 2.0
    upper_q = 1.0 - lower_q
    grouped = results.groupby("target_family")["spearman"]
    summary = grouped.agg(n="count", mean="mean", median="median", std="std").reset_index()
    intervals = grouped.quantile([lower_q, upper_q]).unstack(level=-1).reset_index()
    intervals = intervals.rename(columns={lower_q: "lower", upper_q: "upper"})
    return summary.merge(intervals, on="target_family", how="left")


def target_vector(idata: az.InferenceData, source: str) -> np.ndarray:
    if source == "observed_counts":
        return observed_count_target(idata)
    if source == "mle_log_r_mean":
        return mle_log_r_target(idata)
    raise ValueError(f"Unknown target source: {source}")


def pseudo_targets(idata: az.InferenceData, source: str, n_draws: int, seed: int) -> xr.DataArray:
    if source == "posterior_predictive_counts":
        return posterior_predictive_count_targets(idata, n_draws, seed)
    if source == "posterior_log_r_mean":
        return posterior_log_r_targets(idata, n_draws, seed)
    raise ValueError(f"Unknown pseudo target source: {source}")


def main() -> None:
    args = parse_args()
    metadata = load_cis_metadata(args.metadata_tsv)
    mle_idata = az.from_netcdf(args.mle_nc)
    validate_observed_counts(metadata, mle_idata)

    x_one_hot = one_hot_flatten(metadata["aa_seq"])
    train_idx, test_idx = make_split(len(metadata), args.test_size, args.random_seed)
    ground_truth = target_vector(mle_idata, args.ground_truth_source)

    rows: list[dict[str, object]] = [
        {
            "target_family": "ground_truth",
            "draw": np.nan,
            "model": "Ridge (one-hot)",
            "spearman": fit_ridge_one_hot(
                x_one_hot,
                ground_truth,
                ground_truth,
                train_idx,
                test_idx,
                args.ridge_alpha,
                args.ridge_tol,
                args.ridge_max_iter,
            ),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "ground_truth_source": args.ground_truth_source,
            "pseudo_target_source": "ground_truth",
        }
    ]

    sources: list[tuple[str, az.InferenceData]] = [("mle", mle_idata)]
    if not args.no_hmc:
        sources.append(("hmc", az.from_netcdf(args.hmc_nc)))

    for target_family, idata in sources:
        draws = pseudo_targets(idata, args.pseudo_target_source, args.n_pseudodatasets, args.random_seed)
        for draw in range(draws.sizes["sample"]):
            train_target = draws.isel(sample=draw).values
            rows.append(
                {
                    "target_family": target_family,
                    "draw": draw,
                    "model": "Ridge (one-hot)",
                    "spearman": fit_ridge_one_hot(
                        x_one_hot,
                        train_target,
                        ground_truth,
                        train_idx,
                        test_idx,
                        args.ridge_alpha,
                        args.ridge_tol,
                        args.ridge_max_iter,
                    ),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "ground_truth_source": args.ground_truth_source,
                    "pseudo_target_source": args.pseudo_target_source,
                }
            )

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "correlations.csv"
    summary_path = args.output_dir / "summary.csv"
    results.to_csv(results_path, index=False)
    summarize(results, args.interval).to_csv(summary_path, index=False)

    print(f"Aggregated {len(metadata)} aa_seq variants from {args.metadata_tsv}")
    print(f"One-hot feature shape: {x_one_hot.shape}")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")

if __name__ == "__main__":
    main()
