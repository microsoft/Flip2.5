"""PDZ cis Ridge baseline over posterior target resamples.

Change ``TARGET_VARIABLE`` to one of ``log_r_mean``, ``r``, or ``log_r``. The
script trains Kevin/FLIP-style one-hot Ridge models on posterior draws of that
target from the MLE run, the HMC run, or both, and evaluates every model against
the MLE point estimate for the same target.
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


TARGET_VARIABLE = "log_r_mean"

AAINDEX_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
METADATA_TSV = Path("data/cis_from_bruce_azure.tsv")
MLE_NC = Path("data/pdz_cis_exponential_exponential_samples.nc")
HMC_NC = Path("data/pdz_cis_exponential_exponential_diagnosed.nc")
OUTPUT_DIR = Path("results/pdz_cis_target_resample_baseline")

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
        description="Fit one-hot Ridge baselines on MLE/HMC posterior target resamples."
    )
    parser.add_argument(
        "--resample-source",
        choices=("mle", "hmc", "both"),
        default="both",
        help="Which posterior draws to use as resampled training targets.",
    )
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_cis_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_TSV, sep="\t")
    count_columns = INPUT_COUNT_COLUMNS + OUTPUT_COUNT_COLUMNS
    required = {"aa_seq", *count_columns}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns from {METADATA_TSV}: {missing}")
    return df.groupby("aa_seq", sort=True, as_index=False)[count_columns].sum()


def validate_observed_counts(metadata: pd.DataFrame, idata: az.InferenceData) -> None:
    if len(metadata) != idata.observed_data.sizes["b"]:
        raise ValueError("Grouped aa_seq rows do not match NetCDF b dimension.")

    starting_counts = metadata[INPUT_COUNT_COLUMNS].to_numpy().T
    timepoint_counts = metadata[OUTPUT_COUNT_COLUMNS].to_numpy().T
    if not np.array_equal(starting_counts, idata.observed_data["starting_counts"].values):
        raise ValueError("Grouped starting counts do not match MLE NetCDF observed data.")
    if not np.array_equal(timepoint_counts, idata.observed_data["timepoint_counts"].values):
        raise ValueError("Grouped timepoint counts do not match MLE NetCDF observed data.")


def one_hot_flatten(sequences: pd.Series) -> np.ndarray:
    aa_to_idx = {aa: idx for idx, aa in enumerate(AAINDEX_ALPHABET)}
    max_len = max(sequences.str.len())
    encoded = np.zeros((len(sequences), max_len, len(AAINDEX_ALPHABET)), dtype=np.float32)

    for row, sequence in enumerate(sequences.astype(str).str.upper()):
        for col, aa in enumerate(sequence):
            aa_idx = aa_to_idx.get(aa)
            if aa_idx is not None:
                encoded[row, col, aa_idx] = 1.0

    return encoded.reshape(len(sequences), -1)


def point_target(idata: az.InferenceData) -> np.ndarray:
    if not hasattr(idata, "mle") or TARGET_VARIABLE not in idata.mle:
        raise ValueError(f"MLE group does not contain {TARGET_VARIABLE!r}.")
    return reduce_to_variant_vector(idata.mle[TARGET_VARIABLE])


def posterior_target_draws(
    idata: az.InferenceData, source_name: str, n_resamples: int, seed: int
) -> xr.DataArray:
    if source_name == "mle" and TARGET_VARIABLE == "log_r_mean" and "r" in idata.posterior:
        target = np.log(idata.posterior["r"].mean(dim="a", skipna=True))
    else:
        if TARGET_VARIABLE not in idata.posterior:
            raise ValueError(f"Posterior group does not contain {TARGET_VARIABLE!r}.")
        target = idata.posterior[TARGET_VARIABLE]

    reduce_dims = [dim for dim in target.dims if dim not in {"chain", "draw", "b"}]
    if reduce_dims:
        target = target.mean(dim=reduce_dims, skipna=True)

    target = target.stack(sample=("chain", "draw")).transpose("sample", "b")
    if n_resamples > target.sizes["sample"]:
        raise ValueError(
            f"Requested {n_resamples} resamples, but only {target.sizes['sample']} are available."
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(target.sizes["sample"], size=n_resamples, replace=False)
    return target.isel(sample=selected).reset_index("sample", drop=True).assign_coords(
        sample=np.arange(n_resamples)
    )


def reduce_to_variant_vector(target: xr.DataArray) -> np.ndarray:
    if "b" not in target.dims:
        raise ValueError(f"{TARGET_VARIABLE!r} does not have a b dimension: {target.dims}")
    reduce_dims = [dim for dim in target.dims if dim != "b"]
    if reduce_dims:
        target = target.mean(dim=reduce_dims, skipna=True)
    return target.transpose("b").values


def fit_ridge(
    x_one_hot: np.ndarray,
    train_target: np.ndarray,
    eval_target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    train_idx = train_idx[np.isfinite(train_target[train_idx])]
    test_idx = test_idx[np.isfinite(eval_target[test_idx])]

    scaler = StandardScaler()
    y_train = scaler.fit_transform(train_target[train_idx, None])
    y_test = scaler.transform(eval_target[test_idx, None])

    model = Ridge(solver="auto", tol=1e-5, max_iter=1_000_000, alpha=10)
    model.fit(x_one_hot[train_idx], y_train)
    predictions = model.predict(x_one_hot[test_idx]).reshape(-1, 1)
    return float(spearmanr(y_test, predictions).correlation)


def target_spearman(train_target: np.ndarray, eval_target: np.ndarray, test_idx: np.ndarray) -> float:
    """
    Diagnostic for how closely each resampled target matches the true fitness before model fitting.
    """
    test_idx = test_idx[np.isfinite(train_target[test_idx]) & np.isfinite(eval_target[test_idx])]
    return float(spearmanr(eval_target[test_idx], train_target[test_idx]).correlation)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("resample_source")
        .agg(
            n=("spearman", "count"),
            mean=("spearman", "mean"),
            median=("spearman", "median"),
            std=("spearman", "std"),
            target_spearman_mean=("target_spearman_vs_eval", "mean"),
        )
        .reset_index()
    )


def selected_sources(args: argparse.Namespace, mle_idata: az.InferenceData) -> list[tuple[str, az.InferenceData]]:
    if args.resample_source == "mle":
        return [("mle", mle_idata)]
    if args.resample_source == "hmc":
        return [("hmc", az.from_netcdf(HMC_NC))]
    return [("mle", mle_idata), ("hmc", az.from_netcdf(HMC_NC))]


def main() -> None:
    args = parse_args()
    metadata = load_cis_metadata()
    mle_idata = az.from_netcdf(MLE_NC)
    validate_observed_counts(metadata, mle_idata)

    x_one_hot = one_hot_flatten(metadata["aa_seq"])
    eval_target = point_target(mle_idata)
    train_idx, test_idx = train_test_split(
        np.arange(len(metadata)), test_size=args.test_size, random_state=args.random_seed
    )

    rows = [
        {
            "resample_source": "mle_point_estimate",
            "draw": np.nan,
            "target_variable": TARGET_VARIABLE,
            "spearman": fit_ridge(x_one_hot, eval_target, eval_target, train_idx, test_idx),
            "target_spearman_vs_eval": target_spearman(eval_target, eval_target, test_idx),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
    ]

    for source_name, idata in selected_sources(args, mle_idata):
        draws = posterior_target_draws(idata, source_name, args.n_resamples, args.random_seed)
        for draw in range(draws.sizes["sample"]):
            train_target = draws.isel(sample=draw).values
            rows.append(
                {
                    "resample_source": source_name,
                    "draw": draw,
                    "target_variable": TARGET_VARIABLE,
                    "spearman": fit_ridge(
                        x_one_hot, train_target, eval_target, train_idx, test_idx
                    ),
                    "target_spearman_vs_eval": target_spearman(train_target, eval_target, test_idx),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "correlations.csv", index=False)
    summarize(results).to_csv(args.output_dir / "summary.csv", index=False)

    print(f"TARGET_VARIABLE={TARGET_VARIABLE}")
    print(f"One-hot feature shape: {x_one_hot.shape}")
    print(f"Wrote {args.output_dir / 'correlations.csv'}")
    print(f"Wrote {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()