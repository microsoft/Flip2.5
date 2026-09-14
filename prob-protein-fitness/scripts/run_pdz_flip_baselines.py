"""Run FLIP2-style baselines on published and HMC-resampled PDZ3 targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pdz_trans_hmc_flip_replicates import (
    DEFAULT_FLIP_PATH,
    PROXY_COLUMNS,
    make_hmc_flip_replicates,
)

AAINDEX_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
DEFAULT_SCORE_PATH = ROOT / "data/zero_shot/pdz3_scores.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/pdz_trans_flip_baselines"
ZERO_SHOT_MODELS = ("esm2_650m", "carp_640m", "dayhoff")
PAPER_TEST_SPEARMAN = {
    "Ridge (one-hot)": 0.476,
    "Ridge (one-hot + likelihoods)": 0.478,
    "dayhoff zero-shot": -0.007,
    "esm2_650m zero-shot": 0.048,
    "carp_640m zero-shot": 0.043,
}


def one_hot_flatten(sequences: pd.Series) -> np.ndarray:
    """Flatten zero-padded one-hot encodings after removing the PDZ separator."""
    clean = sequences.astype(str).str.replace(":", "", regex=False).str.upper()
    max_length = clean.str.len().max()
    aa_to_index = {aa: index for index, aa in enumerate(AAINDEX_ALPHABET)}
    encoded = np.zeros((len(clean), max_length, len(AAINDEX_ALPHABET)), dtype=np.float32)
    for row, sequence in enumerate(clean):
        for column, aa in enumerate(sequence):
            if aa in aa_to_index:
                encoded[row, column, aa_to_index[aa]] = 1.0
    return encoded.reshape(len(clean), -1)


def write_zero_shot_template(path: Path, flip_path: Path = DEFAULT_FLIP_PATH) -> None:
    """Write an empty sequence-keyed cache template for missing pLM scores."""
    sequences = pd.read_csv(flip_path, usecols=["sequence"])
    for model in ZERO_SHOT_MODELS:
        sequences[model] = np.nan
        sequences[f"{model}_pdz"] = np.nan
        sequences[f"{model}_cript"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    sequences.to_csv(path, index=False)


def load_zero_shot_scores(path: Path, sequences: pd.Series) -> pd.DataFrame:
    """Load complete aggregate scores, averaging PDZ and CRIPT when needed."""
    if not path.exists():
        return pd.DataFrame({"sequence": sequences})
    cache = pd.read_csv(path)
    if "sequence" not in cache or cache["sequence"].duplicated().any():
        raise ValueError("Zero-shot cache must have one unique row per sequence.")
    scores = pd.DataFrame({"sequence": sequences}).merge(
        cache,
        on="sequence",
        how="left",
        validate="one_to_one",
    )
    available = ["sequence"]
    for model in ZERO_SHOT_MODELS:
        component_columns = [f"{model}_pdz", f"{model}_cript"]
        aggregate_complete = model in scores and scores[model].notna().all()
        components_complete = all(column in scores for column in component_columns) and scores[
            component_columns
        ].notna().all().all()
        if not aggregate_complete and components_complete:
            scores[model] = scores[component_columns].mean(axis=1)
        if model in scores and scores[model].notna().all():
            available.append(model)
    return scores[available]


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(target) & np.isfinite(prediction)
    return cast(float, spearmanr(target[mask], prediction[mask])[0])


def proxy_correlations(
    datasets: list[pd.DataFrame],
    proxy_columns: tuple[str, ...] = PROXY_COLUMNS,
    flip_path: Path = DEFAULT_FLIP_PATH,
) -> pd.DataFrame:
    """Correlate each proxy with the sequence-aligned reported FLIP fitness."""
    if not datasets:
        raise ValueError("At least one proxy dataset is required.")

    flip = pd.read_csv(flip_path, usecols=["sequence", "target"])
    if flip["sequence"].duplicated().any():
        raise ValueError("FLIP reported-fitness data must have unique sequences.")
    sequences = datasets[0]["sequence"].reset_index(drop=True)
    reported = pd.DataFrame({"sequence": sequences}).merge(
        flip.rename(columns={"target": "flip_reported_fitness"}),
        on="sequence",
        how="left",
        validate="one_to_one",
    )
    if reported["flip_reported_fitness"].isna().any():
        raise ValueError("Every proxy sequence must have reported fitness in the FLIP dataset.")
    reported_fitness = reported["flip_reported_fitness"].to_numpy(dtype=float)

    rows = []
    for replicate, dataset in enumerate(datasets):
        missing = [column for column in ("published_target", *proxy_columns) if column not in dataset]
        if missing:
            raise ValueError(f"Replicate {replicate} is missing proxy columns: {missing}")
        if not dataset["sequence"].reset_index(drop=True).equals(sequences):
            raise ValueError("Every proxy replicate must contain sequences in the same order.")
        if not np.allclose(
            dataset["published_target"].to_numpy(dtype=float),
            reported_fitness,
            equal_nan=True,
        ):
            raise ValueError(
                f"Replicate {replicate} published_target does not match the FLIP dataset."
            )
        for proxy in proxy_columns:
            values = dataset[proxy].to_numpy(dtype=float)
            finite = np.isfinite(reported_fitness) & np.isfinite(values)
            rows.append(
                {
                    "replicate": replicate,
                    "proxy": proxy,
                    "reference": "FLIP reported target",
                    "spearman": _spearman(reported_fitness, values),
                    "n_variants": int(finite.sum()),
                }
            )
    return pd.DataFrame(rows)


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Convert each row to average ranks on the interval [0, 1]."""
    if values.shape[1] < 2:
        raise ValueError("Rank-space analysis requires at least two variants.")
    ranks = rankdata(values, axis=1, method="average")
    return ((ranks - 1.0) / (values.shape[1] - 1.0)).astype(np.float32)


def _rowwise_correlation(rows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Calculate Pearson correlations between rows and one reference vector."""
    centered_rows = rows - rows.mean(axis=1, keepdims=True)
    centered_reference = reference - reference.mean()
    denominator = np.linalg.norm(centered_rows, axis=1) * np.linalg.norm(centered_reference)
    return np.divide(
        centered_rows @ centered_reference,
        denominator,
        out=np.full(rows.shape[0], np.nan),
        where=denominator > 0,
    )


def rank_space_compatibility(
    datasets: list[pd.DataFrame],
    proxy_columns: tuple[str, ...] = PROXY_COLUMNS,
    *,
    flip_path: Path = DEFAULT_FLIP_PATH,
    reference_fraction: float = 0.5,
    seed: int = 23,
    rank_sd_floor: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare FLIP ranks with a split-calibrated HMC rank distribution.

    Returns one summary row per proxy, one score row per held-out HMC draw, and
    one variant-diagnostic row per retained sequence and proxy. A fixed
    complete-case sequence set is selected separately for each proxy.
    """
    if len(datasets) < 4:
        raise ValueError("Rank-space analysis requires at least four HMC draws.")
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be between 0 and 1.")

    flip = pd.read_csv(flip_path, usecols=["sequence", "target", "set", "validation"])
    if flip["sequence"].duplicated().any():
        raise ValueError("FLIP reported-fitness data must have unique sequences.")
    sequences = datasets[0]["sequence"].reset_index(drop=True)
    reported = pd.DataFrame({"sequence": sequences}).merge(
        flip.rename(columns={"target": "flip_reported_fitness"}),
        on="sequence",
        how="left",
        validate="one_to_one",
    )
    if reported["flip_reported_fitness"].isna().any():
        raise ValueError("Every proxy sequence must have reported fitness in the FLIP dataset.")
    reported_fitness = reported["flip_reported_fitness"].to_numpy(dtype=float)

    n_draws = len(datasets)
    reference_count = int(round(n_draws * reference_fraction))
    if not 2 <= reference_count <= n_draws - 2:
        raise ValueError("The split must leave at least two reference and two calibration draws.")
    permutation = np.random.default_rng(seed).permutation(n_draws)
    reference_indices = permutation[:reference_count]
    calibration_indices = permutation[reference_count:]

    summary_rows = []
    score_frames = []
    diagnostic_frames = []
    for proxy in proxy_columns:
        values = np.empty((n_draws, len(sequences)), dtype=np.float32)
        for draw, dataset in enumerate(datasets):
            missing = [
                column for column in ("published_target", proxy) if column not in dataset
            ]
            if missing:
                raise ValueError(f"Replicate {draw} is missing proxy columns: {missing}")
            if not dataset["sequence"].reset_index(drop=True).equals(sequences):
                raise ValueError("Every proxy replicate must contain sequences in the same order.")
            if not np.allclose(
                dataset["published_target"].to_numpy(dtype=float),
                reported_fitness,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Replicate {draw} published_target does not match the FLIP dataset."
                )
            values[draw] = dataset[proxy].to_numpy(dtype=np.float32)

        complete = np.isfinite(reported_fitness) & np.isfinite(values).all(axis=0)
        n_variants = int(complete.sum())
        if n_variants < 2:
            raise ValueError(f"{proxy}: fewer than two variants are complete across all draws.")

        proxy_ranks = _normalized_ranks(values[:, complete])
        flip_ranks = _normalized_ranks(reported_fitness[None, complete])[0]
        reference_ranks = proxy_ranks[reference_indices]
        calibration_ranks = proxy_ranks[calibration_indices]
        mean_rank = reference_ranks.mean(axis=0, dtype=np.float64)
        rank_sd = reference_ranks.std(axis=0, ddof=1, dtype=np.float64)
        sd_floor = (
            1.0 / (n_variants - 1.0) if rank_sd_floor is None else float(rank_sd_floor)
        )
        if sd_floor <= 0:
            raise ValueError("rank_sd_floor must be positive.")
        scaled_sd = np.maximum(rank_sd, sd_floor)

        calibration_discrepancy = np.mean(
            ((calibration_ranks - mean_rank) / scaled_sd) ** 2,
            axis=1,
        )
        flip_residual = (flip_ranks - mean_rank) / scaled_sd
        flip_discrepancy = float(np.mean(flip_residual**2))

        consensus_ranks = _normalized_ranks(mean_rank[None, :])[0]
        calibration_consensus = _rowwise_correlation(calibration_ranks, consensus_ranks)
        flip_consensus = float(_rowwise_correlation(flip_ranks[None, :], consensus_ranks)[0])

        discrepancy_percentile = (
            1 + np.count_nonzero(calibration_discrepancy <= flip_discrepancy)
        ) / (len(calibration_indices) + 1)
        upper_tail_fraction = (
            1 + np.count_nonzero(calibration_discrepancy >= flip_discrepancy)
        ) / (len(calibration_indices) + 1)
        consensus_percentile = (
            1 + np.count_nonzero(calibration_consensus <= flip_consensus)
        ) / (len(calibration_indices) + 1)

        rank_intervals = np.quantile(
            reference_ranks,
            [0.025, 0.10, 0.25, 0.75, 0.90, 0.975],
            axis=0,
        )
        coverage_50 = np.mean((flip_ranks >= rank_intervals[2]) & (flip_ranks <= rank_intervals[3]))
        coverage_80 = np.mean((flip_ranks >= rank_intervals[1]) & (flip_ranks <= rank_intervals[4]))
        coverage_95 = np.mean((flip_ranks >= rank_intervals[0]) & (flip_ranks <= rank_intervals[5]))

        summary_rows.append(
            {
                "proxy": proxy,
                "n_draws": n_draws,
                "n_reference": len(reference_indices),
                "n_calibration": len(calibration_indices),
                "n_variants": n_variants,
                "n_excluded_incomplete": len(sequences) - n_variants,
                "rank_sd_floor": sd_floor,
                "n_sd_floored": int((rank_sd < sd_floor).sum()),
                "flip_consensus_spearman": flip_consensus,
                "consensus_spearman_percentile": consensus_percentile,
                "flip_standardized_discrepancy": flip_discrepancy,
                "discrepancy_percentile": discrepancy_percentile,
                "upper_tail_fraction": upper_tail_fraction,
                "coverage_50": coverage_50,
                "coverage_80": coverage_80,
                "coverage_95": coverage_95,
            }
        )
        score_frames.append(
            pd.DataFrame(
                {
                    "proxy": proxy,
                    "replicate": calibration_indices,
                    "standardized_discrepancy": calibration_discrepancy,
                    "consensus_spearman": calibration_consensus,
                }
            )
        )
        retained = reported.loc[complete].reset_index(drop=True)
        retained["proxy"] = proxy
        retained["flip_rank"] = flip_ranks
        retained["hmc_mean_rank"] = mean_rank
        retained["hmc_rank_sd"] = rank_sd
        retained["standardized_rank_residual"] = flip_residual
        retained["squared_contribution"] = flip_residual**2
        diagnostic_frames.append(retained)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(score_frames, ignore_index=True),
        pd.concat(diagnostic_frames, ignore_index=True),
    )


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    scale_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
) -> float:
    finite = np.isfinite(target)
    scale_mask = scale_mask & finite
    train_mask = train_mask & finite
    test_mask = test_mask & finite
    if not scale_mask.any() or not train_mask.any() or test_mask.sum() < 2:
        return np.nan
    scaler = StandardScaler()
    scaler.fit(target[scale_mask, None])
    train_target = scaler.transform(target[train_mask, None])
    test_target = scaler.transform(target[test_mask, None])
    model = Ridge(alpha=alpha, tol=1e-5, max_iter=1_000_000)
    model.fit(features[train_mask], train_target)
    return _spearman(test_target.ravel(), model.predict(features[test_mask]).ravel())


def benchmark_replicates(
    datasets: list[pd.DataFrame],
    *,
    target_column: str = "target",
    zero_shot_scores: pd.DataFrame | None = None,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Evaluate baselines on FLIP and one HMC-derived target proxy."""
    if not datasets:
        raise ValueError("At least one target dataset is required.")
    sequences = datasets[0]["sequence"].reset_index(drop=True)
    published_target = datasets[0]["published_target"].reset_index(drop=True)
    for dataset in datasets[1:]:
        if not dataset["published_target"].reset_index(drop=True).equals(published_target):
            raise ValueError("Published FLIP targets must be identical across replicates.")
    for replicate, dataset in enumerate(datasets):
        if target_column not in dataset:
            raise ValueError(f"Replicate {replicate} is missing target column {target_column!r}.")

    published_dataset = datasets[0].copy()
    published_dataset["target"] = published_target
    evaluation_datasets = [("published", pd.NA, published_dataset)] + [
        ("hmc", replicate, dataset) for replicate, dataset in enumerate(datasets)
    ]

    one_hot = one_hot_flatten(sequences)
    score_columns: list[str] = []
    score_matrix = None
    if zero_shot_scores is not None:
        aligned_scores = pd.DataFrame({"sequence": sequences}).merge(
            zero_shot_scores,
            on="sequence",
            how="left",
            validate="one_to_one",
        )
        score_columns = [column for column in ZERO_SHOT_MODELS if column in aligned_scores]
        if score_columns:
            score_matrix = aligned_scores[score_columns].to_numpy(dtype=float)

    rows = []
    for target_source, replicate, dataset in evaluation_datasets:
        if not dataset["sequence"].reset_index(drop=True).equals(sequences):
            raise ValueError("Every replicate must contain sequences in the same order.")
        target = dataset[
            "target" if target_source == "published" else target_column
        ].to_numpy(dtype=float)
        scale_mask = dataset["set"].eq("train")
        train_mask = dataset["set"].eq("train") & ~dataset["validation"].astype(bool)
        test_mask = dataset["set"].eq("test")
        finite = np.isfinite(target)
        rows.append(
            {
                "target_proxy": target_column,
                "target_source": target_source,
                "replicate": replicate,
                "model": "Ridge (one-hot)",
                "evaluation_set": "test",
                "n_train_finite": int((train_mask.to_numpy() & finite).sum()),
                "n_test_finite": int((test_mask.to_numpy() & finite).sum()),
                "spearman": _fit_ridge(
                    one_hot,
                    target,
                    scale_mask.to_numpy(),
                    train_mask.to_numpy(),
                    test_mask.to_numpy(),
                    alpha,
                ),
            }
        )
        if score_matrix is None:
            continue
        for column, prediction in zip(score_columns, score_matrix.T):
            for evaluation_set, mask in (("full", np.ones(len(dataset), dtype=bool)), ("test", test_mask)):
                rows.append(
                    {
                        "target_proxy": target_column,
                        "target_source": target_source,
                        "replicate": replicate,
                        "model": f"{column} zero-shot",
                        "evaluation_set": evaluation_set,
                        "spearman": _spearman(target[np.asarray(mask)], prediction[np.asarray(mask)]),
                    }
                )
        combined = np.hstack([one_hot, score_matrix])
        rows.append(
            {
                "target_proxy": target_column,
                "target_source": target_source,
                "replicate": replicate,
                "model": "Ridge (one-hot + likelihoods)",
                "evaluation_set": "test",
                "spearman": _fit_ridge(
                    combined,
                    target,
                    scale_mask.to_numpy(),
                    train_mask.to_numpy(),
                    test_mask.to_numpy(),
                    alpha,
                ),
            }
        )
    return pd.DataFrame(rows)


def flip2_random_splits(
    n_variants: int,
    *,
    min_train: int = 50,
    max_fraction: float = 0.8,
    n_training_sizes: int = 10,
    n_repeats: int = 5,
    seed: int = 23,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Reproduce FLIP2's log-spaced random train/test split schedule."""
    if not 1 <= min_train < n_variants:
        raise ValueError("min_train must be between 1 and n_variants - 1.")
    max_train = n_variants * max_fraction
    if not min_train < max_train < n_variants:
        raise ValueError("max_fraction must produce a train size below n_variants.")

    training_sizes = [
        int(value)
        for value in np.logspace(
            np.log10(min_train),
            np.log10(max_train),
            num=n_training_sizes,
        )
    ]
    rng = np.random.RandomState(seed)
    splits = []
    for iteration in range(n_training_sizes * n_repeats):
        n_train = training_sizes[iteration % n_training_sizes]
        indices = np.arange(n_variants)
        rng.shuffle(indices)
        splits.append(
            (
                iteration // n_training_sizes,
                n_train,
                indices[:n_train],
                indices[n_train:],
            )
        )
    return splits


def random_split_learning_curves(
    datasets: list[pd.DataFrame],
    *,
    target_column: str = "target",
    zero_shot_scores: pd.DataFrame | None = None,
    alpha: float = 10.0,
    min_train: int = 50,
    max_fraction: float = 0.8,
    n_training_sizes: int = 10,
    n_split_repeats: int = 5,
    split_seed: int = 23,
) -> pd.DataFrame:
    """Fit fixed random-split Ridge curves for one HMC-derived target proxy."""
    if not datasets:
        raise ValueError("At least one target dataset is required.")
    sequences = datasets[0]["sequence"].reset_index(drop=True)
    published_target = datasets[0]["published_target"].reset_index(drop=True)
    published_dataset = datasets[0].copy()
    published_dataset["target"] = published_target
    for replicate, dataset in enumerate(datasets):
        if target_column not in dataset:
            raise ValueError(f"Replicate {replicate} is missing target column {target_column!r}.")
    evaluation_datasets = [("published", pd.NA, published_dataset)] + [
        ("hmc", replicate, dataset) for replicate, dataset in enumerate(datasets)
    ]

    features = {"Ridge (one-hot)": one_hot_flatten(sequences)}
    if zero_shot_scores is not None:
        aligned_scores = pd.DataFrame({"sequence": sequences}).merge(
            zero_shot_scores,
            on="sequence",
            how="left",
            validate="one_to_one",
        )
        score_columns = [column for column in ZERO_SHOT_MODELS if column in aligned_scores]
        if score_columns:
            features["Ridge (one-hot + likelihoods)"] = np.hstack(
                [features["Ridge (one-hot)"], aligned_scores[score_columns].to_numpy(dtype=float)]
            )

    splits = flip2_random_splits(
        len(sequences),
        min_train=min_train,
        max_fraction=max_fraction,
        n_training_sizes=n_training_sizes,
        n_repeats=n_split_repeats,
        seed=split_seed,
    )
    rows = []
    for target_source, target_replicate, dataset in evaluation_datasets:
        if not dataset["sequence"].reset_index(drop=True).equals(sequences):
            raise ValueError("Every replicate must contain sequences in the same order.")
        target = dataset[
            "target" if target_source == "published" else target_column
        ].to_numpy(dtype=float)
        finite = np.isfinite(target)
        for split_repeat, n_train, train_indices, test_indices in splits:
            train_mask = np.zeros(len(dataset), dtype=bool)
            test_mask = np.zeros(len(dataset), dtype=bool)
            train_mask[train_indices] = True
            test_mask[test_indices] = True
            for model_name, model_features in features.items():
                rows.append(
                    {
                        "target_proxy": target_column,
                        "target_source": target_source,
                        "target_replicate": target_replicate,
                        "split_repeat": split_repeat,
                        "n_train": n_train,
                        "n_test": len(test_indices),
                        "n_train_finite": int((train_mask & finite).sum()),
                        "n_test_finite": int((test_mask & finite).sum()),
                        "model": model_name,
                        "spearman": _fit_ridge(
                            model_features,
                            target,
                            train_mask,
                            train_mask,
                            test_mask,
                            alpha,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_learning_curves(results: pd.DataFrame) -> pd.DataFrame:
    """Separate random-partition variation from HMC target uncertainty."""
    rows = []
    group_columns = ["target_source", "model", "n_train"]
    if "target_proxy" in results:
        group_columns.insert(0, "target_proxy")
    for keys, group in results.groupby(group_columns):
        if "target_proxy" in results:
            target_proxy, target_source, model, n_train = keys
        else:
            target_proxy = None
            target_source, model, n_train = keys
        if target_source == "hmc":
            values = group.groupby("target_replicate")["spearman"].mean()
            uncertainty_source = "HMC targets (split repeats averaged)"
        else:
            values = group["spearman"]
            uncertainty_source = "random partitions"
        row = {
                "target_source": target_source,
                "model": model,
                "n_train": n_train,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "lower": values.quantile(0.025),
                "upper": values.quantile(0.975),
                "uncertainty_source": uncertainty_source,
            }
        if target_proxy is not None:
            row["target_proxy"] = target_proxy
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_learning_curves_pooled(results: pd.DataFrame) -> pd.DataFrame:
    """Pool target draws and random partitions into one empirical distribution."""
    group_columns = ["target_source", "model", "n_train"]
    if "target_proxy" in results:
        group_columns.insert(0, "target_proxy")
    summary = (
        results.groupby(group_columns)["spearman"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            lower=lambda values: values.quantile(0.025),
            upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    summary["uncertainty_source"] = np.where(
        summary["target_source"].eq("hmc"),
        "HMC targets + random partitions (pooled)",
        "random partitions",
    )
    return summary


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Report the published reference separately from HMC uncertainty."""
    group_columns = ["target_source", "model", "evaluation_set"]
    if "target_proxy" in results:
        group_columns.insert(0, "target_proxy")
    summary = (
        results.groupby(group_columns)["spearman"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            lower=lambda values: values.quantile(0.025),
            upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    published_test = summary["target_source"].eq("published") & summary[
        "evaluation_set"
    ].eq("test")
    summary["paper_spearman"] = np.where(
        published_test,
        summary["model"].map(PAPER_TEST_SPEARMAN),
        np.nan,
    )
    summary["difference_from_paper"] = summary["mean"] - summary["paper_spearman"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--zero-shot-scores", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-zero-shot-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_zero_shot_template:
        write_zero_shot_template(args.zero_shot_scores)
        print(f"Wrote empty zero-shot score template to {args.zero_shot_scores}")
        return

    datasets = make_hmc_flip_replicates(args.n_resamples, seed=args.seed)
    scores = load_zero_shot_scores(args.zero_shot_scores, datasets[0]["sequence"])
    if len(scores.columns) == 1:
        print(f"No complete zero-shot scores found at {args.zero_shot_scores}; running Ridge only.")
    results = benchmark_replicates(datasets, zero_shot_scores=scores, alpha=args.alpha)
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "correlations.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()