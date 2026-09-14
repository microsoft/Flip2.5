"""Tsuboyama K50 source inputs and fitted sequence identity."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from flipv3.flip_dsets import load_k50_data

from ..hmc_scanning import file_signature
from ..identity import AnalysisIdentity
from ..score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    ScoreEnsemble,
)

K50_LIBRARIES = ("v1", "v2", "v3", "v4")
K50_PROTEASES = ("chymotrypsin", "trypsin")
K50_PRIMARY_SCORE_VARIABLE = "mean_dg"
K50_SECONDARY_SCORE_VARIABLE = "log_K50"
K50_EXPECTED_SCRAMBLED_COUNT = 64_238
K50_REPORTED_FITNESS_NAME = "dG_ML"
K50_REPORTED_FITNESS_LIMITS = (-1.0, 5.0)
K50_PREDICTIVE_SCORE_NAME = "mean_log_protease_enrichment"
K50_PREDICTIVE_SCORE_CONTRACT = "k50-mean-log-protease-enrichment-v1"
K50_PREDICTIVE_SCORE_NAMES = tuple(
    f"mean_log_{protease}_enrichment" for protease in K50_PROTEASES
)
K50_PREDICTIVE_SCORE_CONTRACTS = {
    score_name: f"k50-{protease}-mean-log-enrichment-v1"
    for protease, score_name in zip(
        K50_PROTEASES,
        K50_PREDICTIVE_SCORE_NAMES,
        strict=True,
    )
}
K50_REPORTED_FITNESS_COLUMNS = (
    "variant_id",
    "source_aa_seq_full",
    "reported_fitness_raw",
    "reported_fitness_95ci_width",
    "reported_fitness_ml",
    "reported_wt_name",
    "reported_wt_cluster",
)


@dataclass(frozen=True)
class K50Library:
    """One K50 synthesis library in its fitted column order."""

    name: str
    global_sequence_indices: np.ndarray
    variants: tuple[str, ...]
    counts_c0: np.ndarray
    counts_cg0: np.ndarray

    @property
    def n_variants(self) -> int:
        return len(self.variants)


@dataclass(frozen=True)
class K50Landscape:
    """All four K50 libraries and shared calibration data."""

    variants: pd.DataFrame
    libraries: dict[str, K50Library]
    ordinal_sequences: np.ndarray
    scrambled_indices: np.ndarray
    qpcr_log_protease_concentration: np.ndarray
    qpcr_log2_survival: np.ndarray
    log_expected_protease_concentration: np.ndarray

    @property
    def n_fitted_variants(self) -> int:
        return len(self.variants)


def _validate_counts(
    library: str,
    sequence_indices: np.ndarray,
    counts_c0: np.ndarray,
    counts_cg0: np.ndarray,
) -> None:
    expected_variants = len(sequence_indices)
    if counts_c0.shape != (2, 2, expected_variants):
        raise ValueError(
            f"{library}: untreated counts must have shape "
            f"(protease=2, replicate=2, variant={expected_variants})."
        )
    if counts_cg0.shape != (2, 2, 11, expected_variants):
        raise ValueError(
            f"{library}: treated counts must have shape "
            f"(protease=2, replicate=2, concentration=11, variant={expected_variants})."
        )
    if np.any(counts_c0 < 0) or np.any(counts_cg0 < 0):
        raise ValueError(f"{library}: count arrays cannot contain negative values.")


def k50_mean_log_enrichment_by_protease(
    counts_c0: np.ndarray,
    counts_cg0: np.ndarray,
) -> np.ndarray:
    """Return smoothed log enrichment averaged within each protease."""
    untreated = np.asarray(counts_c0)
    treated = np.asarray(counts_cg0)
    if untreated.ndim < 3 or treated.ndim != untreated.ndim + 1:
        raise ValueError(
            "K50 counts must be (..., protease, replicate, variant) and "
            "(..., protease, replicate, concentration, variant)."
        )
    if treated.shape[:-4] != untreated.shape[:-3]:
        raise ValueError("K50 untreated and treated sample axes must match.")
    if treated.shape[-4:-2] != untreated.shape[-3:-1]:
        raise ValueError("K50 untreated and treated protease/replicate axes must match.")
    if treated.shape[-1] != untreated.shape[-1]:
        raise ValueError("K50 untreated and treated variant axes must match.")
    if untreated.shape[-3:-1] != (2, 2) or treated.shape[-2] != 11:
        raise ValueError(
            "K50 enrichment requires two proteases, two replicates, and "
            "11 treated concentrations."
        )
    if (
        not np.issubdtype(untreated.dtype, np.number)
        or not np.issubdtype(treated.dtype, np.number)
        or not np.isfinite(untreated).all()
        or not np.isfinite(treated).all()
        or np.any(untreated < 0)
        or np.any(treated < 0)
    ):
        raise ValueError("K50 counts must be finite, numeric, and nonnegative.")

    n_variants = untreated.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_untreated_frequency = np.log(untreated + 0.5) - np.log(
            untreated.sum(axis=-1, keepdims=True, dtype=float) + 0.5 * n_variants
        )
        log_treated_frequency = np.log(treated + 0.5) - np.log(
            treated.sum(axis=-1, keepdims=True, dtype=float) + 0.5 * n_variants
        )
    enrichment = (
        log_treated_frequency - log_untreated_frequency[..., None, :]
    )
    return enrichment.mean(axis=(-3, -2))


def k50_mean_log_protease_enrichment(
    counts_c0: np.ndarray,
    counts_cg0: np.ndarray,
) -> np.ndarray:
    """Return the mean smoothed log enrichment for one K50 library."""
    return k50_mean_log_enrichment_by_protease(counts_c0, counts_cg0).mean(
        axis=-2
    )


def _load_reported_fitness(path: Path) -> pd.DataFrame:
    reported = pd.read_csv(
        path,
        dtype={
            "variant_id": "string",
            "source_aa_seq_full": "string",
            "reported_fitness_ml": "string",
            "reported_wt_name": "string",
            "reported_wt_cluster": "string",
        },
        low_memory=False,
    )
    if tuple(reported.columns) != K50_REPORTED_FITNESS_COLUMNS:
        raise ValueError("K50 reported-fitness projection has an unexpected schema.")
    required = [
        "variant_id",
        "reported_fitness_raw",
        "reported_fitness_95ci_width",
        "reported_fitness_ml",
    ]
    if reported[required].isna().any(axis=None):
        raise ValueError("K50 reported-fitness projection contains missing required values.")
    if reported.empty:
        raise ValueError("K50 reported-fitness projection must contain at least one row.")

    raw = pd.to_numeric(reported["reported_fitness_raw"], errors="raise")
    interval_width = pd.to_numeric(
        reported["reported_fitness_95ci_width"], errors="raise"
    )
    if not np.isfinite(raw).all():
        raise ValueError("K50 raw reported fitness must be finite.")
    if not np.isfinite(interval_width).all() or (interval_width < 0).any():
        raise ValueError("K50 reported-fitness interval widths must be finite and nonnegative.")

    ml = reported["reported_fitness_ml"]
    parsed_ml = pd.to_numeric(ml, errors="coerce")
    special = ml.isin({"-", "<-1", ">5"})
    if (parsed_ml.isna() & ~special).any():
        raise ValueError("K50 dG_ML contains an unsupported value.")
    lower_limit, upper_limit = K50_REPORTED_FITNESS_LIMITS
    published = parsed_ml.clip(lower=lower_limit, upper=upper_limit)
    published = published.mask(ml.eq("<-1"), lower_limit)
    published = published.mask(ml.eq(">5"), upper_limit)
    status = np.select(
        [ml.eq("-"), ml.eq("<-1"), ml.eq(">5")],
        ["unavailable", "below_lower_limit", "above_upper_limit"],
        default="measured",
    )

    normalized = reported.assign(
        reported_fitness=published,
        reported_fitness_raw=raw,
        reported_fitness_95ci_width=interval_width,
        reported_fitness_status=status,
        reported_fitness_is_censored=np.asarray(status) != "measured",
    )
    eligible = normalized.loc[normalized["reported_fitness_status"] != "unavailable"]
    grouped = eligible.groupby("variant_id", sort=False, dropna=False)
    aggregated = grouped.agg(
        reported_fitness=("reported_fitness", "median"),
        reported_fitness_raw=("reported_fitness_raw", "median"),
        reported_fitness_95ci_width=("reported_fitness_95ci_width", "median"),
        reported_measurements=("reported_fitness", "size"),
        reported_fitness_is_censored=("reported_fitness_is_censored", "any"),
        reported_wt_name_count=("reported_wt_name", "nunique"),
        reported_wt_cluster_count=("reported_wt_cluster", "nunique"),
    )
    first_status = grouped["reported_fitness_status"].first()
    status_count = grouped["reported_fitness_status"].nunique()
    aggregated["reported_fitness_status"] = first_status.mask(
        status_count > 1,
        "mixed",
    )
    for column in (
        "reported_fitness",
        "reported_fitness_raw",
        "reported_fitness_95ci_width",
    ):
        aggregated[column] = aggregated[column].astype(float)
    aggregated["reported_measurements"] = aggregated[
        "reported_measurements"
    ].astype(np.int32)
    return aggregated.reset_index()


def load_k50_landscape(
    data_dir: str | Path,
    *,
    reported_fitness_path: str | Path,
    expected_scrambled_count: int | None = K50_EXPECTED_SCRAMBLED_COUNT,
) -> K50Landscape:
    """Load K50 inputs without changing the historical loader's axis order."""
    dataset = load_k50_data(str(data_dir))
    sequence_to_index = {
        str(sequence): int(index) for sequence, index in dataset["seq_ids"].items()
    }
    if len(set(sequence_to_index.values())) != len(sequence_to_index):
        raise ValueError("K50 global sequence indices must be unique.")
    expected_indices = set(range(len(sequence_to_index)))
    if set(sequence_to_index.values()) != expected_indices:
        raise ValueError("K50 global sequence indices must be contiguous from zero.")

    ordered_sequences = [""] * len(sequence_to_index)
    for sequence, index in sequence_to_index.items():
        if not sequence or sequence == "-":
            raise ValueError("K50 fitted amino-acid sequences cannot be empty or dummy values.")
        ordered_sequences[index] = sequence

    ordinal_sequences = np.asarray(dataset["ordinal_seqs"], dtype=np.int64)
    if ordinal_sequences.shape != (len(ordered_sequences), 86):
        raise ValueError("K50 ordinal sequences must have shape (global_variant, 86).")

    scrambled_indices = np.asarray(dataset["scrambled_inds"], dtype=np.int64)
    if scrambled_indices.ndim != 1 or len(np.unique(scrambled_indices)) != len(
        scrambled_indices
    ):
        raise ValueError("K50 scrambled sequence indices must be a unique vector.")
    if expected_scrambled_count is not None and len(scrambled_indices) != expected_scrambled_count:
        raise ValueError(
            f"Expected {expected_scrambled_count:,} scrambled sequences, "
            f"found {len(scrambled_indices):,}."
        )
    if np.any(scrambled_indices < 0) or np.any(scrambled_indices >= len(ordered_sequences)):
        raise ValueError("K50 scrambled sequence indices are outside the global variant axis.")

    libraries = {}
    for library in K50_LIBRARIES:
        sequence_indices = np.asarray(dataset[f"{library}_seqids"], dtype=np.int64)
        if sequence_indices.ndim != 1:
            raise ValueError(f"{library}: sequence indices must be one-dimensional.")
        if np.any(sequence_indices < 0) or np.any(sequence_indices >= len(ordered_sequences)):
            raise ValueError(f"{library}: sequence indices are outside the global variant axis.")
        counts_c0 = np.asarray(dataset[f"{library}_counts_c0"], dtype=np.int64)
        counts_cg0 = np.asarray(dataset[f"{library}_counts_cg0"], dtype=np.int64)
        _validate_counts(library, sequence_indices, counts_c0, counts_cg0)
        libraries[library] = K50Library(
            name=library,
            global_sequence_indices=sequence_indices,
            variants=tuple(ordered_sequences[index] for index in sequence_indices),
            counts_c0=counts_c0,
            counts_cg0=counts_cg0,
        )

    qpcr_log_protease_concentration = np.asarray(
        dataset["qpcr_log_protease_conc"], dtype=float
    )
    qpcr_log2_survival = np.asarray(dataset["qpcr_log2_survival"], dtype=float)
    log_expected_protease_concentration = np.asarray(
        dataset["log_expected_protease_conc"], dtype=float
    )
    if qpcr_log_protease_concentration.shape != (11,):
        raise ValueError("K50 qPCR protease concentrations must have shape (11,).")
    if qpcr_log2_survival.shape != (2, 11, 8):
        raise ValueError(
            "K50 qPCR survival must have shape "
            "(protease=2, concentration=11, protein=8)."
        )
    if log_expected_protease_concentration.shape != (2, 11):
        raise ValueError(
            "K50 expected protease concentrations must have shape (replicate=2, concentration=11)."
        )

    variants = pd.DataFrame(
        {
            "variant_id": ordered_sequences,
            "global_sequence_index": np.arange(len(ordered_sequences), dtype=np.int32),
            "is_scrambled": np.isin(np.arange(len(ordered_sequences)), scrambled_indices),
            "reference_identity": "domain-specific; no global wild type",
        }
    )
    variants["is_source_control"] = variants["is_scrambled"]
    variants["is_ranking_candidate"] = ~variants["is_scrambled"]
    reported = _load_reported_fitness(Path(reported_fitness_path))
    unknown = reported.loc[
        ~reported["variant_id"].isin(sequence_to_index),
        "variant_id",
    ]
    if len(unknown):
        raise ValueError(
            f"K50 reported fitness contains {len(unknown):,} sequences outside "
            f"the fitted landscape; examples: {unknown.head().tolist()}"
        )
    variants = variants.merge(
        reported,
        on="variant_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    variants["reported_fitness_name"] = K50_REPORTED_FITNESS_NAME
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )
    variants["reported_rank_min"] = variants["reported_fitness"].rank(
        method="min",
        ascending=False,
    )
    variants["reported_rank_max"] = variants["reported_fitness"].rank(
        method="max",
        ascending=False,
    )
    variants["reference_fitness"] = variants["reported_fitness"]
    variants["reported_fitness_source"] = np.where(
        variants["reported_fitness"].notna(),
        "Tsuboyama et al. 2023 quality-filtered dG_ML",
        "unavailable",
    )
    variants["reference_fitness_source"] = variants["reported_fitness_source"]

    return K50Landscape(
        variants=variants,
        libraries=libraries,
        ordinal_sequences=ordinal_sequences,
        scrambled_indices=scrambled_indices,
        qpcr_log_protease_concentration=qpcr_log_protease_concentration,
        qpcr_log2_survival=qpcr_log2_survival,
        log_expected_protease_concentration=log_expected_protease_concentration,
    )


def _global_observed_enrichment(landscape: K50Landscape) -> tuple[np.ndarray, np.ndarray]:
    score_sum = np.zeros((len(K50_PROTEASES), landscape.n_fitted_variants), dtype=float)
    occurrences = np.zeros(landscape.n_fitted_variants, dtype=np.int8)
    for library in landscape.libraries.values():
        score = k50_mean_log_enrichment_by_protease(
            library.counts_c0,
            library.counts_cg0,
        )
        score_sum[:, library.global_sequence_indices] += score
        occurrences[library.global_sequence_indices] += 1
    if np.any(occurrences == 0):
        raise ValueError("Every global K50 sequence must occur in a source library.")
    return score_sum / occurrences[None], occurrences


def build_k50_mle_score_ensemble(
    landscape: K50Landscape,
    mle_path: str | Path,
    *,
    fit_id: str | None = None,
    job_id: int | None = None,
    scratch_dir: str | Path | None = None,
    variant_chunk_size: int | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Build the K50 enrichment sensitivity ensemble from sampled-MLE counts."""
    resolved_path = Path(mle_path)
    identity = AnalysisIdentity(
        protein_family="k50",
        landscape="k50",
        inference_method="mle_predictive",
        model_spec="(MASSIVE, only MLE)",
        fit_id=fit_id or resolved_path.stem,
    )
    variants = landscape.variants.copy()
    if not np.array_equal(
        variants["global_sequence_index"].to_numpy(dtype=np.int64),
        np.arange(landscape.n_fitted_variants),
    ):
        raise ValueError("K50 global sequence indices must match the proxy axis.")
    variants["hmc_variant_index"] = variants["global_sequence_index"].astype(
        np.int32
    )
    candidate_mask = variants["is_ranking_candidate"].to_numpy(dtype=bool)
    if not np.array_equal(candidate_mask, ~variants["is_scrambled"].to_numpy()):
        raise ValueError("K50 ranking candidates must exclude exactly the scrambles.")

    observed_proxy, occurrences = _global_observed_enrichment(landscape)
    reported = variants["reported_fitness"].to_numpy(dtype=float)
    agreement_mask = candidate_mask & np.isfinite(reported)
    if agreement_mask.sum() < 2:
        raise ValueError("K50 sensitivity validation requires two reported candidates.")
    reported_values = reported[agreement_mask]
    observed_agreement = {}
    for score_name, observed_score in zip(
        K50_PREDICTIVE_SCORE_NAMES,
        observed_proxy,
        strict=True,
    ):
        observed_reported = observed_score[agreement_mask]
        pearson = float(np.corrcoef(observed_reported, reported_values)[0, 1])
        spearman = float(
            pd.Series(observed_reported).corr(
                pd.Series(reported_values),
                method="spearman",
            )
        )
        if not np.isfinite(spearman) or spearman <= 0:
            raise ValueError(
                f"K50 score {score_name!r} has the wrong direction relative "
                "to dG_ML."
            )
        observed_agreement[score_name] = {
            "source_column": K50_REPORTED_FITNESS_NAME,
            "n_compared": int(agreement_mask.sum()),
            "pearson": pearson,
            "spearman": spearman,
        }

    temporary = tempfile.NamedTemporaryFile(
        prefix="k50_mle_proxy_",
        suffix=".dat",
        dir=scratch_dir,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    sampled_proxy: np.memmap | None = None
    started = time.monotonic()
    try:
        with h5py.File(resolved_path, "r") as handle:
            observed_group = handle.get("observed_data")
            predictive_group = handle.get("posterior_predictive")
            if not isinstance(observed_group, h5py.Group):
                raise ValueError("The sampled-MLE artifact must embed K50 counts.")
            if not isinstance(predictive_group, h5py.Group):
                raise ValueError(
                    "The sampled-MLE artifact is missing posterior_predictive."
                )

            n_samples: int | None = None
            stored_chunk_sizes = []
            for library_name, library in landscape.libraries.items():
                untreated_name = f"{library_name}_counts_c0"
                treated_name = f"{library_name}_counts_cg0"
                observed_untreated = observed_group.get(untreated_name)
                observed_treated = observed_group.get(treated_name)
                predictive_untreated = predictive_group.get(untreated_name)
                predictive_treated = predictive_group.get(treated_name)
                if not all(
                    isinstance(dataset, h5py.Dataset)
                    for dataset in (
                        observed_untreated,
                        observed_treated,
                        predictive_untreated,
                        predictive_treated,
                    )
                ):
                    raise ValueError(
                        f"K50 artifact is missing count variables for {library_name}."
                    )
                if not np.array_equal(
                    observed_untreated[:],
                    library.counts_c0,
                ) or not np.array_equal(
                    observed_treated[:],
                    library.counts_cg0,
                ):
                    raise ValueError(
                        f"{library_name}: embedded K50 counts differ from raw data."
                    )
                if predictive_untreated.ndim != 5 or predictive_untreated.shape[0] != 1:
                    raise ValueError(
                        f"{library_name}: predictive untreated counts have wrong axes."
                    )
                library_samples = int(predictive_untreated.shape[1])
                if n_samples is None:
                    n_samples = library_samples
                elif library_samples != n_samples:
                    raise ValueError("K50 libraries have different predictive draw counts.")
                expected_untreated_shape = (
                    1,
                    library_samples,
                    2,
                    2,
                    library.n_variants,
                )
                expected_treated_shape = (
                    1,
                    library_samples,
                    2,
                    2,
                    11,
                    library.n_variants,
                )
                if predictive_untreated.shape != expected_untreated_shape:
                    raise ValueError(
                        f"{library_name}: predictive untreated shape mismatch."
                    )
                if predictive_treated.shape != expected_treated_shape:
                    raise ValueError(
                        f"{library_name}: predictive treated shape mismatch."
                    )
                for dataset in (predictive_untreated, predictive_treated):
                    if dataset.chunks is not None:
                        stored_chunk_sizes.append(dataset.chunks[-1])

            if n_samples is None or n_samples < 2:
                raise ValueError("K50 proxy analysis requires at least two MLE draws.")
            if variant_chunk_size is None:
                variant_chunk_size = min(
                    stored_chunk_sizes,
                    default=max(
                        library.n_variants
                        for library in landscape.libraries.values()
                    ),
                )
            if variant_chunk_size < 1:
                raise ValueError("variant_chunk_size must be positive.")

            sampled_proxy = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    len(K50_PREDICTIVE_SCORE_NAMES),
                    n_samples,
                    landscape.n_fitted_variants,
                ),
            )
            sampled_proxy[:] = 0
            total_blocks = sum(
                (library.n_variants + variant_chunk_size - 1)
                // variant_chunk_size
                for library in landscape.libraries.values()
            )
            completed_blocks = 0
            if show_progress:
                print(
                    f"K50: {n_samples:,} MLE predictive draws, "
                    f"{landscape.n_fitted_variants:,} global variants, "
                    f"{total_blocks} library chunks",
                    flush=True,
                )

            for library_name, library in landscape.libraries.items():
                untreated_object = predictive_group[f"{library_name}_counts_c0"]
                treated_object = predictive_group[f"{library_name}_counts_cg0"]
                expected_untreated_totals = library.counts_c0.sum(
                    axis=-1,
                    dtype=np.int64,
                )
                expected_treated_totals = library.counts_cg0.sum(
                    axis=-1,
                    dtype=np.int64,
                )
                untreated_totals = np.zeros(
                    (n_samples, 2, 2),
                    dtype=np.int64,
                )
                treated_totals = np.zeros(
                    (n_samples, 2, 2, 11),
                    dtype=np.int64,
                )
                untreated_log_denominator = np.log(
                    expected_untreated_totals + 0.5 * library.n_variants
                ).astype(np.float32)
                treated_log_denominator = np.log(
                    expected_treated_totals + 0.5 * library.n_variants
                ).astype(np.float32)

                for variant_start in range(
                    0,
                    library.n_variants,
                    variant_chunk_size,
                ):
                    variant_stop = min(
                        variant_start + variant_chunk_size,
                        library.n_variants,
                    )
                    untreated = np.asarray(
                        untreated_object[
                            0,
                            :,
                            :,
                            :,
                            variant_start:variant_stop,
                        ]
                    )
                    untreated_totals += untreated.sum(axis=-1, dtype=np.int64)
                    log_untreated = untreated.astype(np.float32)
                    log_untreated += 0.5
                    np.log(log_untreated, out=log_untreated)
                    log_untreated -= untreated_log_denominator[None, :, :, None]
                    score_sum = np.zeros(
                        (
                            len(K50_PREDICTIVE_SCORE_NAMES),
                            n_samples,
                            variant_stop - variant_start,
                        ),
                        dtype=np.float32,
                    )

                    for protease in range(2):
                        for replicate in range(2):
                            for concentration in range(11):
                                treated = np.asarray(
                                    treated_object[
                                        0,
                                        :,
                                        protease,
                                        replicate,
                                        concentration,
                                        variant_start:variant_stop,
                                    ]
                                )
                                treated_totals[
                                    :,
                                    protease,
                                    replicate,
                                    concentration,
                                ] += treated.sum(axis=-1, dtype=np.int64)
                                log_treated = treated.astype(np.float32)
                                log_treated += 0.5
                                np.log(log_treated, out=log_treated)
                                log_treated -= treated_log_denominator[
                                    protease,
                                    replicate,
                                    concentration,
                                ]
                                log_treated -= log_untreated[
                                    :,
                                    protease,
                                    replicate,
                                    :,
                                ]
                                score_sum[protease] += log_treated
                    score_sum /= 22
                    global_indices = library.global_sequence_indices[
                        variant_start:variant_stop
                    ]
                    existing = sampled_proxy[:, :, global_indices]
                    existing += score_sum
                    sampled_proxy[:, :, global_indices] = existing
                    completed_blocks += 1
                    if show_progress:
                        print(
                            f"  {library_name} chunks: {completed_blocks}/{total_blocks} "
                            f"({(time.monotonic() - started) / 60:.1f} min)",
                            flush=True,
                        )

                if not np.all(
                    untreated_totals == expected_untreated_totals[None, :, :]
                ):
                    raise ValueError(
                        f"{library_name}: predictive untreated totals differ "
                        "from observed data."
                    )
                if not np.all(
                    treated_totals == expected_treated_totals[None, :, :, :]
                ):
                    raise ValueError(
                        f"{library_name}: predictive treated totals differ "
                        "from observed data."
                    )

        finite_proxy_values = 0
        finite_per_score_sample = np.zeros(
            (len(K50_PREDICTIVE_SCORE_NAMES), n_samples),
            dtype=np.int64,
        )
        finite_candidates_per_score_sample = np.zeros_like(finite_per_score_sample)
        global_chunk_size = 4_096
        for variant_start in range(
            0,
            landscape.n_fitted_variants,
            global_chunk_size,
        ):
            variant_stop = min(
                variant_start + global_chunk_size,
                landscape.n_fitted_variants,
            )
            block = sampled_proxy[:, :, variant_start:variant_stop]
            block /= occurrences[None, None, variant_start:variant_stop]
            finite = np.isfinite(block)
            finite_proxy_values += int(finite.sum())
            finite_per_score_sample += finite.sum(axis=-1)
            local_candidates = candidate_mask[variant_start:variant_stop]
            finite_candidates_per_score_sample += finite[
                :, :, local_candidates
            ].sum(axis=-1)
        sampled_proxy.flush()
    except Exception:
        if sampled_proxy is not None:
            sampled_proxy.flush()
            del sampled_proxy
        temporary_path.unlink(missing_ok=True)
        raise

    score_names = list(K50_PREDICTIVE_SCORE_NAMES)
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                sampled_proxy,
            ),
            "observed_score": (
                ("score_name", "variant"),
                observed_proxy.astype(np.float32),
            ),
        },
        coords={
            "score_name": score_names,
            "sample": np.arange(n_samples, dtype=np.int32),
            "variant": np.arange(landscape.n_fitted_variants, dtype=np.int32),
        },
    )
    metadata: dict[str, object] = {
        **identity.metadata(),
        "fit_method": "mle",
        "artifact_kind": "sampled_mle",
        "job_id": job_id,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "scores": score_names,
        "score_contracts": dict(K50_PREDICTIVE_SCORE_CONTRACTS),
        "score_status": {score_name: "sensitivity" for score_name in score_names},
        "higher_is_better": {score_name: True for score_name in score_names},
        "reference_score": K50_REPORTED_FITNESS_NAME,
        "reference_score_status": "deferred",
        "reference_top_k_identifiable": False,
        "reference_top_k_tie_size": 41_517,
        "n_samples": n_samples,
        "n_variants": landscape.n_fitted_variants,
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_source_controls": int((~candidate_mask).sum()),
        "n_reported_variants": int(agreement_mask.sum()),
        "variant_chunk_size": variant_chunk_size,
        "finite_proxy_values": finite_proxy_values,
        "finite_ranking_candidates_per_sample": (
            finite_candidates_per_score_sample.min(axis=0).astype(int).tolist()
        ),
        "min_finite_variants_per_sample": int(finite_per_score_sample.min()),
        "max_finite_variants_per_sample": int(finite_per_score_sample.max()),
        "pseudocount": 0.5,
        "within_library_aggregation": (
            "separate by protease; equal mean over 2 replicates and 11 concentrations"
        ),
        "cross_library_aggregation": "equal mean over source-library occurrences",
        "library_occurrence_counts": {
            str(count): int((occurrences == count).sum())
            for count in np.unique(occurrences)
        },
        "count_axes": {
            "counts_c0": [
                "chain",
                "draw",
                "protease",
                "replicate",
                "library_variant",
            ],
            "counts_cg0": [
                "chain",
                "draw",
                "protease",
                "replicate",
                "concentration",
                "library_variant",
            ],
        },
        "observed_counts_status": "matched",
        "observed_agreement": observed_agreement,
        "source_artifact": file_signature(resolved_path),
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [
            "These enrichment sensitivity scores are not fitted K50 or published "
            "dG_ML scores.",
            "Reference top-K overlap is omitted because 41,517 variants share "
            "the censored upper limit.",
        ],
    }
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
        temporary_paths=(temporary_path,),
    )
