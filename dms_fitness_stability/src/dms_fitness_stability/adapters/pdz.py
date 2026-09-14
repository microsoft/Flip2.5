"""PDZ3-CRIPT landscape inputs and shared full-fit stability scanning."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import h5netcdf
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from flipv3.flip_dsets import load_pdz_dataset

from ..analysis import RESULT_SCHEMA_VERSION, _top_k_values
from ..hmc_score_extraction import build_hmc_score_ensemble as _build_hmc_scores
from ..hmc_score_extraction import transform_hmc_score
from ..identity import AnalysisIdentity
from ..inputs import prepare_counts
from ..results import StabilityResult
from ..score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    HMC_COUNT_BASED_TARGET,
    ScoreEnsemble,
)
from ..stability import pairwise_overlap, reference_overlap, summarize_inclusion
from ..streaming import (
    accumulate_tracked_ranks,
    empty_top_k,
    empty_tracked_ranks,
    merge_top_k,
    select_reference_top_k,
)

PDZ_TRANS_LANDSCAPES = ("trans-1", "trans-2")
PDZ_CIS_SOURCE = "cis"
PDZ_CRIPT_LANDSCAPES = ("cript-c", "cript-n")
PDZ_LANDSCAPES = (*PDZ_TRANS_LANDSCAPES, PDZ_CIS_SOURCE, *PDZ_CRIPT_LANDSCAPES)
PDZ_SOURCES = PDZ_TRANS_LANDSCAPES
PDZ_LIBRARY_PROVENANCE = {
    "trans-1": {
        "geo_library": "pdz3cript_b1",
        "variant_molecule": "PDZ3-CRIPT",
        "reported_fitness_file": "GSE265816_pdz3cript_sub.csv.gz",
    },
    "trans-2": {
        "geo_library": "pdz3cript_b2",
        "variant_molecule": "PDZ3-CRIPT",
        "reported_fitness_file": "GSE265816_pdz3cript_sub.csv.gz",
    },
    "cis": {
        "geo_library": "cript2xa3_cis",
        "variant_molecule": "CRIPT",
        "reported_fitness_file": "GSE265816_cript2xa3_sub.csv.gz",
    },
    "cript-c": {
        "geo_library": "cript_c",
        "variant_molecule": "CRIPT",
        "reported_fitness_file": "GSE265816_cript_c_sub.csv.gz",
    },
    "cript-n": {
        "geo_library": "cript_n",
        "variant_molecule": "CRIPT",
        "reported_fitness_file": "GSE265816_cript_n_sub.csv.gz",
    },
}
PDZ_SCORE_NAMES = ("log_mean_growth", "count_ratio", "dimsum_fitness")
PDZ_PREDICTIVE_SCORE_NAME = "dimsum_fitness"
PDZ_PREDICTIVE_SCORE_CONTRACT = "pdz-dimsum-fitness-v1"
PDZGrowthRatePrior = Literal["exponential", "gamma", "lomax"]
PDZGrowthCurve = Literal["exponential", "sigmoid"]
PDZ3_WT_SEQUENCE = (
    "GEEDIPREPRRIVIHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAA"
    "IALKNAGQTVTIIAQYKPEEYSRFEA"
)
CRIPT_WT_SEQUENCE = "KNYKQTSV"
PDZ_LIBRARY_SPLIT = 50


def _format_full_sequence(sequence: str) -> str:
    expected_length = len(PDZ3_WT_SEQUENCE) + len(CRIPT_WT_SEQUENCE)
    if len(sequence) != expected_length:
        raise ValueError(
            f"Expected a {expected_length}-residue PDZ3-CRIPT sequence, "
            f"got {len(sequence)} residues."
        )
    return f"{sequence[: len(PDZ3_WT_SEQUENCE)]}:{sequence[len(PDZ3_WT_SEQUENCE) :]}"


def _canonicalize_sequence(sequence: str) -> str:
    if ":" not in sequence:
        sequence = _format_full_sequence(sequence)
    pdz, cript = sequence.split(":", maxsplit=1)
    return f"{pdz.split('*', maxsplit=1)[0]}:{cript.split('*', maxsplit=1)[0]}"


def pdz_variant_label(sequence: str) -> str:
    """Format a canonical PDZ3:CRIPT sequence as amino-acid changes."""
    if ":" not in sequence:
        raise ValueError("PDZ variant identifiers must contain a ':' separator.")
    pdz, cript = sequence.split(":", maxsplit=1)
    changes = []
    for position, (reference, observed) in enumerate(
        zip(PDZ3_WT_SEQUENCE, pdz),
        start=1,
    ):
        if observed != reference:
            changes.append(f"PDZ {reference}{position}{observed}")
    if len(pdz) < len(PDZ3_WT_SEQUENCE):
        changes.append(f"PDZ stop@{len(pdz) + 1}")
    for position, (reference, observed) in enumerate(
        zip(CRIPT_WT_SEQUENCE, cript),
        start=1,
    ):
        if observed != reference:
            changes.append(f"CRIPT {reference}{position}{observed}")
    if len(cript) < len(CRIPT_WT_SEQUENCE):
        changes.append(f"CRIPT stop@{len(cript) + 1}")
    return ", ".join(changes) if changes else "WT"


def pdz_cis_variant_label(sequence: str) -> str:
    """Format a CRIPT sequence as amino-acid changes from wild type."""
    if len(sequence) != len(CRIPT_WT_SEQUENCE):
        raise ValueError(
            f"Expected an {len(CRIPT_WT_SEQUENCE)}-residue CRIPT sequence, "
            f"got {len(sequence)} residues."
        )
    changes = [
        f"CRIPT {reference}{position}{observed}"
        for position, (reference, observed) in enumerate(
            zip(CRIPT_WT_SEQUENCE, sequence),
            start=1,
        )
        if observed != reference
    ]
    return ", ".join(changes) if changes else "WT"


def _reconstruct_sequences(raw_variants: list[str], source: str) -> list[str]:
    if source not in PDZ_SOURCES:
        raise ValueError(f"source must be one of {PDZ_SOURCES}, not {source!r}.")
    sequences = []
    for variant in raw_variants:
        expected_length = PDZ_LIBRARY_SPLIT + len(CRIPT_WT_SEQUENCE)
        if len(variant) != expected_length:
            raise ValueError(
                f"Expected a {expected_length}-residue fitted PDZ variant, "
                f"got {len(variant)} residues."
            )
        pdz_half = variant[:PDZ_LIBRARY_SPLIT]
        cript = variant[-len(CRIPT_WT_SEQUENCE) :]
        pdz = (
            pdz_half + PDZ3_WT_SEQUENCE[PDZ_LIBRARY_SPLIT:]
            if source == "trans-1"
            else PDZ3_WT_SEQUENCE[:PDZ_LIBRARY_SPLIT] + pdz_half
        )
        sequences.append(_canonicalize_sequence(f"{pdz}:{cript}"))
    return sequences


def _reconstruct_cript_sequences(raw_variants: list[str], source: str) -> list[str]:
    if source not in PDZ_CRIPT_LANDSCAPES:
        raise ValueError(
            f"source must be one of {PDZ_CRIPT_LANDSCAPES}, not {source!r}."
        )
    expected_length = len(CRIPT_WT_SEQUENCE) // 2
    sequences = []
    for variant in raw_variants:
        if len(variant) != expected_length:
            raise ValueError(
                f"Expected a {expected_length}-residue fitted {source} variant, "
                f"got {len(variant)} residues."
            )
        sequence = (
            CRIPT_WT_SEQUENCE[:expected_length] + variant
            if source == "cript-c"
            else variant + CRIPT_WT_SEQUENCE[expected_length:]
        )
        sequences.append(sequence)
    return sequences


def _read_reported_fitness(path: Path) -> pd.DataFrame:
    reported = pd.read_csv(
        path,
        dtype={"aa_seq": "string", "WT": "string"},
        low_memory=False,
    )
    expected_columns = ["aa_seq", "WT", "fitness", "sigma"]
    if reported.columns.tolist() != expected_columns:
        raise ValueError(
            f"Expected PDZ reported-fitness columns {expected_columns}, "
            f"got {reported.columns.tolist()}."
        )
    if reported[["aa_seq", "fitness", "sigma"]].isna().any(axis=None):
        raise ValueError("The PDZ reported-fitness table contains missing required values.")
    return reported


def _load_reported_fitness(path: Path) -> pd.DataFrame:
    reported = _read_reported_fitness(path)
    reported["variant_id"] = (
        reported["aa_seq"].map(_format_full_sequence).map(_canonicalize_sequence)
    )
    return reported


@dataclass(frozen=True)
class PDZLandscape:
    """Aligned raw counts and reported fitness for one PDZ landscape."""

    source: str
    variants: pd.DataFrame
    starting_counts: np.ndarray
    timepoint_counts: np.ndarray
    wild_type_index: int
    fitted_variant_ids: tuple[str, ...] | None = None

    @property
    def n_hmc_variants(self) -> int:
        return self.starting_counts.shape[-1]
PDZTransLandscape = PDZLandscape
PDZCisLandscape = PDZLandscape


def _variant_mapping(source: str, raw_variants: list[str]) -> pd.DataFrame:
    sequences = _reconstruct_sequences(raw_variants, source)
    mapping = pd.DataFrame(
        {
            "hmc_variant_index": np.arange(len(sequences), dtype=np.int32),
            "variant_id": sequences,
        }
    )
    if mapping["variant_id"].duplicated().any():
        raise ValueError(f"{source}: reconstructed sequence identifiers are not unique.")
    return mapping


def _infer_paper_sources(
    sequences: pd.Series,
    source_sequences: dict[str, set[str]],
) -> np.ndarray:
    if set(source_sequences) != set(PDZ_SOURCES):
        raise ValueError(f"count_paths must contain exactly {PDZ_SOURCES}.")
    first_source, second_source = PDZ_SOURCES
    in_first = sequences.isin(source_sequences[first_source]).to_numpy()
    in_second = sequences.isin(source_sequences[second_source]).to_numpy()
    first_only = np.flatnonzero(in_first & ~in_second)
    second_only = np.flatnonzero(in_second & ~in_first)
    if not len(first_only) or not len(second_only):
        raise ValueError("Each PDZ source must have sequences absent from the other source.")

    if first_only.max() + 1 == second_only.min():
        left_source, right_source = first_source, second_source
        boundary = int(second_only.min())
    elif second_only.max() + 1 == first_only.min():
        left_source, right_source = second_source, first_source
        boundary = int(first_only.min())
    else:
        raise ValueError("Reported-fitness rows do not form two unambiguous source blocks.")

    assigned = np.full(len(sequences), left_source, dtype=object)
    assigned[boundary:] = right_source
    if (in_first & ~in_second & (assigned != first_source)).any() or (
        in_second & ~in_first & (assigned != second_source)
    ).any():
        raise ValueError("Reported-fitness source blocks conflict with raw count sequences.")
    return assigned.astype(str)


def _source_reported_fitness(
    source: str,
    *,
    count_paths: dict[str, Path],
    reported_fitness_path: Path,
) -> pd.DataFrame:
    source_sequences = {}
    for source_name, count_path in count_paths.items():
        counts = load_pdz_dataset(str(count_path))
        source_sequences[source_name] = set(
            _variant_mapping(source_name, counts["variants"])["variant_id"]
        )

    reported = _load_reported_fitness(reported_fitness_path).rename(
        columns={"fitness": "reported_fitness"}
    )
    reported["source"] = _infer_paper_sources(
        reported["variant_id"],
        source_sequences,
    )
    reported = reported.loc[
        reported["source"].eq(source),
        ["variant_id", "reported_fitness", "sigma"],
    ].rename(columns={"sigma": "reported_fitness_sigma"})
    if reported["variant_id"].duplicated().any():
        raise ValueError(f"{source}: reported sequence identifiers are not unique.")
    return reported.reset_index(drop=True)


def load_pdz_trans(
    source: str,
    *,
    count_paths: dict[str, Path],
    reported_fitness_path: Path,
) -> PDZTransLandscape:
    """Load one source-specific PDZ landscape using SciStanPy's data loaders."""
    if source not in PDZ_SOURCES:
        raise ValueError(f"source must be one of {PDZ_SOURCES}, not {source!r}.")
    if set(count_paths) != set(PDZ_SOURCES):
        raise ValueError(f"count_paths must contain exactly {PDZ_SOURCES}.")

    counts = load_pdz_dataset(str(count_paths[source]))
    starting_counts = np.asarray(counts["starting_counts"], dtype=np.int64)
    timepoint_counts = np.asarray(counts["timepoint_counts"], dtype=np.int64)
    if starting_counts.shape != timepoint_counts.shape:
        raise ValueError(f"{source}: input and output count arrays must have equal shape.")
    mapping = _variant_mapping(source, counts["variants"])
    reported = _source_reported_fitness(
        source,
        count_paths=count_paths,
        reported_fitness_path=reported_fitness_path,
    )
    candidates = mapping.merge(reported, on="variant_id", how="inner", validate="one_to_one")
    candidates = candidates.sort_values("hmc_variant_index").reset_index(drop=True)
    if len(candidates) != len(reported):
        raise ValueError(
            f"{source}: the HMC count table covers {len(candidates):,} of "
            f"{len(reported):,} reported variants."
        )

    replicate_tables = []
    candidate_indices = candidates["hmc_variant_index"].to_numpy(dtype=np.int64)
    for replicate in range(starting_counts.shape[0]):
        replicate_tables.append(
            pd.DataFrame(
                {
                    "variant_id": candidates["variant_id"],
                    "replicate": replicate + 1,
                    "input_count": starting_counts[replicate, candidate_indices],
                    "output_count": timepoint_counts[replicate, candidate_indices],
                    "reported_fitness": candidates["reported_fitness"],
                }
            )
        )
    variants = prepare_counts(pd.concat(replicate_tables, ignore_index=True))
    variants = variants.merge(
        candidates[["variant_id", "hmc_variant_index", "reported_fitness_sigma"]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    ).sort_values("hmc_variant_index")
    variants = variants.reset_index(drop=True)
    variants.insert(1, "source", source)
    variants.insert(2, "variant_label", variants["variant_id"].map(pdz_variant_label))
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )

    wild_type = f"{PDZ3_WT_SEQUENCE}:{CRIPT_WT_SEQUENCE}"
    wild_type_rows = mapping.loc[mapping["variant_id"].eq(wild_type), "hmc_variant_index"]
    if len(wild_type_rows) != 1:
        raise ValueError(f"{source}: expected exactly one wild-type sequence.")
    return PDZTransLandscape(
        source=source,
        variants=variants,
        starting_counts=starting_counts,
        timepoint_counts=timepoint_counts,
        wild_type_index=int(wild_type_rows.item()),
        fitted_variant_ids=tuple(mapping["variant_id"].astype(str)),
    )


def load_pdz_cis(
    count_path: Path,
    *,
    reported_fitness_path: Path,
) -> PDZCisLandscape:
    """Load the fitted CRIPT pairwise-mutant cis landscape."""
    counts = load_pdz_dataset(str(count_path))
    starting_counts = np.asarray(counts["starting_counts"], dtype=np.int64)
    timepoint_counts = np.asarray(counts["timepoint_counts"], dtype=np.int64)
    if starting_counts.shape != timepoint_counts.shape:
        raise ValueError("cis: input and output count arrays must have equal shape.")

    raw_variants = [str(variant) for variant in counts["variants"]]
    wrong_lengths = {len(variant) for variant in raw_variants} - {len(CRIPT_WT_SEQUENCE)}
    if wrong_lengths:
        raise ValueError(
            f"cis: expected {len(CRIPT_WT_SEQUENCE)}-residue fitted variants; "
            f"found lengths {sorted(wrong_lengths)}."
        )
    mapping = pd.DataFrame(
        {
            "hmc_variant_index": np.arange(len(raw_variants), dtype=np.int32),
            "variant_id": raw_variants,
        }
    )
    if mapping["variant_id"].duplicated().any():
        raise ValueError("cis: fitted sequence identifiers are not unique.")

    reported = _read_reported_fitness(reported_fitness_path).rename(
        columns={
            "aa_seq": "variant_id",
            "fitness": "reported_fitness",
            "sigma": "reported_fitness_sigma",
        }
    )
    if reported["variant_id"].str.len().ne(len(CRIPT_WT_SEQUENCE)).any():
        raise ValueError(
            f"cis: reported variants must contain {len(CRIPT_WT_SEQUENCE)} residues."
        )
    if reported["variant_id"].duplicated().any():
        raise ValueError("cis: reported sequence identifiers are not unique.")
    candidates = mapping.merge(
        reported[["variant_id", "reported_fitness", "reported_fitness_sigma"]],
        on="variant_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("hmc_variant_index")
    candidates = candidates.reset_index(drop=True)
    if len(candidates) != len(reported):
        raise ValueError(
            f"cis: the HMC count table covers {len(candidates):,} of "
            f"{len(reported):,} reported variants."
        )

    replicate_tables = []
    candidate_indices = candidates["hmc_variant_index"].to_numpy(dtype=np.int64)
    for replicate in range(starting_counts.shape[0]):
        replicate_tables.append(
            pd.DataFrame(
                {
                    "variant_id": candidates["variant_id"],
                    "replicate": replicate + 1,
                    "input_count": starting_counts[replicate, candidate_indices],
                    "output_count": timepoint_counts[replicate, candidate_indices],
                    "reported_fitness": candidates["reported_fitness"],
                }
            )
        )
    variants = prepare_counts(pd.concat(replicate_tables, ignore_index=True))
    variants = variants.merge(
        candidates[["variant_id", "hmc_variant_index", "reported_fitness_sigma"]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    ).sort_values("hmc_variant_index")
    variants = variants.reset_index(drop=True)
    variants.insert(1, "source", PDZ_CIS_SOURCE)
    variants.insert(2, "variant_label", variants["variant_id"].map(pdz_cis_variant_label))
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )

    wild_type_rows = mapping.loc[
        mapping["variant_id"].eq(CRIPT_WT_SEQUENCE),
        "hmc_variant_index",
    ]
    if len(wild_type_rows) != 1:
        raise ValueError("cis: expected exactly one wild-type sequence.")
    return PDZCisLandscape(
        source=PDZ_CIS_SOURCE,
        variants=variants,
        starting_counts=starting_counts,
        timepoint_counts=timepoint_counts,
        wild_type_index=int(wild_type_rows.item()),
        fitted_variant_ids=tuple(raw_variants),
    )


def load_pdz_cript(
    source: str,
    count_path: Path,
    *,
    reported_fitness_path: Path,
) -> PDZLandscape:
    """Load one fitted four-residue CRIPT-N or CRIPT-C landscape."""
    if source not in PDZ_CRIPT_LANDSCAPES:
        raise ValueError(
            f"source must be one of {PDZ_CRIPT_LANDSCAPES}, not {source!r}."
        )
    counts = load_pdz_dataset(str(count_path))
    starting_counts = np.asarray(counts["starting_counts"], dtype=np.int64)
    timepoint_counts = np.asarray(counts["timepoint_counts"], dtype=np.int64)
    if starting_counts.shape != timepoint_counts.shape:
        raise ValueError(f"{source}: input and output count arrays must have equal shape.")

    raw_variants = [str(variant) for variant in counts["variants"]]
    full_variants = _reconstruct_cript_sequences(raw_variants, source)
    mapping = pd.DataFrame(
        {
            "hmc_variant_index": np.arange(len(full_variants), dtype=np.int32),
            "variant_id": full_variants,
        }
    )
    if mapping["variant_id"].duplicated().any():
        raise ValueError(f"{source}: reconstructed sequence identifiers are not unique.")

    reported = _read_reported_fitness(reported_fitness_path).rename(
        columns={
            "aa_seq": "variant_id",
            "fitness": "reported_fitness",
            "sigma": "reported_fitness_sigma",
        }
    )
    if reported["variant_id"].str.len().ne(len(CRIPT_WT_SEQUENCE)).any():
        raise ValueError(
            f"{source}: reported variants must contain "
            f"{len(CRIPT_WT_SEQUENCE)} residues."
        )
    if reported["variant_id"].duplicated().any():
        raise ValueError(f"{source}: reported sequence identifiers are not unique.")
    candidates = mapping.merge(
        reported[["variant_id", "reported_fitness", "reported_fitness_sigma"]],
        on="variant_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("hmc_variant_index")
    candidates = candidates.reset_index(drop=True)
    if len(candidates) != len(reported):
        raise ValueError(
            f"{source}: the HMC count table covers {len(candidates):,} of "
            f"{len(reported):,} reported variants."
        )

    replicate_tables = []
    candidate_indices = candidates["hmc_variant_index"].to_numpy(dtype=np.int64)
    for replicate in range(starting_counts.shape[0]):
        replicate_tables.append(
            pd.DataFrame(
                {
                    "variant_id": candidates["variant_id"],
                    "replicate": replicate + 1,
                    "input_count": starting_counts[replicate, candidate_indices],
                    "output_count": timepoint_counts[replicate, candidate_indices],
                    "reported_fitness": candidates["reported_fitness"],
                }
            )
        )
    variants = prepare_counts(pd.concat(replicate_tables, ignore_index=True))
    variants = variants.merge(
        candidates[["variant_id", "hmc_variant_index", "reported_fitness_sigma"]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    ).sort_values("hmc_variant_index")
    variants = variants.reset_index(drop=True)
    variants.insert(1, "source", source)
    variants.insert(2, "variant_label", variants["variant_id"].map(pdz_cis_variant_label))
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )

    wild_type_rows = mapping.loc[
        mapping["variant_id"].eq(CRIPT_WT_SEQUENCE),
        "hmc_variant_index",
    ]
    if len(wild_type_rows) != 1:
        raise ValueError(f"{source}: expected exactly one wild-type sequence.")
    return PDZLandscape(
        source=source,
        variants=variants,
        starting_counts=starting_counts,
        timepoint_counts=timepoint_counts,
        wild_type_index=int(wild_type_rows.item()),
        fitted_variant_ids=tuple(full_variants),
    )


def _growth_target(
    growth_rate_prior: PDZGrowthRatePrior,
) -> tuple[str, Literal["log", "inverse"]]:
    if growth_rate_prior in {"exponential", "lomax"}:
        return "log_r_mean", "log"
    if growth_rate_prior == "gamma":
        return "inv_r_mean", "inverse"
    raise ValueError("growth_rate_prior must be 'exponential', 'gamma', or 'lomax'.")


def _to_log_mean_growth(
    values: np.ndarray,
    transform: Literal["log", "inverse"],
) -> np.ndarray:
    return transform_hmc_score(values, transform)


def _sample_ids(total: int, n_samples: int | None, *, seed: int) -> np.ndarray:
    resolved = total if n_samples is None else n_samples
    if not 1 <= resolved <= total:
        raise ValueError(f"n_samples must be between 1 and {total:,}.")
    if resolved == total:
        return np.arange(total, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(total, resolved, replace=False)).astype(
        np.int64
    )


def _read_native_samples(
    variable,
    sample_ids: np.ndarray,
    *,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
) -> np.ndarray:
    total_samples = variable.shape[0] * draws_per_chain
    if np.array_equal(sample_ids, np.arange(total_samples, dtype=np.int64)):
        selection = (
            slice(None),
            slice(None),
            *(slice(None) for _ in range(variable.ndim - 3)),
            slice(variant_start, variant_stop),
        )
        values = np.asarray(variable[selection])
        return values.reshape(
            total_samples,
            *variable.shape[2:-1],
            variant_stop - variant_start,
        )

    chains = sample_ids // draws_per_chain
    draws = sample_ids % draws_per_chain
    result = np.empty(
        (len(sample_ids), *variable.shape[2:-1], variant_stop - variant_start),
        dtype=variable.dtype,
    )
    for chain in np.unique(chains):
        positions = np.flatnonzero(chains == chain)
        chain_draws = draws[positions]
        order = np.argsort(chain_draws)
        sorted_positions = positions[order]
        selection = (
            int(chain),
            chain_draws[order],
            *(slice(None) for _ in range(variable.ndim - 3)),
            slice(variant_start, variant_stop),
        )
        result[sorted_positions] = variable[selection]
    return result


def _count_proxies(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    wild_type_starting: np.ndarray,
    wild_type_timepoint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starting_total = starting_counts.sum(axis=1, dtype=float)
    timepoint_total = timepoint_counts.sum(axis=1, dtype=float)
    count_ratio = np.divide(
        timepoint_total,
        starting_total,
        out=np.full_like(timepoint_total, np.nan),
        where=starting_total > 0,
    )

    wild_type_starting = wild_type_starting[:, :, None]
    wild_type_timepoint = wild_type_timepoint[:, :, None]
    valid = (
        (starting_counts > 0)
        & (timepoint_counts > 0)
        & (wild_type_starting > 0)
        & (wild_type_timepoint > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        replicate_fitness = np.log(timepoint_counts / starting_counts) - np.log(
            wild_type_timepoint / wild_type_starting
        )
        variance = (
            1 / starting_counts
            + 1 / timepoint_counts
            + 1 / wild_type_starting
            + 1 / wild_type_timepoint
        )
        weights = np.where(valid, 1 / variance, 0.0)
    weighted_fitness = np.zeros_like(replicate_fitness)
    np.multiply(replicate_fitness, weights, out=weighted_fitness, where=valid)
    weight_sum = weights.sum(axis=1)
    dimsum_fitness = np.divide(
        weighted_fitness.sum(axis=1),
        weight_sum,
        out=np.full_like(weight_sum, np.nan),
        where=weight_sum > 0,
    )
    return count_ratio, dimsum_fitness


def pdz_dimsum_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    wild_type_index: int,
) -> np.ndarray:
    """Calculate the repository's DiMSum-style PDZ count proxy."""
    starting = np.asarray(starting_counts)
    timepoint = np.asarray(timepoint_counts)
    if starting.ndim < 2 or starting.shape != timepoint.shape:
        raise ValueError(
            "PDZ starting and timepoint counts must have matching "
            "(..., replicate, variant) shapes."
        )
    if (
        not np.issubdtype(starting.dtype, np.number)
        or not np.issubdtype(timepoint.dtype, np.number)
        or not np.isfinite(starting).all()
        or not np.isfinite(timepoint).all()
        or np.any(starting < 0)
        or np.any(timepoint < 0)
    ):
        raise ValueError("PDZ counts must be finite, numeric, and nonnegative.")
    if not 0 <= wild_type_index < starting.shape[-1]:
        raise ValueError("PDZ wild_type_index lies outside the variant axis.")

    sample_shape = starting.shape[:-2]
    n_replicates, n_variants = starting.shape[-2:]
    flat_starting = starting.reshape(-1, n_replicates, n_variants)
    flat_timepoint = timepoint.reshape(-1, n_replicates, n_variants)
    _, fitness = _count_proxies(
        flat_starting,
        flat_timepoint,
        flat_starting[:, :, wild_type_index],
        flat_timepoint[:, :, wild_type_index],
    )
    return fitness.reshape(*sample_shape, n_variants)


def _proxy_variant_table(landscape: PDZLandscape) -> pd.DataFrame:
    if landscape.fitted_variant_ids is None:
        raise ValueError("PDZ proxy generation requires full fitted variant IDs.")
    if len(landscape.fitted_variant_ids) != landscape.n_hmc_variants:
        raise ValueError("PDZ fitted variant IDs do not match the count axis.")
    if len(set(landscape.fitted_variant_ids)) != landscape.n_hmc_variants:
        raise ValueError("PDZ fitted variant IDs must be unique.")

    n_variants = landscape.n_hmc_variants
    variants = pd.DataFrame(
        {
            "variant_id": landscape.fitted_variant_ids,
            "source": landscape.source,
            "hmc_variant_index": np.arange(n_variants, dtype=np.int32),
            "input_count": landscape.starting_counts.sum(axis=0, dtype=np.int64),
            "output_count": landscape.timepoint_counts.sum(axis=0, dtype=np.int64),
        }
    )
    variants["count_ratio"] = np.divide(
        variants["output_count"],
        variants["input_count"],
        out=np.full(n_variants, np.nan, dtype=float),
        where=variants["input_count"].to_numpy() > 0,
    )
    labeler = (
        pdz_variant_label
        if landscape.source in PDZ_TRANS_LANDSCAPES
        else pdz_cis_variant_label
    )
    variants.insert(2, "variant_label", variants["variant_id"].map(labeler))

    reported_columns = [
        "variant_id",
        "hmc_variant_index",
        "reported_fitness",
        "reference_fitness",
        "reference_fitness_source",
        "reported_fitness_sigma",
        "reported_rank",
    ]
    variants = variants.merge(
        landscape.variants[reported_columns],
        on=["variant_id", "hmc_variant_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    variants["reference_fitness_source"] = variants[
        "reference_fitness_source"
    ].fillna("unavailable")
    variants["is_ranking_candidate"] = variants["reported_fitness"].notna()
    return variants


def _hmc_score_variant_table(landscape: PDZLandscape) -> pd.DataFrame:
    if landscape.fitted_variant_ids is None:
        raise ValueError("PDZ score extraction requires full fitted variant IDs.")
    if len(landscape.fitted_variant_ids) != landscape.n_hmc_variants:
        raise ValueError("PDZ fitted variant IDs do not match the HMC axis.")
    if len(set(landscape.fitted_variant_ids)) != landscape.n_hmc_variants:
        raise ValueError("PDZ fitted variant IDs must be unique.")
    full = pd.DataFrame(
        {
            "variant_id": landscape.fitted_variant_ids,
            "hmc_variant_index": np.arange(
                landscape.n_hmc_variants,
                dtype=np.int32,
            ),
        }
    )
    extra_columns = [
        name
        for name in landscape.variants.columns
        if name not in {"variant_id", "hmc_variant_index"}
    ]
    return full.merge(
        landscape.variants[["variant_id", "hmc_variant_index", *extra_columns]],
        on=["variant_id", "hmc_variant_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )


def build_pdz_hmc_score_ensemble(
    landscape: PDZLandscape,
    hmc_path: str | Path,
    *,
    growth_rate_prior: PDZGrowthRatePrior,
    growth_curve: PDZGrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    artifact_kind: str | None = None,
    n_samples: int | None = None,
    seed: int = 23,
    sample_view: str = "full",
    sample_chunk_size: int = 256,
    variant_chunk_size: int | None = None,
    scratch_dir: str | Path | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Extract the full PDZ posterior log-mean-growth matrix."""
    if landscape.source not in PDZ_LANDSCAPES:
        raise ValueError(f"Unknown PDZ landscape {landscape.source!r}.")
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")
    target_name, target_transform = _growth_target(growth_rate_prior)
    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape=f"pdz-{landscape.source}",
        inference_method="hmc",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    analysis_variants = landscape.variants.copy()
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        analysis_variants["hmc_variant_index"].to_numpy(dtype=np.int64),
    )
    extra_metadata: dict[str, object] = {
        "source": landscape.source,
        **PDZ_LIBRARY_PROVENANCE[landscape.source],
        "model": f"{growth_rate_prior.title()}-{growth_curve.title()}",
        "growth_rate_prior": growth_rate_prior,
        "growth_curve": growth_curve,
        "primary_score": "log_mean_growth",
        "reference_fitness_source": "reported",
        "reported_fitness_available": True,
    }
    if job_id is not None:
        extra_metadata["job_id"] = job_id
    if artifact_kind is not None:
        extra_metadata["artifact_kind"] = artifact_kind
    ensemble = _build_hmc_scores(
        hmc_path=resolved_path,
        identity=identity,
        posterior_variable=target_name,
        posterior_transform=target_transform,
        score_name="log_mean_growth",
        score_contract=f"pdz-{target_name}-{target_transform}-v1",
        variants=_hmc_score_variant_table(landscape),
        analysis_variants=analysis_variants,
        transform=lambda values: _to_log_mean_growth(values, target_transform),
        n_samples=n_samples,
        seed=seed,
        sample_view=sample_view,
        sample_chunk_size=sample_chunk_size,
        variant_chunk_size=variant_chunk_size,
        scratch_dir=scratch_dir,
        extra_metadata=extra_metadata,
        show_progress=show_progress,
    )
    analysis_chunk_size = int(ensemble.metadata["variant_chunk_size"])
    ensemble.metadata["required_block_starts"] = [
        (landscape.wild_type_index // analysis_chunk_size) * analysis_chunk_size
    ]
    ensemble.validate()
    return ensemble


def build_pdz_mle_score_ensemble(
    landscape: PDZLandscape,
    mle_path: str | Path,
    *,
    growth_rate_prior: PDZGrowthRatePrior,
    growth_curve: PDZGrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    variant_chunk_size: int | None = None,
) -> ScoreEnsemble:
    """Build a provisional DiMSum-style proxy from sampled-MLE counts."""
    if landscape.source not in PDZ_LANDSCAPES:
        raise ValueError(f"Unknown PDZ landscape {landscape.source!r}.")
    _growth_target(growth_rate_prior)
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")

    resolved_path = Path(mle_path)
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape=f"pdz-{landscape.source}",
        inference_method="mle_predictive",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    variants = _proxy_variant_table(landscape)
    observed_proxy = pdz_dimsum_fitness(
        landscape.starting_counts,
        landscape.timepoint_counts,
        landscape.wild_type_index,
    )
    candidate_mask = variants["is_ranking_candidate"].to_numpy(dtype=bool)
    reported = variants["reported_fitness"].to_numpy(dtype=float)
    agreement_mask = candidate_mask & np.isfinite(observed_proxy) & np.isfinite(reported)
    if agreement_mask.sum() < 2:
        raise ValueError("PDZ observed agreement requires two finite reported variants.")
    observed_candidate = observed_proxy[agreement_mask]
    reported_candidate = reported[agreement_mask]
    pearson = float(np.corrcoef(observed_candidate, reported_candidate)[0, 1])
    spearman = float(
        pd.Series(observed_candidate).corr(
            pd.Series(reported_candidate),
            method="spearman",
        )
    )
    if not np.isfinite(spearman) or spearman <= 0:
        raise ValueError("PDZ observed proxy has the wrong reported-fitness direction.")
    affine_slope, affine_intercept = np.polyfit(
        observed_candidate,
        reported_candidate,
        1,
    )
    affine_prediction = affine_slope * observed_candidate + affine_intercept

    expected_starting_totals = landscape.starting_counts.sum(axis=-1, dtype=np.int64)
    expected_timepoint_totals = landscape.timepoint_counts.sum(
        axis=-1,
        dtype=np.int64,
    )
    with h5py.File(resolved_path, "r") as handle:
        observed = handle.get("observed_data")
        if not isinstance(observed, h5py.Group):
            raise ValueError("The sampled-MLE artifact must embed observed PDZ counts.")
        if not np.array_equal(
            observed["starting_counts"][:],
            landscape.starting_counts,
        ) or not np.array_equal(
            observed["timepoint_counts"][:],
            landscape.timepoint_counts,
        ):
            raise ValueError("The sampled-MLE artifact embeds different PDZ counts.")
        predictive = handle.get("posterior_predictive")
        if not isinstance(predictive, h5py.Group):
            raise ValueError("The sampled-MLE artifact is missing posterior_predictive.")
        starting_object = predictive.get("starting_counts")
        timepoint_object = predictive.get("timepoint_counts")
        if not isinstance(starting_object, h5py.Dataset) or not isinstance(
            timepoint_object,
            h5py.Dataset,
        ):
            raise ValueError(
                "PDZ posterior_predictive must contain starting_counts and "
                "timepoint_counts."
            )
        if starting_object.ndim != 4 or starting_object.shape[0] != 1:
            raise ValueError(
                "PDZ predictive counts must be chain-by-draw-by-replicate-by-variant."
            )
        n_samples = int(starting_object.shape[1])
        expected_shape = (
            1,
            n_samples,
            landscape.starting_counts.shape[0],
            landscape.n_hmc_variants,
        )
        if starting_object.shape != expected_shape or timepoint_object.shape != expected_shape:
            raise ValueError("PDZ predictive count axes differ from observed counts.")
        if n_samples < 2:
            raise ValueError("PDZ proxy analysis requires at least two MLE count draws.")
        if variant_chunk_size is None:
            stored_chunks = [
                dataset.chunks[-1]
                for dataset in (starting_object, timepoint_object)
                if dataset.chunks is not None
            ]
            variant_chunk_size = min(stored_chunks, default=landscape.n_hmc_variants)
        if variant_chunk_size < 1:
            raise ValueError("variant_chunk_size must be positive.")

        wild_type_starting = np.asarray(
            starting_object[
                0,
                :,
                :,
                landscape.wild_type_index,
            ]
        )
        wild_type_timepoint = np.asarray(
            timepoint_object[
                0,
                :,
                :,
                landscape.wild_type_index,
            ]
        )
        sampled_proxy = np.full(
            (n_samples, landscape.n_hmc_variants),
            np.nan,
            dtype=np.float32,
        )
        starting_totals = np.zeros(
            (n_samples, landscape.starting_counts.shape[0]),
            dtype=np.int64,
        )
        timepoint_totals = np.zeros_like(starting_totals)
        for variant_start in range(
            0,
            landscape.n_hmc_variants,
            variant_chunk_size,
        ):
            variant_stop = min(
                variant_start + variant_chunk_size,
                landscape.n_hmc_variants,
            )
            starting = np.asarray(
                starting_object[0, :, :, variant_start:variant_stop]
            )
            timepoint = np.asarray(
                timepoint_object[0, :, :, variant_start:variant_stop]
            )
            starting_totals += starting.sum(axis=-1, dtype=np.int64)
            timepoint_totals += timepoint.sum(axis=-1, dtype=np.int64)
            _, block_proxy = _count_proxies(
                starting,
                timepoint,
                wild_type_starting,
                wild_type_timepoint,
            )
            sampled_proxy[:, variant_start:variant_stop] = block_proxy.astype(
                np.float32
            )

    if not np.all(starting_totals == expected_starting_totals[None, :]):
        raise ValueError("PDZ predictive starting-count totals differ from observed data.")
    if not np.all(timepoint_totals == expected_timepoint_totals[None, :]):
        raise ValueError("PDZ predictive timepoint-count totals differ from observed data.")
    finite_ranking_candidates = np.zeros(n_samples, dtype=np.int32)
    for variant_start in range(
        0,
        landscape.n_hmc_variants,
        variant_chunk_size,
    ):
        variant_stop = min(
            variant_start + variant_chunk_size,
            landscape.n_hmc_variants,
        )
        local_candidates = candidate_mask[variant_start:variant_stop]
        if local_candidates.any():
            finite_ranking_candidates += np.isfinite(
                sampled_proxy[:, variant_start:variant_stop][:, local_candidates]
            ).sum(axis=-1)

    score_name = PDZ_PREDICTIVE_SCORE_NAME
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                sampled_proxy[None],
            ),
            "observed_score": (
                ("score_name", "variant"),
                observed_proxy[None].astype(np.float32),
            ),
        },
        coords={
            "score_name": [score_name],
            "sample": np.arange(n_samples, dtype=np.int32),
            "variant": np.arange(landscape.n_hmc_variants, dtype=np.int32),
        },
    )
    metadata: dict[str, object] = {
        **identity.metadata(),
        "fit_method": "mle",
        "artifact_kind": "sampled_mle",
        "job_id": job_id,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "scores": [score_name],
        "score_contracts": {score_name: PDZ_PREDICTIVE_SCORE_CONTRACT},
        "score_status": {score_name: "provisional"},
        "higher_is_better": {score_name: True},
        "n_samples": n_samples,
        "n_variants": landscape.n_hmc_variants,
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_excluded_from_ranking": int((~candidate_mask).sum()),
        "variant_chunk_size": variant_chunk_size,
        "finite_proxy_values": int(np.isfinite(sampled_proxy).sum()),
        "min_finite_variants_per_sample": int(
            np.isfinite(sampled_proxy).sum(axis=-1).min()
        ),
        "max_finite_variants_per_sample": int(
            np.isfinite(sampled_proxy).sum(axis=-1).max()
        ),
        "finite_ranking_candidates_per_sample": (
            finite_ranking_candidates.astype(int).tolist()
        ),
        "min_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.min()
        ),
        "max_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.max()
        ),
        "normalization_reference": "fitted wild-type counts in each replicate",
        "replicate_weighting": "inverse Poisson log-ratio variance",
        "count_variables": {
            "input": "starting_counts",
            "output": "timepoint_counts",
        },
        "count_axes": {
            "starting_counts": ["chain", "draw", "replicate", "variant"],
            "timepoint_counts": ["chain", "draw", "replicate", "variant"],
        },
        "observed_counts_status": "matched",
        "observed_agreement": {
            "source_column": "fitness",
            "n_compared": int(agreement_mask.sum()),
            "n_ranking_candidates": int(candidate_mask.sum()),
            "n_missing_observed_proxy": int(
                candidate_mask.sum() - agreement_mask.sum()
            ),
            "exact_match": False,
            "mean_abs_error": float(
                np.mean(np.abs(observed_candidate - reported_candidate))
            ),
            "max_abs_error": float(
                np.max(np.abs(observed_candidate - reported_candidate))
            ),
            "pearson": pearson,
            "spearman": spearman,
            "affine_slope_diagnostic_only": float(affine_slope),
            "affine_intercept_diagnostic_only": float(affine_intercept),
            "affine_mean_abs_error_diagnostic_only": float(
                np.mean(np.abs(affine_prediction - reported_candidate))
            ),
        },
        "source_artifact": _file_signature(resolved_path),
        "warnings": [
            "This raw-count approximation does not exactly reproduce the "
            "published DiMSum fitness scale."
        ],
    }
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
    )


def build_pdz_hmc_count_score_ensemble(
    landscape: PDZLandscape,
    hmc_path: str | Path,
    *,
    growth_rate_prior: PDZGrowthRatePrior,
    growth_curve: PDZGrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    artifact_kind: str | None = None,
    n_samples: int | None = None,
    seed: int = 23,
    sample_chunk_size: int = 64,
    variant_chunk_size: int | None = None,
    scratch_dir: str | Path | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Build a disk-backed DiMSum-style proxy from HMC predictive counts."""
    if landscape.source not in PDZ_LANDSCAPES:
        raise ValueError(f"Unknown PDZ landscape {landscape.source!r}.")
    _growth_target(growth_rate_prior)
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")

    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape=f"pdz-{landscape.source}",
        inference_method="hmc_count_based",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    variants = _proxy_variant_table(landscape)
    observed_proxy = pdz_dimsum_fitness(
        landscape.starting_counts,
        landscape.timepoint_counts,
        landscape.wild_type_index,
    )
    candidate_mask = variants["is_ranking_candidate"].to_numpy(dtype=bool)
    reported = variants["reported_fitness"].to_numpy(dtype=float)
    agreement_mask = candidate_mask & np.isfinite(observed_proxy) & np.isfinite(reported)
    if agreement_mask.sum() < 2:
        raise ValueError("PDZ observed agreement requires two finite reported variants.")
    observed_candidate = observed_proxy[agreement_mask]
    reported_candidate = reported[agreement_mask]
    pearson = float(np.corrcoef(observed_candidate, reported_candidate)[0, 1])
    spearman = float(
        pd.Series(observed_candidate).corr(
            pd.Series(reported_candidate),
            method="spearman",
        )
    )
    if not np.isfinite(spearman) or spearman <= 0:
        raise ValueError("PDZ observed proxy has the wrong reported-fitness direction.")

    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix="pdz_hmc_count_proxy_",
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
            observed = handle.get("observed_data")
            if not isinstance(observed, h5py.Group):
                raise ValueError("The HMC artifact must embed observed PDZ counts.")
            if not np.array_equal(
                observed["starting_counts"][:],
                landscape.starting_counts,
            ) or not np.array_equal(
                observed["timepoint_counts"][:],
                landscape.timepoint_counts,
            ):
                raise ValueError("The HMC artifact embeds different PDZ counts.")
            predictive = handle.get("posterior_predictive")
            if not isinstance(predictive, h5py.Group):
                raise ValueError("The HMC artifact is missing posterior_predictive.")
            starting_object = predictive.get("starting_counts")
            timepoint_object = predictive.get("timepoint_counts")
            if not isinstance(starting_object, h5py.Dataset) or not isinstance(
                timepoint_object,
                h5py.Dataset,
            ):
                raise ValueError(
                    "PDZ posterior_predictive must contain starting_counts and "
                    "timepoint_counts."
                )
            if starting_object.ndim != 4:
                raise ValueError(
                    "PDZ predictive counts must be chain-by-draw-by-replicate-by-variant."
                )
            n_chains, draws_per_chain = map(int, starting_object.shape[:2])
            expected_shape = (
                n_chains,
                draws_per_chain,
                landscape.starting_counts.shape[0],
                landscape.n_hmc_variants,
            )
            if (
                starting_object.shape != expected_shape
                or timepoint_object.shape != expected_shape
            ):
                raise ValueError("PDZ predictive count axes differ from observed counts.")
            selected_sample_ids = _sample_ids(
                n_chains * draws_per_chain,
                n_samples,
                seed=seed,
            )
            if len(selected_sample_ids) < 2:
                raise ValueError("PDZ proxy analysis requires at least two HMC samples.")
            if variant_chunk_size is None:
                stored_chunks = [
                    dataset.chunks[-1]
                    for dataset in (starting_object, timepoint_object)
                    if dataset.chunks is not None
                ]
                variant_chunk_size = min(
                    stored_chunks,
                    default=landscape.n_hmc_variants,
                )
            if variant_chunk_size < 1:
                raise ValueError("variant_chunk_size must be positive.")

            wild_type_starting = _read_native_samples(
                starting_object,
                selected_sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=landscape.wild_type_index,
                variant_stop=landscape.wild_type_index + 1,
            )[..., 0]
            wild_type_timepoint = _read_native_samples(
                timepoint_object,
                selected_sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=landscape.wild_type_index,
                variant_stop=landscape.wild_type_index + 1,
            )[..., 0]
            sampled_proxy = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(selected_sample_ids), landscape.n_hmc_variants),
            )
            sampled_proxy[:] = np.nan
            expected_starting_totals = landscape.starting_counts.sum(
                axis=-1,
                dtype=np.int64,
            )
            expected_timepoint_totals = landscape.timepoint_counts.sum(
                axis=-1,
                dtype=np.int64,
            )
            starting_totals = np.zeros(
                (len(selected_sample_ids), landscape.starting_counts.shape[0]),
                dtype=np.int64,
            )
            timepoint_totals = np.zeros_like(starting_totals)
            finite_per_sample = np.zeros(len(selected_sample_ids), dtype=np.int64)
            finite_ranking_candidates = np.zeros(
                len(selected_sample_ids),
                dtype=np.int64,
            )
            block_starts = list(
                range(0, landscape.n_hmc_variants, variant_chunk_size)
            )
            if show_progress:
                print(
                    f"pdz-{landscape.source}: {len(selected_sample_ids):,} HMC "
                    f"predictive samples, {landscape.n_hmc_variants:,} variants",
                    flush=True,
                )
            for block_number, variant_start in enumerate(block_starts, start=1):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    landscape.n_hmc_variants,
                )
                local_candidates = candidate_mask[variant_start:variant_stop]
                for sample_start in range(
                    0,
                    len(selected_sample_ids),
                    sample_chunk_size,
                ):
                    sample_stop = min(
                        sample_start + sample_chunk_size,
                        len(selected_sample_ids),
                    )
                    block_sample_ids = selected_sample_ids[sample_start:sample_stop]
                    starting = _read_native_samples(
                        starting_object,
                        block_sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    timepoint = _read_native_samples(
                        timepoint_object,
                        block_sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    starting_totals[sample_start:sample_stop] += starting.sum(
                        axis=-1,
                        dtype=np.int64,
                    )
                    timepoint_totals[sample_start:sample_stop] += timepoint.sum(
                        axis=-1,
                        dtype=np.int64,
                    )
                    _, block_proxy = _count_proxies(
                        starting,
                        timepoint,
                        wild_type_starting[sample_start:sample_stop],
                        wild_type_timepoint[sample_start:sample_stop],
                    )
                    block_proxy = block_proxy.astype(np.float32, copy=False)
                    sampled_proxy[
                        sample_start:sample_stop,
                        variant_start:variant_stop,
                    ] = block_proxy
                    finite = np.isfinite(block_proxy)
                    finite_per_sample[sample_start:sample_stop] += finite.sum(axis=-1)
                    finite_ranking_candidates[sample_start:sample_stop] += finite[
                        :,
                        local_candidates,
                    ].sum(axis=-1)
                if show_progress:
                    print(
                        f"  proxy chunks {block_number}/{len(block_starts)} "
                        f"({(time.monotonic() - started) / 60:.1f} min)",
                        flush=True,
                    )
            sampled_proxy.flush()

        if not np.all(starting_totals == expected_starting_totals[None, :]):
            raise ValueError(
                "PDZ predictive starting-count totals differ from observed data."
            )
        if not np.all(timepoint_totals == expected_timepoint_totals[None, :]):
            raise ValueError(
                "PDZ predictive timepoint-count totals differ from observed data."
            )
    except Exception:
        del sampled_proxy
        temporary_path.unlink(missing_ok=True)
        raise

    score_name = PDZ_PREDICTIVE_SCORE_NAME
    scores = xr.Dataset(
        {
            "fitness_score": (
                ("score_name", "sample", "variant"),
                sampled_proxy[None],
            ),
            "observed_score": (
                ("score_name", "variant"),
                observed_proxy[None].astype(np.float32),
            ),
        },
        coords={
            "score_name": [score_name],
            "sample": np.arange(len(selected_sample_ids), dtype=np.int32),
            "variant": np.arange(landscape.n_hmc_variants, dtype=np.int32),
            "source_sample_id": ("sample", selected_sample_ids),
            "source_chain": (
                "sample",
                (selected_sample_ids // draws_per_chain).astype(np.int32),
            ),
            "source_draw": (
                "sample",
                (selected_sample_ids % draws_per_chain).astype(np.int32),
            ),
        },
    )
    metadata: dict[str, object] = {
        **identity.metadata(),
        "fit_method": "hmc_count_based",
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "uncertainty_target": HMC_COUNT_BASED_TARGET,
        "scores": [score_name],
        "score_contracts": {score_name: PDZ_PREDICTIVE_SCORE_CONTRACT},
        "score_status": {score_name: "provisional"},
        "higher_is_better": {score_name: True},
        "n_chains": n_chains,
        "draws_per_chain": draws_per_chain,
        "n_source_samples": n_chains * draws_per_chain,
        "n_samples": len(selected_sample_ids),
        "sample_selection_seed": seed,
        "n_variants": landscape.n_hmc_variants,
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_excluded_from_ranking": int((~candidate_mask).sum()),
        "sample_chunk_size": sample_chunk_size,
        "variant_chunk_size": variant_chunk_size,
        "finite_proxy_values": int(finite_per_sample.sum()),
        "min_finite_variants_per_sample": int(finite_per_sample.min()),
        "max_finite_variants_per_sample": int(finite_per_sample.max()),
        "finite_ranking_candidates_per_sample": finite_ranking_candidates.tolist(),
        "min_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.min()
        ),
        "max_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.max()
        ),
        "normalization_reference": "fitted wild-type counts in each replicate",
        "replicate_weighting": "inverse Poisson log-ratio variance",
        "count_variables": {
            "input": "starting_counts",
            "output": "timepoint_counts",
        },
        "count_axes": {
            "starting_counts": ["chain", "draw", "replicate", "variant"],
            "timepoint_counts": ["chain", "draw", "replicate", "variant"],
        },
        "observed_counts_status": "matched",
        "observed_agreement": {
            "source_column": "fitness",
            "n_compared": int(agreement_mask.sum()),
            "exact_match": False,
            "pearson": pearson,
            "spearman": spearman,
        },
        "source_artifact": _file_signature(resolved_path),
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [
            "This raw-count approximation does not exactly reproduce the "
            "published DiMSum fitness scale."
        ],
    }
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
        temporary_paths=(temporary_path,),
    )


def build_pdz_cis_mle_score_ensemble(
    landscape: PDZLandscape,
    mle_path: str | Path,
    *,
    growth_rate_prior: PDZGrowthRatePrior,
    growth_curve: PDZGrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    variant_chunk_size: int | None = None,
) -> ScoreEnsemble:
    """Build the provisional cis DiMSum-style sampled-MLE proxy."""
    if landscape.source != PDZ_CIS_SOURCE:
        raise ValueError("The cis MLE proxy builder requires the cis landscape.")
    return build_pdz_mle_score_ensemble(
        landscape,
        mle_path,
        growth_rate_prior=growth_rate_prior,
        growth_curve=growth_curve,
        fit_id=fit_id,
        job_id=job_id,
        variant_chunk_size=variant_chunk_size,
    )


def _block_scores(
    *,
    target,
    starting_variable,
    timepoint_variable,
    sample_ids: np.ndarray,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
    local_candidate_indices: np.ndarray,
    wild_type_index: int,
    wild_type_starting: np.ndarray | None,
    wild_type_timepoint: np.ndarray | None,
    score_names: tuple[str, ...],
    target_transform: Literal["log", "inverse"],
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    scores = {}
    if "log_mean_growth" in score_names and len(local_candidate_indices):
        values = _read_native_samples(
            target,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )[:, local_candidate_indices]
        scores["log_mean_growth"] = _to_log_mean_growth(values, target_transform)

    requires_counts = any(name != "log_mean_growth" for name in score_names)
    contains_wild_type = variant_start <= wild_type_index < variant_stop
    if requires_counts and (len(local_candidate_indices) or contains_wild_type):
        starting = _read_native_samples(
            starting_variable,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )
        timepoint = _read_native_samples(
            timepoint_variable,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )
        if contains_wild_type:
            local_wild_type = wild_type_index - variant_start
            wild_type_starting = starting[:, :, local_wild_type]
            wild_type_timepoint = timepoint[:, :, local_wild_type]
        if wild_type_starting is None or wild_type_timepoint is None:
            raise RuntimeError("The wild-type count block must be processed first.")
        if len(local_candidate_indices):
            count_ratio, dimsum_fitness = _count_proxies(
                starting[:, :, local_candidate_indices],
                timepoint[:, :, local_candidate_indices],
                wild_type_starting,
                wild_type_timepoint,
            )
            if "count_ratio" in score_names:
                scores["count_ratio"] = count_ratio.astype(np.float32, copy=False)
            if "dimsum_fitness" in score_names:
                scores["dimsum_fitness"] = dimsum_fitness.astype(np.float32, copy=False)
    return scores, wild_type_starting, wild_type_timepoint


def _observed_counts_status(handle: h5netcdf.File, landscape: PDZLandscape) -> str:
    observed = handle.groups.get("observed_data")
    if observed is None:
        return "not_embedded"
    starting_matches = np.array_equal(
        observed.variables["starting_counts"][:],
        landscape.starting_counts,
    )
    timepoint_matches = np.array_equal(
        observed.variables["timepoint_counts"][:],
        landscape.timepoint_counts,
    )
    if not starting_matches or not timepoint_matches:
        raise ValueError(
            f"{landscape.source}: HMC observed counts differ from the raw count table."
        )
    return "matched"


def _tracked_summary(
    variants: pd.DataFrame,
    tracked_indices: np.ndarray,
    values: np.ndarray,
    ranks: np.ndarray,
    *,
    k: int,
    interval: tuple[float, float],
    include_reported_percentile: bool = True,
) -> pd.DataFrame:
    lower, upper = interval
    if not 0 <= lower < upper <= 1:
        raise ValueError("interval must contain increasing probabilities in [0, 1].")
    n_finite = np.isfinite(ranks).sum(axis=0)
    summary = pd.DataFrame({"variant_index": tracked_indices})
    with np.errstate(invalid="ignore"):
        summary["posterior_value_mean"] = np.nanmean(values, axis=0)
        summary["posterior_value_sd"] = np.nanstd(values, axis=0, ddof=1)
        summary[["posterior_value_lower", "posterior_value_upper"]] = np.nanquantile(
            values,
            [lower, upper],
            axis=0,
        ).T
        summary["posterior_mean_rank"] = np.nanmean(ranks, axis=0)
        summary["posterior_median_rank"] = np.nanmedian(ranks, axis=0)
        summary["posterior_rank_sd"] = np.nanstd(ranks, axis=0, ddof=1)
        summary[["posterior_rank_lower", "posterior_rank_upper"]] = np.nanquantile(
            ranks,
            [lower, upper],
            axis=0,
        ).T
        summary["posterior_top_k_probability"] = np.nanmean(
            np.where(np.isfinite(ranks), ranks <= k, np.nan),
            axis=0,
        )
    if include_reported_percentile:
        reported_rank = variants.iloc[tracked_indices]["reported_rank"].to_numpy(
            dtype=float
        )
        valid = np.isfinite(ranks)
        better = ((ranks < reported_rank[None, :]) & valid).sum(axis=0)
        tied = ((ranks == reported_rank[None, :]) & valid).sum(axis=0)
        summary["reported_rank_percentile"] = np.divide(
            better + 0.5 * tied,
            n_finite,
            out=np.full(len(tracked_indices), np.nan),
            where=n_finite > 0,
        )
    summary["n_finite_samples"] = n_finite
    return summary


def _combined_tracked_summary(
    reported: pd.DataFrame,
    model: pd.DataFrame | None,
) -> pd.DataFrame:
    common_columns = [
        column
        for column in reported.columns
        if column not in {"variant_index", "reported_rank_percentile"}
    ]
    combined = reported.set_index("variant_index").copy()
    combined["is_tracked_reported_cohort"] = True
    combined["is_tracked_model_cohort"] = False
    combined["model_reference_rank"] = np.nan
    combined["model_reference_value"] = np.nan
    if model is None:
        return combined.reset_index()

    model = model.set_index("variant_index")
    overlap = combined.index.intersection(model.index)
    for column in common_columns:
        if not np.allclose(
            combined.loc[overlap, column].to_numpy(dtype=float),
            model.loc[overlap, column].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"Reported and model cohort summaries disagree for {column!r}."
            )
    combined.loc[overlap, "is_tracked_model_cohort"] = True
    combined.loc[overlap, ["model_reference_rank", "model_reference_value"]] = (
        model.loc[overlap, ["model_reference_rank", "model_reference_value"]]
    )

    model_only = model.loc[~model.index.isin(combined.index)].copy()
    model_only["reported_rank_percentile"] = np.nan
    model_only["is_tracked_reported_cohort"] = False
    model_only["is_tracked_model_cohort"] = True
    combined = pd.concat([combined, model_only], axis=0, sort=False)
    return combined.reset_index()


def _plot_tables(
    selection: xr.Dataset,
    variants: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    top_k: tuple[int, ...],
    interval: tuple[float, float],
    n_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    overlap_frames = []
    tracked_indices = np.asarray(
        selection["tracked_variant_index"].values,
        dtype=np.int32,
    )
    for score_name in selection["score_name"].values.astype(str):
        proxy_selection = xr.Dataset(
            {
                "variant_index": selection["variant_index"].sel(score_name=score_name),
                "score": selection["score"].sel(score_name=score_name),
            }
        )
        tracked_values = selection["tracked_score"].sel(score_name=score_name).values
        tracked_ranks = selection["tracked_rank"].sel(score_name=score_name).values
        model_indices = None
        model_values = None
        model_ranks = None
        model_reference_values = None
        if "model_tracked_rank" in selection:
            model_indices = np.asarray(
                selection["model_tracked_variant_index"]
                .sel(score_name=score_name)
                .values,
                dtype=np.int32,
            )
            model_values = selection["model_tracked_score"].sel(
                score_name=score_name
            ).values
            model_ranks = selection["model_tracked_rank"].sel(
                score_name=score_name
            ).values
            model_reference_values = selection["model_reference_score"].sel(
                score_name=score_name
            ).values[: len(model_indices)]
        for selected_k in top_k:
            inclusion = summarize_inclusion(proxy_selection, variants, k=selected_k)
            fixed_indices = (
                tracked_indices
                if model_indices is None
                else np.union1d(tracked_indices, model_indices)
            )
            missing_tracked = np.setdiff1d(
                fixed_indices,
                inclusion["variant_index"].to_numpy(dtype=np.int32),
            )
            if len(missing_tracked):
                missing = variants.iloc[missing_tracked].copy()
                missing.insert(0, "variant_index", missing_tracked)
                missing["k"] = selected_k
                missing["top_k_count"] = 0
                missing["top_k_probability"] = 0.0
                inclusion = pd.concat([inclusion, missing], ignore_index=True)
            tracked = _tracked_summary(
                variants,
                tracked_indices,
                tracked_values,
                tracked_ranks,
                k=selected_k,
                interval=interval,
            )
            model = None
            if model_indices is not None:
                model = _tracked_summary(
                    variants,
                    model_indices,
                    model_values,
                    model_ranks,
                    k=selected_k,
                    interval=interval,
                    include_reported_percentile=False,
                )
                model["model_reference_rank"] = np.arange(
                    1,
                    len(model_indices) + 1,
                )
                model["model_reference_value"] = model_reference_values
            tracked = _combined_tracked_summary(tracked, model)
            inclusion = inclusion.merge(
                tracked,
                on="variant_index",
                how="left",
                validate="one_to_one",
            )
            for column in (
                "is_tracked_reported_cohort",
                "is_tracked_model_cohort",
            ):
                inclusion[column] = inclusion[column].fillna(False).astype(bool)
            inclusion.insert(0, "score", score_name)
            inclusion.insert(0, "fit_method", "hmc_posterior")
            for field, value in reversed(identity.metadata().items()):
                inclusion.insert(0, field, value)
            summary_frames.append(inclusion)

            pairwise = pairwise_overlap(
                proxy_selection,
                k=selected_k,
                n_pairs=n_pairs,
                seed=seed,
            )
            reference = reference_overlap(
                proxy_selection,
                variants["reference_fitness"],
                k=selected_k,
            )
            overlap = pd.concat([pairwise, reference], ignore_index=True)
            overlap.insert(0, "score", score_name)
            overlap.insert(0, "fit_method", "hmc_posterior")
            for field, value in reversed(identity.metadata().items()):
                overlap.insert(0, field, value)
            overlap_frames.append(overlap)
    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(overlap_frames, ignore_index=True),
    )


def _file_signature(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def scan_pdz_hmc(
    landscape: PDZLandscape,
    hmc_path: Path,
    *,
    top_k: tuple[int, ...] = (50, 100, 1000),
    retain_k: int = 1000,
    tracked_top_n: int = 1000,
    tracked_model_top_n: int = 0,
    score_names: tuple[str, ...] = PDZ_SCORE_NAMES,
    growth_rate_prior: PDZGrowthRatePrior = "exponential",
    growth_curve: PDZGrowthCurve = "exponential",
    n_samples: int | None = None,
    seed: int = 23,
    n_pairs: int = 10_000,
    interval: tuple[float, float] = (0.05, 0.95),
    fit_id: str | None = None,
    show_progress: bool = True,
) -> StabilityResult:
    """Scan every reported candidate in a full SciStanPy PDZ HMC fit."""
    hmc_path = Path(hmc_path)
    provenance = PDZ_LIBRARY_PROVENANCE[landscape.source]
    target_name, target_transform = _growth_target(growth_rate_prior)
    model = f"{growth_rate_prior.title()}-{growth_curve.title()}"
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape=f"pdz-{landscape.source}",
        inference_method="hmc",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or hmc_path.stem,
    )
    top_k = _top_k_values(top_k)
    if not top_k[-1] <= retain_k <= len(landscape.variants):
        raise ValueError(
            "retain_k must be at least the largest top_k and no larger than "
            "the candidate count."
        )
    if not 1 <= tracked_top_n <= len(landscape.variants):
        raise ValueError("tracked_top_n must not exceed the candidate count.")
    if not 0 <= tracked_model_top_n <= retain_k:
        raise ValueError("tracked_model_top_n must be between 0 and retain_k.")
    if not score_names or len(set(score_names)) != len(score_names):
        raise ValueError("score_names must be non-empty and unique.")
    unknown_scores = set(score_names) - set(PDZ_SCORE_NAMES)
    if unknown_scores:
        raise ValueError(f"Unknown PDZ scores: {sorted(unknown_scores)}")

    started = time.monotonic()
    variants = landscape.variants
    candidate_hmc_indices = variants["hmc_variant_index"].to_numpy(dtype=np.int64)
    if np.any(np.diff(candidate_hmc_indices) <= 0):
        raise ValueError("PDZ HMC candidate indices must be strictly increasing.")
    reported_order = np.lexsort(
        (
            np.arange(len(variants), dtype=np.int64),
            -variants["reported_fitness"].to_numpy(dtype=float),
        )
    )
    tracked_indices = reported_order[:tracked_top_n].astype(np.int32)
    tracked_hmc_indices = candidate_hmc_indices[tracked_indices]
    model_reference_indices: dict[str, np.ndarray] = {}
    model_reference_scores: dict[str, np.ndarray] = {}
    model_tracked_indices: dict[str, np.ndarray] = {}
    model_tracked_values: dict[str, np.ndarray] = {}
    model_tracked_ranks: dict[str, np.ndarray] = {}

    with h5netcdf.File(hmc_path, "r") as handle:
        observed_counts_status = _observed_counts_status(handle, landscape)
        posterior = handle.groups["posterior"].variables
        predictive = handle.groups["posterior_predictive"].variables
        if target_name not in posterior:
            raise ValueError(f"{model} fit must contain posterior/{target_name}.")
        target = posterior[target_name]
        starting_variable = predictive["starting_counts"]
        timepoint_variable = predictive["timepoint_counts"]
        if target.shape[-1] != landscape.n_hmc_variants:
            raise ValueError("HMC target width differs from the raw count-table width.")
        variant_chunks = {
            int((variable.chunks or variable.shape)[-1])
            for variable in (target, starting_variable, timepoint_variable)
        }
        variant_chunk_size = int(np.gcd.reduce(list(variant_chunks)))
        n_chains, draws_per_chain = target.shape[:2]
        sample_ids = _sample_ids(
            n_chains * draws_per_chain,
            n_samples,
            seed=seed,
        )

        tracked_values = {
            name: np.full((len(sample_ids), tracked_top_n), np.nan, dtype=np.float32)
            for name in score_names
        }
        top_indices = {}
        top_scores = {}
        tracked_ranks = {}
        model_reference_values = {}
        for name in score_names:
            top_indices[name], top_scores[name] = empty_top_k(len(sample_ids), retain_k)
            tracked_ranks[name] = empty_tracked_ranks(len(sample_ids), tracked_top_n)
            if tracked_model_top_n:
                model_reference_values[name] = np.full(
                    len(variants),
                    np.nan,
                    dtype=np.float32,
                )

        block_starts = list(range(0, landscape.n_hmc_variants, variant_chunk_size))
        wild_type_block = (landscape.wild_type_index // variant_chunk_size) * variant_chunk_size
        block_starts.remove(wild_type_block)
        block_starts.insert(0, wild_type_block)
        tracked_blocks = np.unique(
            (tracked_hmc_indices // variant_chunk_size) * variant_chunk_size
        ).tolist()
        if wild_type_block in tracked_blocks:
            tracked_blocks.remove(wild_type_block)
        tracked_blocks.insert(0, wild_type_block)

        if show_progress:
            print(
                f"pdz-{landscape.source}: {len(sample_ids):,} full-fit HMC samples, "
                f"{len(variants):,} reported candidates, {len(block_starts)} native chunks",
                flush=True,
            )
            print(
                f"  pass 1: collect {tracked_top_n:,} fixed reported-cohort values",
                flush=True,
            )
        wild_type_starting = None
        wild_type_timepoint = None
        for block_number, variant_start in enumerate(tracked_blocks, start=1):
            variant_stop = min(variant_start + variant_chunk_size, landscape.n_hmc_variants)
            cohort_columns = np.flatnonzero(
                (tracked_hmc_indices >= variant_start) & (tracked_hmc_indices < variant_stop)
            )
            local_indices = tracked_hmc_indices[cohort_columns] - variant_start
            scores, wild_type_starting, wild_type_timepoint = _block_scores(
                target=target,
                starting_variable=starting_variable,
                timepoint_variable=timepoint_variable,
                sample_ids=sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=variant_start,
                variant_stop=variant_stop,
                local_candidate_indices=local_indices,
                wild_type_index=landscape.wild_type_index,
                wild_type_starting=wild_type_starting,
                wild_type_timepoint=wild_type_timepoint,
                score_names=score_names,
                target_transform=target_transform,
            )
            for name, values in scores.items():
                tracked_values[name][:, cohort_columns] = values
            if show_progress and (block_number == len(tracked_blocks) or block_number % 8 == 0):
                print(
                    f"    tracked chunks {block_number}/{len(tracked_blocks)}",
                    flush=True,
                )

        if show_progress:
            print("  pass 2: scan every reported candidate", flush=True)
        wild_type_starting = None
        wild_type_timepoint = None
        for block_number, variant_start in enumerate(block_starts, start=1):
            variant_stop = min(variant_start + variant_chunk_size, landscape.n_hmc_variants)
            candidate_start = int(
                np.searchsorted(candidate_hmc_indices, variant_start, side="left")
            )
            candidate_stop = int(np.searchsorted(candidate_hmc_indices, variant_stop, side="left"))
            contains_wild_type = variant_start <= landscape.wild_type_index < variant_stop
            if candidate_start == candidate_stop and not contains_wild_type:
                continue
            local_indices = candidate_hmc_indices[candidate_start:candidate_stop] - variant_start
            result_indices = np.arange(candidate_start, candidate_stop, dtype=np.int32)
            scores, wild_type_starting, wild_type_timepoint = _block_scores(
                target=target,
                starting_variable=starting_variable,
                timepoint_variable=timepoint_variable,
                sample_ids=sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=variant_start,
                variant_stop=variant_stop,
                local_candidate_indices=local_indices,
                wild_type_index=landscape.wild_type_index,
                wild_type_starting=wild_type_starting,
                wild_type_timepoint=wild_type_timepoint,
                score_names=score_names,
                target_transform=target_transform,
            )
            for name, values in scores.items():
                top_indices[name], top_scores[name] = merge_top_k(
                    top_indices[name],
                    top_scores[name],
                    result_indices,
                    values,
                )
                accumulate_tracked_ranks(
                    tracked_ranks[name],
                    values,
                    tracked_values[name],
                )
                if tracked_model_top_n:
                    finite = np.isfinite(values)
                    usable = finite.any(axis=0)
                    reference = np.full(values.shape[1], np.nan, dtype=np.float32)
                    reference[usable] = np.nanmedian(values[:, usable], axis=0)
                    model_reference_values[name][candidate_start:candidate_stop] = (
                        reference
                    )
            if show_progress and (block_number == len(block_starts) or block_number % 8 == 0):
                print(
                    f"    full chunks {block_number}/{len(block_starts)} "
                    f"({(time.monotonic() - started) / 60:.1f} min)",
                    flush=True,
                )

        if tracked_model_top_n:
            model_tracked_hmc_indices = {}
            for name in score_names:
                (
                    model_reference_indices[name],
                    model_reference_scores[name],
                ) = select_reference_top_k(
                    model_reference_values[name][None, :],
                    k=retain_k,
                )
                model_tracked_indices[name] = model_reference_indices[name][
                    :tracked_model_top_n
                ]
                model_tracked_hmc_indices[name] = candidate_hmc_indices[
                    model_tracked_indices[name]
                ]
                model_tracked_values[name] = np.full(
                    (len(sample_ids), tracked_model_top_n),
                    np.nan,
                    dtype=np.float32,
                )
                model_tracked_ranks[name] = empty_tracked_ranks(
                    len(sample_ids),
                    tracked_model_top_n,
                )

            target_hmc_indices = np.unique(
                np.concatenate(list(model_tracked_hmc_indices.values()))
            )
            model_blocks = np.unique(
                (target_hmc_indices // variant_chunk_size) * variant_chunk_size
            ).tolist()
            if wild_type_block in model_blocks:
                model_blocks.remove(wild_type_block)
            model_blocks.insert(0, wild_type_block)

            if show_progress:
                print(
                    f"  pass 3: collect {tracked_model_top_n:,} model-cohort values",
                    flush=True,
                )
            wild_type_starting = None
            wild_type_timepoint = None
            for block_number, variant_start in enumerate(model_blocks, start=1):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    landscape.n_hmc_variants,
                )
                block_targets = target_hmc_indices[
                    (target_hmc_indices >= variant_start)
                    & (target_hmc_indices < variant_stop)
                ]
                local_indices = block_targets - variant_start
                scores, wild_type_starting, wild_type_timepoint = _block_scores(
                    target=target,
                    starting_variable=starting_variable,
                    timepoint_variable=timepoint_variable,
                    sample_ids=sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                    local_candidate_indices=local_indices,
                    wild_type_index=landscape.wild_type_index,
                    wild_type_starting=wild_type_starting,
                    wild_type_timepoint=wild_type_timepoint,
                    score_names=score_names,
                    target_transform=target_transform,
                )
                for name, values in scores.items():
                    cohort_columns = np.flatnonzero(
                        (model_tracked_hmc_indices[name] >= variant_start)
                        & (model_tracked_hmc_indices[name] < variant_stop)
                    )
                    positions = np.searchsorted(
                        block_targets,
                        model_tracked_hmc_indices[name][cohort_columns],
                    )
                    model_tracked_values[name][:, cohort_columns] = values[:, positions]
                if show_progress and (
                    block_number == len(model_blocks) or block_number % 8 == 0
                ):
                    print(
                        f"    model chunks {block_number}/{len(model_blocks)}",
                        flush=True,
                    )

            if show_progress:
                print("  pass 4: compute exact model-cohort ranks", flush=True)
            wild_type_starting = None
            wild_type_timepoint = None
            for block_number, variant_start in enumerate(block_starts, start=1):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    landscape.n_hmc_variants,
                )
                candidate_start = int(
                    np.searchsorted(candidate_hmc_indices, variant_start, side="left")
                )
                candidate_stop = int(
                    np.searchsorted(candidate_hmc_indices, variant_stop, side="left")
                )
                contains_wild_type = (
                    variant_start <= landscape.wild_type_index < variant_stop
                )
                if candidate_start == candidate_stop and not contains_wild_type:
                    continue
                local_indices = (
                    candidate_hmc_indices[candidate_start:candidate_stop]
                    - variant_start
                )
                scores, wild_type_starting, wild_type_timepoint = _block_scores(
                    target=target,
                    starting_variable=starting_variable,
                    timepoint_variable=timepoint_variable,
                    sample_ids=sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                    local_candidate_indices=local_indices,
                    wild_type_index=landscape.wild_type_index,
                    wild_type_starting=wild_type_starting,
                    wild_type_timepoint=wild_type_timepoint,
                    score_names=score_names,
                    target_transform=target_transform,
                )
                for name, values in scores.items():
                    accumulate_tracked_ranks(
                        model_tracked_ranks[name],
                        values,
                        model_tracked_values[name],
                    )
                if show_progress and (
                    block_number == len(block_starts) or block_number % 8 == 0
                ):
                    print(
                        f"    rank chunks {block_number}/{len(block_starts)} "
                        f"({(time.monotonic() - started) / 60:.1f} min)",
                        flush=True,
                    )

    for name in score_names:
        if np.any(top_indices[name] < 0):
            raise ValueError(f"{name}: fewer than retain_k finite scores in a sample.")
        tracked_ranks[name][~np.isfinite(tracked_values[name])] = np.nan
        if tracked_model_top_n:
            model_tracked_ranks[name][
                ~np.isfinite(model_tracked_values[name])
            ] = np.nan
    selection_data = {
            "variant_index": (
                ("score_name", "sample", "selection_rank"),
                np.stack([top_indices[name] for name in score_names]),
            ),
            "score": (
                ("score_name", "sample", "selection_rank"),
                np.stack([top_scores[name] for name in score_names]),
            ),
            "tracked_score": (
                ("score_name", "sample", "tracked_variant"),
                np.stack([tracked_values[name] for name in score_names]),
            ),
            "tracked_rank": (
                ("score_name", "sample", "tracked_variant"),
                np.stack([tracked_ranks[name] for name in score_names]),
            ),
        }
    selection_coords = {
            "score_name": list(score_names),
            "sample": np.arange(len(sample_ids), dtype=np.int32),
            "sample_id": ("sample", sample_ids),
            "selection_rank": np.arange(1, retain_k + 1, dtype=np.int32),
            "tracked_variant": np.arange(tracked_top_n, dtype=np.int32),
            "tracked_variant_index": ("tracked_variant", tracked_indices),
        }
    if tracked_model_top_n:
        selection_data.update(
            {
                "model_reference_variant_index": (
                    ("score_name", "model_reference_rank"),
                    np.stack([model_reference_indices[name] for name in score_names]),
                ),
                "model_reference_score": (
                    ("score_name", "model_reference_rank"),
                    np.stack([model_reference_scores[name] for name in score_names]),
                ),
                "model_tracked_variant_index": (
                    ("score_name", "model_tracked_variant"),
                    np.stack([model_tracked_indices[name] for name in score_names]),
                ),
                "model_tracked_score": (
                    ("score_name", "sample", "model_tracked_variant"),
                    np.stack([model_tracked_values[name] for name in score_names]),
                ),
                "model_tracked_rank": (
                    ("score_name", "sample", "model_tracked_variant"),
                    np.stack([model_tracked_ranks[name] for name in score_names]),
                ),
            }
        )
        selection_coords.update(
            {
                "model_reference_rank": np.arange(
                    1,
                    retain_k + 1,
                    dtype=np.int32,
                ),
                "model_tracked_variant": np.arange(
                    tracked_model_top_n,
                    dtype=np.int32,
                ),
            }
        )
    selection = xr.Dataset(
        selection_data,
        coords=selection_coords,
        attrs={
            **identity.metadata(),
            "geo_library": provenance["geo_library"],
            "variant_molecule": provenance["variant_molecule"],
            "fit_method": "hmc_posterior",
            "model": model,
            "retained_k": retain_k,
        },
    )
    summary, overlaps = _plot_tables(
        selection,
        variants,
        identity=identity,
        top_k=top_k,
        interval=interval,
        n_pairs=n_pairs,
        seed=seed,
    )
    elapsed_seconds = time.monotonic() - started
    metadata: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **identity.metadata(),
        "source": landscape.source,
        **provenance,
        "fit_method": "hmc_posterior",
        "model": model,
        "growth_rate_prior": growth_rate_prior,
        "growth_curve": growth_curve,
        "posterior_variable": target_name,
        "posterior_transform": target_transform,
        "scores": list(score_names),
        "n_samples": len(sample_ids),
        "n_variants": len(variants),
        "n_hmc_variants": landscape.n_hmc_variants,
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "interval": list(interval),
        "seed": seed,
        "reference_fitness_source": "reported",
        "reported_fitness_available": True,
        "posterior_predictive_count_source": "stored SciStanPy posterior_predictive",
        "observed_counts_status": observed_counts_status,
        "hmc_variant_block_size": variant_chunk_size,
        "hmc_file": _file_signature(hmc_path),
        "scistanpy_version": version("scistanpy"),
        "elapsed_seconds": elapsed_seconds,
        "warnings": [],
    }
    if tracked_model_top_n:
        metadata.update(
            {
                "model_reference_statistic": "sample_median_score",
                "model_reference_tie_break": "variant_index_ascending",
                "model_reference_retained_k": retain_k,
            }
        )
    return StabilityResult(
        metadata=metadata,
        variants=variants.copy(),
        selection=selection,
        summary=summary,
        overlaps=overlaps,
    )


scan_pdz_trans_hmc = scan_pdz_hmc
