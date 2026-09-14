"""Family-specific predictive-count readers for the fit examples."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from preprocessing.fitness import (
    nuc_g3_highest_gate_enrichment,
    pdz_dimsum_fitness,
)
from preprocessing.scores import (
    publish_scores,
    read_native_samples,
    select_sample_ids,
)


def _temporary_matrix(
    output: str | Path,
    shape: tuple[int, int],
) -> tuple[Path, np.memmap]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}-scores-",
        suffix=".dat",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    return path, np.memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _predictive_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    dataset = handle.get(f"posterior_predictive/{name}")
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Fit is missing posterior_predictive/{name}.")
    return dataset


def _observed_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    dataset = handle.get(f"observed_data/{name}")
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Fit is missing observed_data/{name}.")
    return dataset


def extract_pdz_predictive_scores(
    fit_path: str | Path,
    variants: pd.DataFrame,
    output: str | Path,
    *,
    wild_type_variant_id: str,
    sample_count: int | None,
    seed: int = 23,
    sample_chunk_size: int = 64,
    preserve_chain_draw: bool,
    overwrite: bool = False,
) -> Path:
    """Extract DiMSum-style scores from PDZ predictive counts."""
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    matches = np.flatnonzero(
        variants["variant_id"].to_numpy() == wild_type_variant_id
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one PDZ wild type {wild_type_variant_id!r}.")
    wild_type_index = int(matches[0])
    temporary_path: Path | None = None
    matrix: np.memmap | None = None
    try:
        with h5py.File(fit_path) as handle:
            starting = _predictive_dataset(handle, "starting_counts")
            timepoint = _predictive_dataset(handle, "timepoint_counts")
            observed_starting = _observed_dataset(handle, "starting_counts")[:]
            observed_timepoint = _observed_dataset(handle, "timepoint_counts")[:]
            if (
                starting.ndim != 4
                or timepoint.shape != starting.shape
                or starting.shape[2:] != observed_starting.shape
                or observed_timepoint.shape != observed_starting.shape
                or starting.shape[-1] != len(variants)
            ):
                raise ValueError("PDZ predictive and observed count axes differ.")
            n_chains, draws_per_chain = map(int, starting.shape[:2])
            selected = select_sample_ids(
                n_chains * draws_per_chain,
                sample_count,
                seed=seed,
            )
            block_size = min(
                int((starting.chunks or starting.shape)[-1]),
                int((timepoint.chunks or timepoint.shape)[-1]),
            )
            temporary_path, matrix = _temporary_matrix(
                output,
                (len(selected), len(variants)),
            )
            expected_starting_totals = observed_starting.sum(axis=-1, dtype=np.int64)
            expected_timepoint_totals = observed_timepoint.sum(axis=-1, dtype=np.int64)
            for sample_start in range(0, len(selected), sample_chunk_size):
                sample_stop = min(sample_start + sample_chunk_size, len(selected))
                sample_ids = selected[sample_start:sample_stop]
                wild_type_starting = read_native_samples(
                    starting,
                    sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=wild_type_index,
                    variant_stop=wild_type_index + 1,
                )
                wild_type_timepoint = read_native_samples(
                    timepoint,
                    sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=wild_type_index,
                    variant_stop=wild_type_index + 1,
                )
                starting_totals = np.zeros(
                    (len(sample_ids), observed_starting.shape[0]),
                    dtype=np.int64,
                )
                timepoint_totals = np.zeros_like(starting_totals)
                for variant_start in range(0, len(variants), block_size):
                    variant_stop = min(variant_start + block_size, len(variants))
                    starting_block = read_native_samples(
                        starting,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    timepoint_block = read_native_samples(
                        timepoint,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    starting_totals += starting_block.sum(axis=-1, dtype=np.int64)
                    timepoint_totals += timepoint_block.sum(axis=-1, dtype=np.int64)
                    combined_starting = np.concatenate(
                        (starting_block, wild_type_starting),
                        axis=-1,
                    )
                    combined_timepoint = np.concatenate(
                        (timepoint_block, wild_type_timepoint),
                        axis=-1,
                    )
                    matrix[
                        sample_start:sample_stop,
                        variant_start:variant_stop,
                    ] = pdz_dimsum_fitness(
                        combined_starting,
                        combined_timepoint,
                        combined_starting.shape[-1] - 1,
                    )[..., :-1].astype(np.float32)
                if not np.all(
                    starting_totals == expected_starting_totals[None, :]
                ) or not np.all(
                    timepoint_totals == expected_timepoint_totals[None, :]
                ):
                    raise ValueError("PDZ predictive count totals differ from observed data.")
            matrix.flush()
            return publish_scores(
                variants,
                matrix,
                selected,
                output,
                draws_per_chain=draws_per_chain if preserve_chain_draw else None,
                overwrite=overwrite,
            )
    finally:
        if matrix is not None:
            matrix.flush()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_trpb_predictive_scores(
    fit_path: str | Path,
    variants: pd.DataFrame,
    output: str | Path,
    *,
    times: np.ndarray,
    four_site: bool,
    sample_count: int | None,
    seed: int = 23,
    sample_chunk_size: int = 64,
    preserve_chain_draw: bool,
    overwrite: bool = False,
) -> Path:
    """Extract deposited TrpB fitness from predictive counts."""
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    time_values = np.asarray(times, dtype=float)
    variant_ids = variants["variant_id"].to_numpy(dtype=str)
    stop_mask = np.char.find(variant_ids, "*") >= 0
    if not stop_mask.any():
        raise ValueError("TrpB predictive scores require stop variants.")
    retained_times = time_values != 12 if four_site else np.ones(len(time_values), bool)
    temporary_path: Path | None = None
    matrix: np.memmap | None = None
    try:
        with h5py.File(fit_path) as handle:
            starting = _predictive_dataset(handle, "starting_counts")
            timepoint = _predictive_dataset(handle, "timepoint_counts")
            observed_starting = _observed_dataset(handle, "starting_counts")[:]
            observed_timepoint = _observed_dataset(handle, "timepoint_counts")[:]
            if (
                starting.ndim < 3
                or starting.shape[2:] != observed_starting.shape
                or timepoint.shape[2:] != observed_timepoint.shape
                or starting.shape[:2] != timepoint.shape[:2]
                or starting.shape[-1] != len(variants)
                or timepoint.shape[-1] != len(variants)
                or observed_timepoint.shape[0] != len(time_values)
            ):
                raise ValueError("TrpB predictive and observed count axes differ.")
            if four_site and (
                observed_starting.ndim != 2
                or observed_starting.shape[-2] != 2
                or observed_timepoint.ndim != 3
            ):
                raise ValueError("Four-site TrpB counts require time and replicate axes.")
            if not four_site and (
                observed_starting.ndim != 1
                or observed_timepoint.ndim not in {2, 3}
            ):
                raise ValueError("Three-site TrpB count axes are invalid.")
            n_chains, draws_per_chain = map(int, starting.shape[:2])
            selected = select_sample_ids(
                n_chains * draws_per_chain,
                sample_count,
                seed=seed,
            )
            block_size = min(
                int((starting.chunks or starting.shape)[-1]),
                int((timepoint.chunks or timepoint.shape)[-1]),
            )
            temporary_path, matrix = _temporary_matrix(
                output,
                (len(selected), len(variants)),
            )
            expected_starting_totals = observed_starting.sum(axis=-1, dtype=np.int64)
            expected_timepoint_totals = observed_timepoint.sum(axis=-1, dtype=np.int64)
            for sample_start in range(0, len(selected), sample_chunk_size):
                sample_stop = min(sample_start + sample_chunk_size, len(selected))
                sample_ids = selected[sample_start:sample_stop]
                sample_size = len(sample_ids)
                replicate_count = observed_starting.shape[-2] if four_site else (
                    observed_timepoint.shape[-2] if observed_timepoint.ndim == 3 else 1
                )
                stop_sum = np.zeros(
                    (sample_size, len(time_values), replicate_count),
                    dtype=float,
                )
                stop_count = np.zeros_like(stop_sum, dtype=np.int64)
                maximum = np.full_like(stop_sum, -np.inf)
                starting_totals = np.zeros(
                    (sample_size, *observed_starting.shape[:-1]),
                    dtype=np.int64,
                )
                timepoint_totals = np.zeros(
                    (sample_size, *observed_timepoint.shape[:-1]),
                    dtype=np.int64,
                )
                for variant_start in range(0, len(variants), block_size):
                    variant_stop = min(variant_start + block_size, len(variants))
                    starting_block = read_native_samples(
                        starting,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    timepoint_block = read_native_samples(
                        timepoint,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    starting_totals += starting_block.sum(axis=-1, dtype=np.int64)
                    timepoint_totals += timepoint_block.sum(axis=-1, dtype=np.int64)
                    average_input = (
                        starting_block.mean(axis=-2) if four_site else starting_block
                    )
                    local_stops = stop_mask[variant_start:variant_stop]
                    for time_index in range(len(time_values)):
                        output_block = timepoint_block[:, time_index]
                        if output_block.ndim == 2:
                            output_block = output_block[:, None, :]
                        valid = (average_input >= 10) & (
                            output_block > 0
                        ).all(axis=-2)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            log_ratio = np.log(
                                output_block / average_input[:, None, :]
                            )
                        log_ratio = np.where(valid[:, None, :], log_ratio, np.nan)
                        stop_values = log_ratio[..., local_stops]
                        stop_sum[:, time_index] += np.nansum(stop_values, axis=-1)
                        stop_count[:, time_index] += np.isfinite(stop_values).sum(axis=-1)
                        maximum[:, time_index] = np.maximum(
                            maximum[:, time_index],
                            np.max(
                                np.where(np.isfinite(log_ratio), log_ratio, -np.inf),
                                axis=-1,
                            ),
                        )
                if not np.array_equal(
                    starting_totals,
                    np.broadcast_to(expected_starting_totals, starting_totals.shape),
                ) or not np.array_equal(
                    timepoint_totals,
                    np.broadcast_to(expected_timepoint_totals, timepoint_totals.shape),
                ):
                    raise ValueError("TrpB predictive count totals differ from observed data.")
                if np.any(stop_count == 0):
                    raise ValueError("A TrpB predictive sample has no stop normalization.")
                stop_mean = stop_sum / stop_count
                normalization_range = maximum - stop_mean
                if not np.isfinite(normalization_range).all() or np.any(
                    normalization_range <= 0
                ):
                    raise ValueError("A TrpB predictive sample has invalid normalization.")

                for variant_start in range(0, len(variants), block_size):
                    variant_stop = min(variant_start + block_size, len(variants))
                    starting_block = read_native_samples(
                        starting,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    timepoint_block = read_native_samples(
                        timepoint,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    average_input = (
                        starting_block.mean(axis=-2) if four_site else starting_block
                    )
                    score_sum = np.zeros(
                        (sample_size, variant_stop - variant_start),
                        dtype=float,
                    )
                    score_count = np.zeros_like(score_sum, dtype=np.int16)
                    for time_index in np.flatnonzero(retained_times):
                        output_block = timepoint_block[:, time_index]
                        if output_block.ndim == 2:
                            output_block = output_block[:, None, :]
                        valid = (average_input >= 10) & (
                            output_block > 0
                        ).all(axis=-2)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            log_ratio = np.log(
                                output_block / average_input[:, None, :]
                            )
                        normalized = (
                            log_ratio - stop_mean[:, time_index, :, None]
                        ) / normalization_range[:, time_index, :, None]
                        normalized = np.where(valid[:, None, :], normalized, np.nan)
                        score_sum += np.nansum(normalized, axis=-2)
                        score_count += np.isfinite(normalized).sum(axis=-2)
                    matrix[
                        sample_start:sample_stop,
                        variant_start:variant_stop,
                    ] = np.divide(
                        score_sum,
                        score_count,
                        out=np.full_like(score_sum, np.nan),
                        where=score_count > 0,
                    ).astype(np.float32)
            matrix.flush()
            return publish_scores(
                variants,
                matrix,
                selected,
                output,
                draws_per_chain=draws_per_chain if preserve_chain_draw else None,
                overwrite=overwrite,
            )
    finally:
        if matrix is not None:
            matrix.flush()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_nuc_g3_predictive_scores(
    fit_path: str | Path,
    variants: pd.DataFrame,
    output: str | Path,
    *,
    sample_count: int | None,
    seed: int = 23,
    sample_chunk_size: int = 64,
    preserve_chain_draw: bool,
    overwrite: bool = False,
) -> Path:
    """Extract NucB G3 highest-gate enrichment from predictive counts."""
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    temporary_path: Path | None = None
    matrix: np.memmap | None = None
    try:
        with h5py.File(fit_path) as handle:
            input_counts = _predictive_dataset(handle, "ic1")
            output_counts = _predictive_dataset(handle, "oc1")
            observed_input = _observed_dataset(handle, "ic1")[:]
            observed_output = _observed_dataset(handle, "oc1")[:]
            if (
                input_counts.ndim != 3
                or output_counts.ndim != 4
                or input_counts.shape[:2] != output_counts.shape[:2]
                or input_counts.shape[2:] != observed_input.shape
                or output_counts.shape[2:] != observed_output.shape
                or observed_output.shape != (4, len(variants))
                or observed_input.shape != (len(variants),)
            ):
                raise ValueError("NucB G3 predictive and observed count axes differ.")
            n_chains, draws_per_chain = map(int, input_counts.shape[:2])
            selected = select_sample_ids(
                n_chains * draws_per_chain,
                sample_count,
                seed=seed,
            )
            temporary_path, matrix = _temporary_matrix(
                output,
                (len(selected), len(variants)),
            )
            expected_input_total = observed_input.sum(dtype=np.int64)
            expected_output_totals = observed_output.sum(axis=-1, dtype=np.int64)
            for sample_start in range(0, len(selected), sample_chunk_size):
                sample_stop = min(sample_start + sample_chunk_size, len(selected))
                sample_ids = selected[sample_start:sample_stop]
                input_block = read_native_samples(
                    input_counts,
                    sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=0,
                    variant_stop=len(variants),
                )
                output_block = read_native_samples(
                    output_counts,
                    sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=0,
                    variant_stop=len(variants),
                )
                if not np.all(
                    input_block.sum(axis=-1, dtype=np.int64) == expected_input_total
                ) or not np.all(
                    output_block.sum(axis=-1, dtype=np.int64)
                    == expected_output_totals[None, :]
                ):
                    raise ValueError("NucB predictive count totals differ from observed data.")
                matrix[sample_start:sample_stop] = nuc_g3_highest_gate_enrichment(
                    input_block,
                    output_block,
                ).astype(np.float32)
            matrix.flush()
            return publish_scores(
                variants,
                matrix,
                selected,
                output,
                draws_per_chain=draws_per_chain if preserve_chain_draw else None,
                overwrite=overwrite,
            )
    finally:
        if matrix is not None:
            matrix.flush()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
