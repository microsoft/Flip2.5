"""Count-derived score definitions used by the fit examples."""

from __future__ import annotations

import numpy as np


def pdz_dimsum_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    wild_type_index: int,
) -> np.ndarray:
    """Calculate inverse-variance weighted PDZ log fitness."""
    starting = np.asarray(starting_counts)
    timepoint = np.asarray(timepoint_counts)
    if starting.ndim < 2 or starting.shape != timepoint.shape:
        raise ValueError(
            "PDZ counts must have matching (..., replicate, variant) shapes."
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
    wild_type_starting = flat_starting[:, :, wild_type_index, None]
    wild_type_timepoint = flat_timepoint[:, :, wild_type_index, None]
    valid = (
        (flat_starting > 0)
        & (flat_timepoint > 0)
        & (wild_type_starting > 0)
        & (wild_type_timepoint > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        replicate_fitness = np.log(flat_timepoint / flat_starting) - np.log(
            wild_type_timepoint / wild_type_starting
        )
        variance = (
            1 / flat_starting
            + 1 / flat_timepoint
            + 1 / wild_type_starting
            + 1 / wild_type_timepoint
        )
        weights = np.where(valid, 1 / variance, 0.0)
    weighted_fitness = np.zeros_like(replicate_fitness)
    np.multiply(replicate_fitness, weights, out=weighted_fitness, where=valid)
    weight_sum = weights.sum(axis=1)
    fitness = np.divide(
        weighted_fitness.sum(axis=1),
        weight_sum,
        out=np.full_like(weight_sum, np.nan),
        where=weight_sum > 0,
    )
    return fitness.reshape(*sample_shape, n_variants)


def trpb_three_site_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    variants: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Calculate the deposited TrpB libA-I amino-acid fitness."""
    starting = np.asarray(starting_counts, dtype=float)
    timepoint = np.asarray(timepoint_counts, dtype=float)
    variant_values = np.asarray(variants, dtype=str)
    time_values = np.asarray(times, dtype=float)
    if starting.ndim < 1:
        raise ValueError("Three-site starting counts require a variant axis.")
    if timepoint.ndim == starting.ndim + 1:
        if timepoint.shape[:-2] != starting.shape[:-1]:
            raise ValueError("Three-site starting and timepoint sample axes differ.")
        timepoint = timepoint[..., :, None, :]
    elif timepoint.ndim == starting.ndim + 2:
        if timepoint.shape[:-3] != starting.shape[:-1]:
            raise ValueError("Three-site starting and timepoint sample axes differ.")
    else:
        raise ValueError("Three-site timepoint count axes are invalid.")
    if (
        timepoint.shape[-1] != starting.shape[-1]
        or time_values.shape != (timepoint.shape[-3],)
        or variant_values.shape != (starting.shape[-1],)
    ):
        raise ValueError("Three-site count, time, and variant axes must align.")
    if timepoint.shape[-2] not in {1, 2}:
        raise ValueError("Three-site output counts require one or two replicates.")
    if (
        not np.isfinite(starting).all()
        or not np.isfinite(timepoint).all()
        or np.any(starting < 0)
        or np.any(timepoint < 0)
        or not np.isfinite(time_values).all()
        or np.any(time_values <= 0)
    ):
        raise ValueError("Three-site counts and times must be finite and valid.")

    stop_mask = np.char.find(variant_values, "*") >= 0
    if not stop_mask.any():
        raise ValueError("Three-site fitness requires stop variants.")
    valid = (starting >= 10)[..., None, None, :] & (
        timepoint > 0
    ).all(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(timepoint / starting[..., None, None, :])
    log_ratio = np.where(valid, log_ratio, np.nan)
    stop_mean = np.nanmean(log_ratio[..., stop_mask], axis=-1)
    maximum = np.nanmax(log_ratio, axis=-1)
    normalized = (log_ratio - stop_mean[..., None]) / (
        maximum - stop_mean
    )[..., None]
    finite_count = np.isfinite(normalized).sum(axis=(-3, -2))
    return np.divide(
        np.nansum(normalized, axis=(-3, -2)),
        finite_count,
        out=np.full(starting.shape, np.nan),
        where=finite_count > 0,
    )


def trpb_four_site_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    variants: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Calculate the deposited TrpB four-site amino-acid fitness."""
    starting = np.asarray(starting_counts, dtype=float)
    timepoint = np.asarray(timepoint_counts, dtype=float)
    variant_values = np.asarray(variants, dtype=str)
    time_values = np.asarray(times, dtype=float)
    if starting.ndim < 2 or timepoint.ndim != starting.ndim + 1:
        raise ValueError("Four-site TrpB count axes are invalid.")
    if (
        timepoint.shape[:-3] != starting.shape[:-2]
        or timepoint.shape[-2:] != starting.shape[-2:]
        or time_values.shape != (timepoint.shape[-3],)
        or variant_values.shape != (starting.shape[-1],)
    ):
        raise ValueError("Four-site count, time, and variant axes must align.")
    if starting.shape[-2] != 2:
        raise ValueError("Four-site fitness requires two replicates.")
    if not np.array_equal(time_values, np.array([12, 16, 20, 24, 28, 36])):
        raise ValueError("Four-site fitness requires times 12, 16, 20, 24, 28, 36.")
    if (
        not np.isfinite(starting).all()
        or not np.isfinite(timepoint).all()
        or np.any(starting < 0)
        or np.any(timepoint < 0)
    ):
        raise ValueError("Four-site counts must be finite and nonnegative.")

    stop_mask = np.char.find(variant_values, "*") >= 0
    if not stop_mask.any():
        raise ValueError("Four-site fitness requires stop variants.")
    retained_times = time_values != 12
    average_input = starting.mean(axis=-2)
    valid = (average_input >= 10)[..., None, None, :] & (
        timepoint > 0
    ).all(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(timepoint / average_input[..., None, None, :])
    log_ratio = np.where(valid, log_ratio, np.nan)
    stop_mean = np.nanmean(log_ratio[..., stop_mask], axis=-1)
    maximum = np.nanmax(log_ratio, axis=-1)
    normalized = (log_ratio - stop_mean[..., None]) / (
        maximum - stop_mean
    )[..., None]
    retained = normalized[..., retained_times, :, :]
    finite_count = np.isfinite(retained).sum(axis=(-3, -2))
    return np.divide(
        np.nansum(retained, axis=(-3, -2)),
        finite_count,
        out=np.full(starting.shape[:-2] + (starting.shape[-1],), np.nan),
        where=finite_count > 0,
    )


def nuc_g3_highest_gate_enrichment(
    input_counts: np.ndarray,
    output_counts: np.ndarray,
) -> np.ndarray:
    """Calculate output-gate frequency divided by input frequency for NucB G3."""
    input_values = np.asarray(input_counts, dtype=float)
    output_values = np.asarray(output_counts, dtype=float)
    if (
        input_values.ndim == 0
        or output_values.ndim != input_values.ndim + 1
        or output_values.shape[-2] != 4
        or output_values.shape[:-2] != input_values.shape[:-1]
        or output_values.shape[-1] != input_values.shape[-1]
    ):
        raise ValueError("NucB G3 input and four-gate output axes must align.")
    highest_gate = output_values[..., 3, :]
    if (
        not np.isfinite(input_values).all()
        or not np.isfinite(highest_gate).all()
        or np.any(input_values < 0)
        or np.any(highest_gate < 0)
    ):
        raise ValueError("NucB G3 counts must be finite and nonnegative.")
    input_totals = input_values.sum(axis=-1, keepdims=True)
    output_totals = highest_gate.sum(axis=-1, keepdims=True)
    if np.any(input_totals <= 0) or np.any(output_totals <= 0):
        raise ValueError("Every NucB count sample must have positive totals.")
    input_frequency = input_values / input_totals
    output_frequency = highest_gate / output_totals
    return np.divide(
        output_frequency,
        input_frequency,
        out=np.full_like(output_frequency, np.nan),
        where=input_frequency > 0,
    )
