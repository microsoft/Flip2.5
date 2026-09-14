"""Landscape-agnostic bounded-memory HMC stability scanning."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

from .analysis import _top_k_values
from .candidates import ranking_candidate_indices, ranking_candidate_mask
from .identity import AnalysisIdentity
from .stability import pairwise_overlap, reference_overlap, summarize_inclusion
from .streaming import (
    accumulate_tracked_ranks,
    empty_top_k,
    empty_tracked_ranks,
    merge_top_k,
    select_reference_top_k,
)

ScoreBlockReader = Callable[..., dict[str, np.ndarray]]


@dataclass(frozen=True)
class HMCScan:
    """Compact output from one landscape-independent HMC scan."""

    selection: xr.Dataset
    sample_ids: np.ndarray
    candidate_indices: np.ndarray
    reported_indices: np.ndarray


def sample_ids(total: int, n_samples: int | None, *, seed: int) -> np.ndarray:
    """Choose a sorted, reproducible subset of flattened chain/draw IDs."""
    resolved = total if n_samples is None else n_samples
    if not 1 <= resolved <= total:
        raise ValueError(f"n_samples must be between 1 and {total:,}.")
    if resolved == total:
        return np.arange(total, dtype=np.int64)
    chosen = np.random.default_rng(seed).choice(total, resolved, replace=False)
    return np.sort(chosen).astype(np.int64)


def read_native_samples(
    variable,
    selected_samples: np.ndarray,
    *,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
) -> np.ndarray:
    """Read flattened chain/draw samples while preserving native HDF5 chunks."""
    total_samples = variable.shape[0] * draws_per_chain
    if np.array_equal(selected_samples, np.arange(total_samples, dtype=np.int64)):
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

    chains = selected_samples // draws_per_chain
    draws = selected_samples % draws_per_chain
    result = np.empty(
        (
            len(selected_samples),
            *variable.shape[2:-1],
            variant_stop - variant_start,
        ),
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


def file_signature(path: Path) -> dict[str, str | int]:
    """Record the local identity of an HMC artifact."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def h5netcdf_compatibility(path: Path) -> tuple[bool, str | None]:
    """Check whether standard NetCDF tooling can open an HDF5 artifact."""
    try:
        with h5netcdf.File(path, "r"):
            pass
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


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
    if not len(tracked_indices):
        columns = [
            "posterior_value_mean",
            "posterior_value_sd",
            "posterior_value_lower",
            "posterior_value_upper",
            "posterior_mean_rank",
            "posterior_median_rank",
            "posterior_rank_sd",
            "posterior_rank_lower",
            "posterior_rank_upper",
            "posterior_top_k_probability",
        ]
        if include_reported_percentile:
            columns.append("reported_rank_percentile")
        summary = pd.DataFrame({"variant_index": tracked_indices})
        for column in columns:
            summary[column] = pd.Series(dtype=float)
        summary["n_finite_samples"] = pd.Series(dtype=np.int64)
        return summary
    finite_values = np.isfinite(values)
    finite_ranks = np.isfinite(ranks)
    n_finite_values = finite_values.sum(axis=0)
    n_finite = finite_ranks.sum(axis=0)
    summary = pd.DataFrame({"variant_index": tracked_indices})
    n_tracked = len(tracked_indices)
    value_mean = np.full(n_tracked, np.nan)
    value_sd = np.full(n_tracked, np.nan)
    value_interval = np.full((n_tracked, 2), np.nan)
    rank_mean = np.full(n_tracked, np.nan)
    rank_median = np.full(n_tracked, np.nan)
    rank_sd = np.full(n_tracked, np.nan)
    rank_interval = np.full((n_tracked, 2), np.nan)

    value_usable = n_finite_values > 0
    if value_usable.any():
        usable_values = values[:, value_usable]
        value_mean[value_usable] = np.nansum(usable_values, axis=0) / n_finite_values[
            value_usable
        ]
        value_interval[value_usable] = np.nanquantile(
            usable_values,
            [lower, upper],
            axis=0,
        ).T
    value_sd_usable = n_finite_values > 1
    if value_sd_usable.any():
        value_sd[value_sd_usable] = np.nanstd(
            values[:, value_sd_usable],
            axis=0,
            ddof=1,
        )

    rank_usable = n_finite > 0
    if rank_usable.any():
        usable_ranks = ranks[:, rank_usable]
        rank_mean[rank_usable] = np.nansum(usable_ranks, axis=0) / n_finite[
            rank_usable
        ]
        rank_median[rank_usable] = np.nanmedian(usable_ranks, axis=0)
        rank_interval[rank_usable] = np.nanquantile(
            usable_ranks,
            [lower, upper],
            axis=0,
        ).T
    rank_sd_usable = n_finite > 1
    if rank_sd_usable.any():
        rank_sd[rank_sd_usable] = np.nanstd(
            ranks[:, rank_sd_usable],
            axis=0,
            ddof=1,
        )

    summary["posterior_value_mean"] = value_mean
    summary["posterior_value_sd"] = value_sd
    summary[["posterior_value_lower", "posterior_value_upper"]] = value_interval
    summary["posterior_mean_rank"] = rank_mean
    summary["posterior_median_rank"] = rank_median
    summary["posterior_rank_sd"] = rank_sd
    summary[["posterior_rank_lower", "posterior_rank_upper"]] = rank_interval
    summary["posterior_top_k_probability"] = np.divide(
        ((ranks <= k) & finite_ranks).sum(axis=0),
        n_finite,
        out=np.full(n_tracked, np.nan),
        where=n_finite > 0,
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


def _legacy_tracked_summary(
    variants: pd.DataFrame,
    tracked_indices: np.ndarray,
    values: np.ndarray,
    ranks: np.ndarray,
    *,
    k: int,
    interval: tuple[float, float],
    include_reported_percentile: bool = True,
) -> pd.DataFrame:
    """Reproduce the original direct-HMC NumPy reduction path exactly."""
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


def result_tables(
    selection: xr.Dataset,
    variants: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    top_k: tuple[int, ...],
    interval: tuple[float, float],
    n_pairs: int,
    seed: int,
    fit_method: str = "hmc_posterior",
    include_reference_overlap: bool = True,
    reference_higher_is_better: bool = True,
    legacy_numpy_reductions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build common inclusion, rank, and overlap tables from a streamed scan."""
    tracked_indices = np.asarray(
        selection["tracked_variant_index"].values,
        dtype=np.int32,
    )
    summary_frames = []
    overlap_frames = []
    tracked_summary = (
        _legacy_tracked_summary
        if legacy_numpy_reductions
        else _tracked_summary
    )
    for score_name in selection["score_name"].values.astype(str):
        proxy_selection = xr.Dataset(
            {
                "variant_index": selection["variant_index"].sel(
                    score_name=score_name
                ),
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
            tracked = tracked_summary(
                variants,
                tracked_indices,
                tracked_values,
                tracked_ranks,
                k=selected_k,
                interval=interval,
            )
            model = None
            if model_indices is not None:
                model = tracked_summary(
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
            inclusion.insert(0, "fit_method", fit_method)
            for field, value in reversed(identity.metadata().items()):
                inclusion.insert(0, field, value)
            summary_frames.append(inclusion)

            pairwise = pairwise_overlap(
                proxy_selection,
                k=selected_k,
                n_pairs=n_pairs,
                seed=seed,
            )
            overlap = pairwise
            if include_reference_overlap:
                reference = reference_overlap(
                    proxy_selection,
                    variants["reported_fitness"].where(
                        ranking_candidate_mask(variants)
                    ),
                    k=selected_k,
                    higher_is_better=reference_higher_is_better,
                )
                overlap = pd.concat([pairwise, reference], ignore_index=True)
            overlap.insert(0, "score", score_name)
            overlap.insert(0, "fit_method", fit_method)
            for field, value in reversed(identity.metadata().items()):
                overlap.insert(0, field, value)
            overlap_frames.append(overlap)
    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(overlap_frames, ignore_index=True),
    )


def _ordered_blocks(
    blocks: list[int],
    required_blocks: tuple[int, ...],
) -> list[int]:
    selected = set(blocks)
    selected.update(required_blocks)
    remaining = (block for block in sorted(selected) if block not in required_blocks)
    return [*required_blocks, *remaining]


def scan_hmc_scores(
    variants: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    score_names: tuple[str, ...],
    higher_is_better: Mapping[str, bool] | None = None,
    n_hmc_variants: int,
    n_chains: int,
    draws_per_chain: int,
    variant_chunk_size: int,
    read_scores: ScoreBlockReader,
    top_k: tuple[int, ...],
    retain_k: int,
    tracked_top_n: int,
    tracked_model_top_n: int,
    n_samples: int | None,
    seed: int,
    label: str,
    selection_attrs: Mapping[str, object] | None = None,
    reset_reader: Callable[[], None] | None = None,
    required_block_starts: tuple[int, ...] = (),
    min_model_reference_samples: int = 1,
    require_reported_top_k: bool = True,
    reported_higher_is_better: bool = True,
    show_progress: bool = True,
    fit_method: str = "hmc_posterior",
    sample_kind: str = "HMC",
) -> HMCScan:
    """Scan arbitrary HMC score blocks over a fitted variant axis."""
    if not score_names or len(set(score_names)) != len(score_names):
        raise ValueError("score_names must be non-empty and unique.")
    direction_was_explicit = higher_is_better is not None
    score_directions = (
        {name: True for name in score_names}
        if higher_is_better is None
        else dict(higher_is_better)
    )
    if set(score_directions) != set(score_names) or any(
        not isinstance(value, bool) for value in score_directions.values()
    ):
        raise ValueError(
            "higher_is_better must map every score name exactly once to a boolean."
        )
    if not fit_method.strip() or not sample_kind.strip():
        raise ValueError("fit_method and sample_kind must be non-empty strings.")
    if "reported_fitness" not in variants:
        raise ValueError("variants must contain reported_fitness.")
    if variant_chunk_size < 1 or n_hmc_variants < 1:
        raise ValueError("HMC variant and chunk sizes must be positive.")
    if not 1 <= min_model_reference_samples <= n_chains * draws_per_chain:
        raise ValueError(
            "min_model_reference_samples must be between one and the source "
            "sample count."
        )

    top_k = _top_k_values(top_k)
    mapping_column = next(
        (
            name
            for name in ("matrix_variant_index", "hmc_variant_index")
            if name in variants
        ),
        None,
    )
    all_hmc_indices = (
        variants[mapping_column].to_numpy(dtype=np.int64)
        if mapping_column is not None
        else np.arange(len(variants), dtype=np.int64)
    )
    if len(all_hmc_indices) != len(variants):
        raise ValueError("Every analyzed variant must map to one HMC variant index.")
    if len(all_hmc_indices) and np.any(np.diff(all_hmc_indices) <= 0):
        raise ValueError("Analyzed HMC variant indices must be strictly increasing.")
    if (
        np.any(all_hmc_indices < 0)
        or np.any(all_hmc_indices >= n_hmc_variants)
    ):
        raise ValueError("Analyzed HMC variant indices lie outside the fitted axis.")
    candidate_indices = ranking_candidate_indices(variants)
    candidate_hmc_indices = all_hmc_indices[candidate_indices]
    if retain_k < top_k[-1] or retain_k > len(candidate_indices):
        raise ValueError(
            "retain_k must be at least the largest top_k and no larger than "
            "the ranking-candidate cohort."
        )

    reported_values = variants["reported_fitness"].to_numpy(dtype=float)
    reported_indices = candidate_indices[
        np.isfinite(reported_values[candidate_indices])
    ]
    if require_reported_top_k and top_k[-1] > len(reported_indices):
        raise ValueError("The largest top_k exceeds the reported-fitness cohort.")
    if not 0 <= tracked_top_n <= len(reported_indices):
        raise ValueError("tracked_top_n must not exceed the reported-fitness cohort.")
    if not 0 <= tracked_model_top_n <= retain_k:
        raise ValueError("tracked_model_top_n must be between 0 and retain_k.")
    reported_ordering_values = reported_values[reported_indices]
    if reported_higher_is_better:
        reported_ordering_values = -reported_ordering_values
    reported_order = np.lexsort((reported_indices, reported_ordering_values))
    tracked_indices = reported_indices[reported_order[:tracked_top_n]].astype(np.int32)
    tracked_hmc_indices = all_hmc_indices[tracked_indices]

    selected_samples = sample_ids(
        n_chains * draws_per_chain,
        n_samples,
        seed=seed,
    )
    if len(selected_samples) < 2:
        raise ValueError(
            f"At least two {sample_kind} samples are required for overlap analysis."
        )
    block_starts = list(range(0, n_hmc_variants, variant_chunk_size))
    invalid_required = set(required_block_starts) - set(block_starts)
    if invalid_required:
        raise ValueError(f"Required HMC block starts are invalid: {sorted(invalid_required)}")

    def read_block(
        variant_start: int,
        variant_stop: int,
        local_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        scores = read_scores(
            sample_ids=selected_samples,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
            local_indices=local_indices,
        )
        unknown = set(scores) - set(score_names)
        if unknown:
            raise ValueError(f"Score reader returned unknown scores: {sorted(unknown)}")
        if len(local_indices):
            missing = set(score_names) - set(scores)
            if missing:
                raise ValueError(f"Score reader omitted scores: {sorted(missing)}")
        for name, values in scores.items():
            values = np.asarray(values, dtype=np.float32)
            expected_shape = (len(selected_samples), len(local_indices))
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name}: score block must have shape {expected_shape}, "
                    f"not {values.shape}."
                )
            scores[name] = values
        return scores

    tracked_values = {
        name: np.full(
            (len(selected_samples), tracked_top_n),
            np.nan,
            dtype=np.float32,
        )
        for name in score_names
    }
    top_indices = {}
    top_scores = {}
    tracked_ranks = {}
    model_reference_values = {}
    for name in score_names:
        top_indices[name], top_scores[name] = empty_top_k(
            len(selected_samples),
            retain_k,
            higher_is_better=score_directions[name],
        )
        tracked_ranks[name] = empty_tracked_ranks(
            len(selected_samples), tracked_top_n
        )
        if tracked_model_top_n:
            model_reference_values[name] = np.full(
                len(variants),
                np.nan,
                dtype=np.float32,
            )

    started = time.monotonic()
    tracked_blocks = _ordered_blocks(
        np.unique(
            (tracked_hmc_indices // variant_chunk_size) * variant_chunk_size
        ).tolist(),
        required_block_starts,
    )
    if reset_reader is not None:
        reset_reader()
    if show_progress:
        print(
            f"{label}: {len(selected_samples):,} {sample_kind} samples, "
            f"{len(candidate_indices):,} ranking candidates, "
            f"{len(reported_indices):,} reported variants",
            flush=True,
        )
        print(
            f"  pass 1: collect {tracked_top_n:,} fixed reported-cohort values",
            flush=True,
        )
    for block_number, variant_start in enumerate(tracked_blocks, start=1):
        variant_stop = min(variant_start + variant_chunk_size, n_hmc_variants)
        cohort_columns = np.flatnonzero(
            (tracked_hmc_indices >= variant_start)
            & (tracked_hmc_indices < variant_stop)
        )
        local_indices = tracked_hmc_indices[cohort_columns] - variant_start
        scores = read_block(variant_start, variant_stop, local_indices)
        for name, values in scores.items():
            tracked_values[name][:, cohort_columns] = values
        if show_progress and (
            block_number == len(tracked_blocks) or block_number % 8 == 0
        ):
            print(
                f"    tracked chunks {block_number}/{len(tracked_blocks)}",
                flush=True,
            )

    if reset_reader is not None:
        reset_reader()
    if show_progress:
        print("  pass 2: scan every analyzed variant", flush=True)
    for block_number, variant_start in enumerate(
        _ordered_blocks(block_starts, required_block_starts),
        start=1,
    ):
        variant_stop = min(variant_start + variant_chunk_size, n_hmc_variants)
        candidate_start = int(
            np.searchsorted(candidate_hmc_indices, variant_start, side="left")
        )
        candidate_stop = int(
            np.searchsorted(candidate_hmc_indices, variant_stop, side="left")
        )
        local_indices = (
            candidate_hmc_indices[candidate_start:candidate_stop] - variant_start
        )
        block_candidate_indices = candidate_indices[candidate_start:candidate_stop]
        if not len(local_indices) and variant_start not in required_block_starts:
            continue
        scores = read_block(variant_start, variant_stop, local_indices)
        for name, values in scores.items():
            if not len(local_indices):
                continue
            top_indices[name], top_scores[name] = merge_top_k(
                top_indices[name],
                top_scores[name],
                block_candidate_indices,
                values,
                higher_is_better=score_directions[name],
            )
            accumulate_tracked_ranks(
                tracked_ranks[name],
                values,
                tracked_values[name],
                higher_is_better=score_directions[name],
            )
            if tracked_model_top_n:
                finite = np.isfinite(values)
                usable = finite.sum(axis=0) >= min_model_reference_samples
                reference = np.full(values.shape[1], np.nan, dtype=np.float32)
                reference[usable] = np.nanmedian(values[:, usable], axis=0)
                model_reference_values[name][block_candidate_indices] = reference
        if show_progress and (
            block_number == len(block_starts) or block_number % 8 == 0
        ):
            print(
                f"    full chunks {block_number}/{len(block_starts)} "
                f"({(time.monotonic() - started) / 60:.1f} min)",
                flush=True,
            )

    model_reference_indices = {}
    model_reference_scores = {}
    model_tracked_indices = {}
    model_tracked_values = {}
    model_tracked_ranks = {}
    if tracked_model_top_n:
        for name in score_names:
            (
                model_reference_indices[name],
                model_reference_scores[name],
            ) = select_reference_top_k(
                model_reference_values[name][None, :],
                k=retain_k,
                higher_is_better=score_directions[name],
            )
            model_tracked_indices[name] = model_reference_indices[name][
                :tracked_model_top_n
            ]
            model_tracked_values[name] = np.full(
                (len(selected_samples), tracked_model_top_n),
                np.nan,
                dtype=np.float32,
            )
            model_tracked_ranks[name] = empty_tracked_ranks(
                len(selected_samples),
                tracked_model_top_n,
            )

        target_result_indices = np.unique(
            np.concatenate(list(model_tracked_indices.values()))
        )
        target_hmc_indices = all_hmc_indices[target_result_indices]
        model_blocks = _ordered_blocks(
            np.unique(
                (target_hmc_indices // variant_chunk_size) * variant_chunk_size
            ).tolist(),
            required_block_starts,
        )
        if reset_reader is not None:
            reset_reader()
        if show_progress:
            print(
                f"  pass 3: collect {tracked_model_top_n:,} model-cohort values",
                flush=True,
            )
        for block_number, variant_start in enumerate(model_blocks, start=1):
            variant_stop = min(variant_start + variant_chunk_size, n_hmc_variants)
            block_targets = target_hmc_indices[
                (target_hmc_indices >= variant_start)
                & (target_hmc_indices < variant_stop)
            ]
            scores = read_block(
                variant_start,
                variant_stop,
                block_targets - variant_start,
            )
            for name, values in scores.items():
                model_hmc_indices = all_hmc_indices[model_tracked_indices[name]]
                cohort_columns = np.flatnonzero(
                    (model_hmc_indices >= variant_start)
                    & (model_hmc_indices < variant_stop)
                )
                positions = np.searchsorted(
                    block_targets,
                    model_hmc_indices[cohort_columns],
                )
                model_tracked_values[name][:, cohort_columns] = values[:, positions]
            if show_progress and (
                block_number == len(model_blocks) or block_number % 8 == 0
            ):
                print(
                    f"    model chunks {block_number}/{len(model_blocks)}",
                    flush=True,
                )

        if reset_reader is not None:
            reset_reader()
        if show_progress:
            print("  pass 4: compute exact model-cohort ranks", flush=True)
        for block_number, variant_start in enumerate(
            _ordered_blocks(block_starts, required_block_starts),
            start=1,
        ):
            variant_stop = min(variant_start + variant_chunk_size, n_hmc_variants)
            candidate_start = int(
                np.searchsorted(candidate_hmc_indices, variant_start, side="left")
            )
            candidate_stop = int(
                np.searchsorted(candidate_hmc_indices, variant_stop, side="left")
            )
            local_indices = (
                candidate_hmc_indices[candidate_start:candidate_stop] - variant_start
            )
            if not len(local_indices) and variant_start not in required_block_starts:
                continue
            scores = read_block(variant_start, variant_stop, local_indices)
            for name, values in scores.items():
                if len(local_indices):
                    accumulate_tracked_ranks(
                        model_tracked_ranks[name],
                        values,
                        model_tracked_values[name],
                        higher_is_better=score_directions[name],
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
            raise ValueError(
                f"{name}: fewer than retain_k finite scores in a {sample_kind} sample."
            )
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
    selection_coords: dict[str, object] = {
        "score_name": list(score_names),
        "sample": np.arange(len(selected_samples), dtype=np.int32),
        "sample_id": ("sample", selected_samples),
        "selection_rank": np.arange(1, retain_k + 1, dtype=np.int32),
        "tracked_variant": np.arange(tracked_top_n, dtype=np.int32),
        "tracked_variant_index": ("tracked_variant", tracked_indices),
    }
    if tracked_model_top_n:
        selection_data.update(
            {
                "model_reference_variant_index": (
                    ("score_name", "model_reference_rank"),
                    np.stack(
                        [model_reference_indices[name] for name in score_names]
                    ),
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
            "fit_method": fit_method,
            "retained_k": retain_k,
            "model_reference_min_finite_samples": min_model_reference_samples,
            **(
                {
                    f"higher_is_better_{name}": str(
                        score_directions[name]
                    ).lower()
                    for name in score_names
                }
                if direction_was_explicit
                else {}
            ),
            **dict(selection_attrs or {}),
        },
    )
    return HMCScan(
        selection=selection,
        sample_ids=selected_samples,
        candidate_indices=candidate_indices,
        reported_indices=reported_indices,
    )
