"""Shared primitives for the six fit preprocessing examples."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from dms_rank_stability import SampledScores


def select_sample_ids(total: int, count: int | None, *, seed: int) -> np.ndarray:
    """Select sorted chain-major sample IDs without replacement."""
    selected_count = total if count is None else count
    if not 2 <= selected_count <= total:
        raise ValueError(f"sample count must be between 2 and {total:,}.")
    if selected_count == total:
        return np.arange(total, dtype=np.int64)
    selected = np.random.default_rng(seed).choice(
        total,
        selected_count,
        replace=False,
    )
    return np.sort(selected).astype(np.int64)


def read_native_samples(
    variable: h5py.Dataset,
    sample_ids: np.ndarray,
    *,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
) -> np.ndarray:
    """Read selected chain-major samples from a variant-contiguous block."""
    selected = np.asarray(sample_ids, dtype=np.int64)
    total_samples = variable.shape[0] * draws_per_chain
    if (
        variable.ndim < 3
        or variable.shape[1] != draws_per_chain
        or variant_start < 0
        or variant_stop > variable.shape[-1]
        or variant_start >= variant_stop
    ):
        raise ValueError("Fit array axes or requested variant block are invalid.")
    if (
        selected.ndim != 1
        or len(selected) == 0
        or np.any(selected < 0)
        or np.any(selected >= total_samples)
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("sample IDs must be unique, increasing, and in range.")

    chains = selected // draws_per_chain
    draws = selected % draws_per_chain
    result = np.empty(
        (len(selected), *variable.shape[2:-1], variant_stop - variant_start),
        dtype=variable.dtype,
    )
    for chain in np.unique(chains):
        positions = np.flatnonzero(chains == chain)
        chain_draws = draws[positions]
        selection = (
            int(chain),
            chain_draws,
            *(slice(None) for _ in range(variable.ndim - 3)),
            slice(variant_start, variant_stop),
        )
        result[positions] = variable[selection]
    return result


def transform_scores(values: np.ndarray, transform: str) -> np.ndarray:
    """Convert one native posterior target to finite float32 scores."""
    transformed = np.asarray(values).astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform == "positive":
            transformed = np.log(transformed)
        elif transform == "inverse":
            transformed = -np.log(transformed)
        elif transform != "identity":
            raise ValueError(f"Unknown score transform {transform!r}.")
    return np.where(np.isfinite(transformed), transformed, np.nan).astype(
        np.float32,
        copy=False,
    )


def variant_positions(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Map a target variant table to a strictly increasing fitted axis."""
    for name, variants in (("source", source), ("target", target)):
        if "variant_id" not in variants:
            raise ValueError(f"{name} variants are missing variant_id.")
        if variants["variant_id"].isna().any() or variants["variant_id"].duplicated().any():
            raise ValueError(f"{name} variant IDs must be present and unique.")
    positions = pd.Index(source["variant_id"]).get_indexer(target["variant_id"])
    if np.any(positions < 0):
        missing = target.loc[positions < 0, "variant_id"].tolist()
        raise ValueError(f"Target variants are absent from the fitted axis: {missing[:5]}")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("Target variants must preserve fitted-axis order.")
    return positions.astype(np.int64)


def selection_blocks(
    positions: np.ndarray,
    *,
    source_size: int,
    block_size: int,
    priority_position: int | None = None,
) -> np.ndarray | None:
    """Preserve source block order when a fitted axis is projected."""
    source_positions = np.asarray(positions, dtype=np.int64)
    if np.array_equal(source_positions, np.arange(source_size, dtype=np.int64)):
        return None
    if (
        source_positions.ndim != 1
        or not len(source_positions)
        or source_positions[0] < 0
        or source_positions[-1] >= source_size
        or np.any(np.diff(source_positions) <= 0)
        or block_size < 1
    ):
        raise ValueError("Projected variant positions or source block size are invalid.")

    source_blocks = (source_positions // block_size) * block_size
    observed = {int(value) for value in np.unique(source_blocks)}
    ordered = []
    if priority_position is not None:
        if not 0 <= priority_position < source_size:
            raise ValueError("Priority variant lies outside the fitted axis.")
        priority_block = (priority_position // block_size) * block_size
        if priority_block in observed:
            ordered.append(priority_block)
    ordered.extend(sorted(observed - set(ordered)))
    block_order = {source_block: index for index, source_block in enumerate(ordered)}
    return np.array(
        [block_order[int(source_block)] for source_block in source_blocks],
        dtype=np.int32,
    )


def publish_scores(
    variants: pd.DataFrame,
    values: np.ndarray,
    source_sample_ids: np.ndarray,
    output: str | Path,
    *,
    draws_per_chain: int | None = None,
    blocks: np.ndarray | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically publish one sampled score matrix."""
    matrix = np.asarray(values)
    source_ids = np.asarray(source_sample_ids, dtype=np.int64)
    if matrix.ndim != 2 or matrix.dtype != np.dtype("float32"):
        raise ValueError("Score values must be a two-dimensional float32 array.")
    if source_ids.shape != (matrix.shape[0],):
        raise ValueError("Source sample IDs must match the sampled score axis.")

    coordinates: dict[str, object] = {
        "sample": np.arange(matrix.shape[0], dtype=np.int32),
        "variant": np.arange(matrix.shape[1], dtype=np.int32),
        "source_sample_id": ("sample", source_ids),
    }
    if draws_per_chain is not None:
        if draws_per_chain < 1:
            raise ValueError("draws_per_chain must be positive.")
        coordinates.update(
            {
                "source_chain": (
                    "sample",
                    (source_ids // draws_per_chain).astype(np.int32),
                ),
                "source_draw": (
                    "sample",
                    (source_ids % draws_per_chain).astype(np.int32),
                ),
            }
        )
    if blocks is not None:
        coordinates["selection_block"] = (
            "variant",
            np.asarray(blocks, dtype=np.int32),
        )
    dataset = xr.Dataset(
        {"score": (("sample", "variant"), matrix)},
        coords=coordinates,
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        SampledScores(variants.copy(), dataset).save(temporary)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to replace existing scores: {destination}")
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def extract_parameter_scores(
    fit_path: str | Path,
    source_variants: pd.DataFrame,
    target_variants: pd.DataFrame,
    output: str | Path,
    *,
    posterior_variable: str,
    transform: str,
    sample_count: int | None,
    seed: int = 23,
    priority_variant_id: str | None = None,
    sample_chunk_size: int = 64,
    overwrite: bool = False,
) -> Path:
    """Extract one posterior parameter over an aligned target variant axis."""
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    positions = variant_positions(source_variants, target_variants)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-scores-",
        suffix=".dat",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    matrix: np.memmap | None = None
    try:
        with h5py.File(fit_path) as handle:
            target = handle.get(f"posterior/{posterior_variable}")
            if not isinstance(target, h5py.Dataset) or target.ndim != 3:
                raise ValueError(
                    f"posterior/{posterior_variable} must be "
                    "chain-by-draw-by-variant."
                )
            n_chains, draws_per_chain, source_size = map(int, target.shape)
            if source_size != len(source_variants):
                raise ValueError("Posterior and source variant axes differ.")
            selected = select_sample_ids(
                n_chains * draws_per_chain,
                sample_count,
                seed=seed,
            )
            source_block_size = int((target.chunks or target.shape)[-1])
            priority_position = None
            if priority_variant_id is not None:
                matches = np.flatnonzero(
                    source_variants["variant_id"].to_numpy() == priority_variant_id
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one priority variant {priority_variant_id!r}."
                    )
                priority_position = int(matches[0])
            blocks = selection_blocks(
                positions,
                source_size=source_size,
                block_size=source_block_size,
                priority_position=priority_position,
            )
            matrix = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(selected), len(positions)),
            )
            for sample_start in range(0, len(selected), sample_chunk_size):
                sample_stop = min(sample_start + sample_chunk_size, len(selected))
                sample_ids = selected[sample_start:sample_stop]
                for target_start in range(0, len(positions), source_block_size):
                    target_stop = min(target_start + source_block_size, len(positions))
                    local_positions = positions[target_start:target_stop]
                    source_start = int(local_positions[0])
                    source_stop = int(local_positions[-1]) + 1
                    native = read_native_samples(
                        target,
                        sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=source_start,
                        variant_stop=source_stop,
                    )
                    matrix[
                        sample_start:sample_stop,
                        target_start:target_stop,
                    ] = transform_scores(
                        native[:, local_positions - source_start],
                        transform,
                    )
            matrix.flush()
            return publish_scores(
                target_variants,
                matrix,
                selected,
                destination,
                draws_per_chain=draws_per_chain,
                blocks=blocks,
                overwrite=overwrite,
            )
    finally:
        if matrix is not None:
            matrix.flush()
        temporary_path.unlink(missing_ok=True)


def compare_score_data(
    generated_path: str | Path,
    reference_path: str | Path,
    *,
    sample_chunk_size: int = 64,
    variant_chunk_size: int = 4096,
) -> None:
    """Require exact agreement with selected samples from reference score data."""
    if sample_chunk_size < 1 or variant_chunk_size < 1:
        raise ValueError("Comparison chunk sizes must be positive.")
    generated = SampledScores.load(generated_path)
    reference = SampledScores.load(reference_path)
    try:
        if not generated.variants.equals(reference.variants):
            raise ValueError("Generated and reference variant tables differ.")
        generated_ids = generated.dataset["source_sample_id"].values
        reference_ids = reference.dataset["source_sample_id"].values
        positions = pd.Index(reference_ids).get_indexer(generated_ids)
        if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
            raise ValueError("Generated source samples are absent or reordered.")
        for name in ("source_chain", "source_draw"):
            generated_has = name in generated.dataset.coords
            reference_has = name in reference.dataset.coords
            if generated_has != reference_has:
                raise ValueError(f"Generated and reference {name} coordinates differ.")
            if generated_has and not np.array_equal(
                generated.dataset[name].values,
                reference.dataset[name].values[positions],
            ):
                raise ValueError(f"Generated and reference {name} values differ.")
        generated_has_blocks = "selection_block" in generated.dataset.coords
        reference_has_blocks = "selection_block" in reference.dataset.coords
        if generated_has_blocks != reference_has_blocks:
            raise ValueError("Generated and reference selection blocks differ.")
        if generated_has_blocks and not np.array_equal(
            generated.dataset["selection_block"].values,
            reference.dataset["selection_block"].values,
        ):
            raise ValueError("Generated and reference selection blocks differ.")

        for sample_start in range(0, len(positions), sample_chunk_size):
            sample_stop = min(sample_start + sample_chunk_size, len(positions))
            reference_positions = positions[sample_start:sample_stop]
            for variant_start in range(
                0,
                len(generated.variants),
                variant_chunk_size,
            ):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    len(generated.variants),
                )
                observed = generated.score.isel(
                    sample=slice(sample_start, sample_stop),
                    variant=slice(variant_start, variant_stop),
                ).values
                expected = reference.score.isel(
                    sample=reference_positions,
                    variant=slice(variant_start, variant_stop),
                ).values
                if not np.array_equal(observed, expected, equal_nan=True):
                    raise ValueError(
                        "Generated scores differ from the reference at sample "
                        f"{sample_start}:{sample_stop}, variant "
                        f"{variant_start}:{variant_stop}."
                    )
    finally:
        reference.close()
        generated.close()
