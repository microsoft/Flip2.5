"""Direct readiness checks for posterior-predictive count artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import h5py
import numpy as np


def _file_signature(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def validate_posterior_predictive_counts(
    path: str | Path,
    observed_counts: Mapping[str, np.ndarray],
    *,
    expected_chains: int | None = None,
    expected_draws: int | None = None,
    posterior_variable: str | None = None,
    sample_chunk_size: int = 32,
    variant_chunk_size: int = 16_384,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate embedded and predictive counts against staged observed arrays."""
    resolved_path = Path(path).resolve(strict=True)
    if not observed_counts:
        raise ValueError("At least one observed count variable is required.")
    if expected_chains is not None and expected_chains < 1:
        raise ValueError("expected_chains must be positive when provided.")
    if expected_draws is not None and expected_draws < 1:
        raise ValueError("expected_draws must be positive when provided.")
    if sample_chunk_size < 1 or variant_chunk_size < 1:
        raise ValueError("Count preflight chunk sizes must be positive.")

    variables: dict[str, object] = {}
    sample_shape: tuple[int, int] | None = None
    with h5py.File(resolved_path, "r") as handle:
        observed_group = handle.get("observed_data")
        predictive_group = handle.get("posterior_predictive")
        if not isinstance(observed_group, h5py.Group):
            raise ValueError("sampled-MLE artifact is missing observed_data.")
        if not isinstance(predictive_group, h5py.Group):
            raise ValueError("sampled-MLE artifact is missing posterior_predictive.")

        for name, staged_values in observed_counts.items():
            staged = np.asarray(staged_values)
            if staged.ndim < 1:
                raise ValueError(f"Staged count variable {name!r} needs a variant axis.")
            if not np.issubdtype(staged.dtype, np.integer) or np.any(staged < 0):
                raise ValueError(
                    f"Staged count variable {name!r} must be nonnegative integers."
                )
            embedded = observed_group.get(name)
            predictive = predictive_group.get(name)
            if not isinstance(embedded, h5py.Dataset):
                raise ValueError(f"observed_data/{name} is missing or not a dataset.")
            if not isinstance(predictive, h5py.Dataset):
                raise ValueError(
                    f"posterior_predictive/{name} is missing or not a dataset."
                )
            if embedded.shape != staged.shape or not np.array_equal(embedded[:], staged):
                raise ValueError(
                    f"observed_data/{name} differs from the staged count table."
                )
            if predictive.ndim != staged.ndim + 2 or predictive.shape[2:] != staged.shape:
                raise ValueError(
                    f"posterior_predictive/{name} must have trailing shape "
                    f"{staged.shape}, "
                    f"not {predictive.shape}."
                )
            variable_sample_shape = (int(predictive.shape[0]), int(predictive.shape[1]))
            if sample_shape is None:
                sample_shape = variable_sample_shape
            elif variable_sample_shape != sample_shape:
                raise ValueError(
                    "Posterior-predictive count variables have different chain/draw "
                    "shapes."
                )
            if expected_chains is not None and variable_sample_shape[0] != expected_chains:
                raise ValueError(
                    f"posterior_predictive/{name} must have {expected_chains} chains, "
                    f"not {variable_sample_shape[0]}."
                )
            if expected_draws is not None and variable_sample_shape[1] != expected_draws:
                raise ValueError(
                    f"posterior_predictive/{name} must have {expected_draws} draws, "
                    f"not {variable_sample_shape[1]}."
                )
            if not np.issubdtype(predictive.dtype, np.integer):
                raise ValueError(
                    f"posterior_predictive/{name} must contain integer counts."
                )

            expected_totals = staged.sum(axis=-1, dtype=np.int64)
            n_chains, n_draws = variable_sample_shape
            sample_starts = list(range(0, n_draws, sample_chunk_size))
            progress_stride = max(1, len(sample_starts) // 5)
            for chain in range(n_chains):
                for block_number, sample_start in enumerate(sample_starts, start=1):
                    sample_stop = min(sample_start + sample_chunk_size, n_draws)
                    totals = np.zeros(
                        (sample_stop - sample_start, *staged.shape[:-1]),
                        dtype=np.int64,
                    )
                    for variant_start in range(
                        0,
                        staged.shape[-1],
                        variant_chunk_size,
                    ):
                        variant_stop = min(
                            variant_start + variant_chunk_size,
                            staged.shape[-1],
                        )
                        values = np.asarray(
                            predictive[
                                chain,
                                sample_start:sample_stop,
                                ...,
                                variant_start:variant_stop,
                            ]
                        )
                        if np.any(values < 0):
                            raise ValueError(
                                f"posterior_predictive/{name} contains negative counts."
                            )
                        totals += values.sum(axis=-1, dtype=np.int64)
                    if not np.array_equal(
                        totals,
                        np.broadcast_to(expected_totals, totals.shape),
                    ):
                        raise ValueError(
                            f"posterior_predictive/{name} totals differ from observed "
                            f"data for chain {chain}, draws "
                            f"{sample_start}:{sample_stop}."
                        )
                    if progress is not None and (
                        block_number == len(sample_starts)
                        or block_number % progress_stride == 0
                    ):
                        progress(
                            f"{name}: verified chain {chain + 1}/{n_chains}, "
                            f"draws {sample_stop}/{n_draws}"
                        )
            variables[name] = {
                "observed_shape": list(staged.shape),
                "predictive_shape": list(predictive.shape),
                "dtype": str(predictive.dtype),
                "chunks": list(predictive.chunks) if predictive.chunks else None,
                "observed_totals": np.asarray(expected_totals).tolist(),
            }

        assert sample_shape is not None
        if sample_shape[0] * sample_shape[1] < 2:
            raise ValueError("Posterior-predictive analysis requires at least two samples.")
        if posterior_variable is not None:
            posterior_group = handle.get("posterior")
            if not isinstance(posterior_group, h5py.Group):
                raise ValueError("Artifact is missing posterior.")
            posterior = posterior_group.get(posterior_variable)
            if not isinstance(posterior, h5py.Dataset):
                raise ValueError(f"posterior/{posterior_variable} is missing.")
            variant_count = next(iter(observed_counts.values())).shape[-1]
            if posterior.ndim != 3 or posterior.shape[:2] != sample_shape:
                raise ValueError(
                    f"posterior/{posterior_variable} must match chain/draw shape "
                    f"{sample_shape}."
                )
            if posterior.shape[-1] != variant_count:
                raise ValueError(
                    f"posterior/{posterior_variable} must have {variant_count} variants, "
                    f"not {posterior.shape[-1]}."
                )

    return {
        "status": "ready",
        "source_artifact": _file_signature(resolved_path),
        "n_chains": sample_shape[0],
        "n_draws": sample_shape[1],
        "n_samples": sample_shape[0] * sample_shape[1],
        "posterior_variable": posterior_variable,
        "count_variables": variables,
    }


def validate_sampled_mle_counts(
    path: str | Path,
    observed_counts: Mapping[str, np.ndarray],
    *,
    expected_draws: int = 1_000,
    sample_chunk_size: int = 32,
    variant_chunk_size: int = 16_384,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate one-chain sampled-MLE counts against staged observed arrays."""
    if expected_draws < 2:
        raise ValueError("expected_draws must be at least two.")
    return validate_posterior_predictive_counts(
        path,
        observed_counts,
        expected_chains=1,
        expected_draws=expected_draws,
        sample_chunk_size=sample_chunk_size,
        variant_chunk_size=variant_chunk_size,
        progress=progress,
    )
