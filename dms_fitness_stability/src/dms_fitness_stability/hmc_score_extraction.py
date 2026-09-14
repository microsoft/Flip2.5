"""Bounded-memory extraction of direct-HMC posterior score matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from .hmc_scanning import file_signature, read_native_samples, sample_ids
from .identity import AnalysisIdentity
from .score_ensembles import (
    POSTERIOR_PARAMETER_TARGET,
    SCORE_ENSEMBLE_SCHEMA_VERSION,
    ScoreEnsemble,
    direct_hmc_reference_metadata,
)

ScoreTransform = Callable[[np.ndarray], np.ndarray]


def transform_hmc_score(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply the frozen direct-HMC posterior transform on float32 values."""
    transformed = np.asarray(values).astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform == "positive":
            transformed = np.log(transformed)
        elif transform == "inverse":
            transformed = -np.log(transformed)
        elif transform not in {"identity", "log"}:
            raise ValueError(f"Unknown HMC score transform {transform!r}.")
    return np.where(np.isfinite(transformed), transformed, np.nan).astype(
        np.float32,
        copy=False,
    )


def project_score_bundle_bytes(
    reference_bundle: str | Path,
    *,
    target_samples: int,
    safety_factor: float = 1.15,
) -> int:
    """Project bundle bytes from a measured bundle with the same variant axis."""
    if target_samples < 1:
        raise ValueError("target_samples must be positive.")
    if not np.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be finite and at least one.")
    reference_dir = Path(reference_bundle)
    metadata = json.loads(
        (reference_dir / "metadata.json").read_text(encoding="utf-8")
    )
    reference_samples = metadata.get("n_samples")
    if (
        not isinstance(reference_samples, int)
        or isinstance(reference_samples, bool)
        or reference_samples < 1
    ):
        raise ValueError("Reference bundle has no positive n_samples metadata.")
    score_path = reference_dir / "fitness_scores.nc"
    if not score_path.is_file():
        raise ValueError(f"Reference bundle has no fitness_scores.nc: {reference_dir}")
    score_bytes = score_path.stat().st_size
    fixed_bytes = sum(
        path.stat().st_size
        for path in reference_dir.iterdir()
        if path.is_file() and path != score_path
    )
    projected = score_bytes * target_samples / reference_samples + fixed_bytes
    return math.ceil(projected * safety_factor)


def hmc_score_geometry(
    hmc_path: str | Path,
    *,
    posterior_variable: str,
) -> tuple[int, int, int]:
    """Return chain, draw, and variant sizes for one posterior score target."""
    path = Path(hmc_path)
    with h5py.File(path, "r") as handle:
        posterior = handle.get("posterior")
        if not isinstance(posterior, h5py.Group):
            raise ValueError(f"HMC artifact has no posterior group: {path}")
        target = posterior.get(posterior_variable)
        if not isinstance(target, h5py.Dataset) or target.ndim != 3:
            raise ValueError(
                f"posterior/{posterior_variable} must be "
                "chain-by-draw-by-variant."
            )
        return tuple(map(int, target.shape))


def audit_hmc_score_source(
    ensemble: ScoreEnsemble,
    hmc_path: str | Path,
    *,
    sample_chunk_size: int = 256,
    variant_chunk_size: int | None = None,
) -> dict[str, object]:
    """Require exact equality between a score bundle and its source posterior."""
    ensemble.validate()
    if ensemble.metadata.get("inference_method") != "hmc":
        raise ValueError("Source score audit requires a direct-HMC ensemble.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    path = Path(hmc_path).resolve()
    source = ensemble.metadata.get("source_artifact")
    if not isinstance(source, dict):
        raise ValueError("Score ensemble source_artifact is missing.")
    if (
        Path(str(source.get("path"))).resolve() != path
        or source.get("size") != path.stat().st_size
        or source.get("mtime_ns") != path.stat().st_mtime_ns
    ):
        raise ValueError("Score ensemble source artifact signature changed.")
    posterior_variable = str(ensemble.metadata["posterior_variable"])
    posterior_transform = str(ensemble.metadata["posterior_transform"])
    score_names = ensemble.scores["score_name"].values.astype(str).tolist()
    if len(score_names) != 1:
        raise ValueError("Direct-HMC source audit currently requires one score.")
    selected_sample_ids = np.asarray(
        ensemble.scores["source_sample_id"].values,
        dtype=np.int64,
    )
    source_digest = hashlib.sha256()
    matrix_digest = hashlib.sha256()
    finite_values = 0
    total_values = 0
    with h5py.File(path, "r") as handle:
        posterior = handle.get("posterior")
        if not isinstance(posterior, h5py.Group):
            raise ValueError("HMC artifact must contain a posterior group.")
        target = posterior.get(posterior_variable)
        if not isinstance(target, h5py.Dataset) or target.ndim != 3:
            raise ValueError(
                f"posterior/{posterior_variable} must be "
                "chain-by-draw-by-variant."
            )
        n_chains, draws_per_chain, n_variants = map(int, target.shape)
        if (
            n_chains != ensemble.metadata["n_chains"]
            or draws_per_chain != ensemble.metadata["draws_per_chain"]
            or n_variants != ensemble.scores.sizes["variant"]
        ):
            raise ValueError("Score ensemble and source posterior shapes differ.")
        resolved_variant_chunk_size = (
            int((target.chunks or target.shape)[-1])
            if variant_chunk_size is None
            else variant_chunk_size
        )
        if resolved_variant_chunk_size < 1:
            raise ValueError("variant_chunk_size must be positive.")
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
            for variant_start in range(0, n_variants, resolved_variant_chunk_size):
                variant_stop = min(
                    variant_start + resolved_variant_chunk_size,
                    n_variants,
                )
                native = read_native_samples(
                    target,
                    block_sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                )
                expected = transform_hmc_score(native, posterior_transform)
                observed = np.asarray(
                    ensemble.fitness_score.isel(
                        score_name=0,
                        sample=slice(sample_start, sample_stop),
                        variant=slice(variant_start, variant_stop),
                    ).values,
                    dtype=np.float32,
                )
                if not np.array_equal(expected, observed, equal_nan=True):
                    mismatch = ~(
                        (expected == observed)
                        | (np.isnan(expected) & np.isnan(observed))
                    )
                    raise ValueError(
                        "Persisted HMC scores differ from the source posterior at "
                        f"sample block {sample_start}:{sample_stop}, variant block "
                        f"{variant_start}:{variant_stop}; "
                        f"{int(mismatch.sum())} value(s) differ."
                    )
                expected_bytes = np.ascontiguousarray(expected).tobytes(order="C")
                observed_bytes = np.ascontiguousarray(observed).tobytes(order="C")
                source_digest.update(expected_bytes)
                matrix_digest.update(observed_bytes)
                finite_values += int(np.isfinite(observed).sum())
                total_values += observed.size
    return {
        "analysis_id": ensemble.metadata["analysis_id"],
        "sample_view": ensemble.metadata["sample_view"],
        "posterior_variable": posterior_variable,
        "posterior_transform": posterior_transform,
        "score_name": score_names[0],
        "shape": list(ensemble.fitness_score.shape),
        "finite_values": finite_values,
        "nan_values": total_values - finite_values,
        "source_sha256": source_digest.hexdigest(),
        "matrix_sha256": matrix_digest.hexdigest(),
        "status": "exact",
    }


def build_hmc_score_ensemble(
    *,
    hmc_path: str | Path,
    identity: AnalysisIdentity,
    posterior_variable: str,
    posterior_transform: str,
    score_name: str,
    score_contract: str,
    variants: pd.DataFrame,
    analysis_variants: pd.DataFrame,
    transform: ScoreTransform,
    n_samples: int | None = None,
    seed: int = 23,
    sample_view: str = "full",
    sample_chunk_size: int = 256,
    variant_chunk_size: int | None = None,
    scratch_dir: str | Path | None = None,
    extra_metadata: Mapping[str, object] | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Extract one transformed posterior target over the complete fitted axis."""
    if not posterior_variable.strip() or not posterior_transform.strip():
        raise ValueError("Posterior variable and transform must be non-empty.")
    if not score_name.strip() or not score_contract.strip():
        raise ValueError("Score name and contract must be non-empty.")
    if not sample_view.strip():
        raise ValueError("sample_view must be non-empty.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    hmc_path = Path(hmc_path)
    if not hmc_path.is_file():
        raise FileNotFoundError(f"HMC artifact is missing: {hmc_path}")
    if "hmc_variant_index" in variants and not np.array_equal(
        variants["hmc_variant_index"].to_numpy(dtype=np.int64),
        np.arange(len(variants), dtype=np.int64),
    ):
        raise ValueError("Full variants must follow the fitted HMC axis.")

    temporary_path: Path | None = None
    matrix: np.memmap | None = None
    try:
        with h5py.File(hmc_path, "r") as handle:
            posterior = handle.get("posterior")
            if not isinstance(posterior, h5py.Group):
                raise ValueError("HMC artifact must contain a posterior group.")
            target = posterior.get(posterior_variable)
            if not isinstance(target, h5py.Dataset):
                raise ValueError(
                    f"HMC artifact must contain posterior/{posterior_variable}."
                )
            if target.ndim != 3 or target.shape[-1] != len(variants):
                raise ValueError(
                    f"posterior/{posterior_variable} must be "
                    "chain-by-draw-by-variant and match variants.csv."
                )
            n_chains, draws_per_chain, n_variants = target.shape
            total_samples = n_chains * draws_per_chain
            selected_sample_ids = sample_ids(total_samples, n_samples, seed=seed)
            if len(selected_sample_ids) < 2:
                raise ValueError("Score ensembles require at least two samples.")
            if sample_view == "full" and len(selected_sample_ids) != total_samples:
                raise ValueError("sample_view='full' requires every source sample.")
            resolved_variant_chunk_size = (
                int((target.chunks or target.shape)[-1])
                if variant_chunk_size is None
                else variant_chunk_size
            )
            if resolved_variant_chunk_size < 1:
                raise ValueError("variant_chunk_size must be positive.")

            scratch_path = None if scratch_dir is None else Path(scratch_dir)
            if scratch_path is not None:
                scratch_path.mkdir(parents=True, exist_ok=True)
            descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix="hmc-scores-",
                suffix=".dat",
                dir=scratch_path,
            )
            os.close(descriptor)
            temporary_path = Path(raw_temporary_path)
            matrix = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(1, len(selected_sample_ids), n_variants),
            )

            sample_starts = range(0, len(selected_sample_ids), sample_chunk_size)
            n_sample_blocks = (
                len(selected_sample_ids) + sample_chunk_size - 1
            ) // sample_chunk_size
            for sample_block, sample_start in enumerate(sample_starts, start=1):
                sample_stop = min(
                    sample_start + sample_chunk_size,
                    len(selected_sample_ids),
                )
                block_sample_ids = selected_sample_ids[sample_start:sample_stop]
                for variant_start in range(
                    0,
                    n_variants,
                    resolved_variant_chunk_size,
                ):
                    variant_stop = min(
                        variant_start + resolved_variant_chunk_size,
                        n_variants,
                    )
                    native = read_native_samples(
                        target,
                        block_sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=variant_start,
                        variant_stop=variant_stop,
                    )
                    transformed = np.asarray(transform(native), dtype=np.float32)
                    expected_shape = (
                        len(block_sample_ids),
                        variant_stop - variant_start,
                    )
                    if transformed.shape != expected_shape:
                        raise ValueError(
                            f"Score transform returned shape {transformed.shape}, "
                            f"expected {expected_shape}."
                        )
                    matrix[
                        0,
                        sample_start:sample_stop,
                        variant_start:variant_stop,
                    ] = np.where(np.isfinite(transformed), transformed, np.nan)
                matrix.flush()
                if show_progress:
                    print(
                        f"{identity.analysis_id}: extracted sample block "
                        f"{sample_block}/{n_sample_blocks}",
                        flush=True,
                    )

        scores = xr.Dataset(
            {
                "fitness_score": (
                    ("score_name", "sample", "variant"),
                    matrix,
                )
            },
            coords={
                "score_name": [score_name],
                "sample": np.arange(len(selected_sample_ids), dtype=np.int32),
                "variant": np.arange(n_variants, dtype=np.int32),
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
        metadata = {
            **dict(extra_metadata or {}),
            **identity.metadata(),
            "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
            "fit_method": "hmc_posterior",
            "uncertainty_target": POSTERIOR_PARAMETER_TARGET,
            "posterior_variable": posterior_variable,
            "posterior_transform": posterior_transform,
            "score_contracts": {score_name: score_contract},
            "scores": [score_name],
            "n_chains": n_chains,
            "draws_per_chain": draws_per_chain,
            "n_source_samples": total_samples,
            "n_samples": len(selected_sample_ids),
            "n_variants": n_variants,
            "n_analysis_variants": len(analysis_variants),
            "sample_selection_seed": seed,
            "sample_view": sample_view,
            "sample_chunk_size": sample_chunk_size,
            "variant_chunk_size": resolved_variant_chunk_size,
            "source_artifact": file_signature(hmc_path),
        }
        metadata.update(
            direct_hmc_reference_metadata(metadata, analysis_variants)
        )
        ensemble = ScoreEnsemble(
            metadata=metadata,
            variants=variants.copy(),
            analysis_variants=analysis_variants.copy(),
            scores=scores,
            temporary_paths=(temporary_path,),
        )
        ensemble.validate()
        return ensemble
    except Exception:
        if matrix is not None:
            matrix.flush()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
