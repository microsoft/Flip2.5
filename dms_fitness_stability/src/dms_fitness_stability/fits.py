"""Thin adapters around SciStanPy inference results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

FitMethod = Literal["hmc", "mle"]
ScoreTransform = Literal["identity", "log", "negative_log"]


def load_fit(path: str | Path, *, method: FitMethod, use_dask: bool = False):
    """Load an HMC or sampled-MLE NetCDF artifact through SciStanPy."""
    from scistanpy.model.results import MLEInferenceRes, SampleResults

    resolved_path = str(Path(path))
    if method == "hmc":
        return SampleResults.from_disk(
            resolved_path,
            skip_fit=True,
            use_dask=use_dask,
        )
    if method == "mle":
        return MLEInferenceRes.from_disk(resolved_path, use_dask=use_dask)
    raise ValueError("method must be either 'hmc' or 'mle'.")


def extract_score_samples(
    fit,
    *,
    variable: str,
    variant_dim: str,
    variant_ids: Sequence[object] | None = None,
    transform: ScoreTransform = "identity",
) -> xr.DataArray:
    """Return one fitted score as a labeled sample-by-variant array."""
    posterior = fit.inference_obj.posterior
    if variable not in posterior:
        raise ValueError(f"Posterior does not contain score variable {variable!r}.")
    scores = posterior[variable]
    if variant_dim not in scores.dims:
        raise ValueError(
            f"Score variable {variable!r} does not contain variant dimension {variant_dim!r}."
        )

    if {"chain", "draw"} <= set(scores.dims):
        extra_dims = set(scores.dims) - {"chain", "draw", variant_dim}
        if extra_dims:
            raise ValueError(
                f"Score variable {variable!r} has unsupported dimensions: {sorted(extra_dims)}"
            )
        scores = (
            scores.stack(sample=("chain", "draw"))
            .transpose("sample", variant_dim)
            .reset_index("sample")
            .assign_coords(sample=np.arange(scores.sizes["chain"] * scores.sizes["draw"]))
        )
    elif "sample" in scores.dims:
        extra_dims = set(scores.dims) - {"sample", variant_dim}
        if extra_dims:
            raise ValueError(
                f"Score variable {variable!r} has unsupported dimensions: {sorted(extra_dims)}"
            )
        scores = scores.transpose("sample", variant_dim)
    else:
        raise ValueError(
            f"Score variable {variable!r} must have ('chain', 'draw') or 'sample' dimensions."
        )

    scores = scores.rename({variant_dim: "variant"})
    if variant_ids is not None:
        if len(variant_ids) != scores.sizes["variant"]:
            raise ValueError("variant_ids must match the fitted variant dimension.")
        scores = scores.assign_coords(variant=np.asarray(variant_ids))

    if transform == "log":
        scores = np.log(scores)
    elif transform == "negative_log":
        scores = -np.log(scores)
    elif transform != "identity":
        raise ValueError("transform must be 'identity', 'log', or 'negative_log'.")
    return scores.astype(np.float32).rename(variable)
