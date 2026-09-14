"""TrpB landscape inputs and HMC variant alignment."""

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
from flipv3.flip_dsets import load_trpb_dataset

from ..analysis import RESULT_SCHEMA_VERSION, _top_k_values
from ..hmc_score_extraction import build_hmc_score_ensemble as _build_hmc_scores
from ..hmc_score_extraction import transform_hmc_score
from ..identity import AnalysisIdentity
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
from .pdz import (
    _combined_tracked_summary,
    _file_signature,
    _read_native_samples,
    _sample_ids,
    _tracked_summary,
)

TRPB_LIBRARIES = tuple(f"lib{letter}" for letter in "ABCDEFGHI")
TRPB_LANDSCAPES = (*TRPB_LIBRARIES, "four-site")
TRPB_SCORE_NAMES = ("log_mean_growth", "count_ratio")
TRPB_PREDICTIVE_SCORE_NAME = "johnston_fitness"
TRPB_PREDICTIVE_SCORE_CONTRACT = "trpb-johnston-fitness-v1"
GrowthRatePrior = Literal["exponential", "gamma", "lomax"]
GrowthCurve = Literal["exponential", "sigmoid"]


@dataclass(frozen=True)
class TrpBLandscape:
    """Full fitted TrpB library with aligned reported fitness."""

    library: str
    variants: pd.DataFrame
    starting_counts: np.ndarray
    timepoint_counts: np.ndarray
    times: np.ndarray

    @property
    def n_hmc_variants(self) -> int:
        return self.starting_counts.shape[-1]


def _variant_totals(counts: np.ndarray) -> np.ndarray:
    axes = tuple(range(counts.ndim - 1))
    return counts.sum(axis=axes, dtype=np.int64) if axes else counts.astype(np.int64)


def trpb_johnston_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    variants: list[str] | np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Reproduce the deposited four-site amino-acid fitness calculation."""
    starting = np.asarray(starting_counts, dtype=float)
    timepoint = np.asarray(timepoint_counts, dtype=float)
    time_values = np.asarray(times, dtype=float)
    variant_values = np.asarray(variants, dtype=str)
    if starting.ndim < 2 or timepoint.ndim != starting.ndim + 1:
        raise ValueError(
            "TrpB counts must be (..., replicate, variant) and "
            "(..., time, replicate, variant)."
        )
    if timepoint.shape[:-3] != starting.shape[:-2]:
        raise ValueError("TrpB starting and timepoint sample axes must match.")
    if timepoint.shape[-2:] != starting.shape[-2:]:
        raise ValueError("TrpB starting and timepoint replicate/variant axes must match.")
    if time_values.shape != (timepoint.shape[-3],):
        raise ValueError("TrpB times must match the timepoint count axis.")
    if variant_values.shape != (starting.shape[-1],):
        raise ValueError("TrpB variants must match the count variant axis.")
    if (
        not np.isfinite(starting).all()
        or not np.isfinite(timepoint).all()
        or np.any(starting < 0)
        or np.any(timepoint < 0)
    ):
        raise ValueError("TrpB counts must be finite and nonnegative.")
    if not np.isfinite(time_values).all() or np.any(time_values <= 0):
        raise ValueError("TrpB times must be finite and positive.")
    if starting.shape[-2] != 2:
        raise ValueError("The Johnston fitness contract requires two replicates.")
    if not np.array_equal(time_values, np.array([12, 16, 20, 24, 28, 36])):
        raise ValueError(
            "The Johnston fitness contract requires times 12, 16, 20, 24, 28, 36."
        )

    stop_mask = np.char.find(variant_values, "*") >= 0
    if not stop_mask.any():
        raise ValueError("TrpB fitness normalization requires stop variants.")
    retained_times = time_values != 12
    if not retained_times.any():
        raise ValueError("TrpB fitness requires a retained timepoint after 12 hours.")

    average_input = starting.mean(axis=-2)
    valid = (average_input >= 10)[..., None, None, :] & (
        timepoint > 0
    ).all(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(
            timepoint / average_input[..., None, None, :]
        )
    log_ratio = np.where(valid, log_ratio, np.nan)
    stop_mean = np.nanmean(log_ratio[..., stop_mask], axis=-1)
    maximum = np.nanmax(log_ratio, axis=-1)
    normalized = (log_ratio - stop_mean[..., None]) / (
        maximum - stop_mean
    )[..., None]
    retained = normalized[..., retained_times, :, :]
    finite = np.isfinite(retained)
    return np.divide(
        np.nansum(retained, axis=(-3, -2)),
        finite.sum(axis=(-3, -2)),
        out=np.full(starting.shape[:-2] + (starting.shape[-1],), np.nan),
        where=finite.sum(axis=(-3, -2)) > 0,
    )


def trpb_three_site_fitness(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    variants: list[str] | np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Reproduce the deposited libA-I amino-acid fitness calculation."""
    starting = np.asarray(starting_counts, dtype=float)
    timepoint = np.asarray(timepoint_counts, dtype=float)
    variant_values = np.asarray(variants, dtype=str)
    time_values = np.asarray(times, dtype=float)
    if starting.ndim < 1:
        raise ValueError("Three-site starting counts require a variant axis.")
    if timepoint.ndim == starting.ndim + 1:
        if timepoint.shape[:-2] != starting.shape[:-1]:
            raise ValueError("Three-site starting and timepoint sample axes must match.")
        timepoint = timepoint[..., :, None, :]
    elif timepoint.ndim == starting.ndim + 2:
        if timepoint.shape[:-3] != starting.shape[:-1]:
            raise ValueError("Three-site starting and timepoint sample axes must match.")
    else:
        raise ValueError(
            "Three-site timepoint counts must be (..., time, variant) or "
            "(..., time, replicate, variant)."
        )
    if timepoint.shape[-1] != starting.shape[-1]:
        raise ValueError("Three-site counts must share the variant axis.")
    if time_values.shape != (timepoint.shape[-3],):
        raise ValueError("Three-site times must match the timepoint count axis.")
    if variant_values.shape != (starting.shape[-1],):
        raise ValueError("Three-site variants must match the count variant axis.")
    if timepoint.shape[-2] not in {1, 2}:
        raise ValueError("Three-site output counts require one or two replicates.")
    if (
        not np.isfinite(starting).all()
        or not np.isfinite(timepoint).all()
        or np.any(starting < 0)
        or np.any(timepoint < 0)
    ):
        raise ValueError("Three-site counts must be finite and nonnegative.")
    if not np.isfinite(time_values).all() or np.any(time_values <= 0):
        raise ValueError("Three-site times must be finite and positive.")

    stop_mask = np.char.find(variant_values, "*") >= 0
    if not stop_mask.any():
        raise ValueError("Three-site fitness normalization requires stop variants.")
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
    finite = np.isfinite(normalized)
    finite_count = finite.sum(axis=(-3, -2))
    return np.divide(
        np.nansum(normalized, axis=(-3, -2)),
        finite_count,
        out=np.full(starting.shape, np.nan),
        where=finite_count > 0,
    )


def _load_reported_fitness(path: Path) -> pd.DataFrame:
    reported = pd.read_csv(path)
    required = {"AAs", "fitness"}
    missing = required - set(reported.columns)
    if missing:
        raise ValueError(f"Reported TrpB table is missing columns: {sorted(missing)}")
    if reported.empty:
        raise ValueError("Reported TrpB table must contain at least one variant.")
    if reported["AAs"].isna().any() or reported["fitness"].isna().any():
        raise ValueError("Reported TrpB sequences and fitness values cannot be missing.")
    if reported["AAs"].duplicated().any():
        raise ValueError("Reported TrpB sequences must be unique.")
    reported = reported.copy()
    reported["fitness"] = pd.to_numeric(reported["fitness"], errors="raise")
    if not np.isfinite(reported["fitness"]).all():
        raise ValueError("Reported TrpB fitness values must be finite.")
    return reported


def load_trpb_landscape(
    library: str,
    *,
    count_path: str | Path,
    reported_fitness_path: str | Path,
) -> TrpBLandscape:
    """Load one full TrpB library and align reported fitness to HMC columns."""
    if library not in TRPB_LANDSCAPES:
        raise ValueError(f"library must be one of {TRPB_LANDSCAPES}, not {library!r}.")

    dataset = load_trpb_dataset(str(count_path), libname=library)
    fitted_variants = [str(variant) for variant in dataset["variants"]]
    if len(set(fitted_variants)) != len(fitted_variants):
        raise ValueError(f"{library}: fitted TrpB sequences must be unique.")

    reported = _load_reported_fitness(Path(reported_fitness_path))
    reported_variants = reported["AAs"].astype(str).tolist()
    fitted_variant_set = set(fitted_variants)
    unknown = sorted(set(reported_variants) - fitted_variant_set)
    if unknown:
        raise ValueError(
            f"{library}: {len(unknown)} reported sequences are absent from the fitted "
            f"landscape; examples: {unknown[:5]}"
        )
    reported_variant_set = set(reported_variants)
    ordered_subset = [
        variant for variant in fitted_variants if variant in reported_variant_set
    ]
    if ordered_subset != reported_variants:
        raise ValueError(
            f"{library}: reported sequences are not in fitted HMC column order."
        )

    starting_counts = np.asarray(dataset["starting_counts"], dtype=np.int64)
    timepoint_counts = np.asarray(dataset["timepoint_counts"], dtype=np.int64)
    variants = pd.DataFrame(
        {
            "variant_id": fitted_variants,
            "library": library,
            "variant_label": fitted_variants,
            "hmc_variant_index": np.arange(len(fitted_variants), dtype=np.int32),
            "input_count": _variant_totals(starting_counts),
            "output_count": _variant_totals(timepoint_counts),
        }
    )
    variants["count_ratio"] = np.divide(
        variants["output_count"],
        variants["input_count"],
        out=np.full(len(variants), np.nan, dtype=float),
        where=variants["input_count"].to_numpy() > 0,
    )

    reported_columns = ["AAs", "fitness"]
    reported_renames = {"AAs": "variant_id", "fitness": "reported_fitness"}
    if "active" in reported:
        reported_columns.append("active")
    if "# Stop" in reported:
        reported_columns.append("# Stop")
        reported_renames["# Stop"] = "number_of_stops"
    variants = variants.merge(
        reported[reported_columns].rename(columns=reported_renames),
        on="variant_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )
    variants["reference_fitness"] = variants["reported_fitness"]
    variants["reference_fitness_source"] = np.where(
        variants["reported_fitness"].notna(),
        "reported",
        "unavailable",
    )
    variants["is_ranking_candidate"] = variants["reported_fitness"].notna()

    return TrpBLandscape(
        library=library,
        variants=variants,
        starting_counts=starting_counts,
        timepoint_counts=timepoint_counts,
        times=np.asarray(dataset["times"], dtype=float),
    )


def _growth_target(
    library: str,
    growth_rate_prior: GrowthRatePrior,
) -> tuple[str, Literal["log", "positive", "inverse"]]:
    if library in {"libA", "libB", "libC"}:
        if growth_rate_prior not in {"exponential", "lomax"}:
            raise ValueError(
                "Flat TrpB libraries support exponential or Lomax growth rates."
            )
        return "r", "positive"
    if growth_rate_prior in {"exponential", "lomax"}:
        return "log_r_mean", "log"
    if growth_rate_prior == "gamma":
        return "inv_r_mean", "inverse"
    raise ValueError(
        "growth_rate_prior must be 'exponential', 'gamma', or 'lomax'."
    )


def _to_log_mean_growth(
    values: np.ndarray,
    transform: Literal["log", "positive", "inverse"],
) -> np.ndarray:
    return transform_hmc_score(values, transform)


def build_trpb_hmc_score_ensemble(
    landscape: TrpBLandscape,
    hmc_path: str | Path,
    *,
    growth_rate_prior: GrowthRatePrior,
    growth_curve: GrowthCurve,
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
    """Extract the full TrpB posterior log-mean-growth matrix."""
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")
    target_name, target_transform = _growth_target(
        landscape.library,
        growth_rate_prior,
    )
    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="trpb",
        landscape=f"trpb-{landscape.library}",
        inference_method="hmc",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    analysis_variants = landscape.variants.drop(
        columns="is_ranking_candidate",
        errors="ignore",
    ).copy()
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        analysis_variants["hmc_variant_index"].to_numpy(dtype=np.int64),
    )
    extra_metadata: dict[str, object] = {
        "library": landscape.library,
        "model": f"{growth_rate_prior.title()}-{growth_curve.title()}",
        "growth_rate_prior": growth_rate_prior,
        "growth_curve": growth_curve,
        "primary_score": "log_mean_growth",
        "ranking_universe": "all_fitted_variants",
        "reference_fitness_source": "reported sequence-aligned subset",
        "reported_fitness_available": True,
        "reported_fitness_coverage": float(
            landscape.variants["reported_fitness"].notna().mean()
        ),
    }
    if job_id is not None:
        extra_metadata["job_id"] = job_id
    if artifact_kind is not None:
        extra_metadata["artifact_kind"] = artifact_kind
    return _build_hmc_scores(
        hmc_path=resolved_path,
        identity=identity,
        posterior_variable=target_name,
        posterior_transform=target_transform,
        score_name="log_mean_growth",
        score_contract=(
            f"trpb-{landscape.library.lower()}-{target_name}-{target_transform}-v1"
        ),
        variants=landscape.variants,
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


def _posterior_count_ratio(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
) -> np.ndarray:
    starting_axes = tuple(range(1, starting_counts.ndim - 1))
    timepoint_axes = tuple(range(1, timepoint_counts.ndim - 1))
    starting_total = starting_counts.sum(axis=starting_axes, dtype=float)
    timepoint_total = timepoint_counts.sum(axis=timepoint_axes, dtype=float)
    ratio = np.divide(
        timepoint_total,
        starting_total,
        out=np.full_like(timepoint_total, np.nan),
        where=starting_total > 0,
    )
    return ratio.astype(np.float32, copy=False)


def _block_scores(
    *,
    target,
    starting_variable,
    timepoint_variable,
    sample_ids: np.ndarray,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
    local_indices: np.ndarray,
    score_names: tuple[str, ...],
    transform: Literal["log", "positive", "inverse"],
) -> dict[str, np.ndarray]:
    scores = {}
    if "log_mean_growth" in score_names:
        values = _read_native_samples(
            target,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )[:, local_indices]
        scores["log_mean_growth"] = _to_log_mean_growth(values, transform)
    if "count_ratio" in score_names:
        starting_counts = _read_native_samples(
            starting_variable,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )[..., local_indices]
        timepoint_counts = _read_native_samples(
            timepoint_variable,
            sample_ids,
            draws_per_chain=draws_per_chain,
            variant_start=variant_start,
            variant_stop=variant_stop,
        )[..., local_indices]
        scores["count_ratio"] = _posterior_count_ratio(
            starting_counts,
            timepoint_counts,
        )
    return scores


def _observed_counts_status(handle: h5py.File, landscape: TrpBLandscape) -> str:
    observed = handle.get("observed_data")
    if observed is None:
        return "not_embedded"
    required = {"starting_counts", "timepoint_counts"}
    missing = required - set(observed)
    if missing:
        raise ValueError(f"HMC observed_data is missing variables: {sorted(missing)}")
    if not np.array_equal(
        observed["starting_counts"][:],
        landscape.starting_counts,
    ) or not np.array_equal(
        observed["timepoint_counts"][:],
        landscape.timepoint_counts,
    ):
        raise ValueError(
            f"{landscape.library}: HMC observed counts differ from the raw count table."
        )
    return "matched"


def build_trpb_mle_score_ensemble(
    landscape: TrpBLandscape,
    mle_path: str | Path,
    *,
    growth_rate_prior: GrowthRatePrior,
    growth_curve: GrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    variant_chunk_size: int | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Build a verified Johnston ensemble from sampled-MLE count draws."""
    if landscape.library not in TRPB_LANDSCAPES:
        raise ValueError(f"Unknown TrpB landscape {landscape.library!r}.")
    _growth_target(landscape.library, growth_rate_prior)
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")

    resolved_path = Path(mle_path)
    identity = AnalysisIdentity(
        protein_family="trpb",
        landscape=f"trpb-{landscape.library}",
        inference_method="mle_predictive",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    variant_ids = landscape.variants["variant_id"].to_numpy(dtype=str)
    score_function = (
        trpb_johnston_fitness
        if landscape.library == "four-site"
        else trpb_three_site_fitness
    )
    observed_proxy = score_function(
        landscape.starting_counts,
        landscape.timepoint_counts,
        variant_ids,
        landscape.times,
    )
    reported = landscape.variants["reported_fitness"].to_numpy(dtype=float)
    observed_finite = np.isfinite(observed_proxy)
    reported_finite = np.isfinite(reported)
    if not np.array_equal(observed_finite, reported_finite):
        raise ValueError(
            f"{landscape.library}: observed and reported fitness coverage differs."
        )
    oracle_error = np.abs(observed_proxy[reported_finite] - reported[reported_finite])
    if not np.allclose(
        observed_proxy[reported_finite],
        reported[reported_finite],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{landscape.library}: observed counts do not reproduce reported fitness."
        )
    variants = landscape.variants.copy()
    variants["is_ranking_candidate"] = reported_finite
    candidate_mask = variants["is_ranking_candidate"].to_numpy(dtype=bool)

    n_variants = landscape.n_hmc_variants
    n_times = len(landscape.times)
    four_site = landscape.library == "four-site"
    n_replicates = (
        landscape.starting_counts.shape[-2]
        if four_site
        else (
            landscape.timepoint_counts.shape[-2]
            if landscape.timepoint_counts.ndim == 3
            else 1
        )
    )
    stop_mask = np.char.find(variant_ids, "*") >= 0
    retained_times = (
        landscape.times != 12
        if four_site
        else np.ones(n_times, dtype=bool)
    )
    expected_starting_totals = landscape.starting_counts.sum(axis=-1, dtype=np.int64)
    expected_timepoint_totals = landscape.timepoint_counts.sum(
        axis=-1,
        dtype=np.int64,
    )
    started = time.monotonic()
    with h5py.File(resolved_path, "r") as handle:
        if _observed_counts_status(handle, landscape) != "matched":
            raise ValueError("The sampled-MLE artifact must embed matching counts.")
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
                "TrpB posterior_predictive must contain starting_counts and "
                "timepoint_counts."
            )
        if starting_object.ndim < 3 or starting_object.shape[0] != 1:
            raise ValueError("TrpB predictive starting_counts has invalid axes.")
        n_samples = int(starting_object.shape[1])
        if starting_object.shape != (
            1,
            n_samples,
            *landscape.starting_counts.shape,
        ):
            raise ValueError(
                "TrpB predictive starting_counts does not match the source axes."
            )
        if timepoint_object.shape != (
            1,
            n_samples,
            *landscape.timepoint_counts.shape,
        ):
            raise ValueError(
                "TrpB predictive timepoint_counts does not match the source axes."
            )
        if n_samples < 2:
            raise ValueError("TrpB proxy analysis requires at least two MLE count draws.")
        if variant_chunk_size is None:
            stored_chunk_sizes = [
                dataset.chunks[-1]
                for dataset in (starting_object, timepoint_object)
                if dataset.chunks is not None
            ]
            variant_chunk_size = min(stored_chunk_sizes, default=n_variants)
        if variant_chunk_size < 1:
            raise ValueError("variant_chunk_size must be positive.")

        stop_sum = np.zeros((n_samples, n_times, n_replicates), dtype=float)
        stop_count = np.zeros((n_samples, n_times, n_replicates), dtype=np.int64)
        maximum = np.full((n_samples, n_times, n_replicates), -np.inf)
        starting_totals = np.zeros(
            (n_samples, *landscape.starting_counts.shape[:-1]),
            dtype=np.int64,
        )
        timepoint_totals = np.zeros(
            (n_samples, *landscape.timepoint_counts.shape[:-1]),
            dtype=np.int64,
        )
        block_starts = list(range(0, n_variants, variant_chunk_size))
        if show_progress:
            print(
                f"TrpB {landscape.library}: {n_samples:,} MLE predictive draws, "
                f"{n_variants:,} variants",
                flush=True,
            )
            print("  pass 1: derive stop means and maxima", flush=True)
        for block_number, variant_start in enumerate(block_starts, start=1):
            variant_stop = min(variant_start + variant_chunk_size, n_variants)
            starting = np.asarray(
                starting_object[0, ..., variant_start:variant_stop]
            )
            starting_totals += starting.sum(axis=-1, dtype=np.int64)
            average_input = starting.mean(axis=-2) if four_site else starting
            block_stop_mask = stop_mask[variant_start:variant_stop]
            for time_index in range(n_times):
                output = np.asarray(
                    timepoint_object[
                        0,
                        :,
                        time_index,
                        ...,
                        variant_start:variant_stop,
                    ]
                )
                timepoint_totals[:, time_index] += output.sum(
                    axis=-1,
                    dtype=np.int64,
                )
                if output.ndim == 2:
                    output = output[:, None, :]
                valid = (average_input >= 10) & (output > 0).all(axis=-2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_ratio = np.log(output / average_input[:, None, :])
                valid_log_ratio = np.where(valid[:, None, :], log_ratio, np.nan)
                stop_values = valid_log_ratio[..., block_stop_mask]
                stop_sum[:, time_index] += np.nansum(stop_values, axis=-1)
                stop_count[:, time_index] += np.isfinite(stop_values).sum(axis=-1)
                block_maximum = np.max(
                    np.where(np.isfinite(valid_log_ratio), valid_log_ratio, -np.inf),
                    axis=-1,
                )
                maximum[:, time_index] = np.maximum(
                    maximum[:, time_index],
                    block_maximum,
                )
            if show_progress:
                print(
                    f"    normalization chunks {block_number}/{len(block_starts)}",
                    flush=True,
                )

        if not np.array_equal(
            starting_totals,
            np.broadcast_to(expected_starting_totals, starting_totals.shape),
        ):
            raise ValueError(
                "TrpB predictive starting-count totals differ from observed data."
            )
        if not np.array_equal(
            timepoint_totals,
            np.broadcast_to(expected_timepoint_totals, timepoint_totals.shape),
        ):
            raise ValueError(
                "TrpB predictive timepoint-count totals differ from observed data."
            )
        if np.any(stop_count == 0):
            raise ValueError("A TrpB predictive draw has no valid stop normalization.")
        stop_mean = stop_sum / stop_count
        normalization_range = maximum - stop_mean
        if not np.isfinite(normalization_range).all() or np.any(
            normalization_range <= 0
        ):
            raise ValueError("A TrpB predictive draw has invalid max normalization.")

        sampled_proxy = np.full((n_samples, n_variants), np.nan, dtype=np.float32)
        finite_values = 0
        finite_per_sample = np.zeros(n_samples, dtype=np.int64)
        if show_progress:
            print("  pass 2: calculate variant fitness", flush=True)
        for block_number, variant_start in enumerate(block_starts, start=1):
            variant_stop = min(variant_start + variant_chunk_size, n_variants)
            starting = np.asarray(
                starting_object[0, ..., variant_start:variant_stop]
            )
            average_input = starting.mean(axis=-2) if four_site else starting
            score_sum = np.zeros((n_samples, variant_stop - variant_start), dtype=float)
            score_count = np.zeros(
                (n_samples, variant_stop - variant_start),
                dtype=np.int16,
            )
            for time_index in np.flatnonzero(retained_times):
                output = np.asarray(
                    timepoint_object[
                        0,
                        :,
                        time_index,
                        ...,
                        variant_start:variant_stop,
                    ]
                )
                if output.ndim == 2:
                    output = output[:, None, :]
                valid = (average_input >= 10) & (output > 0).all(axis=-2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_ratio = np.log(output / average_input[:, None, :])
                normalized = (
                    log_ratio - stop_mean[:, time_index, :, None]
                ) / normalization_range[:, time_index, :, None]
                normalized = np.where(valid[:, None, :], normalized, np.nan)
                score_sum += np.nansum(normalized, axis=-2)
                score_count += np.isfinite(normalized).sum(axis=-2)
            block_proxy = np.divide(
                score_sum,
                score_count,
                out=np.full_like(score_sum, np.nan),
                where=score_count > 0,
            ).astype(np.float32)
            sampled_proxy[:, variant_start:variant_stop] = block_proxy
            block_finite = np.isfinite(block_proxy)
            finite_values += int(block_finite.sum())
            finite_per_sample += block_finite[
                :, candidate_mask[variant_start:variant_stop]
            ].sum(axis=-1)
            if show_progress:
                print(
                    f"    fitness chunks {block_number}/{len(block_starts)}",
                    flush=True,
                )

    score_name = TRPB_PREDICTIVE_SCORE_NAME
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
            "variant": np.arange(n_variants, dtype=np.int32),
        },
    )
    metadata: dict[str, object] = {
        **identity.metadata(),
        "fit_method": "mle",
        "artifact_kind": "sampled_mle",
        "job_id": job_id,
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "scores": [score_name],
        "score_contracts": {score_name: TRPB_PREDICTIVE_SCORE_CONTRACT},
        "score_status": {score_name: "verified"},
        "higher_is_better": {score_name: True},
        "n_samples": n_samples,
        "n_variants": n_variants,
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_excluded_from_ranking": int((~candidate_mask).sum()),
        "n_reported_variants": int(reported_finite.sum()),
        "variant_chunk_size": variant_chunk_size,
        "finite_proxy_values": finite_values,
        "min_finite_variants_per_sample": int(finite_per_sample.min()),
        "max_finite_variants_per_sample": int(finite_per_sample.max()),
        "finite_ranking_candidates_per_sample": finite_per_sample.tolist(),
        "normalization_universe": (
            "all fitted variants passing the source filter per draw and timepoint"
        ),
        "source_filter": (
            "mean starting count >= 10 and both replicate outputs > 0"
            if four_site
            else "starting count >= 10 and all output replicates > 0"
        ),
        "excluded_times_hours": [12.0] if four_site else [],
        "count_variables": {
            "input": "starting_counts",
            "output": "timepoint_counts",
        },
        "count_axes": {
            "starting_counts": ["chain", "draw"]
            + (["replicate"] if four_site else [])
            + ["variant"],
            "timepoint_counts": ["chain", "draw", "time"]
            + (["replicate"] if landscape.timepoint_counts.ndim == 3 else [])
            + ["variant"],
        },
        "observed_counts_status": "matched",
        "observed_oracle": {
            "source_column": "fitness",
            "n_compared": int(reported_finite.sum()),
            "coverage_match": True,
            "max_abs_error": float(oracle_error.max(initial=0.0)),
            "rtol": 1e-12,
            "atol": 1e-12,
        },
        "elapsed_seconds": time.monotonic() - started,
        "source_artifact": _file_signature(resolved_path),
    }
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
    )


def build_trpb_hmc_count_score_ensemble(
    landscape: TrpBLandscape,
    hmc_path: str | Path,
    *,
    growth_rate_prior: GrowthRatePrior,
    growth_curve: GrowthCurve,
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
    """Build Johnston proxies from paired HMC posterior-predictive draws."""
    if landscape.library not in TRPB_LANDSCAPES:
        raise ValueError(f"Unknown TrpB landscape {landscape.library!r}.")
    _growth_target(landscape.library, growth_rate_prior)
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")

    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="trpb",
        landscape=f"trpb-{landscape.library}",
        inference_method="hmc_count_based",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or resolved_path.stem,
    )
    variant_ids = landscape.variants["variant_id"].to_numpy(dtype=str)
    score_function = (
        trpb_johnston_fitness
        if landscape.library == "four-site"
        else trpb_three_site_fitness
    )
    observed_proxy = score_function(
        landscape.starting_counts,
        landscape.timepoint_counts,
        variant_ids,
        landscape.times,
    )
    reported = landscape.variants["reported_fitness"].to_numpy(dtype=float)
    observed_finite = np.isfinite(observed_proxy)
    reported_finite = np.isfinite(reported)
    if not np.array_equal(observed_finite, reported_finite):
        raise ValueError(
            f"{landscape.library}: observed and reported fitness coverage differs."
        )
    oracle_error = np.abs(observed_proxy[reported_finite] - reported[reported_finite])
    if not np.allclose(
        observed_proxy[reported_finite],
        reported[reported_finite],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{landscape.library}: observed counts do not reproduce reported fitness."
        )
    variants = landscape.variants.copy()
    variants["is_ranking_candidate"] = reported_finite
    candidate_mask = variants["is_ranking_candidate"].to_numpy(dtype=bool)
    n_variants = landscape.n_hmc_variants
    n_times = len(landscape.times)
    four_site = landscape.library == "four-site"
    n_replicates = (
        landscape.starting_counts.shape[-2]
        if four_site
        else (
            landscape.timepoint_counts.shape[-2]
            if landscape.timepoint_counts.ndim == 3
            else 1
        )
    )
    stop_mask = np.char.find(variant_ids, "*") >= 0
    retained_times = (
        landscape.times != 12
        if four_site
        else np.ones(n_times, dtype=bool)
    )

    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"trpb_{landscape.library}_hmc_count_proxy_",
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
            if _observed_counts_status(handle, landscape) != "matched":
                raise ValueError("The HMC artifact must embed matching TrpB counts.")
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
                    "TrpB posterior_predictive must contain starting_counts and "
                    "timepoint_counts."
                )
            if starting_object.ndim != landscape.starting_counts.ndim + 2:
                raise ValueError("TrpB predictive starting_counts has invalid axes.")
            n_chains, draws_per_chain = map(int, starting_object.shape[:2])
            if starting_object.shape != (
                n_chains,
                draws_per_chain,
                *landscape.starting_counts.shape,
            ):
                raise ValueError(
                    "TrpB predictive starting_counts does not match the source axes."
                )
            if timepoint_object.shape != (
                n_chains,
                draws_per_chain,
                *landscape.timepoint_counts.shape,
            ):
                raise ValueError(
                    "TrpB predictive timepoint_counts does not match the source axes."
                )
            selected_sample_ids = _sample_ids(
                n_chains * draws_per_chain,
                n_samples,
                seed=seed,
            )
            if len(selected_sample_ids) < 2:
                raise ValueError("TrpB proxy analysis requires at least two HMC samples.")
            if variant_chunk_size is None:
                stored_chunks = [
                    dataset.chunks[-1]
                    for dataset in (starting_object, timepoint_object)
                    if dataset.chunks is not None
                ]
                variant_chunk_size = min(stored_chunks, default=n_variants)
            if variant_chunk_size < 1:
                raise ValueError("variant_chunk_size must be positive.")

            expected_starting_totals = landscape.starting_counts.sum(
                axis=-1,
                dtype=np.int64,
            )
            expected_timepoint_totals = landscape.timepoint_counts.sum(
                axis=-1,
                dtype=np.int64,
            )
            starting_totals = np.zeros(
                (len(selected_sample_ids), *landscape.starting_counts.shape[:-1]),
                dtype=np.int64,
            )
            timepoint_totals = np.zeros(
                (len(selected_sample_ids), *landscape.timepoint_counts.shape[:-1]),
                dtype=np.int64,
            )
            stop_sum = np.zeros(
                (len(selected_sample_ids), n_times, n_replicates),
                dtype=float,
            )
            stop_count = np.zeros(
                (len(selected_sample_ids), n_times, n_replicates),
                dtype=np.int64,
            )
            maximum = np.full(
                (len(selected_sample_ids), n_times, n_replicates),
                -np.inf,
            )
            block_starts = list(range(0, n_variants, variant_chunk_size))
            if show_progress:
                print(
                    f"trpb-{landscape.library}: {len(selected_sample_ids):,} HMC "
                    f"predictive samples, {n_variants:,} variants",
                    flush=True,
                )
                print("  pass 1: derive stop means and maxima", flush=True)
            for block_number, variant_start in enumerate(block_starts, start=1):
                variant_stop = min(variant_start + variant_chunk_size, n_variants)
                block_stop_mask = stop_mask[variant_start:variant_stop]
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
                    average_input = starting.mean(axis=-2) if four_site else starting
                    for time_index in range(n_times):
                        output = timepoint[:, time_index]
                        if output.ndim == 2:
                            output = output[:, None, :]
                        valid = (average_input >= 10) & (output > 0).all(axis=-2)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            log_ratio = np.log(output / average_input[:, None, :])
                        valid_log_ratio = np.where(
                            valid[:, None, :],
                            log_ratio,
                            np.nan,
                        )
                        stop_values = valid_log_ratio[..., block_stop_mask]
                        stop_sum[sample_start:sample_stop, time_index] += np.nansum(
                            stop_values,
                            axis=-1,
                        )
                        stop_count[
                            sample_start:sample_stop,
                            time_index,
                        ] += np.isfinite(stop_values).sum(axis=-1)
                        block_maximum = np.max(
                            np.where(
                                np.isfinite(valid_log_ratio),
                                valid_log_ratio,
                                -np.inf,
                            ),
                            axis=-1,
                        )
                        maximum[
                            sample_start:sample_stop,
                            time_index,
                        ] = np.maximum(
                            maximum[sample_start:sample_stop, time_index],
                            block_maximum,
                        )
                if show_progress:
                    print(
                        f"    normalization chunks {block_number}/{len(block_starts)}",
                        flush=True,
                    )

            if not np.array_equal(
                starting_totals,
                np.broadcast_to(expected_starting_totals, starting_totals.shape),
            ):
                raise ValueError(
                    "TrpB predictive starting-count totals differ from observed data."
                )
            if not np.array_equal(
                timepoint_totals,
                np.broadcast_to(expected_timepoint_totals, timepoint_totals.shape),
            ):
                raise ValueError(
                    "TrpB predictive timepoint-count totals differ from observed data."
                )
            if np.any(stop_count == 0):
                raise ValueError("A TrpB predictive draw has no valid stop normalization.")
            stop_mean = stop_sum / stop_count
            normalization_range = maximum - stop_mean
            if not np.isfinite(normalization_range).all() or np.any(
                normalization_range <= 0
            ):
                raise ValueError("A TrpB predictive draw has invalid max normalization.")

            sampled_proxy = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(selected_sample_ids), n_variants),
            )
            sampled_proxy[:] = np.nan
            finite_per_sample = np.zeros(len(selected_sample_ids), dtype=np.int64)
            finite_ranking_candidates = np.zeros(
                len(selected_sample_ids),
                dtype=np.int64,
            )
            if show_progress:
                print("  pass 2: calculate variant fitness", flush=True)
            for block_number, variant_start in enumerate(block_starts, start=1):
                variant_stop = min(variant_start + variant_chunk_size, n_variants)
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
                    average_input = starting.mean(axis=-2) if four_site else starting
                    score_sum = np.zeros(
                        (sample_stop - sample_start, variant_stop - variant_start),
                        dtype=float,
                    )
                    score_count = np.zeros_like(score_sum, dtype=np.int16)
                    for time_index in np.flatnonzero(retained_times):
                        output = timepoint[:, time_index]
                        if output.ndim == 2:
                            output = output[:, None, :]
                        valid = (average_input >= 10) & (output > 0).all(axis=-2)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            log_ratio = np.log(output / average_input[:, None, :])
                        normalized = (
                            log_ratio
                            - stop_mean[
                                sample_start:sample_stop,
                                time_index,
                                :,
                                None,
                            ]
                        ) / normalization_range[
                            sample_start:sample_stop,
                            time_index,
                            :,
                            None,
                        ]
                        normalized = np.where(valid[:, None, :], normalized, np.nan)
                        score_sum += np.nansum(normalized, axis=-2)
                        score_count += np.isfinite(normalized).sum(axis=-2)
                    block_proxy = np.divide(
                        score_sum,
                        score_count,
                        out=np.full_like(score_sum, np.nan),
                        where=score_count > 0,
                    ).astype(np.float32)
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
                        f"    fitness chunks {block_number}/{len(block_starts)} "
                        f"({(time.monotonic() - started) / 60:.1f} min)",
                        flush=True,
                    )
            sampled_proxy.flush()
    except Exception:
        del sampled_proxy
        temporary_path.unlink(missing_ok=True)
        raise

    score_name = TRPB_PREDICTIVE_SCORE_NAME
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
    metadata: dict[str, object] = {
        **identity.metadata(),
        "fit_method": "hmc_count_based",
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "uncertainty_target": HMC_COUNT_BASED_TARGET,
        "scores": [score_name],
        "score_contracts": {score_name: TRPB_PREDICTIVE_SCORE_CONTRACT},
        "score_status": {score_name: "verified"},
        "higher_is_better": {score_name: True},
        "n_chains": n_chains,
        "draws_per_chain": draws_per_chain,
        "n_source_samples": n_chains * draws_per_chain,
        "n_samples": len(selected_sample_ids),
        "sample_selection_seed": seed,
        "n_variants": n_variants,
        "n_ranking_candidates": int(candidate_mask.sum()),
        "n_excluded_from_ranking": int((~candidate_mask).sum()),
        "n_reported_variants": int(reported_finite.sum()),
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
        "normalization_universe": (
            "all fitted variants passing the source filter per draw and timepoint"
        ),
        "source_filter": (
            "mean starting count >= 10 and both replicate outputs > 0"
            if four_site
            else "starting count >= 10 and all output replicates > 0"
        ),
        "excluded_times_hours": [12.0] if four_site else [],
        "count_variables": {
            "input": "starting_counts",
            "output": "timepoint_counts",
        },
        "count_axes": {
            "starting_counts": ["chain", "draw"]
            + (["replicate"] if four_site else [])
            + ["variant"],
            "timepoint_counts": ["chain", "draw", "time"]
            + (["replicate"] if landscape.timepoint_counts.ndim == 3 else [])
            + ["variant"],
        },
        "observed_counts_status": "matched",
        "observed_oracle": {
            "source_column": "fitness",
            "n_compared": int(reported_finite.sum()),
            "coverage_match": True,
            "max_abs_error": float(oracle_error.max(initial=0.0)),
            "rtol": 1e-12,
            "atol": 1e-12,
        },
        "source_artifact": _file_signature(resolved_path),
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [],
    }
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=variants,
        scores=scores,
        temporary_paths=(temporary_path,),
    )


def build_trpb_four_site_mle_score_ensemble(
    landscape: TrpBLandscape,
    mle_path: str | Path,
    *,
    growth_rate_prior: GrowthRatePrior,
    growth_curve: GrowthCurve,
    fit_id: str | None = None,
    job_id: int | None = None,
    variant_chunk_size: int | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Build the verified four-site Johnston ensemble from sampled-MLE counts."""
    if landscape.library != "four-site":
        raise ValueError("The four-site MLE proxy builder requires four-site data.")
    return build_trpb_mle_score_ensemble(
        landscape,
        mle_path,
        growth_rate_prior=growth_rate_prior,
        growth_curve=growth_curve,
        fit_id=fit_id,
        job_id=job_id,
        variant_chunk_size=variant_chunk_size,
        show_progress=show_progress,
    )


def _h5netcdf_compatibility(path: Path) -> tuple[bool, str | None]:
    try:
        with h5netcdf.File(path, "r"):
            pass
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


def _result_tables(
    selection: xr.Dataset,
    variants: pd.DataFrame,
    *,
    identity: AnalysisIdentity,
    top_k: tuple[int, ...],
    interval: tuple[float, float],
    n_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracked_indices = np.asarray(
        selection["tracked_variant_index"].values,
        dtype=np.int32,
    )
    summary_frames = []
    overlap_frames = []
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
                variants["reported_fitness"],
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


def scan_trpb_hmc(
    landscape: TrpBLandscape,
    hmc_path: str | Path,
    *,
    growth_rate_prior: GrowthRatePrior,
    growth_curve: GrowthCurve,
    top_k: tuple[int, ...] = (50, 100, 1000),
    retain_k: int = 1000,
    tracked_top_n: int = 1000,
    tracked_model_top_n: int = 0,
    score_names: tuple[str, ...] = TRPB_SCORE_NAMES,
    n_samples: int | None = None,
    seed: int = 23,
    n_pairs: int = 10_000,
    interval: tuple[float, float] = (0.05, 0.95),
    fit_id: str | None = None,
    show_progress: bool = True,
) -> StabilityResult:
    """Scan all variants in one TrpB HMC fit using bounded memory."""
    if growth_curve not in {"exponential", "sigmoid"}:
        raise ValueError("growth_curve must be 'exponential' or 'sigmoid'.")
    if not score_names or len(set(score_names)) != len(score_names):
        raise ValueError("score_names must be non-empty and unique.")
    unknown_scores = set(score_names) - set(TRPB_SCORE_NAMES)
    if unknown_scores:
        raise ValueError(f"Unknown TrpB scores: {sorted(unknown_scores)}")
    target_name, transform = _growth_target(landscape.library, growth_rate_prior)
    hmc_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="trpb",
        landscape=f"trpb-{landscape.library}",
        inference_method="hmc",
        model_spec=f"{growth_rate_prior}-{growth_curve}",
        fit_id=fit_id or hmc_path.stem,
    )
    top_k = _top_k_values(top_k)
    if retain_k < top_k[-1] or retain_k > len(landscape.variants):
        raise ValueError(
            "retain_k must be at least the largest top_k and no larger than the landscape."
        )

    variants = landscape.variants
    reported_values = variants["reported_fitness"].to_numpy(dtype=float)
    reported_indices = np.flatnonzero(np.isfinite(reported_values))
    if top_k[-1] > len(reported_indices):
        raise ValueError("The largest top_k exceeds the reported-fitness cohort.")
    if not 1 <= tracked_top_n <= len(reported_indices):
        raise ValueError("tracked_top_n must not exceed the reported-fitness cohort.")
    if not 0 <= tracked_model_top_n <= retain_k:
        raise ValueError("tracked_model_top_n must be between 0 and retain_k.")
    reported_order = np.lexsort(
        (reported_indices, -reported_values[reported_indices])
    )
    tracked_indices = reported_indices[reported_order[:tracked_top_n]].astype(np.int32)
    model_reference_indices: dict[str, np.ndarray] = {}
    model_reference_scores: dict[str, np.ndarray] = {}
    model_tracked_indices: dict[str, np.ndarray] = {}
    model_tracked_values: dict[str, np.ndarray] = {}
    model_tracked_ranks: dict[str, np.ndarray] = {}

    started = time.monotonic()
    h5netcdf_compatible, h5netcdf_open_error = _h5netcdf_compatibility(hmc_path)
    count_ratio_available = False
    with h5py.File(hmc_path, "r") as handle:
        observed_counts_status = _observed_counts_status(handle, landscape)
        posterior = handle.get("posterior")
        if posterior is None or target_name not in posterior:
            raise ValueError(
                f"{growth_rate_prior.title()}-{growth_curve.title()} fit must contain "
                f"posterior/{target_name}."
            )
        target = posterior[target_name]
        if target.ndim != 3 or target.shape[-1] != landscape.n_hmc_variants:
            raise ValueError(
                f"posterior/{target_name} must be chain-by-draw-by-variant and match "
                "the raw count-table width."
            )
        predictive = handle.get("posterior_predictive")
        count_ratio_available = (
            predictive is not None
            and "starting_counts" in predictive
            and "timepoint_counts" in predictive
        )
        resolved_score_names = tuple(
            name
            for name in score_names
            if name != "count_ratio" or count_ratio_available
        )
        if not resolved_score_names:
            raise ValueError("The requested TrpB scores are unavailable in this fit.")
        starting_variable = (
            predictive["starting_counts"] if count_ratio_available else None
        )
        timepoint_variable = (
            predictive["timepoint_counts"] if count_ratio_available else None
        )
        score_variables = [target]
        if count_ratio_available:
            score_variables.extend((starting_variable, timepoint_variable))
        variant_chunks = {
            int((variable.chunks or variable.shape)[-1])
            for variable in score_variables
        }
        variant_chunk_size = int(np.gcd.reduce(list(variant_chunks)))
        n_chains, draws_per_chain = target.shape[:2]
        sample_ids = _sample_ids(n_chains * draws_per_chain, n_samples, seed=seed)
        if len(sample_ids) < 2:
            raise ValueError("At least two HMC samples are required for overlap analysis.")

        tracked_values = {
            name: np.full(
                (len(sample_ids), tracked_top_n),
                np.nan,
                dtype=np.float32,
            )
            for name in resolved_score_names
        }
        top_indices = {}
        top_scores = {}
        tracked_ranks = {}
        model_reference_values = {}
        for name in resolved_score_names:
            top_indices[name], top_scores[name] = empty_top_k(
                len(sample_ids), retain_k
            )
            tracked_ranks[name] = empty_tracked_ranks(
                len(sample_ids), tracked_top_n
            )
            if tracked_model_top_n:
                model_reference_values[name] = np.full(
                    len(variants),
                    np.nan,
                    dtype=np.float32,
                )
        block_starts = list(
            range(0, landscape.n_hmc_variants, variant_chunk_size)
        )
        tracked_blocks = np.unique(
            (tracked_indices // variant_chunk_size) * variant_chunk_size
        ).tolist()

        landscape_name = f"trpb-{landscape.library}"
        if show_progress:
            print(
                f"{landscape_name}: {len(sample_ids):,} HMC samples, "
                f"{landscape.n_hmc_variants:,} fitted variants, "
                f"{len(reported_indices):,} reported variants",
                flush=True,
            )
            print(
                f"  pass 1: collect {tracked_top_n:,} fixed reported-cohort values",
                flush=True,
            )
        for block_number, variant_start in enumerate(tracked_blocks, start=1):
            variant_stop = min(
                variant_start + variant_chunk_size,
                landscape.n_hmc_variants,
            )
            cohort_columns = np.flatnonzero(
                (tracked_indices >= variant_start) & (tracked_indices < variant_stop)
            )
            local_indices = tracked_indices[cohort_columns] - variant_start
            scores = _block_scores(
                target=target,
                starting_variable=starting_variable,
                timepoint_variable=timepoint_variable,
                sample_ids=sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=variant_start,
                variant_stop=variant_stop,
                local_indices=local_indices,
                score_names=resolved_score_names,
                transform=transform,
            )
            for name, values in scores.items():
                tracked_values[name][:, cohort_columns] = values
            if show_progress and (
                block_number == len(tracked_blocks) or block_number % 8 == 0
            ):
                print(
                    f"    tracked chunks {block_number}/{len(tracked_blocks)}",
                    flush=True,
                )

        if show_progress:
            print("  pass 2: scan every fitted variant", flush=True)
        for block_number, variant_start in enumerate(block_starts, start=1):
            variant_stop = min(
                variant_start + variant_chunk_size,
                landscape.n_hmc_variants,
            )
            local_indices = np.arange(
                variant_stop - variant_start,
                dtype=np.int64,
            )
            scores = _block_scores(
                target=target,
                starting_variable=starting_variable,
                timepoint_variable=timepoint_variable,
                sample_ids=sample_ids,
                draws_per_chain=draws_per_chain,
                variant_start=variant_start,
                variant_stop=variant_stop,
                local_indices=local_indices,
                score_names=resolved_score_names,
                transform=transform,
            )
            result_indices = np.arange(variant_start, variant_stop, dtype=np.int32)
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
                    model_reference_values[name][variant_start:variant_stop] = reference
            if show_progress and (
                block_number == len(block_starts) or block_number % 8 == 0
            ):
                print(
                    f"    full chunks {block_number}/{len(block_starts)} "
                    f"({(time.monotonic() - started) / 60:.1f} min)",
                    flush=True,
                )

        if tracked_model_top_n:
            for name in resolved_score_names:
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
                model_tracked_values[name] = np.full(
                    (len(sample_ids), tracked_model_top_n),
                    np.nan,
                    dtype=np.float32,
                )
                model_tracked_ranks[name] = empty_tracked_ranks(
                    len(sample_ids),
                    tracked_model_top_n,
                )

            target_indices = np.unique(
                np.concatenate(list(model_tracked_indices.values()))
            )
            model_blocks = np.unique(
                (target_indices // variant_chunk_size) * variant_chunk_size
            ).tolist()
            if show_progress:
                print(
                    f"  pass 3: collect {tracked_model_top_n:,} model-cohort values",
                    flush=True,
                )
            for block_number, variant_start in enumerate(model_blocks, start=1):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    landscape.n_hmc_variants,
                )
                block_targets = target_indices[
                    (target_indices >= variant_start) & (target_indices < variant_stop)
                ]
                scores = _block_scores(
                    target=target,
                    starting_variable=starting_variable,
                    timepoint_variable=timepoint_variable,
                    sample_ids=sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                    local_indices=block_targets - variant_start,
                    score_names=resolved_score_names,
                    transform=transform,
                )
                for name, values in scores.items():
                    cohort_columns = np.flatnonzero(
                        (model_tracked_indices[name] >= variant_start)
                        & (model_tracked_indices[name] < variant_stop)
                    )
                    positions = np.searchsorted(
                        block_targets,
                        model_tracked_indices[name][cohort_columns],
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
            for block_number, variant_start in enumerate(block_starts, start=1):
                variant_stop = min(
                    variant_start + variant_chunk_size,
                    landscape.n_hmc_variants,
                )
                scores = _block_scores(
                    target=target,
                    starting_variable=starting_variable,
                    timepoint_variable=timepoint_variable,
                    sample_ids=sample_ids,
                    draws_per_chain=draws_per_chain,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                    local_indices=np.arange(
                        variant_stop - variant_start,
                        dtype=np.int64,
                    ),
                    score_names=resolved_score_names,
                    transform=transform,
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

    for name in resolved_score_names:
        if np.any(top_indices[name] < 0):
            raise ValueError(
                f"{name}: fewer than retain_k finite scores in an HMC sample."
            )
        tracked_ranks[name][~np.isfinite(tracked_values[name])] = np.nan
        if tracked_model_top_n:
            model_tracked_ranks[name][
                ~np.isfinite(model_tracked_values[name])
            ] = np.nan
    selection_data = {
            "variant_index": (
                ("score_name", "sample", "selection_rank"),
                np.stack([top_indices[name] for name in resolved_score_names]),
            ),
            "score": (
                ("score_name", "sample", "selection_rank"),
                np.stack([top_scores[name] for name in resolved_score_names]),
            ),
            "tracked_score": (
                ("score_name", "sample", "tracked_variant"),
                np.stack([tracked_values[name] for name in resolved_score_names]),
            ),
            "tracked_rank": (
                ("score_name", "sample", "tracked_variant"),
                np.stack([tracked_ranks[name] for name in resolved_score_names]),
            ),
        }
    selection_coords = {
            "score_name": list(resolved_score_names),
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
                    np.stack(
                        [model_reference_indices[name] for name in resolved_score_names]
                    ),
                ),
                "model_reference_score": (
                    ("score_name", "model_reference_rank"),
                    np.stack(
                        [model_reference_scores[name] for name in resolved_score_names]
                    ),
                ),
                "model_tracked_variant_index": (
                    ("score_name", "model_tracked_variant"),
                    np.stack(
                        [model_tracked_indices[name] for name in resolved_score_names]
                    ),
                ),
                "model_tracked_score": (
                    ("score_name", "sample", "model_tracked_variant"),
                    np.stack(
                        [model_tracked_values[name] for name in resolved_score_names]
                    ),
                ),
                "model_tracked_rank": (
                    ("score_name", "sample", "model_tracked_variant"),
                    np.stack(
                        [model_tracked_ranks[name] for name in resolved_score_names]
                    ),
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
            "fit_method": "hmc_posterior",
            "model": f"{growth_rate_prior.title()}-{growth_curve.title()}",
            "retained_k": retain_k,
        },
    )
    summary, overlaps = _result_tables(
        selection,
        variants,
        identity=identity,
        top_k=top_k,
        interval=interval,
        n_pairs=n_pairs,
        seed=seed,
    )
    elapsed_seconds = time.monotonic() - started
    artifact_issues = []
    warning_messages = []
    if observed_counts_status == "not_embedded":
        artifact_issues.append("missing_observed_data")
        warning_messages.append(
            "The fit does not embed observed counts. Variant alignment uses the "
            "source count-table order and should be cross-checked against a paired "
            "fit with matching embedded counts."
        )
    if not h5netcdf_compatible:
        artifact_issues.append("h5netcdf_open_failure")
        warning_messages.append(
            "The fit cannot be opened with h5netcdf in this environment; posterior "
            "arrays were read directly with h5py. Regenerate or reconvert the fit "
            "before relying on standard NetCDF tooling."
        )
    if "count_ratio" in score_names and not count_ratio_available:
        artifact_issues.append("missing_posterior_predictive_counts")
        warning_messages.append(
            "The fit does not contain posterior-predictive starting and timepoint "
            "counts, so count_ratio could not be analyzed."
        )
    metadata: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **identity.metadata(),
        "library": landscape.library,
        "fit_method": "hmc_posterior",
        "model": f"{growth_rate_prior.title()}-{growth_curve.title()}",
        "growth_rate_prior": growth_rate_prior,
        "growth_curve": growth_curve,
        "posterior_variable": target_name,
        "posterior_transform": transform,
        "scores": list(resolved_score_names),
        "n_samples": len(sample_ids),
        "n_variants": landscape.n_hmc_variants,
        "n_reported_variants": len(reported_indices),
        "reported_fitness_coverage": len(reported_indices) / landscape.n_hmc_variants,
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "interval": list(interval),
        "seed": seed,
        "reference_fitness_source": "reported sequence-aligned subset",
        "reported_fitness_available": True,
        "posterior_predictive_count_source": (
            "stored SciStanPy posterior_predictive"
            if count_ratio_available
            else "unavailable"
        ),
        "observed_counts_status": observed_counts_status,
        "count_alignment_evidence": (
            "embedded_observed_counts"
            if observed_counts_status == "matched"
            else "source_count_table_order"
        ),
        "artifact_status": "limited" if artifact_issues else "validated",
        "artifact_issues": artifact_issues,
        "hdf5_reader": "h5py",
        "h5netcdf_compatible": h5netcdf_compatible,
        "h5netcdf_open_error": h5netcdf_open_error,
        "hmc_file": _file_signature(hmc_path),
        "scistanpy_version": version("scistanpy"),
        "elapsed_seconds": elapsed_seconds,
        "warnings": warning_messages,
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
