"""Stability analysis for persisted sampled score ensembles."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from .analysis import RESULT_SCHEMA_VERSION, _top_k_values
from .analysis_config import analysis_config_sha256, validate_analysis_config
from .candidates import ranking_candidate_mask
from .hmc_scanning import result_tables, scan_hmc_scores
from .identity import AnalysisIdentity
from .results import StabilityResult
from .score_ensembles import (
    LEGACY_PROXY_SCHEMA_VERSION,
    ScoreEnsemble,
    direct_hmc_reference_metadata,
    score_method_contract,
)


def _configure_external_reference(
    variants: pd.DataFrame,
    reference: object,
    *,
    retain_k: int,
) -> tuple[bool, bool, int, dict[str, object]]:
    if reference is None:
        variants["reported_fitness"] = np.nan
        variants["reported_rank"] = np.nan
        return (
            False,
            True,
            0,
            {
                "reference_fitness_source": "unavailable",
                "reported_fitness_available": False,
            },
        )
    if not isinstance(reference, Mapping):
        raise ValueError("external_reference must be null or a JSON object.")
    value_column = str(reference["value_column"])
    if value_column not in variants:
        raise ValueError(
            f"External reference column {value_column!r} is absent from "
            f"analysis_variants.csv; available columns are {variants.columns.tolist()}."
        )
    values = pd.to_numeric(variants[value_column], errors="raise").to_numpy(
        dtype=float
    )
    if np.isinf(values).any():
        raise ValueError(f"External reference column {value_column!r} contains infinity.")
    variants["reported_fitness"] = values
    candidate_mask = ranking_candidate_mask(variants)
    higher_is_better = reference["direction"] == "higher"
    rank_column = reference.get("rank_column")
    if rank_column is None:
        variants["reported_rank"] = (
            pd.Series(values)
            .where(candidate_mask)
            .rank(method="min", ascending=not higher_is_better)
        )
    else:
        rank_column = str(rank_column)
        if rank_column not in variants:
            raise ValueError(
                f"External reference rank column {rank_column!r} is absent from "
                f"analysis_variants.csv; available columns are {variants.columns.tolist()}."
            )
        ranks = pd.to_numeric(variants[rank_column], errors="raise").to_numpy(
            dtype=float
        )
        if np.isinf(ranks).any() or np.any(ranks[np.isfinite(ranks)] < 1):
            raise ValueError(
                f"External reference rank column {rank_column!r} must contain "
                "positive finite ranks or missing values."
            )
        variants["reported_rank"] = ranks
    identifiable = bool(reference["identifiable_top_k"])
    finite_count = int(np.isfinite(values[candidate_mask]).sum())
    configured_tracked = reference.get("tracked_top_n")
    if not identifiable and configured_tracked is not None:
        raise ValueError(
            "external_reference.tracked_top_n must be omitted when "
            "identifiable_top_k is false."
        )
    tracked_top_n = (
        min(retain_k, finite_count)
        if configured_tracked is None and identifiable
        else 0
        if configured_tracked is None
        else int(configured_tracked)
    )
    return (
        identifiable,
        higher_is_better,
        tracked_top_n,
        {
            "reference_fitness_source": str(reference["source"]),
            "reported_fitness_available": finite_count > 0,
            "reported_fitness_coverage": float(
                finite_count / int(candidate_mask.sum())
            ),
        },
    )


def analyze_score_ensemble(
    ensemble: ScoreEnsemble,
    *,
    config: Mapping[str, object] | None = None,
    top_k: tuple[int, ...] = (50, 100, 1000),
    retain_k: int = 1000,
    tracked_top_n: int = 1000,
    tracked_model_top_n: int = 100,
    n_samples: int | None = None,
    variant_chunk_size: int | None = None,
    seed: int = 23,
    n_pairs: int = 10_000,
    interval: tuple[float, float] = (0.05, 0.95),
    show_progress: bool = True,
) -> StabilityResult:
    """Build compact ranks and top-K results from a persisted score matrix."""
    ensemble.validate()
    method = score_method_contract(ensemble.metadata)
    resolved_config = (
        None if config is None else validate_analysis_config(config)
    )
    if resolved_config is not None:
        top_k = tuple(int(value) for value in resolved_config["top_k"])
        retain_k = int(resolved_config.get("retain_k") or max(top_k))
        tracked_model_top_n = int(
            resolved_config.get("tracked_model_top_n")
            or min(100, retain_k)
        )
        n_samples = resolved_config.get("n_samples")
        seed = int(resolved_config["seed"])
    is_legacy_proxy = (
        ensemble.source_schema_version == LEGACY_PROXY_SCHEMA_VERSION
    )
    score_dataset = ensemble.scores
    source_score_values = ensemble.fitness_score
    if is_legacy_proxy:
        result_variants = ensemble.variants.copy()
    else:
        matrix_indices = ensemble.analysis_variants[
            "matrix_variant_index"
        ].to_numpy(dtype=np.int64)
        result_variants = ensemble.analysis_variants.copy()
        if resolved_config is None:
            result_variants = result_variants.drop(
                columns="matrix_variant_index"
            )
            if "hmc_variant_index" in result_variants:
                if not np.array_equal(
                    result_variants["hmc_variant_index"].to_numpy(
                        dtype=np.int64
                    ),
                    matrix_indices,
                ):
                    raise ValueError(
                        "hmc_variant_index must match matrix_variant_index."
                    )
            else:
                result_variants.insert(
                    min(1, len(result_variants.columns)),
                    "hmc_variant_index",
                    matrix_indices,
                )
    identity = AnalysisIdentity(
        **{
            field: str(ensemble.metadata[field])
            for field in (
                "protein_family",
                "landscape",
                "inference_method",
                "model_spec",
                "fit_id",
            )
        }
    )
    available_score_names = tuple(score_dataset["score_name"].values.astype(str))
    if resolved_config is None:
        score_names = available_score_names
        score_directions = {name: True for name in score_names}
        reference_higher_is_better = True
        configured_reference_metadata: dict[str, object] = {}
    else:
        score_name = str(resolved_config["score_name"])
        if score_name not in available_score_names:
            raise ValueError(
                f"Configured score {score_name!r} is absent from the score bundle; "
                f"available scores are {list(available_score_names)}."
            )
        score_names = (score_name,)
        score_directions = {
            score_name: resolved_config["direction"] == "higher"
        }
        source_score_values = source_score_values.sel(score_name=[score_name])
        (
            reference_top_k_identifiable,
            reference_higher_is_better,
            tracked_top_n,
            configured_reference_metadata,
        ) = _configure_external_reference(
            result_variants,
            resolved_config.get("external_reference"),
            retain_k=retain_k,
        )
    finite_candidate_counts = ensemble.metadata.get(
        "finite_ranking_candidates_per_sample"
    )
    if finite_candidate_counts is None:
        eligible_source_positions = np.arange(
            source_score_values.sizes["sample"],
            dtype=np.int64,
        )
    else:
        finite_candidate_counts = np.asarray(finite_candidate_counts, dtype=np.int64)
        if finite_candidate_counts.shape != (source_score_values.sizes["sample"],):
            raise ValueError(
                "finite_ranking_candidates_per_sample must match the sample axis."
            )
        eligible_source_positions = np.flatnonzero(
            finite_candidate_counts >= retain_k
        ).astype(np.int64)
    if len(eligible_source_positions) < 2:
        raise ValueError(
            f"Fewer than two samples contain retain_k={retain_k} finite ranking "
            "candidates."
        )
    score_values = source_score_values.isel(sample=eligible_source_positions)
    n_fitted_variants = score_values.sizes["variant"]
    if resolved_config is None:
        reference_top_k_identifiable = bool(
            ensemble.metadata.get("reference_top_k_identifiable", True)
        )
    if not reference_top_k_identifiable and tracked_top_n != 0:
        raise ValueError(
            "tracked_top_n must be zero when the reported top-K set is not "
            "identifiable."
        )
    if variant_chunk_size is None:
        recorded_chunk_size = (
            ensemble.metadata.get("variant_chunk_size")
            if not is_legacy_proxy
            else None
        )
        if isinstance(recorded_chunk_size, int) and not isinstance(
            recorded_chunk_size,
            bool,
        ):
            variant_chunk_size = recorded_chunk_size
        else:
            chunksizes = score_values.encoding.get("chunksizes")
            variant_chunk_size = (
                int(chunksizes[-1])
                if chunksizes is not None
                else n_fitted_variants
            )
    if variant_chunk_size < 1:
        raise ValueError("variant_chunk_size must be positive.")

    def read_scores(
        *,
        sample_ids: np.ndarray,
        variant_start: int,
        variant_stop: int,
        local_indices: np.ndarray,
        **_: object,
    ) -> dict[str, np.ndarray]:
        scores = {}
        for score_name in score_names:
            values = score_values.sel(score_name=score_name).isel(
                sample=sample_ids,
                variant=slice(variant_start, variant_stop),
            )
            scores[score_name] = np.asarray(values)[:, local_indices]
        return scores

    resolved_top_k = _top_k_values(top_k)
    required_block_starts = tuple(
        int(value)
        for value in ensemble.metadata.get("required_block_starts", [])
    )
    scan = scan_hmc_scores(
        result_variants,
        identity=identity,
        score_names=score_names,
        higher_is_better=(
            score_directions if resolved_config is not None else None
        ),
        n_hmc_variants=n_fitted_variants,
        n_chains=1,
        draws_per_chain=score_values.sizes["sample"],
        variant_chunk_size=variant_chunk_size,
        read_scores=read_scores,
        top_k=resolved_top_k,
        retain_k=retain_k,
        tracked_top_n=tracked_top_n,
        tracked_model_top_n=tracked_model_top_n,
        n_samples=n_samples,
        seed=seed,
        label=identity.landscape,
        selection_attrs={
            (
                "proxy_ensemble_schema_version"
                if is_legacy_proxy
                else "score_ensemble_schema_version"
            ): ensemble.metadata["schema_version"],
            "uncertainty_target": method["uncertainty_target"],
        },
        required_block_starts=required_block_starts,
        min_model_reference_samples=(
            int(resolved_config.get("model_reference_min_finite_samples") or 2)
            if resolved_config is not None
            else 1
            if ensemble.metadata["inference_method"] == "hmc"
            else 2
        ),
        require_reported_top_k=reference_top_k_identifiable,
        reported_higher_is_better=reference_higher_is_better,
        fit_method=method["fit_method"],
        sample_kind=method["sample_kind"],
        show_progress=show_progress,
    )
    selected_source_positions = eligible_source_positions[scan.sample_ids]
    source_sample_ids = np.asarray(
        score_dataset.coords.get(
            "source_sample_id",
            xr.DataArray(
                np.arange(source_score_values.sizes["sample"], dtype=np.int64),
                dims=("sample",),
            ),
        ).values,
        dtype=np.int64,
    )
    scan.selection.coords["sample_id"] = (
        "sample",
        source_sample_ids[selected_source_positions],
    )
    if ensemble.metadata["inference_method"] == "hmc_count_based":
        coordinate_names = ("source_chain", "source_draw")
    else:
        coordinate_names = ()
    for coordinate_name in coordinate_names:
        if coordinate_name in score_dataset.coords:
            coordinate = np.asarray(
                score_dataset[coordinate_name].values,
                dtype=np.int32,
            )
            scan.selection.coords[coordinate_name] = (
                "sample",
                coordinate[selected_source_positions],
            )
    summary, overlaps = result_tables(
        scan.selection,
        result_variants,
        identity=identity,
        top_k=resolved_top_k,
        interval=interval,
        n_pairs=n_pairs,
        seed=seed,
        fit_method=method["fit_method"],
        include_reference_overlap=reference_top_k_identifiable,
        reference_higher_is_better=reference_higher_is_better,
        legacy_numpy_reductions=ensemble.metadata["inference_method"] == "hmc",
    )
    storage_metadata = {
        (
            "proxy_ensemble_schema_version"
            if is_legacy_proxy
            else "score_ensemble_schema_version"
        ): ensemble.metadata["schema_version"]
    }
    sample_metadata = (
        {"n_proxy_samples": source_score_values.sizes["sample"]}
        if is_legacy_proxy
        else {"n_score_samples": source_score_values.sizes["sample"]}
    )
    model_reference_min_finite_samples = (
        int(resolved_config.get("model_reference_min_finite_samples") or 2)
        if resolved_config is not None
        else 1
        if ensemble.metadata["inference_method"] == "hmc"
        else 2
    )
    reference_metadata: dict[str, object] = {}
    if resolved_config is not None:
        reference_metadata = configured_reference_metadata
    elif ensemble.metadata["inference_method"] == "hmc":
        reference_metadata = direct_hmc_reference_metadata(
            ensemble.metadata,
            result_variants,
        )
    metadata: dict[str, object] = {
        **ensemble.metadata,
        **reference_metadata,
        **storage_metadata,
        "schema_version": RESULT_SCHEMA_VERSION,
        "fit_method": method["fit_method"],
        "n_samples": len(scan.sample_ids),
        "n_source_samples": ensemble.metadata.get(
            "n_source_samples",
            source_score_values.sizes["sample"],
        ),
        **sample_metadata,
        "n_eligible_samples": len(eligible_source_positions),
        "n_excluded_samples": (
            source_score_values.sizes["sample"] - len(eligible_source_positions)
        ),
        "sample_eligibility_min_finite_ranking_candidates": retain_k,
        "n_variants": len(scan.candidate_indices),
        "n_fitted_variants": n_fitted_variants,
        "n_excluded_from_ranking": n_fitted_variants - len(scan.candidate_indices),
        "n_reported_variants": len(scan.reported_indices),
        "reference_top_k_identifiable": reference_top_k_identifiable,
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "model_reference_min_finite_samples": model_reference_min_finite_samples,
        "top_k": list(resolved_top_k),
        "interval": list(interval),
        "seed": seed,
    }
    if resolved_config is not None:
        metadata.update(
            {
                "analysis_config": resolved_config,
                "analysis_config_sha256": analysis_config_sha256(
                    resolved_config
                ),
                "higher_is_better": score_directions,
            }
        )
    return StabilityResult(
        metadata=metadata,
        variants=result_variants,
        selection=scan.selection,
        summary=summary,
        overlaps=overlaps,
    )
