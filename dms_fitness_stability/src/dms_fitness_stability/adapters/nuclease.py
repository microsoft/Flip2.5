"""NucB generation inputs and reported activity alignment."""

from __future__ import annotations

import ast
import re
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from flipv3.flip_dsets import load_nuclease_data

from ..analysis import RESULT_SCHEMA_VERSION, _top_k_values
from ..candidates import RANKING_CANDIDATE_COLUMN, ranking_candidate_indices
from ..hmc_scanning import (
    file_signature,
    h5netcdf_compatibility,
    read_native_samples,
    result_tables,
    scan_hmc_scores,
)
from ..hmc_score_extraction import build_hmc_score_ensemble as _build_hmc_scores
from ..identity import AnalysisIdentity
from ..results import StabilityResult
from ..score_ensembles import (
    CONDITIONAL_REPEATABILITY_TARGET,
    HMC_COUNT_BASED_TARGET,
    ScoreEnsemble,
)

NucleaseGeneration = Literal["G1", "G2", "G3", "G4"]
NUCLEASE_GENERATIONS = ("G1", "G2", "G3", "G4")
NUCB_WT_SEQUENCE = (
    "MIKKWAVHLLFSALVLLGLSGGAAYSPQHAEGAARYDDVLYFPASRYPETGAHISDAIKAGHADVCTIERSG"
    "ADKRRQESLKGIPTKPGFDRDEWPMAMCEEGGKGASVRYVSSSDNRGAGSWVGNRLNGYADGTRILFIVQ"
)
NUCLEASE_SCORE_VARIABLE = {
    "G1": "absolute_mean_log_fluorescence",
    "G2": "log_fluorescence",
    "G3": "log_fluorescence",
    "G4": "log_fluorescence",
}
NUCLEASE_SCORE_NAME = "model_fluorescence"
NUCLEASE_PREDICTIVE_SCORE_NAME = "highest_gate_enrichment"
NUCLEASE_PREDICTIVE_SCORE_CONTRACT = "nuc-g3-highest-gate-enrichment-v1"
NUCLEASE_PREDICTIVE_SCORE_CONTRACTS = {
    "G1": "nuc-g1-highest-gate-enrichment-v1",
    "G2": "nuc-g2-highest-gate-enrichment-v1",
    "G3": NUCLEASE_PREDICTIVE_SCORE_CONTRACT,
    "G4": "nuc-g4-highest-gate-enrichment-v1",
}
NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS = {
    "input": 6_000_424,
    "output": 1_937_171,
}
NUCLEASE_REPORTED_FITNESS_COLUMN = {
    "G1": "ef_2_high_reseq_g1",
    "G2": "ef_1_97.5_g2",
    "G3": "ef_2_99_g3",
    "G4": "ef_4_99.5_g4",
}

_MUTATION_PATTERN = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")
_COUNT_NAMES = {
    "G1": ("ic1", "ic2", "ic3", "lc", "hc1", "hc2", "hc3"),
    "G2": ("ic1", "ic2", "c86", "c93", "c975"),
    "G3": ("ic1", "oc1"),
    "G4": ("ic1", "oc1"),
}
_INPUT_NAMES = {
    "G1": ("ic1", "ic2", "ic3"),
    "G2": ("ic1", "ic2"),
    "G3": ("ic1",),
    "G4": ("ic1",),
}
_OUTPUT_NAMES = {
    "G1": ("lc", "hc1", "hc2", "hc3"),
    "G2": ("c86", "c93", "c975"),
    "G3": ("oc1",),
    "G4": ("oc1",),
}
_THRESHOLD_NAMES = {"G1": ("lt", "ht"), "G2": ("ft",), "G3": ("ft",), "G4": ("ft",)}


@dataclass(frozen=True)
class NucleaseLandscape:
    """One fitted NucB generation with source activity annotations."""

    generation: str
    variants: pd.DataFrame
    counts: dict[str, np.ndarray]
    thresholds: dict[str, np.ndarray]
    wild_type_index: int

    @property
    def n_fitted_variants(self) -> int:
        return len(self.variants)


def _parse_mutation(value: object) -> tuple[str, int, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Malformed mutation entry: {value!r}")
    old, position, new = value
    match = _MUTATION_PATTERN.fullmatch(f"{old}{position}{new}")
    if match is None or int(match.group(2)) <= 0:
        raise ValueError(f"Malformed mutation entry: {value!r}")
    return match.group(1), int(match.group(2)), match.group(3)


def mutation_signature(value: object) -> str:
    """Return the mutation signature used by ``flipv3.load_nuclease_data``."""
    parsed: object
    if not isinstance(value, str):
        parsed = value
    else:
        stripped = value.strip()
        if stripped in {"", "[]", "()"}:
            return ""
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            tokens = re.split(r"[,_]", stripped)
            parsed = []
            for token in tokens:
                match = _MUTATION_PATTERN.fullmatch(token.strip())
                if match is None:
                    raise ValueError(f"Malformed mutation string: {value!r}")
                parsed.append((match.group(1), int(match.group(2)), match.group(3)))

    if parsed in ([], ()):
        return ""
    if isinstance(parsed, (list, tuple)) and len(parsed) == 3 and not isinstance(
        parsed[0], (list, tuple)
    ):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Malformed mutation value: {value!r}")
    mutations = sorted((_parse_mutation(item) for item in parsed), key=lambda item: item[1])
    positions = [position for _, position, _ in mutations]
    if len(set(positions)) != len(positions):
        raise ValueError(f"Mutation positions must be unique: {value!r}")
    return "_".join(f"{old}{position}{new}" for old, position, new in mutations)


def _variant_totals(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    totals = None
    for name in names:
        array = arrays[name]
        axes = tuple(range(array.ndim - 1))
        values = array.sum(axis=axes, dtype=np.int64) if axes else array.astype(np.int64)
        totals = values if totals is None else totals + values
    assert totals is not None
    return totals


def normalized_count_enrichment(
    input_counts: np.ndarray,
    output_counts: np.ndarray,
) -> np.ndarray:
    """Return output frequency divided by input frequency along the variant axis."""
    input_values = np.asarray(input_counts, dtype=float)
    output_values = np.asarray(output_counts, dtype=float)
    if input_values.ndim == 0 or input_values.shape != output_values.shape:
        raise ValueError("Input and output counts must have the same non-scalar shape.")
    if (
        not np.isfinite(input_values).all()
        or not np.isfinite(output_values).all()
        or np.any(input_values < 0)
        or np.any(output_values < 0)
    ):
        raise ValueError("Input and output counts must be finite and nonnegative.")

    input_totals = input_values.sum(axis=-1, keepdims=True)
    output_totals = output_values.sum(axis=-1, keepdims=True)
    if np.any(input_totals <= 0) or np.any(output_totals <= 0):
        raise ValueError("Every count sample must have positive input and output totals.")

    input_frequencies = input_values / input_totals
    output_frequencies = output_values / output_totals
    return np.divide(
        output_frequencies,
        input_frequencies,
        out=np.full_like(output_frequencies, np.nan),
        where=input_frequencies > 0,
    )


def nuclease_highest_gate_enrichment(
    generation: NucleaseGeneration,
    counts: dict[str, np.ndarray],
) -> np.ndarray:
    """Reproduce verified highest-gate enrichment from count arrays."""
    required = {
        "G1": {"ic1", "ic2", "ic3", "hc2"},
        "G2": {"ic1", "ic2", "c975"},
        "G3": {"ic1", "oc1"},
        "G4": {"ic1", "oc1"},
    }[generation]
    missing = required - set(counts)
    if missing:
        raise ValueError(f"{generation} counts are missing variables: {sorted(missing)}")

    if generation == "G1":
        input_counts = (
            np.asarray(counts["ic1"]).sum(axis=-2)
            + np.asarray(counts["ic2"]).sum(axis=-2)
            + np.asarray(counts["ic3"])
        )
        output_counts = np.asarray(counts["hc2"])[..., 1, :]
        if input_counts.shape != output_counts.shape:
            raise ValueError("G1 pooled inputs and resequenced high gate must align.")
        if (
            not np.isfinite(input_counts).all()
            or not np.isfinite(output_counts).all()
            or np.any(input_counts < 0)
            or np.any(output_counts < 0)
        ):
            raise ValueError("G1 pooled input and output counts must be nonnegative.")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                output_counts,
                input_counts,
                out=np.full_like(input_counts, np.nan, dtype=float),
                where=input_counts > 0,
            ) * (
                NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS["input"]
                / NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS["output"]
            )
    if generation == "G2":
        input_counts = np.asarray(counts["ic1"]).sum(axis=-2) + np.asarray(
            counts["ic2"]
        )
        output_counts = np.asarray(counts["c975"])
        return normalized_count_enrichment(input_counts, output_counts)

    input_counts = np.asarray(counts["ic1"])
    output_counts = np.asarray(counts["oc1"])
    if output_counts.ndim != input_counts.ndim + 1:
        raise ValueError(
            f"{generation} oc1 must have exactly one more dimension than ic1."
        )
    if output_counts.shape[-2] != 4 or output_counts.shape[:-2] != input_counts.shape[:-1]:
        raise ValueError(
            f"{generation} oc1 must have four gates and match the ic1 sample axes."
        )
    if output_counts.shape[-1] != input_counts.shape[-1]:
        raise ValueError(f"{generation} ic1 and oc1 must share the fitted variant axis.")
    return normalized_count_enrichment(input_counts, output_counts[..., 3, :])


def nuclease_g3_highest_gate_enrichment(
    counts: dict[str, np.ndarray],
) -> np.ndarray:
    """Reproduce G3 ``ef_2_99_g3`` from observed or predictive count arrays."""
    return nuclease_highest_gate_enrichment("G3", counts)


def _mutation_count(signature: object) -> int:
    if not isinstance(signature, str):
        raise ValueError(f"Mutation signature must be a string, not {signature!r}.")
    return 0 if not signature else signature.count("_") + 1


def _load_reported_activity(path: Path) -> pd.DataFrame:
    reported = pd.read_csv(path)
    required = {"mutations", "activity_level", "is_functional"}
    missing = required - set(reported.columns)
    if missing:
        raise ValueError(f"Nuclease activity table is missing columns: {sorted(missing)}")
    if reported.empty:
        raise ValueError("Nuclease activity table must contain at least one variant.")
    reported = reported.copy()
    reported["variant_id"] = reported["mutations"].map(mutation_signature)
    if reported["variant_id"].duplicated().any():
        raise ValueError("Nuclease activity mutation signatures must be unique.")
    return reported


def _load_reported_fitness(generation: str, count_dir: Path) -> pd.DataFrame:
    score_column = NUCLEASE_REPORTED_FITNESS_COLUMN[generation]
    path = count_dir / f"{generation.lower()}.csv"
    reported = pd.read_csv(path, usecols=["mutations", score_column])
    reported["variant_id"] = reported["mutations"].map(mutation_signature)
    if reported["variant_id"].duplicated().any():
        raise ValueError(f"{generation}: reported-fitness mutation signatures must be unique.")
    values = pd.to_numeric(reported[score_column], errors="raise")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            f"{generation}: reported enrichment factors must be finite and nonnegative."
        )
    return reported[["variant_id"]].assign(reported_fitness=values)


def _source_variant_roles(
    generation: str,
    count_dir: Path,
    fitted_variants: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    main_path = count_dir / f"{generation.lower()}.csv"
    control_path = count_dir / f"{generation.lower()}_neg_control.csv"
    main = pd.read_csv(main_path, usecols=["mutations"])
    controls = pd.read_csv(control_path, usecols=["mutations"])
    main_ids = main["mutations"].map(mutation_signature).tolist()
    control_ids = controls["mutations"].map(mutation_signature).tolist()
    source_ids = [*main_ids, *control_ids]
    if source_ids != fitted_variants:
        raise ValueError(
            f"{generation}: main and negative-control source rows do not match "
            "the fitted variant order."
        )
    is_source_control = np.r_[
        np.zeros(len(main_ids), dtype=bool),
        np.ones(len(control_ids), dtype=bool),
    ]
    return is_source_control, ~is_source_control


def load_nuclease_landscape(
    generation: NucleaseGeneration,
    *,
    count_dir: str | Path,
    reported_activity_path: str | Path,
) -> NucleaseLandscape:
    """Load one NucB generation in the exact fitted variant order."""
    if generation not in NUCLEASE_GENERATIONS:
        raise ValueError(
            f"generation must be one of {NUCLEASE_GENERATIONS}, not {generation!r}."
        )

    dataset = load_nuclease_data(str(count_dir), gen=generation)
    fitted_variants = [str(variant) for variant in dataset["variants"]]
    if len(set(fitted_variants)) != len(fitted_variants):
        raise ValueError(f"{generation}: fitted mutation signatures must be unique.")

    counts = {
        name: np.asarray(dataset[name], dtype=np.int64) for name in _COUNT_NAMES[generation]
    }
    for name, array in counts.items():
        if array.ndim == 0 or array.shape[-1] != len(fitted_variants):
            raise ValueError(f"{generation}: count array {name!r} has an invalid variant axis.")
        if np.any(array < 0):
            raise ValueError(f"{generation}: count array {name!r} contains negative values.")
    thresholds = {
        name: np.asarray(dataset[name], dtype=float)
        for name in _THRESHOLD_NAMES[generation]
    }

    wild_type_rows = [index for index, variant in enumerate(fitted_variants) if variant == ""]
    if len(wild_type_rows) != 1:
        raise ValueError(f"{generation}: expected exactly one empty-signature wild type.")
    wild_type_index = wild_type_rows[0]
    is_source_control, is_ranking_candidate = _source_variant_roles(
        generation,
        Path(count_dir),
        fitted_variants,
    )

    variants = pd.DataFrame(
        {
            "variant_id": fitted_variants,
            "generation": generation,
            "hmc_variant_index": np.arange(len(fitted_variants), dtype=np.int32),
            "is_wild_type": np.arange(len(fitted_variants)) == wild_type_index,
            "is_source_control": is_source_control,
            "is_ranking_candidate": is_ranking_candidate,
            "input_count": _variant_totals(counts, _INPUT_NAMES[generation]),
            "output_count": _variant_totals(counts, _OUTPUT_NAMES[generation]),
        }
    )
    variants["num_mutations"] = variants["variant_id"].map(_mutation_count)

    reported = _load_reported_activity(Path(reported_activity_path))
    reported_columns = ["variant_id", "activity_level", "is_functional"]
    for optional in ("generations", "sequence", "sublibrary_names"):
        if optional in reported:
            reported_columns.append(optional)
    variants = variants.merge(
        reported[reported_columns].rename(
            columns={
                "activity_level": "reported_activity_level",
                "is_functional": "reported_is_functional",
                "generations": "reported_generations",
            }
        ),
        on="variant_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    variants["reported_activity_source"] = np.where(
        variants["reported_activity_level"].notna(),
        "Thomas et al. 2025 categorical activity",
        "unavailable",
    )

    reported_fitness = _load_reported_fitness(generation, Path(count_dir))
    variants = variants.merge(
        reported_fitness,
        on="variant_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    variants["reported_fitness_name"] = NUCLEASE_REPORTED_FITNESS_COLUMN[generation]
    variants["reported_rank"] = variants["reported_fitness"].rank(
        method="average",
        ascending=False,
    )
    variants["reference_fitness"] = variants["reported_fitness"]
    variants["reference_fitness_source"] = np.where(
        variants["reported_fitness"].notna(),
        "Thomas et al. 2025 highest-gate enrichment factor",
        "unavailable",
    )

    return NucleaseLandscape(
        generation=generation,
        variants=variants,
        counts=counts,
        thresholds=thresholds,
        wild_type_index=wild_type_index,
    )


def _observed_counts_status(
    handle: h5py.File,
    landscape: NucleaseLandscape,
) -> str:
    observed_object = handle.get("observed_data")
    if observed_object is None:
        return "not_embedded"
    if not isinstance(observed_object, h5py.Group):
        raise ValueError("NucB HMC observed_data must be an HDF5 group.")
    observed = observed_object
    missing = set(landscape.counts) - set(observed)
    if missing:
        raise ValueError(f"NucB HMC observed_data is missing variables: {sorted(missing)}")
    mismatched = []
    for name, counts in landscape.counts.items():
        dataset = observed.get(name)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"NucB HMC observed_data/{name} must be a dataset.")
        if not np.array_equal(dataset[:], counts):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            f"{landscape.generation}: HMC observed counts differ from the raw "
            f"count tables for {mismatched}."
        )
    return "matched"


def build_nuclease_mle_score_ensemble(
    landscape: NucleaseLandscape,
    mle_path: str | Path,
    *,
    fitness_distribution: Literal["exponential", "lomax"],
    fit_id: str | None = None,
    job_id: int | None = None,
    sample_chunk_size: int = 64,
) -> ScoreEnsemble:
    """Build a verified NucB highest-gate ensemble from sampled-MLE counts."""
    generation = landscape.generation
    if generation not in NUCLEASE_PREDICTIVE_SCORE_CONTRACTS:
        raise ValueError(f"Unsupported NucB generation {generation!r}.")
    if fitness_distribution not in {"exponential", "lomax"}:
        raise ValueError("fitness_distribution must be 'exponential' or 'lomax'.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")

    resolved_path = Path(mle_path)
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape=f"nuc-{generation}",
        inference_method="mle_predictive",
        model_spec=fitness_distribution,
        fit_id=fit_id or resolved_path.stem,
    )
    observed_proxy = nuclease_highest_gate_enrichment(generation, landscape.counts)
    candidate_indices = ranking_candidate_indices(landscape.variants)
    reported = landscape.variants["reported_fitness"].to_numpy(dtype=float)
    oracle_indices = candidate_indices[np.isfinite(reported[candidate_indices])]
    oracle_error = np.abs(observed_proxy[oracle_indices] - reported[oracle_indices])
    if not np.allclose(
        observed_proxy[oracle_indices],
        reported[oracle_indices],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{generation} observed counts do not reproduce reported fitness."
        )

    required_count_names = {
        "G1": ("ic1", "ic2", "ic3", "hc2"),
        "G2": ("ic1", "ic2", "c975"),
        "G3": ("ic1", "oc1"),
        "G4": ("ic1", "oc1"),
    }[generation]
    with h5py.File(resolved_path, "r") as handle:
        if _observed_counts_status(handle, landscape) != "matched":
            raise ValueError("The sampled-MLE artifact must embed matching counts.")
        predictive_object = handle.get("posterior_predictive")
        if not isinstance(predictive_object, h5py.Group):
            raise ValueError("The sampled-MLE artifact is missing posterior_predictive.")
        predictive_counts = {
            name: predictive_object.get(name) for name in required_count_names
        }
        invalid = [
            name
            for name, dataset in predictive_counts.items()
            if not isinstance(dataset, h5py.Dataset)
        ]
        if invalid:
            raise ValueError(
                f"{generation} posterior_predictive is missing datasets: {invalid}."
            )
        n_variants = landscape.n_fitted_variants
        first_object = predictive_counts[required_count_names[0]]
        assert isinstance(first_object, h5py.Dataset)
        if first_object.shape[0] != 1:
            raise ValueError(f"{generation} predictive counts require one chain.")
        n_samples = int(first_object.shape[1])
        for name, dataset in predictive_counts.items():
            assert isinstance(dataset, h5py.Dataset)
            expected_shape = (1, n_samples, *landscape.counts[name].shape)
            if dataset.shape != expected_shape:
                raise ValueError(
                    f"{generation} predictive {name} must have shape "
                    f"{expected_shape}, not {dataset.shape}."
                )
        if n_samples < 2:
            raise ValueError(
                f"{generation} proxy analysis requires at least two MLE count draws."
            )

        sampled_proxy = np.empty((n_samples, n_variants), dtype=np.float32)
        zero_input_values = 0
        for sample_start in range(0, n_samples, sample_chunk_size):
            sample_stop = min(sample_start + sample_chunk_size, n_samples)
            chunk_counts = {}
            for name, dataset in predictive_counts.items():
                assert isinstance(dataset, h5py.Dataset)
                values = np.asarray(dataset[0, sample_start:sample_stop])
                expected_totals = landscape.counts[name].sum(axis=-1, dtype=np.int64)
                if not np.all(
                    values.sum(axis=-1, dtype=np.int64)
                    == expected_totals[None, ...]
                ):
                    raise ValueError(
                        f"{generation} predictive {name} totals differ from "
                        "observed data."
                    )
                chunk_counts[name] = values
            sampled = nuclease_highest_gate_enrichment(generation, chunk_counts)
            zero_input_values += int(np.count_nonzero(~np.isfinite(sampled)))
            sampled_proxy[sample_start:sample_stop] = sampled.astype(np.float32)

    finite_ranking_candidates = np.isfinite(
        sampled_proxy[:, candidate_indices]
    ).sum(axis=-1)

    score_name = NUCLEASE_PREDICTIVE_SCORE_NAME
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
        "score_contracts": {
            score_name: NUCLEASE_PREDICTIVE_SCORE_CONTRACTS[generation]
        },
        "score_status": {score_name: "verified"},
        "higher_is_better": {score_name: True},
        "n_samples": n_samples,
        "n_variants": n_variants,
        "n_ranking_candidates": len(candidate_indices),
        "n_source_controls": n_variants - len(candidate_indices),
        "sample_chunk_size": sample_chunk_size,
        "zero_input_values": zero_input_values,
        "finite_ranking_candidates_per_sample": (
            finite_ranking_candidates.astype(int).tolist()
        ),
        "min_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.min()
        ),
        "max_finite_ranking_candidates_per_sample": int(
            finite_ranking_candidates.max()
        ),
        "normalization_universe": "all fitted variants including source controls",
        "count_variables": {
            "G1": {
                "input": "sum(ic1[0], ic1[1], ic2[0], ic2[1], ic3)",
                "output": "hc2[1]",
            },
            "G2": {"input": "sum(ic1[0], ic1[1], ic2)", "output": "c975"},
            "G3": {"input": "ic1", "output": "oc1[3]"},
            "G4": {"input": "ic1", "output": "oc1[3]"},
        }[generation],
        "count_axes": {
            name: ["chain", "draw", *(["component"] if array.ndim == 2 else []), "variant"]
            for name, array in (
                (name, landscape.counts[name]) for name in required_count_names
            )
        },
        "observed_counts_status": "matched",
        "observed_oracle": {
            "source_column": NUCLEASE_REPORTED_FITNESS_COLUMN[generation],
            "n_compared": len(oracle_indices),
            "max_abs_error": float(oracle_error.max(initial=0.0)),
            "rtol": 1e-12,
            "atol": 1e-12,
        },
        "source_artifact": file_signature(resolved_path),
    }
    if generation == "G1":
        metadata["source_normalization_totals"] = dict(
            NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS
        )
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=landscape.variants.copy(),
        scores=scores,
    )


def build_nuclease_hmc_count_score_ensemble(
    landscape: NucleaseLandscape,
    hmc_path: str | Path,
    *,
    fitness_distribution: Literal["exponential", "lomax"],
    fit_id: str | None = None,
    job_id: int | None = None,
    artifact_kind: str | None = None,
    n_samples: int | None = None,
    seed: int = 23,
    sample_chunk_size: int = 64,
    scratch_dir: str | Path | None = None,
    show_progress: bool = True,
) -> ScoreEnsemble:
    """Build highest-gate proxies from paired HMC posterior-predictive draws."""
    generation = landscape.generation
    if generation not in NUCLEASE_PREDICTIVE_SCORE_CONTRACTS:
        raise ValueError(f"Unsupported NucB generation {generation!r}.")
    if fitness_distribution not in {"exponential", "lomax"}:
        raise ValueError("fitness_distribution must be 'exponential' or 'lomax'.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")

    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape=f"nuc-{generation}",
        inference_method="hmc_count_based",
        model_spec=fitness_distribution,
        fit_id=fit_id or resolved_path.stem,
    )
    observed_proxy = nuclease_highest_gate_enrichment(generation, landscape.counts)
    candidate_indices = ranking_candidate_indices(landscape.variants)
    candidate_mask = np.zeros(landscape.n_fitted_variants, dtype=bool)
    candidate_mask[candidate_indices] = True
    reported = landscape.variants["reported_fitness"].to_numpy(dtype=float)
    oracle_indices = candidate_indices[np.isfinite(reported[candidate_indices])]
    oracle_error = np.abs(observed_proxy[oracle_indices] - reported[oracle_indices])
    if not np.allclose(
        observed_proxy[oracle_indices],
        reported[oracle_indices],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{generation} observed counts do not reproduce reported fitness."
        )
    required_count_names = {
        "G1": ("ic1", "ic2", "ic3", "hc2"),
        "G2": ("ic1", "ic2", "c975"),
        "G3": ("ic1", "oc1"),
        "G4": ("ic1", "oc1"),
    }[generation]

    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"nuc_{generation.lower()}_hmc_count_proxy_",
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
                raise ValueError("The HMC artifact must embed matching NucB counts.")
            predictive_object = handle.get("posterior_predictive")
            if not isinstance(predictive_object, h5py.Group):
                raise ValueError("The HMC artifact is missing posterior_predictive.")
            predictive_counts = {
                name: predictive_object.get(name) for name in required_count_names
            }
            invalid = [
                name
                for name, dataset in predictive_counts.items()
                if not isinstance(dataset, h5py.Dataset)
            ]
            if invalid:
                raise ValueError(
                    f"{generation} posterior_predictive is missing datasets: {invalid}."
                )
            first_object = predictive_counts[required_count_names[0]]
            assert isinstance(first_object, h5py.Dataset)
            n_chains, draws_per_chain = map(int, first_object.shape[:2])
            for name, dataset in predictive_counts.items():
                assert isinstance(dataset, h5py.Dataset)
                expected_shape = (
                    n_chains,
                    draws_per_chain,
                    *landscape.counts[name].shape,
                )
                if dataset.shape != expected_shape:
                    raise ValueError(
                        f"{generation} predictive {name} must have shape "
                        f"{expected_shape}, not {dataset.shape}."
                    )
            total_samples = n_chains * draws_per_chain
            resolved_n_samples = total_samples if n_samples is None else n_samples
            if not 1 <= resolved_n_samples <= total_samples:
                raise ValueError(
                    f"n_samples must be between 1 and {total_samples:,}."
                )
            if resolved_n_samples == total_samples:
                selected_sample_ids = np.arange(total_samples, dtype=np.int64)
            else:
                selected_sample_ids = np.sort(
                    np.random.default_rng(seed).choice(
                        total_samples,
                        resolved_n_samples,
                        replace=False,
                    )
                ).astype(np.int64)
            if len(selected_sample_ids) < 2:
                raise ValueError("NucB proxy analysis requires at least two HMC samples.")

            sampled_proxy = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(selected_sample_ids), landscape.n_fitted_variants),
            )
            sampled_proxy[:] = np.nan
            finite_per_sample = np.zeros(len(selected_sample_ids), dtype=np.int64)
            finite_ranking_candidates = np.zeros(
                len(selected_sample_ids),
                dtype=np.int64,
            )
            expected_totals = {
                name: landscape.counts[name].sum(axis=-1, dtype=np.int64)
                for name in required_count_names
            }
            if show_progress:
                print(
                    f"nuc-{generation}: {len(selected_sample_ids):,} HMC predictive "
                    f"samples, {landscape.n_fitted_variants:,} variants",
                    flush=True,
                )
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
                chunk_counts = {}
                for name, dataset in predictive_counts.items():
                    assert isinstance(dataset, h5py.Dataset)
                    values = read_native_samples(
                        dataset,
                        block_sample_ids,
                        draws_per_chain=draws_per_chain,
                        variant_start=0,
                        variant_stop=landscape.n_fitted_variants,
                    )
                    totals = values.sum(axis=-1, dtype=np.int64)
                    if not np.array_equal(
                        totals,
                        np.broadcast_to(expected_totals[name], totals.shape),
                    ):
                        raise ValueError(
                            f"{generation} predictive {name} totals differ from "
                            "observed data."
                        )
                    chunk_counts[name] = values
                sampled = nuclease_highest_gate_enrichment(
                    generation,
                    chunk_counts,
                ).astype(np.float32, copy=False)
                sampled_proxy[sample_start:sample_stop] = sampled
                finite = np.isfinite(sampled)
                finite_per_sample[sample_start:sample_stop] = finite.sum(axis=-1)
                finite_ranking_candidates[sample_start:sample_stop] = finite[
                    :,
                    candidate_mask,
                ].sum(axis=-1)
                if show_progress:
                    print(
                        f"  samples {sample_stop}/{len(selected_sample_ids)} "
                        f"({(time.monotonic() - started) / 60:.1f} min)",
                        flush=True,
                    )
            sampled_proxy.flush()
    except Exception:
        del sampled_proxy
        temporary_path.unlink(missing_ok=True)
        raise

    score_name = NUCLEASE_PREDICTIVE_SCORE_NAME
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
            "variant": np.arange(landscape.n_fitted_variants, dtype=np.int32),
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
        "score_contracts": {
            score_name: NUCLEASE_PREDICTIVE_SCORE_CONTRACTS[generation]
        },
        "score_status": {score_name: "verified"},
        "higher_is_better": {score_name: True},
        "n_chains": n_chains,
        "draws_per_chain": draws_per_chain,
        "n_source_samples": n_chains * draws_per_chain,
        "n_samples": len(selected_sample_ids),
        "sample_selection_seed": seed,
        "n_variants": landscape.n_fitted_variants,
        "n_ranking_candidates": len(candidate_indices),
        "n_source_controls": landscape.n_fitted_variants - len(candidate_indices),
        "sample_chunk_size": sample_chunk_size,
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
        "normalization_universe": "all fitted variants including source controls",
        "count_variables": {
            "G1": {
                "input": "sum(ic1[0], ic1[1], ic2[0], ic2[1], ic3)",
                "output": "hc2[1]",
            },
            "G2": {"input": "sum(ic1[0], ic1[1], ic2)", "output": "c975"},
            "G3": {"input": "ic1", "output": "oc1[3]"},
            "G4": {"input": "ic1", "output": "oc1[3]"},
        }[generation],
        "count_axes": {
            name: [
                "chain",
                "draw",
                *(["component"] if array.ndim == 2 else []),
                "variant",
            ]
            for name, array in (
                (name, landscape.counts[name]) for name in required_count_names
            )
        },
        "observed_counts_status": "matched",
        "observed_oracle": {
            "source_column": NUCLEASE_REPORTED_FITNESS_COLUMN[generation],
            "n_compared": len(oracle_indices),
            "max_abs_error": float(oracle_error.max(initial=0.0)),
            "rtol": 1e-12,
            "atol": 1e-12,
        },
        "source_artifact": file_signature(resolved_path),
        "elapsed_seconds": time.monotonic() - started,
        "warnings": [],
    }
    if generation == "G1":
        metadata["source_normalization_totals"] = dict(
            NUCLEASE_G1_SOURCE_NORMALIZATION_TOTALS
        )
    return ScoreEnsemble.from_analysis_view(
        metadata=metadata,
        variants=landscape.variants.copy(),
        scores=scores,
        temporary_paths=(temporary_path,),
    )


def build_nuclease_g3_mle_score_ensemble(
    landscape: NucleaseLandscape,
    mle_path: str | Path,
    *,
    fitness_distribution: Literal["exponential", "lomax"],
    fit_id: str | None = None,
    job_id: int | None = None,
    sample_chunk_size: int = 64,
) -> ScoreEnsemble:
    """Build the verified G3 highest-gate ensemble from sampled-MLE counts."""
    if landscape.generation != "G3":
        raise ValueError("The G3 MLE proxy builder requires a G3 landscape.")
    return build_nuclease_mle_score_ensemble(
        landscape,
        mle_path,
        fitness_distribution=fitness_distribution,
        fit_id=fit_id,
        job_id=job_id,
        sample_chunk_size=sample_chunk_size,
    )


def _score_block(
    target,
    sample_ids: np.ndarray,
    *,
    draws_per_chain: int,
    variant_start: int,
    variant_stop: int,
    local_indices: np.ndarray,
) -> np.ndarray:
    values = read_native_samples(
        target,
        sample_ids,
        draws_per_chain=draws_per_chain,
        variant_start=variant_start,
        variant_stop=variant_stop,
    )[:, local_indices]
    values = values.astype(np.float32, copy=False)
    return np.where(np.isfinite(values), values, np.nan).astype(
        np.float32,
        copy=False,
    )


def build_nuclease_hmc_score_ensemble(
    landscape: NucleaseLandscape,
    hmc_path: str | Path,
    *,
    fitness_distribution: Literal["exponential", "lomax"],
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
    """Extract the full NucB posterior fluorescence matrix."""
    if fitness_distribution not in {"exponential", "lomax"}:
        raise ValueError("fitness_distribution must be 'exponential' or 'lomax'.")
    resolved_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape=f"nuc-{landscape.generation}",
        inference_method="hmc",
        model_spec=fitness_distribution,
        fit_id=fit_id or resolved_path.stem,
    )
    target_name = NUCLEASE_SCORE_VARIABLE[landscape.generation]
    analysis_variants = landscape.variants.copy()
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        np.arange(len(analysis_variants), dtype=np.int64),
    )
    extra_metadata: dict[str, object] = {
        "generation": landscape.generation,
        "model": fitness_distribution.title(),
        "fitness_distribution": fitness_distribution,
        "primary_score": NUCLEASE_SCORE_NAME,
        "reference_fitness_source": (
            "Thomas et al. 2025 generation-specific highest-gate enrichment factor"
        ),
        "reported_fitness_available": True,
        "reported_fitness_coverage": 1.0,
        "ranking_candidate_column": RANKING_CANDIDATE_COLUMN,
        "ranking_candidate_definition": (
            "main generation rows; source controls excluded"
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
        posterior_transform="identity",
        score_name=NUCLEASE_SCORE_NAME,
        score_contract=(
            f"nuc-{landscape.generation.lower()}-{target_name}-identity-v1"
        ),
        variants=landscape.variants,
        analysis_variants=analysis_variants,
        transform=lambda values: values.astype(np.float32, copy=False),
        n_samples=n_samples,
        seed=seed,
        sample_view=sample_view,
        sample_chunk_size=sample_chunk_size,
        variant_chunk_size=variant_chunk_size,
        scratch_dir=scratch_dir,
        extra_metadata=extra_metadata,
        show_progress=show_progress,
    )


def scan_nuclease_hmc(
    landscape: NucleaseLandscape,
    hmc_path: str | Path,
    *,
    fitness_distribution: Literal["exponential", "lomax"],
    top_k: tuple[int, ...] = (50, 100, 1000),
    retain_k: int = 1000,
    tracked_top_n: int = 1000,
    tracked_model_top_n: int = 0,
    n_samples: int | None = None,
    seed: int = 23,
    n_pairs: int = 10_000,
    interval: tuple[float, float] = (0.05, 0.95),
    fit_id: str | None = None,
    job_id: int | None = None,
    artifact_kind: str | None = None,
    show_progress: bool = True,
) -> StabilityResult:
    """Scan one generation's fluorescence posterior using bounded memory."""
    if fitness_distribution not in {"exponential", "lomax"}:
        raise ValueError("fitness_distribution must be 'exponential' or 'lomax'.")
    hmc_path = Path(hmc_path)
    identity = AnalysisIdentity(
        protein_family="nuc",
        landscape=f"nuc-{landscape.generation}",
        inference_method="hmc",
        model_spec=fitness_distribution,
        fit_id=fit_id or hmc_path.stem,
    )
    top_k = _top_k_values(top_k)
    variants = landscape.variants
    candidate_indices = ranking_candidate_indices(variants)
    if retain_k < top_k[-1] or retain_k > len(candidate_indices):
        raise ValueError(
            "retain_k must be at least the largest top_k and no larger than "
            "the ranking-candidate cohort."
        )

    started = time.monotonic()
    target_name = NUCLEASE_SCORE_VARIABLE[landscape.generation]
    h5netcdf_compatible, h5netcdf_open_error = h5netcdf_compatibility(hmc_path)
    with h5py.File(hmc_path, "r") as handle:
        observed_counts_status = _observed_counts_status(handle, landscape)
        posterior_object = handle.get("posterior")
        if not isinstance(posterior_object, h5py.Group):
            raise ValueError(
                f"{landscape.generation} {fitness_distribution} fit must contain "
                f"posterior/{target_name}."
            )
        target_object = posterior_object.get(target_name)
        if not isinstance(target_object, h5py.Dataset):
            raise ValueError(
                f"{landscape.generation} {fitness_distribution} fit must contain "
                f"posterior/{target_name}."
            )
        target = target_object
        if target.ndim != 3 or target.shape[-1] != landscape.n_fitted_variants:
            raise ValueError(
                f"posterior/{target_name} must be chain-by-draw-by-variant and "
                "match the raw count-table width."
            )
        variant_chunk_size = int((target.chunks or target.shape)[-1])
        n_chains, draws_per_chain = target.shape[:2]
        scan = scan_hmc_scores(
            variants,
            identity=identity,
            score_names=(NUCLEASE_SCORE_NAME,),
            n_hmc_variants=landscape.n_fitted_variants,
            n_chains=n_chains,
            draws_per_chain=draws_per_chain,
            variant_chunk_size=variant_chunk_size,
            read_scores=lambda **block: {
                NUCLEASE_SCORE_NAME: _score_block(target, **block)
            },
            top_k=top_k,
            retain_k=retain_k,
            tracked_top_n=tracked_top_n,
            tracked_model_top_n=tracked_model_top_n,
            n_samples=n_samples,
            seed=seed,
            label=f"nuc-{landscape.generation}",
            selection_attrs={
                "model": fitness_distribution.title(),
                "n_fitted_variants": landscape.n_fitted_variants,
                "n_ranking_candidates": len(candidate_indices),
                "ranking_candidate_column": RANKING_CANDIDATE_COLUMN,
            },
            show_progress=show_progress,
        )

    summary, overlaps = result_tables(
        scan.selection,
        variants,
        identity=identity,
        top_k=top_k,
        interval=interval,
        n_pairs=n_pairs,
        seed=seed,
    )
    artifact_issues = []
    warning_messages = []
    if observed_counts_status == "not_embedded":
        artifact_issues.append("missing_observed_data")
        warning_messages.append(
            "The fit does not embed observed counts. Variant alignment uses the "
            "source count-table order."
        )
    if not h5netcdf_compatible:
        artifact_issues.append("h5netcdf_open_failure")
        warning_messages.append(
            "The fit cannot be opened with h5netcdf in this environment; posterior "
            "arrays were read directly with h5py."
        )
    metadata: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **identity.metadata(),
        "generation": landscape.generation,
        "fit_method": "hmc_posterior",
        "model": fitness_distribution.title(),
        "fitness_distribution": fitness_distribution,
        "posterior_variable": target_name,
        "posterior_transform": "identity",
        "scores": [NUCLEASE_SCORE_NAME],
        "n_samples": len(scan.sample_ids),
        "n_variants": len(scan.candidate_indices),
        "n_fitted_variants": landscape.n_fitted_variants,
        "n_excluded_from_ranking": (
            landscape.n_fitted_variants - len(scan.candidate_indices)
        ),
        "ranking_candidate_column": RANKING_CANDIDATE_COLUMN,
        "ranking_candidate_definition": "main generation rows; source controls excluded",
        "n_reported_variants": len(scan.reported_indices),
        "reported_fitness_coverage": len(scan.reported_indices)
        / len(scan.candidate_indices),
        "retained_k": retain_k,
        "tracked_reported_top_n": tracked_top_n,
        "tracked_model_top_n": tracked_model_top_n,
        "top_k": list(top_k),
        "interval": list(interval),
        "seed": seed,
        "reference_fitness_source": (
            "Thomas et al. 2025 generation-specific highest-gate enrichment factor"
        ),
        "reported_fitness_available": True,
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
        "hmc_file": file_signature(hmc_path),
        "scistanpy_version": version("scistanpy"),
        "elapsed_seconds": time.monotonic() - started,
        "warnings": warning_messages,
    }
    if job_id is not None:
        metadata["job_id"] = job_id
    if artifact_kind is not None:
        metadata["artifact_kind"] = artifact_kind
    if tracked_model_top_n:
        metadata.update(
            {
                "model_reference_definition": "posterior median score",
                "model_reference_tie_break": "fitted variant index",
            }
        )
    return StabilityResult(
        metadata=metadata,
        variants=variants.copy(),
        selection=scan.selection,
        summary=summary,
        overlaps=overlaps,
    )
