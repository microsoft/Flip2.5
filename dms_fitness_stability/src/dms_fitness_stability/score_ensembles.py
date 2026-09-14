"""Generic persistence for sampled variant-score matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import is_integer_dtype

from .candidates import ranking_candidate_mask
from .identity import AnalysisIdentity
from .results import _read_variants_csv

SCORE_ENSEMBLE_SCHEMA_VERSION = 2
LEGACY_PROXY_SCHEMA_VERSION = 1
POSTERIOR_PARAMETER_TARGET = "posterior_parameter_landscape"
CONDITIONAL_REPEATABILITY_TARGET = "conditional_repeated_experiment"
HMC_COUNT_BASED_TARGET = "posterior_predictive_count_landscape"

_METHOD_CONTRACTS = {
    "hmc": {
        "uncertainty_target": POSTERIOR_PARAMETER_TARGET,
        "fit_method": "hmc_posterior",
        "sample_kind": "HMC posterior",
    },
    "mle_predictive": {
        "uncertainty_target": CONDITIONAL_REPEATABILITY_TARGET,
        "fit_method": "mle_predictive",
        "sample_kind": "MLE predictive",
    },
    "hmc_count_based": {
        "uncertainty_target": HMC_COUNT_BASED_TARGET,
        "fit_method": "hmc_count_based",
        "sample_kind": "HMC count-based",
    },
}


def score_method_contract(metadata: dict[str, object]) -> dict[str, str]:
    """Return the analysis contract for one persisted score method."""
    inference_method = metadata.get("inference_method")
    contract = _METHOD_CONTRACTS.get(inference_method)
    if contract is not None:
        return contract
    resolved = {
        field: metadata.get(field)
        for field in ("uncertainty_target", "fit_method", "sample_kind")
    }
    invalid = [
        field
        for field, value in resolved.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(
            "Score ensembles with a new inference_method must declare non-empty "
            f"metadata fields: {invalid}."
        )
    return {field: str(value) for field, value in resolved.items()}


def direct_hmc_reference_metadata(
    metadata: dict[str, object],
    analysis_variants: pd.DataFrame,
) -> dict[str, object]:
    """Derive the frozen family-specific reference metadata for direct HMC."""
    if metadata.get("inference_method") != "hmc":
        return {}
    family = str(metadata.get("protein_family"))
    reference_sources = {
        "pdz": "reported",
        "trpb": "reported sequence-aligned subset",
        "nuc": (
            "Thomas et al. 2025 generation-specific highest-gate enrichment factor"
        ),
    }
    if family not in reference_sources:
        raise ValueError(f"Unsupported direct-HMC reference family: {family!r}.")
    candidate_mask = ranking_candidate_mask(analysis_variants)
    reported = analysis_variants["reported_fitness"].to_numpy(dtype=float)
    result: dict[str, object] = {
        "reference_fitness_source": reference_sources[family],
        "reported_fitness_available": bool(np.isfinite(reported[candidate_mask]).any()),
        "reference_metadata_contract": "direct-hmc-reference-v1",
    }
    if family in {"trpb", "nuc"}:
        result["reported_fitness_coverage"] = float(
            np.isfinite(reported[candidate_mask]).mean()
        )
    if family == "nuc":
        result.update(
            {
                "ranking_candidate_column": "is_ranking_candidate",
                "ranking_candidate_definition": (
                    "main generation rows; source controls excluded"
                ),
            }
        )
    return result


@dataclass
class ScoreEnsemble:
    """A full sampled score axis and its exact analysis-facing variant view."""

    metadata: dict[str, object]
    variants: pd.DataFrame
    analysis_variants: pd.DataFrame
    scores: xr.Dataset
    temporary_paths: tuple[Path, ...] = field(default_factory=tuple, repr=False)

    @classmethod
    def from_analysis_view(
        cls,
        *,
        metadata: dict[str, object],
        variants: pd.DataFrame,
        scores: xr.Dataset,
        temporary_paths: tuple[Path, ...] = (),
    ) -> ScoreEnsemble:
        """Build a schema-v2 bundle whose analysis view is the full score axis."""
        analysis_variants = variants.copy()
        analysis_variants.insert(
            0,
            "matrix_variant_index",
            np.arange(len(variants), dtype=np.int64),
        )
        ensemble = cls(
            metadata={
                **metadata,
                "schema_version": SCORE_ENSEMBLE_SCHEMA_VERSION,
                "fit_method": score_method_contract(metadata)["fit_method"],
                "n_analysis_variants": len(analysis_variants),
            },
            variants=variants.copy(),
            analysis_variants=analysis_variants,
            scores=scores,
            temporary_paths=temporary_paths,
        )
        ensemble.validate()
        return ensemble

    @property
    def fitness_score(self) -> xr.DataArray:
        """Return sampled scores with dimensions score_name, sample, variant."""
        return self.scores["fitness_score"]

    @property
    def observed_score(self) -> xr.DataArray | None:
        """Return an optional fixed observed score aligned to the full axis."""
        return self.scores.get("observed_score")

    @property
    def source_schema_version(self) -> int:
        """Return the on-disk schema version from which this object was loaded."""
        return int(self.metadata["schema_version"])

    def validate(self) -> None:
        """Validate dimensions, metadata, sample identity, and both variant axes."""
        schema_version = self.metadata.get("schema_version")
        if schema_version not in {
            LEGACY_PROXY_SCHEMA_VERSION,
            SCORE_ENSEMBLE_SCHEMA_VERSION,
        }:
            raise ValueError(
                "Score ensemble schema_version must be "
                f"{LEGACY_PROXY_SCHEMA_VERSION} or {SCORE_ENSEMBLE_SCHEMA_VERSION}."
            )
        method = score_method_contract(self.metadata)
        if schema_version == LEGACY_PROXY_SCHEMA_VERSION and self.metadata.get(
            "inference_method"
        ) not in {"mle_predictive", "hmc_count_based"}:
            raise ValueError("Legacy proxy bundles cannot contain direct HMC scores.")
        if self.metadata.get("uncertainty_target") != method["uncertainty_target"]:
            raise ValueError(
                "Score ensemble uncertainty_target does not match "
                f"inference_method={self.metadata.get('inference_method')!r}."
            )
        if schema_version == SCORE_ENSEMBLE_SCHEMA_VERSION:
            if self.metadata.get("fit_method") != method["fit_method"]:
                raise ValueError(
                    "Score ensemble fit_method does not match its inference method."
                )
            self._validate_identity()

        self._validate_variant_table(self.variants, name="variants")
        self._validate_variant_table(
            self.analysis_variants,
            name="analysis_variants",
        )
        self._validate_analysis_mapping()

        variable_names = set(self.scores.data_vars)
        if "fitness_score" not in variable_names:
            raise ValueError("scores must contain fitness_score.")
        extra_variables = variable_names - {"fitness_score", "observed_score"}
        if extra_variables:
            raise ValueError(
                f"scores contain unsupported variables: {sorted(extra_variables)}."
            )
        if schema_version == LEGACY_PROXY_SCHEMA_VERSION and (
            "observed_score" not in variable_names
        ):
            raise ValueError("Legacy proxy bundles require observed_score.")
        if self.fitness_score.dims != ("score_name", "sample", "variant"):
            raise ValueError(
                "fitness_score dimensions must be score_name, sample, variant."
            )
        if self.fitness_score.dtype != np.dtype("float32"):
            raise ValueError("fitness_score must have dtype float32.")
        if self.observed_score is not None:
            if self.observed_score.dims != ("score_name", "variant"):
                raise ValueError(
                    "observed_score dimensions must be score_name, variant."
                )
            if self.observed_score.dtype != np.dtype("float32"):
                raise ValueError("observed_score must have dtype float32.")

        score_names = self.scores["score_name"].values.astype(str).tolist()
        if not score_names or len(set(score_names)) != len(score_names):
            raise ValueError("score_name coordinates must be non-empty and unique.")
        if any(not name.strip() for name in score_names):
            raise ValueError("score_name coordinates cannot be blank.")
        if self.scores.sizes["sample"] < 2:
            raise ValueError("Score ensembles require at least two samples.")
        if self.scores.sizes["variant"] != len(self.variants):
            raise ValueError("The score variant axis must match variants.csv rows.")
        if not np.array_equal(
            self.scores["variant"].values,
            np.arange(len(self.variants)),
        ):
            raise ValueError("The score variant coordinate must match variants.csv.")
        if not np.array_equal(
            self.scores["sample"].values,
            np.arange(self.scores.sizes["sample"]),
        ):
            raise ValueError("The score sample coordinate must be contiguous from zero.")

        contracts = self.metadata.get("score_contracts")
        if not isinstance(contracts, dict) or set(contracts) != set(score_names):
            raise ValueError("score_contracts must map every score_name exactly once.")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in contracts.values()
        ):
            raise ValueError("Every score contract must be a non-empty string.")
        if self.metadata.get("n_samples") != self.scores.sizes["sample"]:
            raise ValueError("metadata n_samples does not match fitness_score.")
        if self.metadata.get("n_variants") != self.scores.sizes["variant"]:
            raise ValueError("metadata n_variants does not match fitness_score.")
        if self.metadata.get("scores") != score_names:
            raise ValueError("metadata scores must match score_name coordinate order.")
        if schema_version == SCORE_ENSEMBLE_SCHEMA_VERSION and self.metadata.get(
            "n_analysis_variants"
        ) != len(self.analysis_variants):
            raise ValueError(
                "metadata n_analysis_variants does not match analysis_variants.csv."
            )
        required_block_starts = self.metadata.get("required_block_starts", [])
        if (
            not isinstance(required_block_starts, list)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= self.scores.sizes["variant"]
                for value in required_block_starts
            )
            or len(set(required_block_starts)) != len(required_block_starts)
        ):
            raise ValueError(
                "required_block_starts must contain unique in-range integers."
            )
        if self.metadata.get("inference_method") in {"hmc", "hmc_count_based"}:
            self._validate_hmc_sample_coordinates()

    def _validate_identity(self) -> None:
        fields = (
            "protein_family",
            "landscape",
            "inference_method",
            "model_spec",
            "fit_id",
        )
        if any(
            not isinstance(self.metadata.get(name), str)
            or not str(self.metadata[name]).strip()
            for name in fields
        ):
            raise ValueError("Score ensemble identity fields must be non-empty strings.")
        identity = AnalysisIdentity(
            **{name: str(self.metadata[name]) for name in fields}
        )
        if self.metadata.get("analysis_id") != identity.analysis_id:
            raise ValueError("Score ensemble analysis_id does not match its identity.")

    @staticmethod
    def _validate_variant_table(variants: pd.DataFrame, *, name: str) -> None:
        if "variant_id" not in variants:
            raise ValueError(f"{name} must contain variant_id.")
        if variants["variant_id"].isna().any():
            raise ValueError(f"{name} variant_id values cannot be missing.")
        if variants["variant_id"].duplicated().any():
            raise ValueError(f"{name} variant_id values must be unique.")

    def _validate_analysis_mapping(self) -> None:
        mapping_name = "matrix_variant_index"
        if mapping_name not in self.analysis_variants:
            raise ValueError(f"analysis_variants must contain {mapping_name}.")
        mapping = self.analysis_variants[mapping_name]
        if not is_integer_dtype(mapping.dtype):
            raise ValueError("matrix_variant_index must have an integer dtype.")
        values = mapping.to_numpy(dtype=np.int64)
        if (
            len(values) == 0
            or values[0] < 0
            or values[-1] >= len(self.variants)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(
                "matrix_variant_index must be non-empty, in range, and "
                "strictly increasing."
            )
        expected_ids = self.variants.iloc[values]["variant_id"].reset_index(drop=True)
        observed_ids = self.analysis_variants["variant_id"].reset_index(drop=True)
        if not expected_ids.equals(observed_ids):
            raise ValueError(
                "analysis_variants variant IDs do not match matrix_variant_index."
            )

    def _validate_hmc_sample_coordinates(self) -> None:
        required = {"source_sample_id", "source_chain", "source_draw"}
        missing = required - set(self.scores.coords)
        if missing:
            raise ValueError(
                f"HMC score ensembles require sample coordinates: {sorted(missing)}."
            )
        n_chains = self.metadata.get("n_chains")
        draws_per_chain = self.metadata.get("draws_per_chain")
        if (
            not isinstance(n_chains, int)
            or isinstance(n_chains, bool)
            or n_chains < 1
            or not isinstance(draws_per_chain, int)
            or isinstance(draws_per_chain, bool)
            or draws_per_chain < 1
        ):
            raise ValueError(
                "HMC score ensembles require positive integer n_chains and "
                "draws_per_chain metadata."
            )
        coordinates = {
            name: np.asarray(self.scores[name].values, dtype=np.int64)
            for name in required
        }
        expected_shape = (self.scores.sizes["sample"],)
        if any(self.scores[name].dims != ("sample",) for name in required) or any(
            values.shape != expected_shape for values in coordinates.values()
        ):
            raise ValueError("HMC source coordinates must use the sample axis.")
        source_sample_id = coordinates["source_sample_id"]
        source_chain = coordinates["source_chain"]
        source_draw = coordinates["source_draw"]
        if (
            np.any(source_chain < 0)
            or np.any(source_chain >= n_chains)
            or np.any(source_draw < 0)
            or np.any(source_draw >= draws_per_chain)
        ):
            raise ValueError("HMC source chain/draw coordinates lie outside their axes.")
        if not np.array_equal(
            source_sample_id,
            source_chain * draws_per_chain + source_draw,
        ):
            raise ValueError("HMC source sample IDs do not match chain/draw coordinates.")
        if len(np.unique(source_sample_id)) != len(source_sample_id):
            raise ValueError("HMC source sample IDs must be unique.")

    def save(self, path: str | Path) -> Path:
        """Write a schema-v2 score bundle."""
        if self.metadata.get("schema_version") != SCORE_ENSEMBLE_SCHEMA_VERSION:
            raise ValueError("Only schema-v2 score ensembles can be saved.")
        self.validate()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.variants.to_csv(output_dir / "variants.csv", index=False)
        self.analysis_variants.to_csv(
            output_dir / "analysis_variants.csv",
            index=False,
        )
        n_samples = self.scores.sizes["sample"]
        n_variants = self.scores.sizes["variant"]
        encoding = {
            "fitness_score": {
                "dtype": "float32",
                "chunksizes": (1, min(64, n_samples), min(4_096, n_variants)),
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
            }
        }
        if self.observed_score is not None:
            encoding["observed_score"] = {
                "dtype": "float32",
                "chunksizes": (1, min(4_096, n_variants)),
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
            }
        self.scores.to_netcdf(
            output_dir / "fitness_scores.nc",
            engine="h5netcdf",
            encoding=encoding,
        )
        return output_dir

    @classmethod
    def load(cls, path: str | Path) -> ScoreEnsemble:
        """Open a schema-v2 score bundle or lazily normalize a v1 proxy bundle."""
        output_dir = Path(path)
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Score ensemble directory is missing metadata.json.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        schema_version = metadata.get("schema_version")
        if schema_version == LEGACY_PROXY_SCHEMA_VERSION:
            required = {"variants.csv", "fitness_proxies.nc"}
            missing = sorted(
                name for name in required if not (output_dir / name).is_file()
            )
            if missing:
                raise ValueError(
                    f"Legacy proxy ensemble directory is missing files: {missing}"
                )
            variants = _read_variants_csv(output_dir / "variants.csv")
            analysis_variants = variants.copy()
            analysis_variants.insert(
                0,
                "matrix_variant_index",
                np.arange(len(variants), dtype=np.int64),
            )
            scores = xr.open_dataset(
                output_dir / "fitness_proxies.nc",
                engine="h5netcdf",
            ).rename(
                {
                    "fitness_proxy": "fitness_score",
                    "observed_proxy": "observed_score",
                }
            )
        elif schema_version == SCORE_ENSEMBLE_SCHEMA_VERSION:
            required = {
                "variants.csv",
                "analysis_variants.csv",
                "fitness_scores.nc",
            }
            missing = sorted(
                name for name in required if not (output_dir / name).is_file()
            )
            if missing:
                raise ValueError(
                    f"Score ensemble directory is missing files: {missing}"
                )
            variants = _read_variants_csv(output_dir / "variants.csv")
            analysis_variants = _read_variants_csv(
                output_dir / "analysis_variants.csv"
            )
            scores = xr.open_dataset(
                output_dir / "fitness_scores.nc",
                engine="h5netcdf",
            )
        else:
            raise ValueError(f"Unsupported score ensemble schema_version: {schema_version}")

        ensemble = cls(
            metadata=metadata,
            variants=variants,
            analysis_variants=analysis_variants,
            scores=scores,
        )
        try:
            ensemble.validate()
        except Exception:
            scores.close()
            raise
        return ensemble

    def close(self) -> None:
        """Close lazy resources and remove builder-owned temporary files."""
        self.scores.close()
        for path in self.temporary_paths:
            path.unlink(missing_ok=True)
