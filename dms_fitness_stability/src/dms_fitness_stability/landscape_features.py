"""Materialized observed-count features used by mechanism analyses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .adapters.k50 import K50_PROTEASES
from .identity import AnalysisIdentity
from .mechanisms import grouped_output_disagreement, paired_replicate_features
from .migration_audit import sha256_file
from .results import _read_variants_csv

LANDSCAPE_FEATURE_SCHEMA_VERSION = 1
LANDSCAPE_FEATURE_NAMES = (
    "observation_count",
    "input_count",
    "output_count",
    "log_input_support",
    "zero_input_fraction",
    "zero_output_fraction",
    "reduced_heterogeneity",
    "enrichment_range",
    "enrichment_mad",
)
LANDSCAPE_FEATURE_DTYPES = {
    "observation_count": np.dtype("int32"),
    **{
        name: np.dtype("float64")
        for name in LANDSCAPE_FEATURE_NAMES
        if name != "observation_count"
    },
}
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _slug(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    slug = _SLUG_SEPARATOR.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{field} must contain an ASCII letter or number.")
    return slug


def landscape_feature_id(
    protein_family: str,
    landscape: str,
    *,
    score_name: str | None = None,
) -> str:
    """Return the stable directory identifier for one feature view."""
    fields = [
        _slug(protein_family, field="protein_family"),
        _slug(landscape, field="landscape"),
    ]
    if score_name is not None:
        fields.append(_slug(score_name, field="score_name"))
    return "__".join(fields)


def _json_shape(values: np.ndarray) -> str:
    return json.dumps(list(np.asarray(values).shape))


def _json_depths(values: np.ndarray) -> str:
    counts = np.asarray(values)
    if counts.ndim == 1:
        depths = [int(counts.sum())]
    else:
        depths = counts.sum(axis=-1, dtype=np.int64).tolist()
    return json.dumps(depths)


def extract_landscape_count_features(
    identity: AnalysisIdentity,
    landscape,
    *,
    score_name: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Calculate the exact observed-count materialized view used by figures."""
    if identity.protein_family == "pdz":
        features = dict(
            paired_replicate_features(
                landscape.starting_counts,
                landscape.timepoint_counts,
            )
        )
        inventory = {
            "observation_structure": "paired_input_output_replicates",
            "group_count": 1,
            "replicate_count": int(landscape.starting_counts.shape[0]),
            "discordance_available": True,
        }
        starting_count_shape = _json_shape(landscape.starting_counts)
        output_count_shape = _json_shape(landscape.timepoint_counts)
        input_library_depths = _json_depths(landscape.starting_counts)
        output_library_depths = _json_depths(landscape.timepoint_counts)
        zero_input_fraction = float(np.mean(landscape.starting_counts == 0))
        zero_output_fraction = float(np.mean(landscape.timepoint_counts == 0))
    elif identity.protein_family == "trpb":
        starting = np.asarray(landscape.starting_counts, dtype=float)
        timepoint = np.asarray(landscape.timepoint_counts, dtype=float)
        n_variant = starting.shape[-1]
        input_count = starting.sum(axis=0) if starting.ndim == 2 else starting
        zero_input_fraction_by_variant = (
            np.mean(starting == 0, axis=0)
            if starting.ndim == 2
            else (starting == 0).astype(float)
        )
        features = {
            "observation_count": np.full(
                n_variant,
                timepoint.shape[0],
                dtype=np.int32,
            ),
            "input_count": input_count,
            "output_count": timepoint.sum(
                axis=tuple(range(timepoint.ndim - 1)),
            ),
            "log_input_support": np.log1p(input_count),
            "zero_input_fraction": zero_input_fraction_by_variant,
            "zero_output_fraction": np.mean(
                timepoint == 0,
                axis=tuple(range(timepoint.ndim - 1)),
            ),
            "reduced_heterogeneity": np.full(n_variant, np.nan),
            "enrichment_range": np.full(n_variant, np.nan),
            "enrichment_mad": np.full(n_variant, np.nan),
        }
        if timepoint.ndim == 3:
            disagreement = grouped_output_disagreement(timepoint)
            for name in (
                "observation_count",
                "output_count",
                "zero_output_fraction",
                "reduced_heterogeneity",
                "enrichment_range",
                "enrichment_mad",
            ):
                features[name] = disagreement[name]
            inventory = {
                "observation_structure": (
                    "time_series_output_replicates_two_inputs"
                    if starting.ndim == 2
                    else "time_series_output_replicates_shared_input"
                ),
                "group_count": int(timepoint.shape[0]),
                "replicate_count": int(timepoint.shape[1]),
                "input_replicate_count": (
                    int(starting.shape[0]) if starting.ndim == 2 else 1
                ),
                "discordance_available": True,
            }
        else:
            inventory = {
                "observation_structure": "time_series_single_output",
                "group_count": int(timepoint.shape[0]),
                "replicate_count": 1,
                "input_replicate_count": 1,
                "discordance_available": False,
            }
        starting_count_shape = _json_shape(landscape.starting_counts)
        output_count_shape = _json_shape(landscape.timepoint_counts)
        input_library_depths = _json_depths(landscape.starting_counts)
        output_library_depths = _json_depths(landscape.timepoint_counts)
        zero_input_fraction = float(np.mean(landscape.starting_counts == 0))
        zero_output_fraction = float(np.mean(landscape.timepoint_counts == 0))
    elif identity.protein_family == "nuc":
        input_count = landscape.variants["input_count"].to_numpy(dtype=float)
        output_count = landscape.variants["output_count"].to_numpy(dtype=float)
        n_variants = len(landscape.variants)
        features = {
            "observation_count": np.ones(n_variants, dtype=np.int32),
            "input_count": input_count,
            "output_count": output_count,
            "log_input_support": np.log1p(input_count),
            "zero_input_fraction": (input_count == 0).astype(float),
            "zero_output_fraction": (output_count == 0).astype(float),
            "reduced_heterogeneity": np.full(n_variants, np.nan),
            "enrichment_range": np.full(n_variants, np.nan),
            "enrichment_mad": np.full(n_variants, np.nan),
        }
        inventory = {
            "observation_structure": (
                "pooled_generation_counts_no_replicate_contrast"
            ),
            "group_count": len(landscape.counts),
            "replicate_count": 1,
            "discordance_available": False,
        }
        count_shapes = {
            name: list(np.asarray(values).shape)
            for name, values in landscape.counts.items()
        }
        starting_count_shape = json.dumps(count_shapes, sort_keys=True)
        output_count_shape = starting_count_shape
        input_library_depths = json.dumps([int(input_count.sum())])
        output_library_depths = json.dumps([int(output_count.sum())])
        zero_input_fraction = float(np.mean(input_count == 0))
        zero_output_fraction = float(np.mean(output_count == 0))
    elif identity.protein_family == "k50":
        if score_name is None:
            raise ValueError("K50 mechanism features require a score name.")
        protease_name = score_name.removeprefix("mean_log_").removesuffix(
            "_enrichment"
        )
        if protease_name not in K50_PROTEASES:
            raise ValueError(f"Unsupported K50 score: {score_name!r}.")
        protease_index = K50_PROTEASES.index(protease_name)
        n_variants = landscape.n_fitted_variants
        input_count = np.zeros(n_variants, dtype=np.float64)
        output_count = np.zeros(n_variants, dtype=np.float64)
        observation_count = np.zeros(n_variants, dtype=np.int32)
        input_observation_count = np.zeros(n_variants, dtype=np.int32)
        zero_input_count = np.zeros(n_variants, dtype=np.int32)
        zero_output_count = np.zeros(n_variants, dtype=np.int32)
        count_shapes = {}
        input_library_depths = {}
        output_library_depths = {}
        for library_name, library in landscape.libraries.items():
            indices = library.global_sequence_indices
            starting = np.asarray(library.counts_c0[protease_index])
            timepoint = np.asarray(library.counts_cg0[protease_index])
            np.add.at(input_count, indices, starting.sum(axis=0))
            np.add.at(output_count, indices, timepoint.sum(axis=(0, 1)))
            np.add.at(
                observation_count,
                indices,
                timepoint.shape[0] * timepoint.shape[1],
            )
            np.add.at(input_observation_count, indices, starting.shape[0])
            np.add.at(zero_input_count, indices, (starting == 0).sum(axis=0))
            np.add.at(
                zero_output_count,
                indices,
                (timepoint == 0).sum(axis=(0, 1)),
            )
            count_shapes[library_name] = {
                "input": list(starting.shape),
                "output": list(timepoint.shape),
            }
            input_library_depths[library_name] = starting.sum(
                axis=-1,
                dtype=np.int64,
            ).tolist()
            output_library_depths[library_name] = timepoint.sum(
                axis=-1,
                dtype=np.int64,
            ).tolist()
        zero_input_fraction_by_variant = zero_input_count / input_observation_count
        zero_output_fraction_by_variant = zero_output_count / observation_count
        features = {
            "observation_count": observation_count,
            "input_count": input_count,
            "output_count": output_count,
            "log_input_support": np.log1p(input_count),
            "zero_input_fraction": zero_input_fraction_by_variant,
            "zero_output_fraction": zero_output_fraction_by_variant,
            "reduced_heterogeneity": np.full(n_variants, np.nan),
            "enrichment_range": np.full(n_variants, np.nan),
            "enrichment_mad": np.full(n_variants, np.nan),
        }
        inventory = {
            "observation_structure": (
                "protease_specific_concentration_series_two_replicates"
            ),
            "protease": protease_name,
            "group_count": 11,
            "replicate_count": 2,
            "input_replicate_count": 2,
            "library_count": len(landscape.libraries),
            "discordance_available": False,
        }
        starting_count_shape = json.dumps(count_shapes, sort_keys=True)
        output_count_shape = starting_count_shape
        input_library_depths = json.dumps(input_library_depths, sort_keys=True)
        output_library_depths = json.dumps(output_library_depths, sort_keys=True)
        zero_input_fraction = float(
            zero_input_count.sum() / input_observation_count.sum()
        )
        zero_output_fraction = float(
            zero_output_count.sum() / observation_count.sum()
        )
    else:
        raise ValueError(
            f"Unsupported mechanism family: {identity.protein_family!r}."
        )
    inventory.update(
        {
            "protein_family": identity.protein_family,
            "landscape": identity.landscape,
            "starting_count_shape": starting_count_shape,
            "output_count_shape": output_count_shape,
            "input_library_depths": input_library_depths,
            "output_library_depths": output_library_depths,
            "zero_input_fraction_all_variants": zero_input_fraction,
            "zero_output_fraction_all_variants": zero_output_fraction,
        }
    )
    return features, inventory


def landscape_feature_variants(
    identity: AnalysisIdentity,
    landscape,
) -> pd.DataFrame:
    """Return the full ordered source axis for one feature view."""
    if identity.protein_family == "pdz":
        variant_ids = landscape.fitted_variant_ids
        if variant_ids is None:
            raise ValueError("PDZ feature extraction requires full fitted variant IDs.")
    else:
        if "variant_id" not in landscape.variants:
            raise ValueError("Landscape variants must contain variant_id.")
        variant_ids = tuple(landscape.variants["variant_id"].astype(str))
    variants = pd.DataFrame(
        {
            "source_variant_index": np.arange(len(variant_ids), dtype=np.int64),
            "variant_id": variant_ids,
        }
    )
    if variants["variant_id"].duplicated().any():
        raise ValueError("Landscape feature variant IDs must be unique.")
    if hasattr(landscape, "n_hmc_variants"):
        expected = int(landscape.n_hmc_variants)
    elif hasattr(landscape, "n_fitted_variants"):
        expected = int(landscape.n_fitted_variants)
    else:
        expected = len(landscape.variants)
    if len(variants) != expected:
        raise ValueError("Landscape feature variants do not match the source axis.")
    return variants


@dataclass
class LandscapeFeatureBundle:
    """Exact terminal observed-count features on one full source variant axis."""

    metadata: dict[str, object]
    variants: pd.DataFrame
    features: xr.Dataset

    def validate(self) -> None:
        """Validate schema, feature dtypes, metadata, and variant alignment."""
        if self.metadata.get("schema_version") != LANDSCAPE_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Landscape feature schema_version must be "
                f"{LANDSCAPE_FEATURE_SCHEMA_VERSION}."
            )
        protein_family = self.metadata.get("protein_family")
        landscape = self.metadata.get("landscape")
        score_name = self.metadata.get("score_name")
        expected_id = landscape_feature_id(
            str(protein_family),
            str(landscape),
            score_name=None if score_name is None else str(score_name),
        )
        if self.metadata.get("feature_id") != expected_id:
            raise ValueError("feature_id does not match its landscape identity.")
        required_variant_columns = {"source_variant_index", "variant_id"}
        if set(self.variants.columns) != required_variant_columns:
            raise ValueError(
                "Feature variants must contain exactly source_variant_index and "
                "variant_id."
            )
        if self.variants["variant_id"].isna().any():
            raise ValueError("Feature variant IDs cannot be missing.")
        if self.variants["variant_id"].duplicated().any():
            raise ValueError("Feature variant IDs must be unique.")
        source_indices = self.variants["source_variant_index"].to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(source_indices, np.arange(len(self.variants))):
            raise ValueError(
                "source_variant_index must be contiguous and match row order."
            )
        if set(self.features.data_vars) != set(LANDSCAPE_FEATURE_NAMES):
            raise ValueError(
                "Feature dataset must contain exactly the terminal feature contract."
            )
        if self.features.sizes.get("variant") != len(self.variants):
            raise ValueError("Feature arrays must match the variant table length.")
        if not np.array_equal(
            self.features["variant"].values,
            np.arange(len(self.variants)),
        ):
            raise ValueError("Feature variant coordinate must match row order.")
        for name, dtype in LANDSCAPE_FEATURE_DTYPES.items():
            if self.features[name].dims != ("variant",):
                raise ValueError(f"{name} must use only the variant dimension.")
            if self.features[name].dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}.")
        if self.metadata.get("n_variants") != len(self.variants):
            raise ValueError("metadata n_variants does not match the feature axis.")
        if self.metadata.get("derivation_contract") != "landscape-count-features-v1":
            raise ValueError("Unsupported landscape feature derivation contract.")
        inventory_fields = self.metadata.get("inventory_fields")
        if (
            not isinstance(inventory_fields, list)
            or len(set(inventory_fields)) != len(inventory_fields)
            or any(
                not isinstance(name, str) or name not in self.metadata
                for name in inventory_fields
            )
        ):
            raise ValueError(
                "inventory_fields must uniquely identify persisted metadata fields."
            )

    @classmethod
    def from_landscape(
        cls,
        identity: AnalysisIdentity,
        landscape,
        *,
        score_name: str | None = None,
        extra_metadata: dict[str, object] | None = None,
    ) -> LandscapeFeatureBundle:
        """Materialize the current terminal feature transformation."""
        feature_values, inventory = extract_landscape_count_features(
            identity,
            landscape,
            score_name=score_name,
        )
        variants = landscape_feature_variants(identity, landscape)
        features = xr.Dataset(
            {
                name: (
                    "variant",
                    np.asarray(feature_values[name], dtype=dtype),
                )
                for name, dtype in LANDSCAPE_FEATURE_DTYPES.items()
            },
            coords={"variant": np.arange(len(variants), dtype=np.int32)},
        )
        metadata = {
            **dict(extra_metadata or {}),
            **inventory,
            "schema_version": LANDSCAPE_FEATURE_SCHEMA_VERSION,
            "feature_id": landscape_feature_id(
                identity.protein_family,
                identity.landscape,
                score_name=score_name,
            ),
            "score_name": score_name,
            "n_variants": len(variants),
            "derivation_contract": "landscape-count-features-v1",
            "inventory_fields": list(inventory),
        }
        bundle = cls(metadata=metadata, variants=variants, features=features)
        bundle.validate()
        return bundle

    def save(self, path: str | Path) -> Path:
        """Write a checksummed feature bundle directory."""
        self.validate()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        variant_path = output_dir / "variants.csv"
        feature_path = output_dir / "features.nc"
        self.variants.to_csv(variant_path, index=False)
        encoding = {
            name: {
                "dtype": str(dtype),
                "chunksizes": (min(4_096, len(self.variants)),),
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
            }
            for name, dtype in LANDSCAPE_FEATURE_DTYPES.items()
        }
        self.features.to_netcdf(
            feature_path,
            engine="h5netcdf",
            encoding=encoding,
        )
        self.metadata.update(
            {
                "variants_sha256": sha256_file(variant_path),
                "features_sha256": sha256_file(feature_path),
                "variant_id_sha256": hashlib.sha256(
                    json.dumps(
                        self.variants["variant_id"].tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_dir

    @classmethod
    def load(cls, path: str | Path) -> LandscapeFeatureBundle:
        """Open and checksum-verify a persisted feature bundle."""
        output_dir = Path(path)
        required = {"metadata.json", "variants.csv", "features.nc"}
        missing = sorted(
            name for name in required if not (output_dir / name).is_file()
        )
        if missing:
            raise ValueError(f"Feature bundle directory is missing files: {missing}")
        metadata = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )
        for name, filename in (
            ("variants_sha256", "variants.csv"),
            ("features_sha256", "features.nc"),
        ):
            observed = sha256_file(output_dir / filename)
            if metadata.get(name) != observed:
                raise ValueError(f"Feature bundle checksum mismatch for {filename}.")
        bundle = cls(
            metadata=metadata,
            variants=_read_variants_csv(output_dir / "variants.csv"),
            features=xr.open_dataset(output_dir / "features.nc", engine="h5netcdf"),
        )
        try:
            bundle.validate()
        except Exception:
            bundle.features.close()
            raise
        return bundle

    def close(self) -> None:
        """Close the lazy feature dataset."""
        self.features.close()

    def align_result_variants(self, variants: pd.DataFrame) -> np.ndarray:
        """Validate and return source indices for an analysis result table."""
        if "variant_id" not in variants:
            raise ValueError(
                "Result variants are missing feature alignment column: variant_id."
            )
        mapping_columns = [
            name
            for name in ("matrix_variant_index", "hmc_variant_index")
            if name in variants
        ]
        if not mapping_columns:
            raise ValueError(
                "Result variants must contain matrix_variant_index; "
                "hmc_variant_index is accepted only for legacy results."
            )
        source_indices = variants[mapping_columns[0]].to_numpy(dtype=np.int64)
        if len(mapping_columns) == 2 and not np.array_equal(
            source_indices,
            variants[mapping_columns[1]].to_numpy(dtype=np.int64),
        ):
            raise ValueError(
                "matrix_variant_index and legacy hmc_variant_index disagree."
            )
        if (
            np.any(source_indices < 0)
            or np.any(source_indices >= len(self.variants))
            or len(np.unique(source_indices)) != len(source_indices)
        ):
            raise ValueError("Result feature indices must be unique and in range.")
        expected_ids = self.variants.iloc[source_indices]["variant_id"].reset_index(
            drop=True
        )
        observed_ids = variants["variant_id"].astype(str).reset_index(drop=True)
        if not expected_ids.equals(observed_ids):
            raise ValueError(
                "Result variant IDs do not match the landscape feature axis."
            )
        return source_indices


def audit_landscape_feature_source(
    bundle: LandscapeFeatureBundle,
    identity: AnalysisIdentity,
    landscape,
    *,
    score_name: str | None = None,
) -> dict[str, object]:
    """Require exact equality with features recomputed from observed inputs."""
    bundle.validate()
    expected = LandscapeFeatureBundle.from_landscape(
        identity,
        landscape,
        score_name=score_name,
    )
    pd.testing.assert_frame_equal(bundle.variants, expected.variants, check_exact=True)
    for name in LANDSCAPE_FEATURE_NAMES:
        np.testing.assert_array_equal(
            bundle.features[name].values,
            expected.features[name].values,
            err_msg=f"Feature source mismatch for {bundle.metadata['feature_id']}:{name}",
        )
    inventory_fields = bundle.metadata["inventory_fields"]
    if inventory_fields != expected.metadata["inventory_fields"]:
        raise ValueError("Feature inventory field ordering differs from its source.")
    mismatched_inventory = {
        name: (bundle.metadata.get(name), expected.metadata.get(name))
        for name in inventory_fields
        if bundle.metadata.get(name) != expected.metadata.get(name)
    }
    if mismatched_inventory:
        raise ValueError(f"Feature inventory differs: {mismatched_inventory}")
    finite_values = sum(
        int(np.isfinite(bundle.features[name].values).sum())
        for name in LANDSCAPE_FEATURE_NAMES
    )
    total_values = len(bundle.variants) * len(LANDSCAPE_FEATURE_NAMES)
    return {
        "feature_id": bundle.metadata["feature_id"],
        "n_variants": len(bundle.variants),
        "finite_values": finite_values,
        "nan_values": total_values - finite_values,
        "status": "exact",
    }
