"""Strict persisted inputs for rank-stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCORE_FILE = "scores.nc"
VARIANT_FILE = "variants.csv"
FEATURE_FILE = "features.nc"

FEATURE_DTYPES = {
    "observation_count": np.dtype("int32"),
    "input_count": np.dtype("float64"),
    "output_count": np.dtype("float64"),
    "log_input_support": np.dtype("float64"),
    "zero_input_fraction": np.dtype("float64"),
    "zero_output_fraction": np.dtype("float64"),
    "reduced_heterogeneity": np.dtype("float64"),
    "enrichment_range": np.dtype("float64"),
    "enrichment_mad": np.dtype("float64"),
}


def _read_variants(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    na_values = {column: [""] for column in columns if column != "variant_id"}
    return pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
        na_values=na_values,
    )


def _validate_variants(variants: pd.DataFrame, *, expected_rows: int) -> None:
    required = {"variant_id", "is_candidate"}
    missing = required - set(variants)
    if missing:
        raise ValueError(f"Variant table is missing columns: {sorted(missing)}")
    if len(variants) != expected_rows:
        raise ValueError("Variant rows must match the persisted variant axis.")
    if variants["variant_id"].isna().any() or variants["variant_id"].duplicated().any():
        raise ValueError("variant_id values must be present and unique.")
    if variants["is_candidate"].dtype != bool:
        raise ValueError("is_candidate must contain only boolean values.")
    if not variants["is_candidate"].any():
        raise ValueError("At least one variant must be eligible for ranking.")


def _validate_coordinate(dataset: xr.Dataset, name: str, size: int) -> None:
    if name not in dataset.coords:
        raise ValueError(f"Dataset is missing the {name!r} coordinate.")
    if dataset[name].dims != (name,) or not np.array_equal(
        dataset[name].values,
        np.arange(size),
    ):
        raise ValueError(f"{name} coordinates must be contiguous from zero.")


@dataclass
class SampledScores:
    """One sampled score matrix and its aligned biological variants."""

    variants: pd.DataFrame
    dataset: xr.Dataset

    @property
    def score(self) -> xr.DataArray:
        return self.dataset["score"]

    def validate(self) -> None:
        if set(self.dataset.data_vars) != {"score"}:
            raise ValueError("Score data must contain exactly one variable named 'score'.")
        if self.score.dims != ("sample", "variant"):
            raise ValueError("score dimensions must be sample and variant.")
        if self.score.dtype != np.dtype("float32"):
            raise ValueError("score values must use float32 storage.")
        if self.score.sizes["sample"] < 2:
            raise ValueError("At least two sampled score landscapes are required.")
        _validate_coordinate(self.dataset, "sample", self.score.sizes["sample"])
        _validate_coordinate(self.dataset, "variant", self.score.sizes["variant"])
        _validate_variants(self.variants, expected_rows=self.score.sizes["variant"])

        source_coordinates = {
            name
            for name in ("source_sample_id", "source_chain", "source_draw")
            if name in self.dataset.coords
        }
        if "source_sample_id" not in source_coordinates:
            raise ValueError("Score data must contain stable source_sample_id values.")
        for name in source_coordinates:
            if self.dataset[name].dims != ("sample",):
                raise ValueError(f"{name} must be aligned to the sample axis.")
        if "source_sample_id" in source_coordinates:
            source_ids = np.asarray(self.dataset["source_sample_id"].values)
            if len(np.unique(source_ids)) != len(source_ids):
                raise ValueError("source_sample_id values must be unique.")
        chain_draw = {"source_chain", "source_draw"} & source_coordinates
        if chain_draw and chain_draw != {"source_chain", "source_draw"}:
            raise ValueError("Source chain and draw coordinates must be provided together.")

        if "selection_block" in self.dataset.coords:
            blocks = self.dataset["selection_block"]
            if blocks.dims != ("variant",) or not np.issubdtype(
                blocks.dtype,
                np.integer,
            ):
                raise ValueError("selection_block must be an integer variant coordinate.")
            values = np.asarray(blocks.values, dtype=np.int64)
            labels = np.unique(values)
            if not np.array_equal(labels, np.arange(len(labels))):
                raise ValueError("selection_block labels must be contiguous from zero.")
            for label in labels:
                positions = np.flatnonzero(values == label)
                if positions[-1] - positions[0] + 1 != len(positions):
                    raise ValueError("Each selection_block must occupy contiguous variants.")

    def save(self, path: str | Path) -> Path:
        self.validate()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.variants.to_csv(root / VARIANT_FILE, index=False)
        self.dataset.to_netcdf(
            root / SCORE_FILE,
            engine="h5netcdf",
            encoding={
                "score": {
                    "dtype": "float32",
                    "chunksizes": (
                        min(64, self.score.sizes["sample"]),
                        min(4096, self.score.sizes["variant"]),
                    ),
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                }
            },
        )
        return root

    @classmethod
    def load(cls, path: str | Path) -> SampledScores:
        root = Path(path)
        required = (root / VARIANT_FILE, root / SCORE_FILE)
        missing = [item.name for item in required if not item.is_file()]
        if missing:
            raise ValueError(f"Score data are missing files: {missing}")
        dataset = xr.open_dataset(root / SCORE_FILE, engine="h5netcdf")
        scores = cls(variants=_read_variants(root / VARIANT_FILE), dataset=dataset)
        try:
            scores.validate()
        except Exception:
            dataset.close()
            raise
        return scores

    def close(self) -> None:
        self.dataset.close()


@dataclass
class LandscapeFeatures:
    """Count-derived landscape features aligned by biological variant identity."""

    variants: pd.DataFrame
    dataset: xr.Dataset

    def validate(self) -> None:
        if set(self.dataset.data_vars) != set(FEATURE_DTYPES):
            raise ValueError("Feature data do not contain the required feature set.")
        variant_count = self.dataset.sizes.get("variant")
        if variant_count is None:
            raise ValueError("Feature data must contain a variant dimension.")
        _validate_coordinate(self.dataset, "variant", variant_count)
        _validate_variants(self.variants, expected_rows=variant_count)
        for name, dtype in FEATURE_DTYPES.items():
            values = self.dataset[name]
            if values.dims != ("variant",) or values.dtype != dtype:
                raise ValueError(f"{name} must be a {dtype} vector over variants.")

    def save(self, path: str | Path) -> Path:
        self.validate()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.variants.to_csv(root / VARIANT_FILE, index=False)
        self.dataset.to_netcdf(
            root / FEATURE_FILE,
            engine="h5netcdf",
            encoding={
                name: {
                    "dtype": str(dtype),
                    "chunksizes": (min(4096, len(self.variants)),),
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                }
                for name, dtype in FEATURE_DTYPES.items()
            },
        )
        return root

    @classmethod
    def load(cls, path: str | Path) -> LandscapeFeatures:
        root = Path(path)
        required = (root / VARIANT_FILE, root / FEATURE_FILE)
        missing = [item.name for item in required if not item.is_file()]
        if missing:
            raise ValueError(f"Feature data are missing files: {missing}")
        dataset = xr.open_dataset(root / FEATURE_FILE, engine="h5netcdf")
        features = cls(variants=_read_variants(root / VARIANT_FILE), dataset=dataset)
        try:
            features.validate()
        except Exception:
            dataset.close()
            raise
        return features

    def indices_for(self, variants: pd.DataFrame) -> np.ndarray:
        if "variant_id" not in variants:
            raise ValueError("Analysis variants must contain variant_id.")
        if variants["variant_id"].isna().any() or variants["variant_id"].duplicated().any():
            raise ValueError("Analysis variant IDs must be present and unique.")
        feature_index = pd.Series(
            np.arange(len(self.variants), dtype=np.int64),
            index=self.variants["variant_id"],
        )
        indices = feature_index.reindex(variants["variant_id"])
        if indices.isna().any():
            missing = variants.loc[indices.isna().to_numpy(), "variant_id"].tolist()
            raise ValueError(f"Feature data are missing variant IDs: {missing[:5]}")
        return indices.to_numpy(dtype=np.int64)

    def close(self) -> None:
        self.dataset.close()
