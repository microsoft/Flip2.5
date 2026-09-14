"""Persistence for compact stability results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr


def _read_variants_csv(path: Path) -> pd.DataFrame:
    """Read variants while preserving an empty-string biological identifier."""
    columns = pd.read_csv(path, nrows=0).columns
    na_values = {column: [""] for column in columns if column != "variant_id"}
    return pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
        na_values=na_values,
    )


@dataclass
class StabilityResult:
    """Compact draw-level selections and plot-ready tables."""

    metadata: dict[str, object]
    variants: pd.DataFrame
    selection: xr.Dataset
    summary: pd.DataFrame
    overlaps: pd.DataFrame

    def save(self, path: str | Path) -> Path:
        """Write the five-file result bundle to a directory."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.variants.to_csv(output_dir / "variants.csv", index=False)
        self.selection.to_netcdf(output_dir / "stability.nc", engine="h5netcdf")
        self.summary.to_csv(output_dir / "summary.csv", index=False)
        self.overlaps.to_csv(output_dir / "overlaps.csv", index=False)
        return output_dir

    @classmethod
    def load(cls, path: str | Path) -> "StabilityResult":
        """Load a previously saved five-file result bundle."""
        output_dir = Path(path)
        required = {
            "metadata.json",
            "variants.csv",
            "stability.nc",
            "summary.csv",
            "overlaps.csv",
        }
        missing = sorted(name for name in required if not (output_dir / name).exists())
        if missing:
            raise ValueError(f"Result directory is missing files: {missing}")
        return cls(
            metadata=json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")),
            variants=_read_variants_csv(output_dir / "variants.csv"),
            selection=xr.load_dataset(output_dir / "stability.nc", engine="h5netcdf"),
            summary=pd.read_csv(output_dir / "summary.csv", low_memory=False),
            overlaps=pd.read_csv(output_dir / "overlaps.csv"),
        )
