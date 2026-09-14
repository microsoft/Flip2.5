import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability.score_ensembles import ScoreEnsemble


@pytest.fixture
def write_schema_v1_bundle() -> Callable[..., Path]:
    def _write(
        path: str | Path,
        *,
        metadata: dict[str, object],
        variants: pd.DataFrame,
        legacy_dataset: xr.Dataset,
    ) -> Path:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        variants.to_csv(output_dir / "variants.csv", index=False)

        n_samples = legacy_dataset.sizes["sample"]
        n_variants = legacy_dataset.sizes["variant"]
        legacy_dataset.to_netcdf(
            output_dir / "fitness_proxies.nc",
            engine="h5netcdf",
            encoding={
                "fitness_proxy": {
                    "dtype": "float32",
                    "chunksizes": (1, min(64, n_samples), min(4_096, n_variants)),
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                },
                "observed_proxy": {
                    "dtype": "float32",
                    "chunksizes": (1, min(4_096, n_variants)),
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                },
            },
        )

        loaded = ScoreEnsemble.load(output_dir)
        loaded.close()
        return output_dir

    return _write
