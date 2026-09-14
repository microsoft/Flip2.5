"""Validate the canonical external PDZ cis data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5netcdf
import numpy as np
from flipv3.flip_dsets import load_pdz_dataset

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data(data_root: Path) -> None:
    """Validate file identity, count alignment, and NetCDF structure."""
    manifest = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    pdz_root = data_root / "pdz_cis"

    for filename, expected in manifest["raw"].items():
        path = pdz_root / "raw" / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected a regular copied raw file: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Unexpected byte size for {path}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        print(f"raw OK: {path}")

    counts = load_pdz_dataset(str(pdz_root / "raw" / "cis_from_bruce_azure.tsv"))
    for filename, expected in manifest["fits"].items():
        path = pdz_root / "fits" / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected a regular copied fit file: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Unexpected byte size for {path}")
        with h5netcdf.File(path, "r") as handle:
            posterior = handle.groups["posterior"].variables["log_r_mean"]
            predictive = handle.groups["posterior_predictive"].variables
            if list(posterior.shape) != expected["posterior_log_r_mean_shape"]:
                raise ValueError(f"Unexpected posterior shape in {path}")
            for variable_name in ("starting_counts", "timepoint_counts"):
                variable = predictive[variable_name]
                if list(variable.shape) != expected["posterior_predictive_count_shape"]:
                    raise ValueError(f"Unexpected {variable_name} shape in {path}")
                _ = variable[0, 0, 0, 0]
            observed = handle.groups["observed_data"].variables
            for raw_name, variable_name in (
                ("starting_counts", "starting_counts"),
                ("timepoint_counts", "timepoint_counts"),
            ):
                if not np.array_equal(observed[variable_name][:], counts[raw_name]):
                    raise ValueError(f"Embedded {variable_name} differs from {pdz_root / 'raw'}")
            _ = posterior[0, 0, 0]
        print(f"fit OK: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    validate_data(args.data_root)


if __name__ == "__main__":
    main()
