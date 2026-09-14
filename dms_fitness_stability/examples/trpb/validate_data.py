"""Validate TrpB sequence alignment and canonical HMC fit structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5netcdf
import h5py
import numpy as np

from dms_fitness_stability.adapters.trpb import (
    TRPB_LIBRARIES,
    load_trpb_landscape,
    trpb_johnston_fitness,
    trpb_three_site_fitness,
)

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


def _validate_file(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular raw data file: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Unexpected byte size for {path}")
    if _sha256(path) != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")


def validate_data(data_root: Path, *, raw_only: bool) -> None:
    trpb_root = data_root / "trpb"
    data_manifest = json.loads(
        (HERE / "data_manifest.json").read_text(encoding="utf-8")
    )
    _validate_file(trpb_root / data_manifest["archive"]["filename"], data_manifest["archive"])
    for group_name in ("counts", "reported_fitness"):
        for filename, expected in data_manifest[group_name]["files"].items():
            _validate_file(trpb_root / "raw" / group_name / filename, expected)

    landscapes = {}
    for library in TRPB_LIBRARIES:
        landscape = load_trpb_landscape(
            library,
            count_path=trpb_root / "raw" / "counts" / f"{library}.csv",
            reported_fitness_path=(
                trpb_root / "raw" / "reported_fitness" / f"{library}.csv"
            ),
        )
        landscapes[library] = landscape
        n_reported = int(landscape.variants["reported_fitness"].notna().sum())
        observed_fitness = trpb_three_site_fitness(
            landscape.starting_counts,
            landscape.timepoint_counts,
            landscape.variants["variant_id"].to_numpy(dtype=str),
            landscape.times,
        )
        reported_fitness = landscape.variants["reported_fitness"].to_numpy(
            dtype=float
        )
        if not np.array_equal(
            np.isfinite(observed_fitness),
            np.isfinite(reported_fitness),
        ) or not np.allclose(
            observed_fitness[np.isfinite(reported_fitness)],
            reported_fitness[np.isfinite(reported_fitness)],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"{library}: observed counts do not reproduce reported fitness."
            )
        print(
            f"raw OK: {library} ({landscape.n_hmc_variants:,} fitted; "
            f"{n_reported:,} reported; Johnston oracle matched)"
        )
    four_site = load_trpb_landscape(
        "four-site",
        count_path=trpb_root / "raw/counts/four-site.csv",
        reported_fitness_path=trpb_root / "raw/reported_fitness/four-site.csv",
    )
    n_reported = int(four_site.variants["reported_fitness"].notna().sum())
    observed_fitness = trpb_johnston_fitness(
        four_site.starting_counts,
        four_site.timepoint_counts,
        four_site.variants["variant_id"].to_numpy(dtype=str),
        four_site.times,
    )
    reported_fitness = four_site.variants["reported_fitness"].to_numpy(dtype=float)
    if not np.array_equal(
        np.isfinite(observed_fitness),
        np.isfinite(reported_fitness),
    ) or not np.allclose(
        observed_fitness[np.isfinite(reported_fitness)],
        reported_fitness[np.isfinite(reported_fitness)],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Four-site observed counts do not reproduce reported fitness.")
    print(
        f"raw OK: four-site ({four_site.n_hmc_variants:,} fitted; "
        f"{n_reported:,} reported; Johnston oracle matched)"
    )
    if raw_only:
        return

    manifest = json.loads((HERE / "fit_jobs.json").read_text(encoding="utf-8"))
    embedded_count_matches = dict.fromkeys(TRPB_LIBRARIES, False)
    for spec in manifest["fits"]:
        landscape = landscapes[spec["library"]]
        path = (
            trpb_root
            / "fits"
            / spec["library"]
            / spec["canonical_filename"]
        )
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected a regular canonical HMC file: {path}")
        observed_issues = set()
        try:
            with h5netcdf.File(path, "r"):
                pass
        except Exception:
            observed_issues.add("h5netcdf_open_failure")
        with h5py.File(path, "r") as handle:
            observed = handle.get("observed_data")
            if observed is None:
                observed_issues.add("missing_observed_data")
                count_status = "counts not embedded"
            else:
                if not np.array_equal(
                    observed["starting_counts"][:], landscape.starting_counts
                ) or not np.array_equal(
                    observed["timepoint_counts"][:], landscape.timepoint_counts
                ):
                    raise ValueError(f"Observed counts differ from raw data in {path}")
                embedded_count_matches[spec["library"]] = True
                count_status = "counts matched"
            target = handle["posterior"][spec["posterior_variable"]]
            if target.ndim != 3 or target.shape[-1] != landscape.n_hmc_variants:
                raise ValueError(f"Unexpected posterior target shape in {path}")
            edge_values = np.asarray([target[0, 0, 0], target[-1, -1, -1]])
            if not np.isfinite(edge_values).all():
                raise ValueError(f"Non-finite posterior target edge values in {path}")
        declared_issues = set(spec.get("known_issues", []))
        if observed_issues != declared_issues:
            raise ValueError(
                f"{spec['job']}: observed issues {sorted(observed_issues)} differ "
                f"from manifest issues {sorted(declared_issues)}"
            )
        status = "LIMITED" if observed_issues else "OK"
        issue_text = (
            f"; known issues: {', '.join(sorted(observed_issues))}"
            if observed_issues
            else ""
        )
        print(f"fit {status}: {spec['job']} ({count_status}{issue_text}) -> {path}")

    missing_count_checks = [
        library for library, matched in embedded_count_matches.items() if not matched
    ]
    if missing_count_checks:
        raise ValueError(
            "No fit embedded observed counts for libraries: "
            f"{missing_count_checks}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    validate_data(args.data_root, raw_only=args.raw_only)


if __name__ == "__main__":
    main()
