"""Validate the canonical source-only NucB data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dms_fitness_stability.adapters.nuclease import (
    NUCB_WT_SEQUENCE,
    NUCLEASE_REPORTED_FITNESS_COLUMN,
    load_nuclease_landscape,
    nuclease_highest_gate_enrichment,
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


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def _validate_file(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular source file: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Unexpected byte size for {path}")
    if _sha256(path) != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    if _row_count(path) != expected["rows"]:
        raise ValueError(f"Unexpected row count for {path}")


def _reject_fit_artifacts(root: Path) -> None:
    forbidden = [
        path
        for path in root.rglob("*")
        if (path.is_dir() and path.name == "fits")
        or (path.is_file() and path.suffix.lower() == ".nc")
    ]
    if forbidden:
        raise ValueError(f"Source-only directory contains fit artifacts: {forbidden}")


def validate_data(data_root: Path) -> None:
    manifest = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    nuc_root = data_root / "nuc"
    raw_root = nuc_root / "raw"
    for relative_path, expected in manifest["raw"].items():
        _validate_file(raw_root / relative_path, expected)

    if len(NUCB_WT_SEQUENCE) != manifest["reference"]["sequence_length"]:
        raise ValueError("NucB reference sequence length differs from the manifest.")
    if NUCB_WT_SEQUENCE != manifest["reference"]["sequence"]:
        raise ValueError("NucB reference sequence differs from the manifest.")

    for generation, expected in manifest["generations"].items():
        landscape = load_nuclease_landscape(
            generation,
            count_dir=raw_root / "counts",
            reported_activity_path=raw_root / "reported_fitness/landscape.csv",
        )
        if landscape.n_fitted_variants != expected["fitted_variants"]:
            raise ValueError(f"{generation}: unexpected fitted variant count.")
        if landscape.wild_type_index != expected["wild_type_index"]:
            raise ValueError(f"{generation}: unexpected wild-type index.")
        if landscape.variants["variant_id"].duplicated().any():
            raise ValueError(f"{generation}: variant identities are not unique.")
        is_control = landscape.variants["is_source_control"].to_numpy(dtype=bool)
        is_candidate = landscape.variants["is_ranking_candidate"].to_numpy(dtype=bool)
        if not np.array_equal(is_candidate, ~is_control):
            raise ValueError(f"{generation}: candidate and source-control roles overlap.")
        if int(is_candidate.sum()) != expected["main_rows"]:
            raise ValueError(f"{generation}: unexpected ranking-candidate count.")
        if int(is_control.sum()) != expected["negative_control_rows"]:
            raise ValueError(f"{generation}: unexpected source-control count.")
        if is_control[: expected["main_rows"]].any() or is_candidate[
            expected["main_rows"] :
        ].any():
            raise ValueError(f"{generation}: source roles do not follow fitted order.")
        reported_count = int(landscape.variants["reported_activity_level"].notna().sum())
        if reported_count != expected["reported_variants"]:
            raise ValueError(f"{generation}: unexpected reported-activity coverage.")
        reported_fitness = landscape.variants["reported_fitness"].dropna()
        if len(reported_fitness) != expected["reported_variants"]:
            raise ValueError(f"{generation}: unexpected reported-fitness coverage.")
        if NUCLEASE_REPORTED_FITNESS_COLUMN[generation] != expected[
            "reported_fitness_column"
        ]:
            raise ValueError(f"{generation}: unexpected reported-fitness column.")
        if reported_fitness.min() != expected["reported_fitness_min"]:
            raise ValueError(f"{generation}: unexpected reported-fitness minimum.")
        if reported_fitness.max() != expected["reported_fitness_max"]:
            raise ValueError(f"{generation}: unexpected reported-fitness maximum.")
        if reported_fitness.nunique() != expected["reported_fitness_unique_values"]:
            raise ValueError(f"{generation}: unexpected reported-fitness cardinality.")
        if int(reported_fitness.eq(0).sum()) != expected["reported_fitness_zero_values"]:
            raise ValueError(f"{generation}: unexpected zero-enrichment count.")
        recalculated = nuclease_highest_gate_enrichment(
            generation,
            landscape.counts,
        )
        score_column = NUCLEASE_REPORTED_FITNESS_COLUMN[generation]
        all_reported_fitness = pd.concat(
            [
                pd.read_csv(
                    raw_root / "counts" / f"{generation.lower()}.csv",
                    usecols=[score_column],
                ),
                pd.read_csv(
                    raw_root
                    / "counts"
                    / f"{generation.lower()}_neg_control.csv",
                    usecols=[score_column],
                ),
            ],
            ignore_index=True,
        )[score_column].to_numpy(dtype=float)
        if not np.allclose(
            recalculated,
            all_reported_fitness,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"{generation}: count-derived enrichment differs from the "
                "full main/control source table."
            )
        candidate_values = recalculated[is_candidate]
        if not np.allclose(
            candidate_values,
            landscape.variants.loc[is_candidate, "reported_fitness"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"{generation}: count-derived enrichment differs from "
                "reported fitness."
            )
        for top_k, boundary in expected["top_k_boundaries"].items():
            top_k = int(top_k)
            observed_boundary = reported_fitness.nlargest(top_k).iloc[-1]
            if observed_boundary != boundary:
                raise ValueError(f"{generation}: unexpected top-{top_k} boundary.")
            if int(reported_fitness.eq(boundary).sum()) != 1:
                raise ValueError(f"{generation}: top-{top_k} boundary is not unique.")
        count_shapes = {name: list(array.shape) for name, array in landscape.counts.items()}
        if count_shapes != expected["count_shapes"]:
            raise ValueError(f"{generation}: count-array shapes differ from the manifest.")
        threshold_shapes = {
            name: list(array.shape) for name, array in landscape.thresholds.items()
        }
        if threshold_shapes != expected["threshold_shapes"]:
            raise ValueError(f"{generation}: threshold shapes differ from the manifest.")
        if expected["main_rows"] + expected["negative_control_rows"] != expected[
            "fitted_variants"
        ]:
            raise ValueError(f"{generation}: source rows do not sum to fitted variants.")
        print(
            f"raw OK: {generation} ({landscape.n_fitted_variants:,} fitted; "
            f"{is_candidate.sum():,} ranking candidates; "
            f"{is_control.sum():,} source controls; "
            f"{reported_count:,} reported {expected['reported_fitness_column']}; "
            f"WT index {landscape.wild_type_index})"
        )

    _reject_fit_artifacts(nuc_root)
    print("fit artifacts: none (source-only stage)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    validate_data(args.data_root)


if __name__ == "__main__":
    main()
