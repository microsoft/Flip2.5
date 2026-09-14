"""Validate the canonical source-only Tsuboyama K50 data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

from dms_fitness_stability.adapters.k50 import load_k50_landscape

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
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
    if _hash(path, "sha256") != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    if _row_count(path) != expected["rows"]:
        raise ValueError(f"Unexpected row count for {path}")


def _validate_raw_schema(path: Path, library: int) -> None:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"name", "aa_seq", "dna_seq"}
    required.update(
        f"v{library}_{protease}{index:02d}"
        for protease in "CT"
        for index in range(1, 25)
    )
    if set(columns) != required or len(columns) != 51:
        raise ValueError(f"Unexpected K50 count schema in {path}")


def _reject_fit_artifacts(root: Path) -> None:
    deferred_names = {"K50_dG_tables.zip", "Processed_K50_dG_datasets.zip"}
    forbidden = [
        path
        for path in root.rglob("*")
        if (path.is_dir() and path.name == "fits")
        or (path.is_file() and (path.suffix.lower() == ".nc" or path.name in deferred_names))
    ]
    if forbidden:
        raise ValueError(f"Source-only directory contains fit artifacts: {forbidden}")


def validate_data(data_root: Path) -> None:
    manifest = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    k50_root = data_root / "k50"
    raw_root = k50_root / "raw"

    archive_spec = manifest["archive"]
    archive_path = k50_root / archive_spec["filename"]
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError(f"Expected a regular source archive: {archive_path}")
    if archive_path.stat().st_size != archive_spec["bytes"]:
        raise ValueError("Unexpected K50 source archive byte size.")
    if _hash(archive_path, "md5") != archive_spec["md5"]:
        raise ValueError("K50 source archive MD5 mismatch.")
    if _hash(archive_path, "sha256") != archive_spec["sha256"]:
        raise ValueError("K50 source archive SHA-256 mismatch.")

    with zipfile.ZipFile(archive_path) as archive:
        for library in range(1, 5):
            filename = f"NGS_count_lib{library}.csv"
            expected = manifest["raw"][filename]
            matches = [info for info in archive.infolist() if Path(info.filename).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected one {filename} archive member.")
            info = matches[0]
            if info.file_size != expected["bytes"] or f"{info.CRC:08x}" != expected[
                "zip_crc32"
            ]:
                raise ValueError(f"Archive identity mismatch for {filename}.")

    for filename, expected in manifest["raw"].items():
        path = raw_root / filename
        _validate_file(path, expected)
        if filename.startswith("NGS_count_lib"):
            library = int(
                filename.removeprefix("NGS_count_lib").removesuffix(".csv")
            )
            _validate_raw_schema(path, library)

    expected_qpcr_columns = [
        "protease_con",
        "ec5.1",
        "ec50_4.9",
        "ec50_4.4",
        "ec50_4.2",
        "ec50_3",
        "ec50_2",
        "ec50_0",
        "4cuts",
        "protease",
    ]
    qpcr_columns = pd.read_csv(
        raw_root / "Raw_qPCR_data_FigS1.csv", nrows=0
    ).columns.tolist()
    if qpcr_columns != expected_qpcr_columns:
        raise ValueError("Unexpected qPCR source schema.")
    expected_scramble_columns = ["aa_seq_experimental"]
    scramble_columns = pd.read_csv(
        raw_root / "K50_scrambles_for_STEP3.csv", nrows=0
    ).columns.tolist()
    if scramble_columns != expected_scramble_columns:
        raise ValueError("Unexpected K50 scramble source schema.")

    reported_spec = manifest["reported_fitness"]["projection"]
    reported_path = raw_root / reported_spec["path"]
    _validate_file(reported_path, reported_spec)
    reported_columns = pd.read_csv(reported_path, nrows=0).columns.tolist()
    if reported_columns != reported_spec["columns"]:
        raise ValueError("Unexpected K50 reported-fitness projection schema.")

    landscape = load_k50_landscape(
        raw_root,
        reported_fitness_path=reported_path,
    )
    expected_output = manifest["loader_output"]
    if landscape.n_fitted_variants != expected_output["global_fitted_variants"]:
        raise ValueError("Unexpected global K50 fitted variant count.")
    if len(landscape.scrambled_indices) != expected_output["scrambled_variants"]:
        raise ValueError("Unexpected K50 scramble count.")
    if list(landscape.ordinal_sequences.shape) != expected_output["ordinal_sequence_shape"]:
        raise ValueError("Unexpected K50 ordinal sequence shape.")
    if list(landscape.qpcr_log_protease_concentration.shape) != expected_output[
        "qpcr_log_protease_concentration_shape"
    ]:
        raise ValueError("Unexpected K50 qPCR concentration shape.")
    if list(landscape.qpcr_log2_survival.shape) != expected_output[
        "qpcr_log2_survival_shape"
    ]:
        raise ValueError("Unexpected K50 qPCR survival shape.")
    if list(landscape.log_expected_protease_concentration.shape) != expected_output[
        "log_expected_protease_concentration_shape"
    ]:
        raise ValueError("Unexpected K50 expected concentration shape.")
    if landscape.variants["variant_id"].duplicated().any():
        raise ValueError("K50 global sequence identities must be unique.")
    n_reported = int(landscape.variants["reported_fitness"].notna().sum())
    if n_reported != manifest["reported_fitness"]["projected_rows"][
        "eligible_unique_sequences"
    ]:
        raise ValueError("Unexpected K50 aligned reported-fitness count.")
    expected_boundaries = manifest["reported_fitness"]["top_k_boundaries"]
    if set(expected_boundaries) != {"50", "100", "1000"}:
        raise ValueError("K50 top-K boundary manifest must cover K=50, 100, and 1,000.")
    reported_fitness = landscape.variants["reported_fitness"].dropna()
    for top_k_text, expected_boundary in expected_boundaries.items():
        top_k = int(top_k_text)
        score = float(reported_fitness.nlargest(top_k).iloc[-1])
        strictly_above = int(reported_fitness.gt(score).sum())
        tied_at_boundary = int(reported_fitness.eq(score).sum())
        observed_boundary = {
            "score": score,
            "strictly_above": strictly_above,
            "tied_at_boundary": tied_at_boundary,
            "best_rank": strictly_above + 1,
            "worst_rank": strictly_above + tied_at_boundary,
            "exact_top_k_identifiable": strictly_above + tied_at_boundary == top_k,
        }
        if observed_boundary != expected_boundary:
            raise ValueError(f"Unexpected K50 top-{top_k} reported-fitness boundary.")
        tied = landscape.variants["reported_fitness"].eq(score)
        if not landscape.variants.loc[tied, "reported_rank_min"].eq(
            expected_boundary["best_rank"]
        ).all():
            raise ValueError(f"Unexpected K50 top-{top_k} best reported rank.")
        if not landscape.variants.loc[tied, "reported_rank_max"].eq(
            expected_boundary["worst_rank"]
        ).all():
            raise ValueError(f"Unexpected K50 top-{top_k} worst reported rank.")

    for library, expected in expected_output["libraries"].items():
        loaded = landscape.libraries[library]
        if loaded.n_variants != expected["fitted_variants"]:
            raise ValueError(f"{library}: unexpected fitted variant count.")
        if list(loaded.counts_c0.shape) != expected["counts_c0_shape"]:
            raise ValueError(f"{library}: unexpected untreated count shape.")
        if list(loaded.counts_cg0.shape) != expected["counts_cg0_shape"]:
            raise ValueError(f"{library}: unexpected treated count shape.")
        print(
            f"raw OK: {library} ({loaded.n_variants:,} fitted; "
            f"c0 {loaded.counts_c0.shape}; cg0 {loaded.counts_cg0.shape})"
        )

    _reject_fit_artifacts(k50_root)
    print(
        f"global OK: {landscape.n_fitted_variants:,} unique sequences; "
        f"{len(landscape.scrambled_indices):,} scrambles"
    )
    print(f"reported fitness OK: {n_reported:,} quality-eligible dG_ML sequences")
    print("reported top-K: K=50, 100, and 1,000 are tied within 41,517 sequences")
    print("job-63 MLE artifacts: not retained in this source-only K50 directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    validate_data(args.data_root)


if __name__ == "__main__":
    main()
