"""Retrieve K50 observations and the published reported-fitness reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterator

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
ZENODO_RECORD = "7992926"
RAW_ARCHIVE = "Raw_NGS_count_tables.zip"
RAW_ARCHIVE_BYTES = 242_934_040
RAW_ARCHIVE_MD5 = "f0a2c5882439001dbbb8d81b4e64fe24"
SCRAMBLE_ZENODO_RECORD = "7401275"
PIPELINE_ARCHIVE = "Pipeline_K50_dG.zip"
PIPELINE_ARCHIVE_BYTES = 198_777_302
PIPELINE_ARCHIVE_MD5 = "57a81806aa701e2c6e5976f70b0f88a7"
SCRAMBLE_MEMBER = "K50_scrambles_for_STEP3.csv"
SCRAMBLE_MEMBER_BYTES = 59_028_290
SCRAMBLE_MEMBER_CRC32 = 0xCFC3D4DF
SCRAMBLE_MEMBER_SHA256 = "4512ce79de6bfae51bb761419918fcc67c18d51c03c20836e3f0242ebd6dfd5b"
SCRAMBLE_ROWS = 127_717
SCRAMBLE_SEQUENCE_COLUMN = "aa_seq_experimental"
SCRAMBLE_PROJECTION_BYTES = 9_323_361
SCRAMBLE_PROJECTION_SHA256 = (
    "fe15f9c3cedd57b4227115ac47e6db81baa86fda42ff06084ac3f50f6b560451"
)
REPORTED_ARCHIVE = "Processed_K50_dG_datasets.zip"
REPORTED_ARCHIVE_BYTES = 1_013_678_473
REPORTED_ARCHIVE_MD5 = "f7e8c553efee734cf161ee6f2b0a09cf"
REPORTED_MEMBER = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
REPORTED_MEMBER_BYTES = 697_658_024
REPORTED_MEMBER_COMPRESSED_BYTES = 209_854_627
REPORTED_MEMBER_CRC32 = 0x9AC87977
REPORTED_MEMBER_SHA256 = (
    "0c7b2bf1f162122d83d5bb39b901cc446e436b5643684ca268f934be145d8398"
)
REPORTED_SOURCE_ROWS = 776_298
REPORTED_PROJECTION_ROWS: int | None = 537_021
REPORTED_PROJECTION_BYTES: int | None = 112_916_008
REPORTED_PROJECTION_SHA256: str | None = (
    "a5ae968d7f2e9a81119bb1fb74d6268864cbf76dc25f79ad7db54114fc97eecb"
)
REPORTED_PROJECTION_COLUMNS = (
    "variant_id",
    "source_aa_seq_full",
    "reported_fitness_raw",
    "reported_fitness_95ci_width",
    "reported_fitness_ml",
    "reported_wt_name",
    "reported_wt_cluster",
)
LEGACY_REPORTED_PROJECTION_COLUMNS = (
    "variant_id",
    "reported_fitness_raw",
    "reported_fitness_95ci_width",
    "reported_fitness_ml",
    "reported_wt_name",
    "reported_wt_cluster",
)
IDENTITY_PROJECTION_COLUMNS = (
    "identity_key",
    "source_aa_seq_full",
    "reported_fitness_raw",
    "reported_fitness_95ci_width",
    "reported_fitness_ml",
    "reported_wt_name",
    "reported_wt_cluster",
)
K50_SOURCE_COMMIT = "597442a33f931a5e86003134c279b6f2a58ffd33"


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, expected_bytes: int, expected_md5: str) -> None:
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == expected_bytes
            and _hash(destination, "md5") == expected_md5
        ):
            print(f"source OK: {destination}")
            return
        raise ValueError(f"Existing source file does not match its manifest: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "dms-fitness-stability/0.1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size != expected_bytes or _hash(temporary, "md5") != expected_md5:
            raise ValueError(f"Downloaded source file failed validation: {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"retrieved: {destination}")


def _extract_member(archive: zipfile.ZipFile, basename: str, destination: Path) -> None:
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {basename!r} member, found {matches}")
    if destination.exists():
        if destination.is_file() and not destination.is_symlink():
            print(f"source present: {destination}")
            return
        raise ValueError(f"Expected a regular source file: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with archive.open(matches[0]) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"extracted: {destination}")


def _project_scramble(source: BinaryIO, destination: Path) -> None:
    source_digest = hashlib.sha256()

    def decoded_lines() -> Iterator[str]:
        for line in source:
            source_digest.update(line)
            yield line.decode("utf-8")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        reader = csv.DictReader(decoded_lines())
        if reader.fieldnames is None or SCRAMBLE_SEQUENCE_COLUMN not in reader.fieldnames:
            raise ValueError("K50 scramble source is missing its experimental sequence column.")
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow([SCRAMBLE_SEQUENCE_COLUMN])
            row_count = 0
            for row in reader:
                writer.writerow([row[SCRAMBLE_SEQUENCE_COLUMN]])
                row_count += 1
        if row_count != SCRAMBLE_ROWS:
            raise ValueError(f"Unexpected K50 scramble source row count: {row_count:,}.")
        if source_digest.hexdigest() != SCRAMBLE_MEMBER_SHA256:
            raise ValueError("K50 scramble source SHA-256 mismatch.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"projected sequence identity: {destination}")


def _scramble_columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as source:
        return next(csv.reader(source))


def _validate_projected_scramble(path: Path) -> None:
    if _scramble_columns(path) != [SCRAMBLE_SEQUENCE_COLUMN]:
        raise ValueError("Projected K50 scramble source has an unexpected schema.")
    if path.stat().st_size != SCRAMBLE_PROJECTION_BYTES:
        raise ValueError("Projected K50 scramble source has an unexpected byte size.")
    if _hash(path, "sha256") != SCRAMBLE_PROJECTION_SHA256:
        raise ValueError("Projected K50 scramble source SHA-256 mismatch.")
    with path.open("rb") as source:
        row_count = sum(1 for _ in source) - 1
    if row_count != SCRAMBLE_ROWS:
        raise ValueError("Projected K50 scramble source has an unexpected row count.")


def _extract_remote_scramble(url: str, destination: Path) -> None:
    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError("Install remotezip or run with `uv run --with remotezip`.") from error
    with RemoteZip(url) as archive:
        matches = [
            info for info in archive.infolist() if Path(info.filename).name == SCRAMBLE_MEMBER
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {SCRAMBLE_MEMBER!r} member.")
        member = matches[0]
        if member.file_size != SCRAMBLE_MEMBER_BYTES or member.CRC != SCRAMBLE_MEMBER_CRC32:
            raise ValueError("K50 scramble archive member identity mismatch.")
        with archive.open(member) as source:
            _project_scramble(source, destination)


def _identity_key(name: str, dna_seq: str) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dna_seq.encode("ascii"))
    return digest.digest()


def _project_reported_fitness(
    source: BinaryIO,
    destination: Path,
    raw_root: Path,
) -> str:
    source_digest = hashlib.sha256()

    def decoded_lines() -> Iterator[str]:
        for line in source:
            source_digest.update(line)
            yield line.decode("utf-8")

    identity_temporary = destination.with_suffix(destination.suffix + ".identity.part")
    temporary = destination.with_suffix(destination.suffix + ".part")
    identity_temporary.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        reader = csv.DictReader(decoded_lines())
        required = {
            "name",
            "dna_seq",
            "aa_seq_full",
            "deltaG",
            "deltaG_95CI",
            "dG_ML",
            "WT_name",
            "WT_cluster",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("K50 reported-fitness source has an unexpected schema.")
        identity_keys = set()
        with identity_temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=IDENTITY_PROJECTION_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            row_count = 0
            for row in reader:
                variant_id = row["aa_seq_full"]
                raw_fitness = float(row["deltaG"])
                sigma = float(row["deltaG_95CI"])
                if not variant_id or not math.isfinite(raw_fitness):
                    raise ValueError("K50 reported fitness contains an invalid variant row.")
                if not math.isfinite(sigma) or sigma < 0:
                    raise ValueError("K50 reported fitness contains an invalid interval width.")
                ml_value = row["dG_ML"]
                if ml_value not in {"-", "<-1", ">5"}:
                    parsed_ml_value = float(ml_value)
                    if not math.isfinite(parsed_ml_value):
                        raise ValueError("K50 dG_ML contains a non-finite value.")
                key = _identity_key(row["name"], row["dna_seq"])
                identity_keys.add(key)
                writer.writerow(
                    {
                        "identity_key": key.hex(),
                        "source_aa_seq_full": variant_id,
                        "reported_fitness_raw": row["deltaG"],
                        "reported_fitness_95ci_width": row["deltaG_95CI"],
                        "reported_fitness_ml": ml_value,
                        "reported_wt_name": row["WT_name"],
                        "reported_wt_cluster": row["WT_cluster"],
                    }
                )
                row_count += 1
        if row_count != REPORTED_SOURCE_ROWS:
            raise ValueError(f"Unexpected K50 reported-fitness row count: {row_count:,}.")
        digest = source_digest.hexdigest()
        if digest != REPORTED_MEMBER_SHA256:
            raise ValueError("K50 reported-fitness source SHA-256 mismatch.")

        identity_map: dict[bytes, str] = {}
        for library in range(1, 5):
            path = raw_root / f"NGS_count_lib{library}.csv"
            with path.open(encoding="utf-8", newline="") as raw_source:
                raw_reader = csv.DictReader(raw_source)
                if raw_reader.fieldnames is None or not {
                    "name",
                    "dna_seq",
                    "aa_seq",
                } <= set(raw_reader.fieldnames):
                    raise ValueError(f"Unexpected K50 raw identity schema in {path}.")
                for row in raw_reader:
                    key = _identity_key(row["name"], row["dna_seq"])
                    if key not in identity_keys:
                        continue
                    variant_id = sys.intern(row["aa_seq"])
                    previous = identity_map.setdefault(key, variant_id)
                    if previous != variant_id:
                        raise ValueError(
                            "One K50 construct identity maps to multiple fitted sequences."
                        )
        missing_keys = identity_keys - identity_map.keys()

        with identity_temporary.open(encoding="utf-8", newline="") as identity_source, (
            temporary.open("w", encoding="utf-8", newline="")
        ) as output:
            identity_reader = csv.DictReader(identity_source)
            writer = csv.DictWriter(
                output,
                fieldnames=REPORTED_PROJECTION_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            projected_rows = 0
            for row in identity_reader:
                key = bytes.fromhex(row.pop("identity_key"))
                if key not in identity_map:
                    continue
                row["variant_id"] = identity_map[key]
                writer.writerow(row)
                projected_rows += 1
        temporary.replace(destination)
    finally:
        identity_temporary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    print(f"projected reported fitness: {destination}")
    print(
        f"construct alignment: {projected_rows:,} rows retained; "
        f"{len(missing_keys):,} source constructs absent from job-63 inputs"
    )
    print(f"source member sha256:{digest}")
    return digest


def _validate_reported_projection(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular K50 reported-fitness file: {path}")
    if (
        REPORTED_PROJECTION_BYTES is not None
        and path.stat().st_size != REPORTED_PROJECTION_BYTES
    ):
        raise ValueError("Projected K50 reported fitness has an unexpected byte size.")
    if (
        REPORTED_PROJECTION_SHA256 is not None
        and _hash(path, "sha256") != REPORTED_PROJECTION_SHA256
    ):
        raise ValueError("Projected K50 reported fitness SHA-256 mismatch.")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if tuple(next(reader)) != REPORTED_PROJECTION_COLUMNS:
            raise ValueError("Projected K50 reported fitness has an unexpected schema.")
    with path.open("rb") as source:
        row_count = sum(1 for _ in source) - 1
    if REPORTED_PROJECTION_ROWS is not None and row_count != REPORTED_PROJECTION_ROWS:
        raise ValueError("Projected K50 reported fitness has an unexpected row count.")


def _reported_projection_needs_rebuild(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as source:
        columns = tuple(next(csv.reader(source)))
    if columns == LEGACY_REPORTED_PROJECTION_COLUMNS:
        return True
    if columns != REPORTED_PROJECTION_COLUMNS:
        raise ValueError("Existing K50 reported-fitness projection has an unknown schema.")
    return False


def _extract_remote_reported_fitness(
    url: str,
    destination: Path,
    raw_root: Path,
) -> None:
    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError("Install remotezip or run with `uv run --with remotezip`.") from error
    with RemoteZip(url) as archive:
        matches = [
            info for info in archive.infolist() if Path(info.filename).name == REPORTED_MEMBER
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {REPORTED_MEMBER!r} member.")
        member = matches[0]
        if (
            member.file_size != REPORTED_MEMBER_BYTES
            or member.compress_size != REPORTED_MEMBER_COMPRESSED_BYTES
            or member.CRC != REPORTED_MEMBER_CRC32
        ):
            raise ValueError("K50 reported-fitness archive member identity mismatch.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source:
            _project_reported_fitness(source, destination, raw_root)


def retrieve(data_root: Path) -> None:
    k50_root = data_root / "k50"
    raw_root = k50_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    raw_archive_path = k50_root / RAW_ARCHIVE
    _download(
        f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{RAW_ARCHIVE}/content",
        raw_archive_path,
        expected_bytes=RAW_ARCHIVE_BYTES,
        expected_md5=RAW_ARCHIVE_MD5,
    )
    with zipfile.ZipFile(raw_archive_path) as archive:
        for library in range(1, 5):
            filename = f"NGS_count_lib{library}.csv"
            _extract_member(archive, filename, raw_root / filename)

    qpcr_url = (
        "https://raw.githubusercontent.com/Rocklin-Lab/"
        f"cdna-display-proteolysis-pipeline/{K50_SOURCE_COMMIT}/"
        "Pipeline_qPCR_data/Raw_qPCR_data_FigS1.csv"
    )
    qpcr_path = raw_root / "Raw_qPCR_data_FigS1.csv"
    if not qpcr_path.exists():
        request = urllib.request.Request(
            qpcr_url,
            headers={"User-Agent": "dms-fitness-stability/0.1"},
        )
        temporary = qpcr_path.with_suffix(".csv.part")
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != 2_858:
                raise ValueError("Unexpected qPCR source byte size.")
            temporary.replace(qpcr_path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"retrieved: {qpcr_path}")
    elif not qpcr_path.is_file() or qpcr_path.is_symlink():
        raise ValueError(f"Expected a regular qPCR source file: {qpcr_path}")

    pipeline_url = (
        f"https://zenodo.org/api/records/{SCRAMBLE_ZENODO_RECORD}/files/"
        f"{PIPELINE_ARCHIVE}/content"
    )
    scramble_path = raw_root / SCRAMBLE_MEMBER
    if scramble_path.exists():
        if not scramble_path.is_file() or scramble_path.is_symlink():
            raise ValueError(f"Expected a regular scramble source file: {scramble_path}")
        columns = _scramble_columns(scramble_path)
        if columns == [SCRAMBLE_SEQUENCE_COLUMN]:
            _validate_projected_scramble(scramble_path)
            print(f"source OK: {scramble_path}")
        elif SCRAMBLE_SEQUENCE_COLUMN in columns:
            with scramble_path.open("rb") as source:
                _project_scramble(source, scramble_path)
            _validate_projected_scramble(scramble_path)
        else:
            raise ValueError("Existing K50 scramble source has an unexpected schema.")
    else:
        _extract_remote_scramble(pipeline_url, scramble_path)
        _validate_projected_scramble(scramble_path)
    print(
        f"remote parent: {PIPELINE_ARCHIVE} ({PIPELINE_ARCHIVE_BYTES} bytes; "
        f"md5:{PIPELINE_ARCHIVE_MD5})"
    )

    reported_path = raw_root / "reported_fitness" / REPORTED_MEMBER
    if reported_path.exists() and _reported_projection_needs_rebuild(reported_path):
        reported_path.unlink()
        print(f"rebuilding reported-fitness identity: {reported_path}")
    if reported_path.exists():
        _validate_reported_projection(reported_path)
        print(f"source present: {reported_path}")
    else:
        reported_url = (
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
            f"{REPORTED_ARCHIVE}/content"
        )
        _extract_remote_reported_fitness(reported_url, reported_path, raw_root)
        _validate_reported_projection(reported_path)
    print(
        f"remote parent: {REPORTED_ARCHIVE} ({REPORTED_ARCHIVE_BYTES} bytes; "
        f"md5:{REPORTED_ARCHIVE_MD5})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    retrieve(args.data_root)


if __name__ == "__main__":
    main()
