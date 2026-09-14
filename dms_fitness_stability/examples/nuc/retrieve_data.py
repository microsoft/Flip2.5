"""Retrieve the compact NucB count and reported-activity source files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
BUCKET = "nuclease_design"


@dataclass(frozen=True)
class SourceObject:
    object_name: str
    generation: str
    bytes: int
    md5_base64: str
    relative_path: Path


SOURCE_OBJECTS = (
    SourceObject(
        "processed_data/g1.csv",
        "1708601525023646",
        2_437_712,
        "hmRoCGJhi4g/txcPslw64Q==",
        Path("raw/counts/g1.csv"),
    ),
    SourceObject(
        "processed_data/g2.csv",
        "1708601525323079",
        3_163_331,
        "0JkQwUKu4geRbKOhDAQ+zQ==",
        Path("raw/counts/g2.csv"),
    ),
    SourceObject(
        "processed_data/g3.csv",
        "1708601525400044",
        4_788_940,
        "Eix9XwUrmdLxTZZoYVs6WQ==",
        Path("raw/counts/g3.csv"),
    ),
    SourceObject(
        "processed_data/g4.csv",
        "1708601525342547",
        5_897_118,
        "AIyITT68b9GOXeim29Px8g==",
        Path("raw/counts/g4.csv"),
    ),
    SourceObject(
        "processed_fiducial_data/g1_neg_control.csv",
        "1708601525169780",
        268_885,
        "4jVXcM7HTs149FM4yS8ILg==",
        Path("raw/counts/g1_neg_control.csv"),
    ),
    SourceObject(
        "processed_fiducial_data/g2_neg_control.csv",
        "1708601525250145",
        26_482,
        "QotdvpRPXDryoVzaOYrLDQ==",
        Path("raw/counts/g2_neg_control.csv"),
    ),
    SourceObject(
        "processed_fiducial_data/g3_neg_control.csv",
        "1708601525724852",
        24_529,
        "WFX3R07CrqL3bxESK10wUQ==",
        Path("raw/counts/g3_neg_control.csv"),
    ),
    SourceObject(
        "processed_fiducial_data/g4_neg_control.csv",
        "1708601526130403",
        34_708,
        "Ms29Gs/QRq5xUcfh4kzQ5w==",
        Path("raw/counts/g4_neg_control.csv"),
    ),
    SourceObject(
        "processed_data/landscape.csv",
        "1708601526510660",
        17_326_572,
        "+GOap1hprBSJaIMJvXztZw==",
        Path("raw/reported_fitness/landscape.csv"),
    ),
)


def _md5(path: Path) -> bytes:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _validate(path: Path, source: SourceObject) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular source file: {path}")
    if path.stat().st_size != source.bytes:
        raise ValueError(f"Unexpected byte size for {path}")
    if _md5(path) != base64.b64decode(source.md5_base64):
        raise ValueError(f"GCS MD5 mismatch for {path}")


def _source_url(source: SourceObject) -> str:
    object_name = urllib.parse.quote(source.object_name, safe="")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/"
        f"{object_name}?generation={source.generation}&alt=media"
    )


def retrieve(data_root: Path) -> None:
    nuc_root = data_root / "nuc"
    for source in SOURCE_OBJECTS:
        destination = nuc_root / source.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _validate(destination, source)
            print(f"source OK: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(
            _source_url(source),
            headers={"User-Agent": "dms-fitness-stability/0.1"},
        )
        try:
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            _validate(temporary, source)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"retrieved: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    retrieve(args.data_root)


if __name__ == "__main__":
    main()
