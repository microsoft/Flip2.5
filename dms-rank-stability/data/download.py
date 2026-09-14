"""Download and verify the prepared paper data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dms_rank_stability import LandscapeFeatures, SampledScores
from dms_rank_stability.runs import SAMPLING_METHODS

FAMILIES = {"pdz", "trpb", "nuc"}
FORMATS = {"tar.gz", "zip"}
CONTENTS = {"scores", "features"}
FILE_FIELDS = {
    "name",
    "url",
    "doi",
    "bytes",
    "sha256",
    "format",
    "content",
    "sampling",
    "families",
    "landscapes",
}
CITATION_FIELDS = {"family", "study", "doi", "data"}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strings(value: object, name: str, allowed: set[str]) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or item not in allowed for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a non-empty list of unique supported values.")
    return tuple(value)


@dataclass(frozen=True)
class DataFile:
    """One independently downloadable archive part."""

    name: str
    url: str | None
    doi: str | None
    bytes: int | None
    sha256: str | None
    format: str
    content: str
    sampling: tuple[str, ...]
    families: tuple[str, ...]
    landscapes: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.url is not None

    @classmethod
    def from_dict(cls, payload: object) -> DataFile:
        if not isinstance(payload, dict) or set(payload) != FILE_FIELDS:
            fields = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "Each data file must contain exactly the required fields; "
                f"missing={sorted(FILE_FIELDS - fields)}, "
                f"extra={sorted(fields - FILE_FIELDS)}."
            )
        name = payload["name"]
        if (
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).name != name
            or "\\" in name
        ):
            raise ValueError("Data file names must be non-empty basenames.")
        archive_format = payload["format"]
        if archive_format not in FORMATS:
            raise ValueError(f"Unsupported archive format: {archive_format!r}.")
        expected_suffix = ".tar.gz" if archive_format == "tar.gz" else ".zip"
        if not name.endswith(expected_suffix):
            raise ValueError(f"Data file {name!r} does not match format {archive_format!r}.")
        content = payload["content"]
        if content not in CONTENTS:
            raise ValueError(f"Unsupported archive content: {content!r}.")
        families = _strings(payload["families"], "families", FAMILIES)
        landscapes = payload["landscapes"]
        if (
            not isinstance(landscapes, list)
            or not landscapes
            or any(not isinstance(item, str) or not item for item in landscapes)
            or len(set(landscapes)) != len(landscapes)
        ):
            raise ValueError("landscapes must be a non-empty list of unique names.")
        sampling_payload = payload["sampling"]
        if content == "scores":
            sampling = _strings(
                sampling_payload,
                "sampling",
                SAMPLING_METHODS,
            )
            if len(sampling) != 1:
                raise ValueError("Each score archive must contain one sampling method.")
        else:
            if sampling_payload != []:
                raise ValueError("Feature archives must use an empty sampling list.")
            sampling = ()

        url = payload["url"]
        doi = payload["doi"]
        expected_bytes = payload["bytes"]
        expected_sha256 = payload["sha256"]
        complete = (
            isinstance(url, str)
            and bool(url)
            and (doi is None or isinstance(doi, str))
            and isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes > 0
            and isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256)
        )
        pending = all(
            value is None
            for value in (url, doi, expected_bytes, expected_sha256)
        )
        if not complete and not pending:
            raise ValueError(
                f"Data file {name!r} publication fields must be complete or all null."
            )
        return cls(
            name=name,
            url=url,
            doi=doi,
            bytes=expected_bytes,
            sha256=expected_sha256,
            format=archive_format,
            content=content,
            sampling=sampling,
            families=families,
            landscapes=tuple(landscapes),
        )


@dataclass(frozen=True)
class FileIndex:
    """Publication status and ordered archive parts."""

    status: str
    doi: str | None
    citations: tuple[dict[str, str], ...]
    files: tuple[DataFile, ...]


def load_file_index(path: str | Path) -> FileIndex:
    """Load the strict archive index through its single supported path."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "publication",
        "citations",
        "files",
    }:
        raise ValueError("File index must contain publication, citations, and files.")
    publication = payload["publication"]
    if not isinstance(publication, dict) or set(publication) != {"status", "doi"}:
        raise ValueError("publication must contain only status and doi.")
    status_value = publication["status"]
    if status_value not in {"staging", "published"}:
        raise ValueError("publication.status must be 'staging' or 'published'.")
    doi = publication["doi"]
    if doi is not None and (not isinstance(doi, str) or not doi.strip()):
        raise ValueError("publication.doi must be null or a non-empty string.")
    file_payload = payload["files"]
    if not isinstance(file_payload, list) or not file_payload:
        raise ValueError("File index must contain a non-empty files list.")
    files = tuple(DataFile.from_dict(item) for item in file_payload)
    names = [item.name for item in files]
    if len(set(names)) != len(names):
        raise ValueError("File index contains duplicate archive names.")
    if status_value == "published" and (doi is None or not all(item.available for item in files)):
        raise ValueError("Published data require a DOI and complete archive records.")
    citation_payload = payload["citations"]
    if not isinstance(citation_payload, list) or len(citation_payload) != len(FAMILIES):
        raise ValueError("File index must contain one citation for each family.")
    citations = []
    for citation in citation_payload:
        if (
            not isinstance(citation, dict)
            or set(citation) != CITATION_FIELDS
            or citation.get("family") not in FAMILIES
            or any(
                not isinstance(citation.get(name), str) or not citation[name].strip()
                for name in CITATION_FIELDS - {"family"}
            )
        ):
            raise ValueError(f"Malformed family citation: {citation!r}")
        citations.append({name: str(citation[name]) for name in CITATION_FIELDS})
    if {citation["family"] for citation in citations} != FAMILIES:
        raise ValueError("File index citations must cover PDZ, TrpB, and NucB exactly.")
    return FileIndex(
        status=status_value,
        doi=doi,
        citations=tuple(citations),
        files=files,
    )


def select_files(
    index: FileIndex,
    *,
    families: Iterable[str] = (),
    sampling_methods: Iterable[str] = (),
) -> tuple[DataFile, ...]:
    """Select score parts and their shared feature parts."""
    requested_families = set(families)
    requested_sampling = set(sampling_methods)
    unknown_families = requested_families - FAMILIES
    unknown_sampling = requested_sampling - SAMPLING_METHODS
    if unknown_families:
        raise ValueError(f"Unknown families: {sorted(unknown_families)}")
    if unknown_sampling:
        raise ValueError(f"Unknown sampling methods: {sorted(unknown_sampling)}")
    selected = tuple(
        item
        for item in index.files
        if (not requested_families or requested_families.intersection(item.families))
        and (
            item.content == "features"
            or not requested_sampling
            or requested_sampling.intersection(item.sampling)
        )
    )
    if not selected:
        raise ValueError("No data files match the requested filters.")
    return selected


def _safe_member(name: str, expected_prefix: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"Archive contains an unsafe path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Archive contains an unsafe path: {name!r}")
    if path.parts[0] != expected_prefix:
        raise ValueError(
            f"Archive member {name!r} must begin with {expected_prefix!r}."
        )
    return path


def _extract_archive(archive_path: Path, item: DataFile, destination: Path) -> None:
    expected_prefix = item.content
    if item.format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _safe_member(member.filename, expected_prefix)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Archive contains a symbolic link: {member.filename!r}")
            archive.extractall(destination)
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_member(member.name, expected_prefix)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive contains a non-file member: {member.name!r}")
        archive.extractall(destination, filter="data")


def _leaf_bundles(root: Path, item: DataFile) -> tuple[Path, ...]:
    content = item.content
    expected = (
        {"scores.nc", "variants.csv"}
        if content == "scores"
        else {"features.nc", "variants.csv"}
    )
    content_root = root / content
    if not content_root.is_dir():
        raise ValueError(f"Archive does not contain a {content}/ directory.")
    bundles = tuple(
        path
        for path in sorted(content_root.rglob("*"))
        if path.is_dir() and {item.name for item in path.iterdir() if item.is_file()}
    )
    if not bundles:
        raise ValueError(f"Archive contains no {content} bundles.")
    for bundle in bundles:
        names = {item.name for item in bundle.iterdir() if item.is_file()}
        if names != expected or any(item.is_dir() for item in bundle.iterdir()):
            raise ValueError(f"Archive bundle has unexpected contents: {bundle}")
        relative = bundle.relative_to(content_root)
        if content == "scores":
            if len(relative.parts) != 2 or relative.parts[0] not in item.sampling:
                raise ValueError(f"Score bundle is outside the declared sampling scope: {bundle}")
            sampling_slug = relative.parts[0].replace("_", "-")
            if not any(
                relative.parts[1].startswith(f"{landscape}-{sampling_slug}-")
                for landscape in item.landscapes
            ):
                raise ValueError(f"Score bundle is outside the declared landscape scope: {bundle}")
        elif len(relative.parts) != 1 or relative.parts[0] not in item.landscapes:
            raise ValueError(f"Feature bundle is outside the declared landscape scope: {bundle}")
        if content == "scores":
            loaded = SampledScores.load(bundle)
        else:
            loaded = LandscapeFeatures.load(bundle)
        loaded.close()
    return bundles


def _same_files(first: Path, second: Path) -> bool:
    first_files = {item.name: item for item in first.iterdir() if item.is_file()}
    second_files = {item.name: item for item in second.iterdir() if item.is_file()}
    if set(first_files) != set(second_files):
        return False
    return all(
        first_files[name].stat().st_size == second_files[name].stat().st_size
        and sha256_file(first_files[name]) == sha256_file(second_files[name])
        for name in first_files
    )


def _publish_bundles(extracted: Path, destination: Path, item: DataFile) -> int:
    bundles = _leaf_bundles(extracted, item)
    planned = []
    for source in bundles:
        relative = source.relative_to(extracted)
        target = destination / relative
        if target.exists() and not _same_files(source, target):
            raise FileExistsError(f"Existing data differ from archive contents: {target}")
        planned.append((source, target))
    published = 0
    for source, target in planned:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        published += 1
    return published


def _cached_archive(item: DataFile, cache_dir: Path) -> Path:
    if not item.available or item.bytes is None or item.sha256 is None:
        raise ValueError(f"Data file has not been published: {item.name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / item.name
    if (
        cached.is_file()
        and cached.stat().st_size == item.bytes
        and sha256_file(cached) == item.sha256
    ):
        return cached
    temporary = cached.with_suffix(cached.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        item.url,
        headers={"User-Agent": "dms-rank-stability-data"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size != item.bytes or sha256_file(temporary) != item.sha256:
            raise ValueError(f"Downloaded data failed verification: {item.name}")
        temporary.replace(cached)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return cached


def retrieve_files(
    files: Sequence[DataFile],
    *,
    destination: str | Path,
    cache_dir: str | Path,
) -> dict[str, int]:
    """Download, verify, safely extract, and incrementally publish archive parts."""
    destination = Path(destination).resolve()
    cache_dir = Path(cache_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"archives": 0, "bundles": 0}
    for position, item in enumerate(files, start=1):
        print(f"[{position}/{len(files)}] {item.name}", flush=True)
        archive_path = _cached_archive(item, cache_dir)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{item.name}-", dir=destination.parent)
        )
        try:
            _extract_archive(archive_path, item, temporary)
            counts["bundles"] += _publish_bundles(temporary, destination, item)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        counts["archives"] += 1
    return counts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--family", nargs="+", choices=sorted(FAMILIES), default=())
    parser.add_argument(
        "--sampling",
        nargs="+",
        choices=sorted(SAMPLING_METHODS),
        default=(),
    )
    parser.add_argument("--index", type=Path, default=root / "data/files.json")
    parser.add_argument("--destination", type=Path, default=root / "data/downloads")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public data downloader."""
    args = _parse_args(argv)
    if not args.paper:
        raise ValueError("Pass --paper to select the prepared paper data.")
    index = load_file_index(args.index)
    selected = select_files(
        index,
        families=args.family,
        sampling_methods=args.sampling,
    )
    for item in selected:
        state = "available" if item.available else "awaiting publication"
        print(f"{state}: {item.name}")
    if args.list:
        return
    unavailable = [item.name for item in selected if not item.available]
    if unavailable:
        raise ValueError(
            "Selected data are not yet published: " + ", ".join(unavailable)
        )
    cache_dir = args.cache_dir or args.destination / ".cache"
    counts = retrieve_files(
        selected,
        destination=args.destination,
        cache_dir=cache_dir,
    )
    print(
        f"Verified {counts['archives']} archive(s); "
        f"published {counts['bundles']} new bundle(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
