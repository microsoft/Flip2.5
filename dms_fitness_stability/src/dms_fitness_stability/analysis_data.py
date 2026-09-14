"""Manifest generation and validation for released analysis-data bundles."""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import h5py

from .landscape_features import LandscapeFeatureBundle, landscape_feature_id
from .migration_audit import sha256_file
from .score_ensembles import ScoreEnsemble

ANALYSIS_DATA_MANIFEST_SCHEMA_VERSION = 1


def _file_record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def score_bundle_record(
    bundle_dir: str | Path,
    *,
    data_root: str | Path,
    sample_view: str,
) -> dict[str, object]:
    """Build one checksummed score-bundle manifest record."""
    bundle_dir = Path(bundle_dir).resolve()
    data_root = Path(data_root).resolve()
    if not bundle_dir.is_relative_to(data_root):
        raise ValueError(f"Score bundle lies outside the data root: {bundle_dir}")
    ensemble = ScoreEnsemble.load(bundle_dir)
    try:
        metadata = ensemble.metadata
        score_names = ensemble.scores["score_name"].values.astype(str).tolist()
        if metadata.get("inference_method") == "hmc" and metadata.get(
            "sample_view"
        ) != sample_view:
            raise ValueError(f"Score bundle sample view mismatch: {bundle_dir}")
        feature_ids = {
            score_name: landscape_feature_id(
                str(metadata["protein_family"]),
                str(metadata["landscape"]),
                score_name=(
                    score_name
                    if metadata["protein_family"] == "k50"
                    else None
                ),
            )
            for score_name in score_names
        }
        data_filename = (
            "fitness_scores.nc"
            if ensemble.source_schema_version == 2
            else "fitness_proxies.nc"
        )
        variable_name = (
            "fitness_score"
            if ensemble.source_schema_version == 2
            else "fitness_proxy"
        )
        with h5py.File(bundle_dir / data_filename, "r") as handle:
            variable = handle[variable_name]
            storage = {
                "filename": data_filename,
                "variable": variable_name,
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "chunks": None if variable.chunks is None else list(variable.chunks),
                "compression": variable.compression,
            }
    finally:
        ensemble.close()
    files = sorted(
        (_file_record(path) for path in bundle_dir.iterdir() if path.is_file()),
        key=lambda record: str(record["name"]),
    )
    return {
        "analysis_id": metadata["analysis_id"],
        "protein_family": metadata["protein_family"],
        "landscape": metadata["landscape"],
        "inference_method": metadata["inference_method"],
        "sample_view": sample_view,
        "schema_version": ensemble.source_schema_version,
        "relative_path": str(bundle_dir.relative_to(data_root)),
        "scores": score_names,
        "score_contracts": metadata["score_contracts"],
        "feature_ids": feature_ids,
        "storage": storage,
        "files": files,
    }


def feature_bundle_record(
    bundle_dir: str | Path,
    *,
    data_root: str | Path,
) -> dict[str, object]:
    """Build one checksummed landscape-feature manifest record."""
    bundle_dir = Path(bundle_dir).resolve()
    data_root = Path(data_root).resolve()
    if not bundle_dir.is_relative_to(data_root):
        raise ValueError(f"Feature bundle lies outside the data root: {bundle_dir}")
    bundle = LandscapeFeatureBundle.load(bundle_dir)
    try:
        metadata = bundle.metadata
        dtypes = {
            name: str(bundle.features[name].dtype)
            for name in bundle.features.data_vars
        }
    finally:
        bundle.close()
    files = sorted(
        (_file_record(path) for path in bundle_dir.iterdir() if path.is_file()),
        key=lambda record: str(record["name"]),
    )
    return {
        "feature_id": metadata["feature_id"],
        "protein_family": metadata["protein_family"],
        "landscape": metadata["landscape"],
        "score_name": metadata["score_name"],
        "schema_version": metadata["schema_version"],
        "relative_path": str(bundle_dir.relative_to(data_root)),
        "n_variants": metadata["n_variants"],
        "derivation_contract": metadata["derivation_contract"],
        "dtypes": dtypes,
        "source_files": metadata["source_files"],
        "files": files,
    }


def build_analysis_data_manifest(
    *,
    data_root: str | Path,
    direct_hmc_root: str | Path,
    mle_score_root: str | Path,
    hmc_count_score_root: str | Path,
    feature_root: str | Path,
    direct_hmc_views: tuple[str, ...] = ("release_1000",),
) -> dict[str, object]:
    """Build a deterministic local inventory for all analysis-data methods."""
    data_root = Path(data_root).resolve()
    if not direct_hmc_views or len(set(direct_hmc_views)) != len(direct_hmc_views):
        raise ValueError("direct_hmc_views must be non-empty and unique.")
    roots = (
        *((Path(direct_hmc_root) / view, view) for view in direct_hmc_views),
        (Path(mle_score_root), "canonical"),
        (Path(hmc_count_score_root), "canonical"),
    )
    score_bundles = []
    for root, sample_view in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Score bundle root is missing: {root}")
        score_bundles.extend(
            score_bundle_record(
                path,
                data_root=data_root,
                sample_view=sample_view,
            )
            for path in sorted(root.iterdir())
            if path.is_dir()
        )
    features = [
        feature_bundle_record(path, data_root=data_root)
        for path in sorted(Path(feature_root).iterdir())
        if path.is_dir()
    ]
    feature_ids = {record["feature_id"] for record in features}
    missing_feature_ids = sorted(
        {
            feature_id
            for record in score_bundles
            for feature_id in record["feature_ids"].values()
        }
        - feature_ids
    )
    if missing_feature_ids:
        raise ValueError(
            f"Score bundles reference missing feature IDs: {missing_feature_ids}"
        )
    method_counts = {}
    for record in score_bundles:
        method = str(record["inference_method"])
        method_counts[method] = method_counts.get(method, 0) + 1
    return {
        "schema_version": ANALYSIS_DATA_MANIFEST_SCHEMA_VERSION,
        "publication": {
            "status": "local_staging",
            "doi": None,
            "archive_base_url": None,
        },
        "archives": [],
        "canonical_direct_hmc_view": "full",
        "available_direct_hmc_views": list(direct_hmc_views),
        "method_counts": method_counts,
        "score_bundles": score_bundles,
        "feature_bundles": features,
    }


def _validate_file_records(
    bundle_dir: Path,
    files: object,
    *,
    verify_checksums: bool,
) -> None:
    if not isinstance(files, list) or not files:
        raise ValueError(f"Bundle has no file records: {bundle_dir}")
    expected_names = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError(f"Malformed file record at {bundle_dir}.")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Malformed file name at {bundle_dir}.")
        if name in expected_names:
            raise ValueError(f"Duplicate file record {name!r} at {bundle_dir}.")
        expected_names.add(name)
        path = bundle_dir / name
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Bundle file is missing or changed: {path}")
        if verify_checksums and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Bundle file checksum changed: {path}")
    observed_names = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ValueError(
            f"Bundle file inventory changed at {bundle_dir}: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}."
        )


def validate_analysis_data_manifest(
    manifest: object,
    *,
    data_root: str | Path,
    verify_checksums: bool = True,
) -> dict[str, int]:
    """Validate every local bundle and relationship in an analysis manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Analysis data manifest schema_version must be 1.")
    data_root = Path(data_root).resolve()
    score_records = manifest.get("score_bundles")
    feature_records = manifest.get("feature_bundles")
    if not isinstance(score_records, list) or not isinstance(feature_records, list):
        raise ValueError("Analysis data manifest bundle records must be lists.")
    feature_ids = []
    for record in feature_records:
        if not isinstance(record, dict):
            raise ValueError("Malformed feature bundle record.")
        feature_id = str(record["feature_id"])
        feature_ids.append(feature_id)
        bundle_dir = data_root / str(record["relative_path"])
        _validate_file_records(
            bundle_dir,
            record["files"],
            verify_checksums=verify_checksums,
        )
        bundle = LandscapeFeatureBundle.load(bundle_dir)
        try:
            if bundle.metadata["feature_id"] != feature_id:
                raise ValueError(f"Feature bundle identity changed: {bundle_dir}")
        finally:
            bundle.close()
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Analysis data manifest contains duplicate feature IDs.")

    score_keys = []
    referenced_features = set()
    for record in score_records:
        if not isinstance(record, dict):
            raise ValueError("Malformed score bundle record.")
        key = (record["inference_method"], record["sample_view"], record["analysis_id"])
        score_keys.append(key)
        bundle_dir = data_root / str(record["relative_path"])
        _validate_file_records(
            bundle_dir,
            record["files"],
            verify_checksums=verify_checksums,
        )
        ensemble = ScoreEnsemble.load(bundle_dir)
        try:
            if ensemble.metadata["analysis_id"] != record["analysis_id"]:
                raise ValueError(f"Score bundle identity changed: {bundle_dir}")
            if list(ensemble.fitness_score.shape) != record["storage"]["shape"]:
                raise ValueError(f"Score bundle shape changed: {bundle_dir}")
        finally:
            ensemble.close()
        referenced_features.update(record["feature_ids"].values())
    if len(set(score_keys)) != len(score_keys):
        raise ValueError("Analysis data manifest contains duplicate score bundle keys.")
    missing_features = referenced_features - set(feature_ids)
    if missing_features:
        raise ValueError(
            f"Analysis data manifest references missing features: {sorted(missing_features)}"
        )
    return {
        "score_bundles": len(score_records),
        "feature_bundles": len(feature_records),
    }


def load_and_validate_analysis_data_manifest(
    path: str | Path,
    *,
    data_root: str | Path,
    verify_checksums: bool = True,
) -> dict[str, int]:
    """Load and validate a persisted analysis-data manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_analysis_data_manifest(
        payload,
        data_root=data_root,
        verify_checksums=verify_checksums,
    )


def _safe_archive_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Archive contains an unsafe path: {name!r}")


def _extract_archive(path: Path, *, archive_format: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                _safe_archive_member(member)
            archive.extractall(destination)
        return
    if archive_format == "tar.gz":
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                _safe_archive_member(member.name)
            archive.extractall(destination, filter="data")
        return
    raise ValueError(f"Unsupported analysis archive format: {archive_format!r}.")


def retrieve_analysis_data_archives(
    manifest: object,
    *,
    destination: str | Path,
    cache_dir: str | Path,
    archive_names: set[str] | None = None,
) -> list[Path]:
    """Download, verify, and safely extract published analysis archives."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Analysis data manifest schema_version must be 1.")
    publication = manifest.get("publication")
    archives = manifest.get("archives")
    if not isinstance(publication, dict) or publication.get("status") != "published":
        raise ValueError("Analysis data retrieval requires publication status 'published'.")
    if not isinstance(archives, list) or not archives:
        raise ValueError("Published analysis data manifest must contain archives.")
    selected = [
        archive
        for archive in archives
        if archive_names is None or archive.get("name") in archive_names
    ]
    if not selected:
        raise ValueError("No analysis archives match the requested names.")
    observed_names = {str(archive.get("name")) for archive in selected}
    if archive_names is not None and observed_names != archive_names:
        raise ValueError(
            f"Unknown requested archives: {sorted(archive_names - observed_names)}"
        )

    destination = Path(destination)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for archive in selected:
        if not isinstance(archive, dict):
            raise ValueError("Malformed analysis archive record.")
        name = archive.get("name")
        url = archive.get("url")
        expected_bytes = archive.get("bytes")
        expected_sha256 = archive.get("sha256")
        archive_format = archive.get("format")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(url, str)
            or not url
            or not isinstance(expected_bytes, int)
            or expected_bytes < 1
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or archive_format not in {"zip", "tar.gz"}
        ):
            raise ValueError(f"Malformed analysis archive record: {archive}")
        cached = cache_dir / name
        if not (
            cached.is_file()
            and cached.stat().st_size == expected_bytes
            and sha256_file(cached) == expected_sha256
        ):
            temporary = cached.with_suffix(cached.suffix + ".tmp")
            temporary.unlink(missing_ok=True)
            with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if (
                temporary.stat().st_size != expected_bytes
                or sha256_file(temporary) != expected_sha256
            ):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Downloaded analysis archive failed validation: {name}")
            temporary.replace(cached)
        _extract_archive(
            cached,
            archive_format=str(archive_format),
            destination=destination,
        )
        extracted.append(cached)
    return extracted
