"""Deterministic semantic hashes for matrix-migration baselines."""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .analysis_manifest import AnalysisSpec, load_analysis_manifest
from .results import _read_variants_csv
from .score_ensembles import ScoreEnsemble

MIGRATION_AUDIT_SCHEMA_VERSION = 1
RESULT_FILENAMES = (
    "metadata.json",
    "variants.csv",
    "stability.nc",
    "summary.csv",
    "overlaps.csv",
)
HMC_TABLE_FILENAMES = (
    "functionality_scores.csv.gz",
    "inventory.csv",
    "mechanism_diagnostics.csv",
    "mechanism_effects.csv",
    "mechanism_inventory.csv",
    "mechanism_runs.csv",
    "mechanism_variants.csv",
    "model_rank_intervals.csv",
    "rank_agreement.csv",
    "rank_percentiles.csv",
    "reference_category_sensitivity.csv",
    "reference_diagnostics.csv",
    "top_k_candidates.csv",
    "top_k_curves.csv",
)


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        allow_nan=True,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalized_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "S", "U"}:
        return _canonical_json(array.tolist())
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array).tobytes(order="C")


def _array_chunks(array: xr.DataArray, *, sample_chunk_size: int) -> Iterator[bytes]:
    if "sample" not in array.dims:
        yield _normalized_bytes(array.values)
        return
    sample_axis = array.dims.index("sample")
    prefix_dims = array.dims[:sample_axis]
    prefix_ranges = [range(array.sizes[name]) for name in prefix_dims]
    for prefix in itertools.product(*prefix_ranges):
        prefix_indexers = dict(zip(prefix_dims, prefix, strict=True))
        for start in range(0, array.sizes["sample"], sample_chunk_size):
            indexers = {
                **prefix_indexers,
                "sample": slice(
                    start,
                    min(start + sample_chunk_size, array.sizes["sample"]),
                ),
            }
            yield _normalized_bytes(array.isel(indexers).values)


def data_array_record(
    array: xr.DataArray,
    *,
    sample_chunk_size: int = 256,
) -> dict[str, object]:
    """Describe and hash one labeled array in deterministic logical order."""
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive.")
    header = {
        "dims": list(array.dims),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    digest = hashlib.sha256(_canonical_json(header))
    for block in _array_chunks(array, sample_chunk_size=sample_chunk_size):
        digest.update(block)
    return {**header, "sha256": digest.hexdigest()}


def _table_record(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "sha256": sha256_file(path),
    }


def csv_content_record(path: str | Path) -> dict[str, object]:
    """Hash logical CSV text, ignoring gzip container metadata."""
    csv_path = Path(path)
    binary_open = gzip.open if csv_path.suffix == ".gz" else Path.open
    digest = hashlib.sha256()
    with binary_open(csv_path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)

    text_open = gzip.open if csv_path.suffix == ".gz" else Path.open
    with text_open(csv_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV table is empty: {csv_path}") from error
        rows = sum(1 for _ in reader)
    return {
        "rows": rows,
        "columns": columns,
        "bytes": csv_path.stat().st_size,
        "content_sha256": digest.hexdigest(),
    }


def audit_dataset(
    path: str | Path,
    *,
    sample_chunk_size: int = 256,
) -> dict[str, object]:
    """Audit the labeled content of one NetCDF dataset."""
    dataset_path = Path(path)
    with xr.open_dataset(dataset_path, engine="h5netcdf") as dataset:
        return {
            "bytes": dataset_path.stat().st_size,
            "sizes": {name: int(size) for name, size in dataset.sizes.items()},
            "attrs": _jsonable(dict(dataset.attrs)),
            "attrs_sha256": hashlib.sha256(
                _canonical_json(dict(dataset.attrs))
            ).hexdigest(),
            "coordinates": {
                name: data_array_record(value, sample_chunk_size=sample_chunk_size)
                for name, value in dataset.coords.items()
            },
            "variables": {
                name: data_array_record(value, sample_chunk_size=sample_chunk_size)
                for name, value in dataset.data_vars.items()
            },
        }


def audit_result_bundle(
    path: str | Path,
    *,
    sample_chunk_size: int = 256,
) -> dict[str, object]:
    """Audit one persisted five-file ``StabilityResult`` without opening a fit."""
    result_dir = Path(path)
    missing = [
        name for name in RESULT_FILENAMES if not (result_dir / name).is_file()
    ]
    if missing:
        raise ValueError(f"Result directory is missing files: {missing}")

    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    analysis_id = metadata.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError(f"Result metadata has no analysis_id: {result_dir}")
    if result_dir.name != analysis_id:
        raise ValueError(
            f"Result directory name does not match analysis_id: "
            f"{result_dir.name!r} != {analysis_id!r}."
        )

    variants_path = result_dir / "variants.csv"
    variants = _read_variants_csv(variants_path)
    if "variant_id" not in variants:
        raise ValueError(f"Result variants lack variant_id: {result_dir}")
    if variants["variant_id"].duplicated().any():
        raise ValueError(f"Result variant IDs are not unique: {result_dir}")

    summary_path = result_dir / "summary.csv"
    overlaps_path = result_dir / "overlaps.csv"
    summary = pd.read_csv(summary_path, low_memory=False)
    overlaps = pd.read_csv(overlaps_path, low_memory=False)
    selection_record = audit_dataset(
        result_dir / "stability.nc",
        sample_chunk_size=sample_chunk_size,
    )
    if selection_record["attrs"].get("analysis_id") != analysis_id:
        raise ValueError(f"Selection identity mismatch at {result_dir}.")
    coordinates = selection_record["coordinates"]
    variables = selection_record["variables"]

    integer_selection_sha256 = {
        name: record["sha256"]
        for name, record in variables.items()
        if np.dtype(str(record["dtype"])).kind in {"i", "u"}
    }
    return {
        "analysis_id": analysis_id,
        "metadata_sha256": hashlib.sha256(_canonical_json(metadata)).hexdigest(),
        "run_settings": {
            name: metadata.get(name)
            for name in (
                "n_samples",
                "retained_k",
                "tracked_reported_top_n",
                "tracked_model_top_n",
                "top_k",
                "seed",
                "interval",
            )
        },
        "variants": {
            **_table_record(variants_path, variants),
            "variant_id_sha256": hashlib.sha256(
                _canonical_json(variants["variant_id"].tolist())
            ).hexdigest(),
            "semantic_sha256": hashlib.sha256(
                _canonical_json(variants.to_dict(orient="list"))
            ).hexdigest(),
        },
        "summary": _table_record(summary_path, summary),
        "overlaps": _table_record(overlaps_path, overlaps),
        "selection": selection_record,
        "sample_id_sha256": coordinates.get("sample_id", {}).get("sha256"),
        "integer_selection_sha256": integer_selection_sha256,
    }


def _frame_semantic_record(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "sha256": hashlib.sha256(
            _canonical_json(frame.to_dict(orient="list"))
        ).hexdigest(),
    }


def audit_score_ensemble_bundle(
    path: str | Path,
    *,
    sample_chunk_size: int = 256,
) -> dict[str, object]:
    """Audit schema-neutral score content and schema-specific source files."""
    bundle_dir = Path(path)
    ensemble = ScoreEnsemble.load(bundle_dir)
    try:
        score_names = ensemble.scores["score_name"].values.astype(str).tolist()
        source_sample_id = ensemble.scores.coords.get(
            "source_sample_id",
            ensemble.scores["sample"],
        )
        semantic = {
            "analysis_id": ensemble.metadata["analysis_id"],
            "uncertainty_target": ensemble.metadata["uncertainty_target"],
            "score_contracts": ensemble.metadata["score_contracts"],
            "scores": score_names,
            "variants": _frame_semantic_record(ensemble.variants),
            "analysis_variants": _frame_semantic_record(
                ensemble.analysis_variants
            ),
            "source_sample_id": data_array_record(
                source_sample_id,
                sample_chunk_size=sample_chunk_size,
            ),
            "source_chain": (
                data_array_record(
                    ensemble.scores["source_chain"],
                    sample_chunk_size=sample_chunk_size,
                )
                if "source_chain" in ensemble.scores.coords
                else None
            ),
            "source_draw": (
                data_array_record(
                    ensemble.scores["source_draw"],
                    sample_chunk_size=sample_chunk_size,
                )
                if "source_draw" in ensemble.scores.coords
                else None
            ),
            "fitness_scores": {
                score_name: data_array_record(
                    ensemble.fitness_score.sel(
                        score_name=score_name,
                        drop=True,
                    ),
                    sample_chunk_size=sample_chunk_size,
                )
                for score_name in score_names
            },
            "observed_scores": (
                {
                    score_name: data_array_record(
                        ensemble.observed_score.sel(
                            score_name=score_name,
                            drop=True,
                        ),
                        sample_chunk_size=sample_chunk_size,
                    )
                    for score_name in score_names
                }
                if ensemble.observed_score is not None
                else None
            ),
        }
        files = {
            file_path.name: {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
            for file_path in sorted(bundle_dir.iterdir())
            if file_path.is_file()
        }
        return {
            "analysis_id": ensemble.metadata["analysis_id"],
            "source_schema_version": ensemble.source_schema_version,
            "metadata_sha256": hashlib.sha256(
                _canonical_json(ensemble.metadata)
            ).hexdigest(),
            "semantic": semantic,
            "semantic_sha256": hashlib.sha256(
                _canonical_json(semantic)
            ).hexdigest(),
            "files": files,
        }
    finally:
        ensemble.close()


def _source_score_record(
    spec: AnalysisSpec,
    fit_audit_by_job: dict[int, dict[str, object]],
) -> dict[str, object]:
    record = fit_audit_by_job.get(spec.job_id)
    if record is None:
        raise ValueError(f"Job {spec.job_id} is absent from the fit audit.")
    artifacts = record.get(spec.artifact_kind)
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError(
            f"Job {spec.job_id} must have one {spec.artifact_kind} artifact."
        )
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or artifact.get("status") != "validated":
        raise ValueError(f"Job {spec.job_id} source artifact is not validated.")
    score_shape = artifact.get("score_shape")
    if (
        not isinstance(score_shape, list)
        or len(score_shape) != 3
        or any(not isinstance(size, int) or size < 1 for size in score_shape)
    ):
        raise ValueError(f"Job {spec.job_id} has no valid source score shape.")
    return {
        "artifact_kind": spec.artifact_kind,
        "path": artifact.get("path"),
        "bytes": artifact.get("bytes"),
        "score_variable": artifact.get("score_variable"),
        "score_shape": score_shape,
    }


def _require_exact_entries(
    root: Path,
    *,
    expected: set[str],
    directories: bool,
) -> None:
    observed = {
        path.name
        for path in root.iterdir()
        if path.is_dir() == directories and (directories or path.is_file())
    }
    if observed != expected:
        raise ValueError(
            f"Baseline inventory mismatch at {root}: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


def build_hmc_migration_audit(
    *,
    manifest_path: str | Path,
    fit_audit_path: str | Path,
    result_root: str | Path,
    cache_root: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    code_paths: Mapping[str, str | Path] | None = None,
    repository_commit: str | None = None,
    sample_chunk_size: int = 256,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Build the complete fit-driven direct-HMC migration baseline."""
    manifest_path = Path(manifest_path)
    fit_audit_path = Path(fit_audit_path)
    result_root = Path(result_root)
    cache_root = Path(cache_root)
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    specs = load_analysis_manifest(manifest_path)
    if {spec.identity.inference_method for spec in specs} != {"hmc"}:
        raise ValueError("The migration baseline requires an HMC-only manifest.")

    fit_records = json.loads(fit_audit_path.read_text(encoding="utf-8"))
    if not isinstance(fit_records, list):
        raise ValueError("Fit audit must be a JSON list.")
    fit_audit_by_job = {int(record["job_id"]): record for record in fit_records}
    if len(fit_audit_by_job) != len(fit_records):
        raise ValueError("Fit audit contains duplicate job IDs.")

    expected_ids = {spec.analysis_id for spec in specs}
    _require_exact_entries(result_root, expected=expected_ids, directories=True)
    _require_exact_entries(
        cache_root,
        expected={f"{analysis_id}.nc" for analysis_id in expected_ids},
        directories=False,
    )
    _require_exact_entries(
        table_dir,
        expected=set(HMC_TABLE_FILENAMES),
        directories=False,
    )

    figure_paths = sorted(
        path for path in figure_dir.iterdir() if path.suffix in {".pdf", ".png"}
    )
    if not figure_paths:
        raise ValueError(f"No baseline figures found at {figure_dir}.")
    png_stems = {path.stem for path in figure_paths if path.suffix == ".png"}
    pdf_stems = {path.stem for path in figure_paths if path.suffix == ".pdf"}
    if png_stems != pdf_stems:
        raise ValueError(
            "Baseline PNG/PDF figure pairs differ: "
            f"missing_png={sorted(pdf_stems - png_stems)}, "
            f"missing_pdf={sorted(png_stems - pdf_stems)}."
        )

    analyses = []
    caches = {}
    for index, spec in enumerate(specs, start=1):
        bundle = audit_result_bundle(
            result_root / spec.analysis_id,
            sample_chunk_size=sample_chunk_size,
        )
        metadata = json.loads(
            (result_root / spec.analysis_id / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        expected_identity = {
            **spec.identity.metadata(),
            "job_id": spec.job_id,
            "artifact_kind": spec.artifact_kind,
            "primary_score": spec.primary_score,
        }
        mismatched = {
            name: (metadata.get(name), value)
            for name, value in expected_identity.items()
            if metadata.get(name) != value
        }
        if mismatched:
            raise ValueError(
                f"Result metadata mismatch for {spec.analysis_id}: {mismatched}."
            )
        source_score = _source_score_record(spec, fit_audit_by_job)
        if metadata.get("posterior_variable") != source_score["score_variable"]:
            raise ValueError(
                f"Posterior variable mismatch for {spec.analysis_id}: "
                f"{metadata.get('posterior_variable')!r} != "
                f"{source_score['score_variable']!r}."
            )
        analyses.append(
            {
                "manifest": spec.to_dict(),
                "source_score": source_score,
                "posterior_transform": metadata.get("posterior_transform"),
                "result": bundle,
            }
        )
        caches[spec.analysis_id] = audit_dataset(
            cache_root / f"{spec.analysis_id}.nc",
            sample_chunk_size=sample_chunk_size,
        )
        if progress is not None:
            progress(index, len(specs), spec.analysis_id)

    code_inputs = {}
    for label, code_path_value in (code_paths or {}).items():
        code_path = Path(code_path_value)
        if not code_path.is_file():
            raise ValueError(f"Baseline code input is missing: {code_path}")
        code_inputs[label] = sha256_file(code_path)

    return {
        "schema_version": MIGRATION_AUDIT_SCHEMA_VERSION,
        "baseline_kind": "direct_hmc_fit_driven",
        "repository_commit": repository_commit,
        "manifest": {
            "sha256": sha256_file(manifest_path),
            "analysis_count": len(specs),
        },
        "fit_audit_sha256": sha256_file(fit_audit_path),
        "code_inputs": code_inputs,
        "analyses": analyses,
        "mechanism_caches": caches,
        "tables": {
            name: csv_content_record(table_dir / name)
            for name in HMC_TABLE_FILENAMES
        },
        "figures": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in figure_paths
        },
    }


def build_score_method_baseline(
    *,
    manifest_path: str | Path,
    score_root: str | Path,
    result_root: str | Path,
    cache_root: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    inference_method: str,
    repository_commit: str | None = None,
    sample_chunk_size: int = 256,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Freeze all persisted surfaces for one schema-v1 score method."""
    manifest_path = Path(manifest_path)
    score_root = Path(score_root)
    result_root = Path(result_root)
    cache_root = Path(cache_root)
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    specs = load_analysis_manifest(manifest_path)
    if {spec.identity.inference_method for spec in specs} != {inference_method}:
        raise ValueError(
            f"Manifest analyses must all use inference_method={inference_method!r}."
        )
    expected_ids = {spec.analysis_id for spec in specs}
    _require_exact_entries(score_root, expected=expected_ids, directories=True)
    _require_exact_entries(result_root, expected=expected_ids, directories=True)
    expected_cache_ids = {
        spec.score_identity(score_name).analysis_id
        for spec in specs
        for score_name in spec.scores
    }
    _require_exact_entries(
        cache_root,
        expected={f"{analysis_id}.nc" for analysis_id in expected_cache_ids},
        directories=False,
    )
    _require_exact_entries(
        table_dir,
        expected=set(HMC_TABLE_FILENAMES),
        directories=False,
    )

    figure_paths = sorted(
        path for path in figure_dir.iterdir() if path.suffix in {".pdf", ".png"}
    )
    if not figure_paths:
        raise ValueError(f"No baseline figures found at {figure_dir}.")
    png_stems = {path.stem for path in figure_paths if path.suffix == ".png"}
    pdf_stems = {path.stem for path in figure_paths if path.suffix == ".pdf"}
    if png_stems != pdf_stems:
        raise ValueError(
            "Baseline PNG/PDF figure pairs differ: "
            f"missing_png={sorted(pdf_stems - png_stems)}, "
            f"missing_pdf={sorted(png_stems - pdf_stems)}."
        )

    analyses = []
    caches = {}
    for index, spec in enumerate(specs, start=1):
        score_dir = score_root / spec.analysis_id
        score_metadata = json.loads(
            (score_dir / "metadata.json").read_text(encoding="utf-8")
        )
        expected_score_metadata = {
            **spec.identity.metadata(),
            "job_id": spec.job_id,
            "primary_score": spec.primary_score,
            "scores": list(spec.scores),
        }
        score_mismatches = {
            name: (score_metadata.get(name), value)
            for name, value in expected_score_metadata.items()
            if score_metadata.get(name) != value
        }
        if score_mismatches:
            raise ValueError(
                f"Score metadata mismatch for {spec.analysis_id}: "
                f"{score_mismatches}."
            )
        result_dir = result_root / spec.analysis_id
        result_metadata = json.loads(
            (result_dir / "metadata.json").read_text(encoding="utf-8")
        )
        expected_result_metadata = {
            **spec.identity.metadata(),
            "job_id": spec.job_id,
            "primary_score": spec.primary_score,
        }
        result_mismatches = {
            name: (result_metadata.get(name), value)
            for name, value in expected_result_metadata.items()
            if result_metadata.get(name) != value
        }
        if result_mismatches:
            raise ValueError(
                f"Result metadata mismatch for {spec.analysis_id}: "
                f"{result_mismatches}."
            )
        cache_ids = [
            spec.score_identity(score_name).analysis_id
            for score_name in spec.scores
        ]
        analyses.append(
            {
                "manifest": spec.to_dict(),
                "source_artifact": score_metadata.get("source_artifact"),
                "score_bundle": audit_score_ensemble_bundle(
                    score_dir,
                    sample_chunk_size=sample_chunk_size,
                ),
                "result": audit_result_bundle(
                    result_dir,
                    sample_chunk_size=sample_chunk_size,
                ),
                "mechanism_cache_ids": cache_ids,
            }
        )
        for cache_id in cache_ids:
            caches[cache_id] = audit_dataset(
                cache_root / f"{cache_id}.nc",
                sample_chunk_size=sample_chunk_size,
            )
        if progress is not None:
            progress(index, len(specs), spec.analysis_id)

    return {
        "schema_version": MIGRATION_AUDIT_SCHEMA_VERSION,
        "baseline_kind": f"{inference_method}_schema_v1_pre_conversion",
        "repository_commit": repository_commit,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "analysis_count": len(specs),
            "score_view_count": len(expected_cache_ids),
        },
        "score_root": str(score_root),
        "result_root": str(result_root),
        "cache_root": str(cache_root),
        "analyses": analyses,
        "mechanism_caches": caches,
        "tables": {
            name: csv_content_record(table_dir / name)
            for name in HMC_TABLE_FILENAMES
        },
        "figures": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in figure_paths
        },
    }


def migration_audit_sha256(audit: object) -> str:
    """Return a canonical digest for a complete migration audit."""
    return hashlib.sha256(_canonical_json(audit)).hexdigest()


def migration_artifact_sha256(audit: object) -> str:
    """Return a digest excluding code provenance that changes during migration."""
    if not isinstance(audit, dict):
        raise ValueError("Migration audit must be a JSON object.")
    artifact_audit = {
        name: value
        for name, value in audit.items()
        if name not in {"code_inputs", "repository_commit"}
    }
    return migration_audit_sha256(artifact_audit)


def verify_score_method_conversion(
    expected: object,
    observed: object,
) -> None:
    """Allow only the intended physical score-schema differences."""
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        raise ValueError("Score conversion audits must be JSON objects.")
    expected_analyses = expected.get("analyses")
    observed_analyses = observed.get("analyses")
    if not isinstance(expected_analyses, list) or not isinstance(
        observed_analyses,
        list,
    ):
        raise ValueError("Score conversion audits must contain analyses lists.")

    def normalized(audit: dict[str, object], *, schema_version: int) -> object:
        analyses = audit["analyses"]
        assert isinstance(analyses, list)
        normalized_analyses = []
        for analysis in analyses:
            if not isinstance(analysis, dict):
                raise ValueError("Score conversion audit contains a malformed analysis.")
            score_bundle = analysis.get("score_bundle")
            if not isinstance(score_bundle, dict) or score_bundle.get(
                "source_schema_version"
            ) != schema_version:
                raise ValueError(
                    f"Expected schema {schema_version} score bundle for "
                    f"{analysis.get('manifest')}."
                )
            normalized_analyses.append(
                {
                    "manifest": analysis.get("manifest"),
                    "source_artifact": analysis.get("source_artifact"),
                    "score_bundle": {
                        "analysis_id": score_bundle.get("analysis_id"),
                        "semantic": score_bundle.get("semantic"),
                        "semantic_sha256": score_bundle.get("semantic_sha256"),
                    },
                    "result": analysis.get("result"),
                    "mechanism_cache_ids": analysis.get("mechanism_cache_ids"),
                }
            )
        return {
            "schema_version": audit.get("schema_version"),
            "baseline_kind": audit.get("baseline_kind"),
            "manifest": audit.get("manifest"),
            "result_root": audit.get("result_root"),
            "cache_root": audit.get("cache_root"),
            "analyses": normalized_analyses,
            "mechanism_caches": audit.get("mechanism_caches"),
            "tables": audit.get("tables"),
            "figures": audit.get("figures"),
        }

    expected_normalized = normalized(expected, schema_version=1)
    observed_normalized = normalized(observed, schema_version=2)
    expected_digest = migration_audit_sha256(expected_normalized)
    observed_digest = migration_audit_sha256(observed_normalized)
    if expected_digest != observed_digest:
        raise ValueError(
            "Score conversion changed a frozen scientific artifact: "
            f"expected {expected_digest}, observed {observed_digest}."
        )


def verify_hmc_migration_audit(
    expected: object,
    observed: object,
    *,
    strict_code_inputs: bool = False,
) -> None:
    """Reject scientific artifact drift from a frozen migration baseline."""
    digest = migration_audit_sha256 if strict_code_inputs else migration_artifact_sha256
    expected_digest = digest(expected)
    observed_digest = digest(observed)
    if expected_digest != observed_digest:
        raise ValueError(
            "HMC migration baseline changed: "
            f"expected {expected_digest}, observed {observed_digest}."
        )
