"""Provenance manifest for observed files used to derive feature bundles."""

from __future__ import annotations

import json
from pathlib import Path

from .migration_audit import sha256_file

COUNT_SOURCE_MANIFEST_SCHEMA_VERSION = 1
FAMILY_GATES = {
    "pdz": {
        "status": "blocked",
        "reason": (
            "Upstream source locations and redistribution terms for the staged "
            "PDZ count tables are not yet recorded."
        ),
        "provenance_manifest": None,
        "license": None,
    },
    "trpb": {
        "status": "blocked",
        "reason": (
            "The staged TrpB count source is internal and redistribution terms "
            "are not yet recorded."
        ),
        "provenance_manifest": None,
        "license": None,
    },
    "nuc": {
        "status": "public_source",
        "reason": None,
        "provenance_manifest": "examples/nuc/data_manifest.json",
        "license": "CC-BY-4.0",
    },
    "k50": {
        "status": "public_source",
        "reason": None,
        "provenance_manifest": "examples/k50/data_manifest.json",
        "license": "CC-BY-4.0",
    },
}


def _public_source_spec(
    family: str,
    relative_path: str,
    *,
    repository_root: Path,
) -> object:
    gate = FAMILY_GATES[family]
    manifest_name = gate["provenance_manifest"]
    if manifest_name is None:
        return None
    manifest = json.loads(
        (repository_root / str(manifest_name)).read_text(encoding="utf-8")
    )
    prefix = f"{family}/raw/"
    family_path = relative_path.removeprefix(prefix)
    if family == "nuc":
        return manifest["raw"].get(family_path)
    if family == "k50":
        if family_path in manifest["raw"]:
            return manifest["raw"][family_path]
        projection = manifest["reported_fitness"]["projection"]
        if family_path == projection["path"]:
            return projection
    return None


def build_count_sources_manifest(
    *,
    data_root: str | Path,
    feature_root: str | Path,
    repository_root: str | Path,
) -> dict[str, object]:
    """Deduplicate all feature-producer inputs and attach release gates."""
    data_root = Path(data_root).resolve()
    feature_root = Path(feature_root)
    repository_root = Path(repository_root).resolve()
    sources: dict[str, dict[str, object]] = {}
    feature_count = 0
    for bundle_dir in sorted(feature_root.iterdir()):
        if not bundle_dir.is_dir():
            continue
        metadata = json.loads(
            (bundle_dir / "metadata.json").read_text(encoding="utf-8")
        )
        feature_id = str(metadata["feature_id"])
        family = str(metadata["protein_family"])
        if family not in FAMILY_GATES:
            raise ValueError(f"Unknown feature source family {family!r}.")
        feature_count += 1
        source_files = metadata.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError(f"Feature bundle has no source files: {bundle_dir}")
        for source in source_files:
            if not isinstance(source, dict):
                raise ValueError(f"Malformed feature source record: {bundle_dir}")
            path = Path(str(source["path"])).resolve()
            if not path.is_relative_to(data_root):
                raise ValueError(f"Feature source lies outside data root: {path}")
            relative_path = str(path.relative_to(data_root))
            existing = sources.get(relative_path)
            signature = {
                "bytes": source["bytes"],
                "sha256": source["sha256"],
            }
            if existing is not None:
                if any(existing[name] != value for name, value in signature.items()):
                    raise ValueError(f"Conflicting source signatures: {relative_path}")
                existing["feature_ids"].append(feature_id)
                continue
            sources[relative_path] = {
                "family": family,
                "relative_path": relative_path,
                **signature,
                "feature_ids": [feature_id],
                "upstream": _public_source_spec(
                    family,
                    relative_path,
                    repository_root=repository_root,
                ),
            }
    for source in sources.values():
        source["feature_ids"] = sorted(set(source["feature_ids"]))
        if (
            FAMILY_GATES[str(source["family"])]["status"] == "public_source"
            and source["upstream"] is None
        ):
            raise ValueError(
                f"Public family source lacks upstream provenance: "
                f"{source['relative_path']}"
            )
    blocked_families = sorted(
        family for family, gate in FAMILY_GATES.items() if gate["status"] == "blocked"
    )
    return {
        "schema_version": COUNT_SOURCE_MANIFEST_SCHEMA_VERSION,
        "publication_status": (
            "blocked_pending_provenance" if blocked_families else "ready"
        ),
        "blocked_families": blocked_families,
        "feature_bundle_count": feature_count,
        "family_gates": FAMILY_GATES,
        "sources": [sources[name] for name in sorted(sources)],
    }


def validate_count_sources_manifest(
    manifest: object,
    *,
    data_root: str | Path,
    verify_checksums: bool = True,
) -> dict[str, int]:
    """Validate count-source files and prevent overstated publication status."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Count source manifest schema_version must be 1.")
    sources = manifest.get("sources")
    gates = manifest.get("family_gates")
    if not isinstance(sources, list) or not isinstance(gates, dict):
        raise ValueError("Count source manifest sources and family_gates are required.")
    blocked = sorted(
        family
        for family, gate in gates.items()
        if isinstance(gate, dict) and gate.get("status") == "blocked"
    )
    if manifest.get("blocked_families") != blocked:
        raise ValueError("Count source blocked_families do not match family gates.")
    if blocked and manifest.get("publication_status") == "ready":
        raise ValueError("Count source publication cannot be ready with blocked families.")

    data_root = Path(data_root).resolve()
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Malformed count source record.")
        relative_path = str(source["relative_path"])
        if relative_path in seen:
            raise ValueError(f"Duplicate count source path: {relative_path}")
        seen.add(relative_path)
        path = data_root / relative_path
        if not path.is_file() or path.stat().st_size != source.get("bytes"):
            raise ValueError(f"Count source is missing or changed: {path}")
        if verify_checksums and sha256_file(path) != source.get("sha256"):
            raise ValueError(f"Count source checksum changed: {path}")
        family = str(source["family"])
        gate = gates.get(family)
        if not isinstance(gate, dict):
            raise ValueError(f"Count source has no family gate: {family}")
        if gate.get("status") == "public_source" and source.get("upstream") is None:
            raise ValueError(f"Public count source has no upstream record: {path}")
    return {"sources": len(sources), "blocked_families": len(blocked)}
