import json

import pytest

from dms_fitness_stability.count_sources import (
    build_count_sources_manifest,
    validate_count_sources_manifest,
)
from dms_fitness_stability.migration_audit import sha256_file


def test_count_source_manifest_deduplicates_sources_and_enforces_blocked_gate(tmp_path):
    data_root = tmp_path / "data"
    source = data_root / "pdz_cis/raw/counts.tsv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"counts")
    feature_root = data_root / "paper/analysis_data/landscape_features"
    for feature_id in ("pdz__pdz-cis", "pdz__pdz-cis-second"):
        bundle = feature_root / feature_id
        bundle.mkdir(parents=True)
        (bundle / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_id": feature_id,
                    "protein_family": "pdz",
                    "source_files": [
                        {
                            "path": str(source.resolve()),
                            "bytes": source.stat().st_size,
                            "sha256": sha256_file(source),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    manifest = build_count_sources_manifest(
        data_root=data_root,
        feature_root=feature_root,
        repository_root=tmp_path,
    )
    counts = validate_count_sources_manifest(manifest, data_root=data_root)

    assert counts == {"sources": 1, "blocked_families": 2}
    assert manifest["publication_status"] == "blocked_pending_provenance"
    assert manifest["sources"][0]["feature_ids"] == [
        "pdz__pdz-cis",
        "pdz__pdz-cis-second",
    ]
    manifest["publication_status"] = "ready"
    with pytest.raises(ValueError, match="cannot be ready"):
        validate_count_sources_manifest(manifest, data_root=data_root)
