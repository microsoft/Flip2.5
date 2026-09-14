import json
import shutil
import zipfile
from pathlib import Path

import pytest

from data.download import (
    load_file_index,
    retrieve_files,
    select_files,
    sha256_file,
)
from dms_rank_stability import LandscapeFeatures, SampledScores
from dms_rank_stability.runs import load_run_config

PUBLIC_INDEX = Path(__file__).parents[1] / "data/files.json"
PAPER_RUNS = Path(__file__).parents[1] / "analysis/runs.json"


def _archive(source: Path, path: Path, content: str) -> dict[str, object]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted((source / content).rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source).as_posix())
    return {
        "name": path.name,
        "url": path.as_uri(),
        "doi": "10.0000/example",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": "zip",
        "content": content,
        "sampling": ["parameter_posterior"] if content == "scores" else [],
        "families": ["pdz"],
        "landscapes": ["pdz-example"],
    }


def _index(tmp_path: Path, example_root: Path):
    source = tmp_path / "source"
    score_target = (
        source
        / "scores/parameter_posterior/pdz-example-parameter-posterior-example"
    )
    feature_target = source / "features/pdz-example"
    score_target.parent.mkdir(parents=True)
    feature_target.parent.mkdir(parents=True)
    shutil.copytree(example_root / "scores/toy-parameter-posterior-example", score_target)
    shutil.copytree(example_root / "features/toy", feature_target)
    files = [
        _archive(source, tmp_path / "scores.zip", "scores"),
        _archive(source, tmp_path / "features.zip", "features"),
        {
            "name": "trpb-predictive.tar.gz",
            "url": None,
            "doi": None,
            "bytes": None,
            "sha256": None,
            "format": "tar.gz",
            "content": "scores",
            "sampling": ["posterior_predictive"],
            "families": ["trpb"],
            "landscapes": ["trpb-example"],
        },
    ]
    path = tmp_path / "files.json"
    path.write_text(
        json.dumps(
            {
                "publication": {"status": "staging", "doi": None},
                "citations": [
                    {
                        "family": family,
                        "study": f"{family} example study",
                        "doi": f"10.0000/{family}",
                        "data": f"{family} example data",
                    }
                    for family in ("pdz", "trpb", "nuc")
                ],
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return load_file_index(path)


def test_download_selects_shared_features_and_repairs_tampered_cache(
    tmp_path,
    example_root,
):
    index = _index(tmp_path, example_root)
    selected = select_files(
        index,
        families=("pdz",),
        sampling_methods=("parameter_posterior",),
    )
    assert [item.content for item in selected] == ["scores", "features"]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "scores.zip").write_bytes(b"changed")

    counts = retrieve_files(
        selected,
        destination=tmp_path / "downloaded",
        cache_dir=cache,
    )

    assert counts == {"archives": 2, "bundles": 2}
    score = SampledScores.load(
        tmp_path
        / "downloaded/scores/parameter_posterior/"
        "pdz-example-parameter-posterior-example"
    )
    feature = LandscapeFeatures.load(tmp_path / "downloaded/features/pdz-example")
    try:
        assert "source_sample_id" in score.dataset.coords
    finally:
        score.close()
        feature.close()
    assert sha256_file(cache / "scores.zip") == selected[0].sha256


def test_public_file_index_locks_planned_paper_scope():
    index = load_file_index(PUBLIC_INDEX)

    assert index.status == "staging"
    assert index.doi is None
    assert {citation["family"] for citation in index.citations} == {
        "pdz",
        "trpb",
        "nuc",
    }
    assert len(index.files) == 20
    assert not any(item.available for item in index.files)
    assert {family for item in index.files for family in item.families} == {
        "pdz",
        "trpb",
        "nuc",
    }
    assert {
        sampling for item in index.files for sampling in item.sampling
    } == {
        "parameter_posterior",
        "mle_predictive",
        "posterior_predictive",
    }
    pdz_parameter = select_files(
        index,
        families=("pdz",),
        sampling_methods=("parameter_posterior",),
    )
    assert len(pdz_parameter) == 6
    assert sum(item.content == "features" for item in pdz_parameter) == 1
    trpb_mle = select_files(
        index,
        families=("trpb",),
        sampling_methods=("mle_predictive",),
    )
    assert {item.name for item in trpb_mle} == {
        "features-trpb.tar.gz",
        "mle-predictive-trpb.tar.gz",
    }


def test_public_file_parts_cover_each_declared_bundle_exactly_once():
    index = load_file_index(PUBLIC_INDEX)
    config = load_run_config(PAPER_RUNS)

    for run in config.runs:
        matching_scores = [
            item
            for item in index.files
            if item.content == "scores"
            and run.sampling in item.sampling
            and run.family in item.families
            and run.landscape.lower() in item.landscapes
        ]
        assert len(matching_scores) == 1, run.run_id

    feature_keys = {
        (run.family, run.landscape.lower())
        for run in config.runs
    }
    for family, landscape in feature_keys:
        matching_features = [
            item
            for item in index.files
            if item.content == "features"
            and family in item.families
            and landscape in item.landscapes
        ]
        assert len(matching_features) == 1, (family, landscape)

    covered_score_runs = sum(
        sum(
            run.sampling in item.sampling
            and run.family in item.families
            and run.landscape.lower() in item.landscapes
            for run in config.runs
        )
        for item in index.files
        if item.content == "scores"
    )
    covered_feature_keys = sum(
        sum(
            family in item.families and landscape in item.landscapes
            for family, landscape in feature_keys
        )
        for item in index.files
        if item.content == "features"
    )
    assert covered_score_runs == 158
    assert covered_feature_keys == 19


def test_unpublished_and_unsafe_archives_are_rejected(tmp_path, example_root):
    index = _index(tmp_path, example_root)
    unpublished = select_files(
        index,
        families=("trpb",),
        sampling_methods=("posterior_predictive",),
    )
    with pytest.raises(ValueError, match="not been published"):
        retrieve_files(
            unpublished,
            destination=tmp_path / "downloaded",
            cache_dir=tmp_path / "cache",
        )

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    record = {
        **index.files[0].__dict__,
        "name": unsafe.name,
        "url": unsafe.as_uri(),
        "bytes": unsafe.stat().st_size,
        "sha256": sha256_file(unsafe),
    }
    path = tmp_path / "unsafe-files.json"
    path.write_text(
        json.dumps(
            {
                "publication": {"status": "staging", "doi": None},
                "citations": [
                    {
                        "family": family,
                        "study": f"{family} example study",
                        "doi": f"10.0000/{family}",
                        "data": f"{family} example data",
                    }
                    for family in ("pdz", "trpb", "nuc")
                ],
                "files": [
                    {
                        **record,
                        "sampling": list(record["sampling"]),
                        "families": list(record["families"]),
                        "landscapes": list(record["landscapes"]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    unsafe_item = load_file_index(path).files
    with pytest.raises(ValueError, match="unsafe path"):
        retrieve_files(
            unsafe_item,
            destination=tmp_path / "unsafe-output",
            cache_dir=tmp_path / "unsafe-cache",
        )


def test_archive_bundle_paths_must_match_declared_scope(tmp_path, example_root):
    source = tmp_path / "source"
    target = source / "scores/mle_predictive/pdz-example-mle-predictive-example"
    target.parent.mkdir(parents=True)
    shutil.copytree(
        example_root / "scores/toy-parameter-posterior-example",
        target,
    )
    record = _archive(source, tmp_path / "wrong-scope.zip", "scores")
    path = tmp_path / "files.json"
    path.write_text(
        json.dumps(
            {
                "publication": {"status": "staging", "doi": None},
                "citations": [
                    {
                        "family": family,
                        "study": f"{family} example study",
                        "doi": f"10.0000/{family}",
                        "data": f"{family} example data",
                    }
                    for family in ("pdz", "trpb", "nuc")
                ],
                "files": [record],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling scope"):
        retrieve_files(
            load_file_index(path).files,
            destination=tmp_path / "output",
            cache_dir=tmp_path / "cache",
        )
