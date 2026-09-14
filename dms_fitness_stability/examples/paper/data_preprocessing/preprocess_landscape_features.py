"""Materialize observed-count features used by current mechanism figures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dms_fitness_stability.adapters.k50 import load_k50_landscape
from dms_fitness_stability.adapters.nuclease import load_nuclease_landscape
from dms_fitness_stability.adapters.pdz import (
    load_pdz_cis,
    load_pdz_cript,
    load_pdz_trans,
)
from dms_fitness_stability.adapters.trpb import load_trpb_landscape
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.identity import AnalysisIdentity
from dms_fitness_stability.landscape_features import (
    LandscapeFeatureBundle,
    landscape_feature_id,
)
from dms_fitness_stability.migration_audit import sha256_file

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


@dataclass(frozen=True)
class FeaturePlan:
    """One unique landscape/score feature view."""

    identity: AnalysisIdentity
    score_name: str | None

    @property
    def feature_id(self) -> str:
        return landscape_feature_id(
            self.identity.protein_family,
            self.identity.landscape,
            score_name=self.score_name,
        )


def _plans(manifest_paths: tuple[Path, ...]) -> tuple[FeaturePlan, ...]:
    plans: dict[str, FeaturePlan] = {}
    for manifest_path in manifest_paths:
        for spec in load_analysis_manifest(manifest_path):
            for score_name in spec.scores:
                feature_score = (
                    score_name if spec.identity.protein_family == "k50" else None
                )
                plan = FeaturePlan(spec.identity, feature_score)
                plans.setdefault(plan.feature_id, plan)
    return tuple(plans.values())


def _pdz_paths(data_root: Path, source: str) -> tuple[Path, ...]:
    if source in {"trans-1", "trans-2"}:
        root = data_root / "pdz_trans/raw"
        return (
            root / "trans-1.tsv",
            root / "trans-2.tsv",
            root / "GSE265816_pdz3cript_sub.csv.gz",
        )
    root = data_root / "pdz_cis/raw"
    if source == "cis":
        return (
            root / "cis_from_bruce_azure.tsv",
            root / "GSE265816_cript2xa3_sub.csv.gz",
        )
    return (
        root / f"{source}.tsv",
        root / f"GSE265816_{source.replace('-', '_')}_sub.csv.gz",
    )


def _source_paths(plan: FeaturePlan, data_root: Path) -> tuple[Path, ...]:
    family = plan.identity.protein_family
    source = plan.identity.landscape.removeprefix(f"{family}-")
    if family == "pdz":
        return _pdz_paths(data_root, source)
    if family == "trpb":
        root = data_root / "trpb/raw"
        return (
            root / "counts" / f"{source}.csv",
            root / "reported_fitness" / f"{source}.csv",
        )
    if family == "nuc":
        root = data_root / "nuc/raw"
        generation = source.lower()
        return (
            root / "counts" / f"{generation}.csv",
            root / "counts" / f"{generation}_neg_control.csv",
            root / "reported_fitness/landscape.csv",
        )
    if family == "k50":
        root = data_root / "k50/raw"
        return (
            root / "NGS_count_lib1.csv",
            root / "NGS_count_lib2.csv",
            root / "NGS_count_lib3.csv",
            root / "NGS_count_lib4.csv",
            root / "K50_scrambles_for_STEP3.csv",
            root / "Raw_qPCR_data_FigS1.csv",
            root / "reported_fitness/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
        )
    raise ValueError(f"Unsupported mechanism family {family!r}.")


def _source_records(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Landscape feature source is missing: {path}")
        records.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _load_landscape(plan: FeaturePlan, data_root: Path):
    family = plan.identity.protein_family
    source = plan.identity.landscape.removeprefix(f"{family}-")
    if family == "pdz":
        if source in {"trans-1", "trans-2"}:
            root = data_root / "pdz_trans/raw"
            return load_pdz_trans(
                source,
                count_paths={
                    name: root / f"{name}.tsv"
                    for name in ("trans-1", "trans-2")
                },
                reported_fitness_path=root / "GSE265816_pdz3cript_sub.csv.gz",
            )
        root = data_root / "pdz_cis/raw"
        if source == "cis":
            return load_pdz_cis(
                root / "cis_from_bruce_azure.tsv",
                reported_fitness_path=root / "GSE265816_cript2xa3_sub.csv.gz",
            )
        return load_pdz_cript(
            source,
            root / f"{source}.tsv",
            reported_fitness_path=(
                root / f"GSE265816_{source.replace('-', '_')}_sub.csv.gz"
            ),
        )
    if family == "trpb":
        root = data_root / "trpb/raw"
        return load_trpb_landscape(
            source,
            count_path=root / "counts" / f"{source}.csv",
            reported_fitness_path=root / "reported_fitness" / f"{source}.csv",
        )
    if family == "nuc":
        root = data_root / "nuc/raw"
        return load_nuclease_landscape(
            source.upper(),
            count_dir=root / "counts",
            reported_activity_path=root / "reported_fitness/landscape.csv",
        )
    if family == "k50":
        root = data_root / "k50/raw"
        return load_k50_landscape(
            root,
            reported_fitness_path=(
                root
                / "reported_fitness/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
            ),
        )
    raise ValueError(f"Unsupported mechanism family {family!r}.")


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sources_match(
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> bool:
    return metadata.get("source_files") == records


def _existing_matches(
    output_dir: Path,
    plan: FeaturePlan,
    records: list[dict[str, object]],
) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("feature_id") != plan.feature_id
        or not _sources_match(metadata, records)
    ):
        return False
    bundle = LandscapeFeatureBundle.load(output_dir)
    bundle.close()
    return True


def _publish(bundle: LandscapeFeatureBundle, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        bundle.save(staging_dir)
        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            output_dir.replace(backup_dir)
            try:
                staging_dir.replace(output_dir)
            except Exception:
                backup_dir.replace(output_dir)
                raise
            shutil.rmtree(backup_dir)
        else:
            staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="Repeat for additional analysis manifests (default: HMC, MLE, HMC count).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/analysis_data/landscape_features",
    )
    parser.add_argument("--family", choices=("pdz", "trpb", "nuc", "k50"), nargs="+")
    parser.add_argument("--feature-id", nargs="+", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = tuple(
        args.manifest
        or (
            PAPER_ROOT / "analyses_hmc.json",
            PAPER_ROOT / "analyses_mle.json",
            PAPER_ROOT / "analyses_hmc_count_based.json",
        )
    )
    plans = _plans(manifests)
    families = set(args.family or ())
    requested_ids = set(args.feature_id or ())
    selected = tuple(
        plan
        for plan in plans
        if (not families or plan.identity.protein_family in families)
        and (not requested_ids or plan.feature_id in requested_ids)
    )
    if not selected:
        raise ValueError("No landscape feature views match the requested filters.")
    if requested_ids and {plan.feature_id for plan in selected} != requested_ids:
        raise ValueError("A requested feature ID is absent from the selected manifests.")

    commit = _repository_commit()
    tasks = []
    for plan in selected:
        records = _source_records(_source_paths(plan, args.data_root))
        output_dir = args.output_root / plan.feature_id
        reuse = False if args.overwrite else _existing_matches(output_dir, plan, records)
        state = "reuse" if reuse else "build"
        print(f"{state}: {plan.feature_id} -> {output_dir}")
        tasks.append((plan, records, output_dir, reuse))
    if not args.execute:
        print(
            f"Dry run: {len(tasks)} landscape feature plan(s); "
            "add --execute to build."
        )
        return

    landscape_cache = {}
    for index, (plan, records, output_dir, reuse) in enumerate(tasks, start=1):
        if reuse:
            print(f"[{index}/{len(tasks)}] Reusing {output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing feature bundle does not match: {output_dir}. "
                "Use --overwrite only after reviewing its metadata."
            )
        cache_key = (plan.identity.protein_family, plan.identity.landscape)
        if cache_key not in landscape_cache:
            landscape_cache[cache_key] = _load_landscape(plan, args.data_root)
        print(f"[{index}/{len(tasks)}] Building {plan.feature_id}", flush=True)
        bundle = LandscapeFeatureBundle.from_landscape(
            plan.identity,
            landscape_cache[cache_key],
            score_name=plan.score_name,
            extra_metadata={
                "producer_code_commit": commit,
                "source_files": records,
            },
        )
        _publish(bundle, output_dir)
        validated = LandscapeFeatureBundle.load(output_dir)
        validated.close()
        print(f"Published {output_dir}", flush=True)


if __name__ == "__main__":
    main()
