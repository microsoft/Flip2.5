"""Build tidy paper tables for every analysis run in the checked-in manifest."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dms_fitness_stability import AnalysisIdentity, StabilityResult, build_paper_tables
from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.reporting import PAPER_TABLE_NAMES

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
IDENTITY_FIELDS = (
    "protein_family",
    "landscape",
    "inference_method",
    "model_spec",
    "fit_id",
)


@dataclass(frozen=True)
class SummaryRun:
    identity: AnalysisIdentity
    source_identity: AnalysisIdentity
    score: str


def load_manifest(path: Path) -> list[AnalysisIdentity]:
    """Load and validate the ordered analysis-run manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Analysis manifest schema_version must be 1.")
    entries = payload.get("analyses")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Analysis manifest must contain a non-empty analyses list.")

    identities = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Analysis manifest entry {index} must be an object.")
        missing = set(IDENTITY_FIELDS) - set(entry)
        extra = set(entry) - set(IDENTITY_FIELDS)
        if missing or extra:
            raise ValueError(
                f"Analysis manifest entry {index} has missing={sorted(missing)} "
                f"and extra={sorted(extra)} fields."
            )
        identities.append(AnalysisIdentity(**entry))
    analysis_ids = [identity.analysis_id for identity in identities]
    if len(set(analysis_ids)) != len(analysis_ids):
        raise ValueError("Analysis manifest contains duplicate analysis_id values.")
    return identities


def load_summary_runs(
    path: Path,
    *,
    score_override: str | None = None,
) -> list[SummaryRun]:
    """Load legacy or curated manifests with one declared score per run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        score = score_override or "log_mean_growth"
        return [SummaryRun(identity, identity, score) for identity in load_manifest(path)]
    if schema_version == 2:
        specs = load_analysis_manifest(path)
        return [
            SummaryRun(
                spec.score_identity(score),
                spec.identity,
                score,
            )
            for spec in specs
            for score in ((score_override,) if score_override else spec.scores)
        ]
    raise ValueError("Analysis manifest schema_version must be 1 or 2.")


def _validate_identity(result: StabilityResult, expected: AnalysisIdentity) -> None:
    metadata = result.metadata
    for field, expected_value in expected.metadata().items():
        if metadata.get(field) != expected_value:
            raise ValueError(
                f"{expected.analysis_id}: metadata {field}={metadata.get(field)!r}, "
                f"expected {expected_value!r}."
            )


def summarize_manifest(
    manifest_path: Path,
    *,
    result_root: Path,
    score: str | None,
    main_k: int,
    k_values: tuple[int, ...],
    analysis_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every manifested result and concatenate its paper tables."""
    runs = load_summary_runs(manifest_path, score_override=score)
    if analysis_ids:
        runs = [run for run in runs if run.identity.analysis_id in analysis_ids]
        missing_requested = analysis_ids - {
            run.identity.analysis_id for run in runs
        }
        if missing_requested:
            raise ValueError(
                f"Unknown requested analysis IDs: {sorted(missing_requested)}"
            )
    if not runs:
        raise ValueError("No analyses selected for summarization.")
    missing = [
        run.source_identity.analysis_id
        for run in runs
        if not (
            result_root / run.source_identity.analysis_id / "metadata.json"
        ).is_file()
    ]
    if missing:
        examples = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} manifested result bundles under {result_root}: "
            f"{examples}{suffix}."
        )

    collected = {name: [] for name in PAPER_TABLE_NAMES}
    for analysis_order, run in enumerate(runs):
        identity = run.identity
        result_dir = result_root / run.source_identity.analysis_id
        result = StabilityResult.load(result_dir)
        _validate_identity(result, run.source_identity)
        tables = build_paper_tables(
            result,
            score=run.score,
            main_k=main_k,
            k_values=k_values,
            percentile_cohorts=(main_k, 1000),
        )
        for name, table in tables.items():
            table.insert(1, "source_analysis_id", run.source_identity.analysis_id)
            table["analysis_id"] = identity.analysis_id
            table["landscape"] = identity.landscape
            table.insert(0, "analysis_order", analysis_order)
            collected[name].append(table)
        warnings = result.metadata.get("warnings", [])
        status = result.metadata.get("artifact_status", "unrecorded")
        print(
            f"[{analysis_order + 1:02d}/{len(runs):02d}] "
            f"{identity.analysis_id} [{run.score}]: {status}, "
            f"{len(warnings)} warning(s)",
            flush=True,
        )
    return {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in collected.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "paper/hmc/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "output/hmc/tables",
    )
    parser.add_argument(
        "--score",
        default=None,
        help="Override every manifest primary_score (legacy default: log_mean_growth).",
    )
    parser.add_argument("--analysis-id", nargs="+")
    parser.add_argument("--main-k", type=int, default=100)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = summarize_manifest(
        args.manifest,
        result_root=args.result_root,
        score=args.score,
        main_k=args.main_k,
        k_values=tuple(args.k_values),
        analysis_ids=set(args.analysis_id) if args.analysis_id else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        output = args.output_dir / f"{name}.csv"
        table.to_csv(output, index=False)
        print(f"Saved {len(table):,} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
