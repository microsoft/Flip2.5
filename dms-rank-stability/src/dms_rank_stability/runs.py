"""Strict run configuration and method-neutral analysis execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .data import SampledScores
from .ranking import analyze_scores
from .results import IDENTITY_FIELDS, RankResult

SAMPLING_METHODS = {
    "parameter_posterior",
    "mle_predictive",
    "posterior_predictive",
}
SETTING_FIELDS = {
    "k_values",
    "retain_k",
    "decision_k",
    "external_top_n",
    "model_top_n",
    "interval",
    "challenger_probability",
    "seed",
}
RUN_FIELDS = {
    *IDENTITY_FIELDS,
    "score_path",
    "feature_path",
    "score_higher_is_better",
    "external_value_column",
    "external_rank_column",
    "external_higher_is_better",
    "min_reference_samples",
    "variant_chunk_size",
}
RUN_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace.")
    return value


def _relative_path(value: object, name: str) -> PurePosixPath:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without parent traversal.")
    if path.as_posix() != text:
        raise ValueError(f"{name} must use normalized forward-slash separators.")
    return path


@dataclass(frozen=True)
class AnalysisSettings:
    """Reviewed choices shared by every declared run."""

    k_values: tuple[int, ...]
    retain_k: int
    decision_k: int
    external_top_n: int
    model_top_n: int
    interval: tuple[float, float]
    challenger_probability: float
    seed: int

    @classmethod
    def from_dict(cls, payload: object) -> AnalysisSettings:
        if not isinstance(payload, dict) or set(payload) != SETTING_FIELDS:
            fields = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "settings must contain exactly the required fields; "
                f"missing={sorted(SETTING_FIELDS - fields)}, "
                f"extra={sorted(fields - SETTING_FIELDS)}."
            )
        k_payload = payload["k_values"]
        if not isinstance(k_payload, list) or not k_payload:
            raise ValueError("settings.k_values must be a non-empty list.")
        k_values = tuple(
            _integer(value, "settings.k_values item", minimum=1) for value in k_payload
        )
        if tuple(sorted(set(k_values))) != k_values:
            raise ValueError("settings.k_values must be unique and increasing.")
        retain_k = _integer(payload["retain_k"], "settings.retain_k", minimum=1)
        decision_k = _integer(payload["decision_k"], "settings.decision_k", minimum=1)
        external_top_n = _integer(
            payload["external_top_n"],
            "settings.external_top_n",
        )
        model_top_n = _integer(payload["model_top_n"], "settings.model_top_n", minimum=1)
        if not decision_k <= model_top_n <= retain_k:
            raise ValueError("settings must satisfy decision_k <= model_top_n <= retain_k.")
        if k_values[-1] > retain_k or decision_k not in k_values:
            raise ValueError("settings.k_values must include decision_k and not exceed retain_k.")

        interval_payload = payload["interval"]
        if not isinstance(interval_payload, list) or len(interval_payload) != 2:
            raise ValueError("settings.interval must contain two probabilities.")
        interval = (
            _number(interval_payload[0], "settings.interval lower"),
            _number(interval_payload[1], "settings.interval upper"),
        )
        if not 0 <= interval[0] < interval[1] <= 1:
            raise ValueError("settings.interval must increase within [0, 1].")
        challenger_probability = _number(
            payload["challenger_probability"],
            "settings.challenger_probability",
        )
        if not 0 < challenger_probability <= 1:
            raise ValueError("settings.challenger_probability must lie in (0, 1].")
        return cls(
            k_values=k_values,
            retain_k=retain_k,
            decision_k=decision_k,
            external_top_n=external_top_n,
            model_top_n=model_top_n,
            interval=interval,
            challenger_probability=challenger_probability,
            seed=_integer(payload["seed"], "settings.seed"),
        )


@dataclass(frozen=True)
class RunSpec:
    """One matrix-first scientific analysis."""

    run_id: str
    family: str
    landscape: str
    sampling: str
    model: str
    score: str
    score_path: PurePosixPath
    feature_path: PurePosixPath
    score_higher_is_better: bool
    external_value_column: str
    external_rank_column: str
    external_higher_is_better: bool
    min_reference_samples: int
    variant_chunk_size: int

    @classmethod
    def from_dict(cls, payload: object) -> RunSpec:
        if not isinstance(payload, dict) or set(payload) != RUN_FIELDS:
            fields = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "Each run must contain exactly the required fields; "
                f"missing={sorted(RUN_FIELDS - fields)}, "
                f"extra={sorted(fields - RUN_FIELDS)}."
            )
        run_id = _text(payload["run_id"], "run.run_id")
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "run.run_id must contain only lowercase letters, numbers, and hyphens."
            )
        sampling = _text(payload["sampling"], "run.sampling")
        if sampling not in SAMPLING_METHODS:
            raise ValueError(f"Unsupported sampling method: {sampling!r}.")
        for name in ("score_higher_is_better", "external_higher_is_better"):
            if not isinstance(payload[name], bool):
                raise ValueError(f"run.{name} must be boolean.")
        return cls(
            run_id=run_id,
            family=_text(payload["family"], "run.family"),
            landscape=_text(payload["landscape"], "run.landscape"),
            sampling=sampling,
            model=_text(payload["model"], "run.model"),
            score=_text(payload["score"], "run.score"),
            score_path=_relative_path(payload["score_path"], "run.score_path"),
            feature_path=_relative_path(payload["feature_path"], "run.feature_path"),
            score_higher_is_better=payload["score_higher_is_better"],
            external_value_column=_text(
                payload["external_value_column"],
                "run.external_value_column",
            ),
            external_rank_column=_text(
                payload["external_rank_column"],
                "run.external_rank_column",
            ),
            external_higher_is_better=payload["external_higher_is_better"],
            min_reference_samples=_integer(
                payload["min_reference_samples"],
                "run.min_reference_samples",
                minimum=1,
            ),
            variant_chunk_size=_integer(
                payload["variant_chunk_size"],
                "run.variant_chunk_size",
                minimum=1,
            ),
        )

    def identity(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in IDENTITY_FIELDS}


@dataclass(frozen=True)
class RunConfig:
    """Validated settings, runs, and the root used to resolve data paths."""

    root: Path
    settings: AnalysisSettings
    runs: tuple[RunSpec, ...]
    data_root: Path | None = None

    def input_path(self, relative_path: PurePosixPath) -> Path:
        """Resolve an input without changing its recorded relative path."""
        if self.data_root is None:
            return self.root / relative_path
        if relative_path.is_relative_to("data/downloads"):
            relative_path = relative_path.relative_to("data/downloads")
        return self.data_root / relative_path


@dataclass(frozen=True)
class RunOutcome:
    """One executed or reused result path."""

    run_id: str
    state: str
    result_path: Path


def load_run_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> RunConfig:
    """Load a strict run file with an optional external input directory."""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"settings", "runs"}:
        raise ValueError("Run configuration must contain only settings and runs.")
    settings = AnalysisSettings.from_dict(payload["settings"])
    run_payload = payload["runs"]
    if not isinstance(run_payload, list) or not run_payload:
        raise ValueError("Run configuration must contain a non-empty runs list.")
    runs = tuple(RunSpec.from_dict(item) for item in run_payload)
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Run configuration contains duplicate run IDs.")
    identities = [tuple(run.identity().values()) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Run configuration contains duplicate scientific identities.")
    score_paths = [run.score_path for run in runs]
    if len(set(score_paths)) != len(score_paths):
        raise ValueError("Every run must have a distinct score path.")
    return RunConfig(
        root=config_path.parent.parent,
        settings=settings,
        runs=runs,
        data_root=Path(data_root).expanduser().resolve() if data_root is not None else None,
    )


def select_runs(
    runs: Sequence[RunSpec],
    *,
    families: Iterable[str] = (),
    sampling_methods: Iterable[str] = (),
    run_ids: Iterable[str] = (),
) -> tuple[RunSpec, ...]:
    """Filter runs while rejecting misspelled requested values."""
    requested_families = set(families)
    requested_sampling = set(sampling_methods)
    requested_ids = set(run_ids)
    available_families = {run.family for run in runs}
    available_sampling = {run.sampling for run in runs}
    available_ids = {run.run_id for run in runs}
    for label, requested, available in (
        ("families", requested_families, available_families),
        ("sampling methods", requested_sampling, available_sampling),
        ("run IDs", requested_ids, available_ids),
    ):
        missing = requested - available
        if missing:
            raise ValueError(f"Unknown requested {label}: {sorted(missing)}")
    selected = tuple(
        run
        for run in runs
        if (not requested_families or run.family in requested_families)
        and (not requested_sampling or run.sampling in requested_sampling)
        and (not requested_ids or run.run_id in requested_ids)
    )
    if not selected:
        raise ValueError("No runs match the requested filters.")
    return selected


def _analysis_metadata(config: RunConfig, run: RunSpec) -> dict[str, object]:
    settings = config.settings
    return {
        **run.identity(),
        "score_path": run.score_path.as_posix(),
        "feature_path": run.feature_path.as_posix(),
        "k_values": list(settings.k_values),
        "interval": list(settings.interval),
        "seed": settings.seed,
        "external_value_column": run.external_value_column,
        "external_higher_is_better": run.external_higher_is_better,
    }


def _result_matches(
    result_path: Path,
    *,
    config: RunConfig,
    run: RunSpec,
    scores: SampledScores,
) -> bool:
    if not result_path.is_dir():
        return False
    try:
        result = RankResult.load(result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    try:
        expected_metadata = {
            **_analysis_metadata(config, run),
            "higher_is_better": run.score_higher_is_better,
            "retained_k": config.settings.retain_k,
            "decision_k": config.settings.decision_k,
            "external_top_n": config.settings.external_top_n,
            "model_top_n": config.settings.model_top_n,
            "challenger_probability": config.settings.challenger_probability,
            "variant_chunk_size": run.variant_chunk_size,
            "min_reference_samples": run.min_reference_samples,
            "external_rank_column": run.external_rank_column,
        }
        if any(result.metadata.get(name) != value for name, value in expected_metadata.items()):
            return False
        if not result.variants.equals(scores.variants):
            return False
        result_source_ids = result.stability["source_sample_id"].values
        score_source_ids = scores.dataset["source_sample_id"].values
        source_positions = pd.Index(score_source_ids).get_indexer(result_source_ids)
        return (
            result.metadata.get("n_source_samples") == len(score_source_ids)
            and np.all(source_positions >= 0)
            and np.all(np.diff(source_positions) > 0)
        )
    finally:
        result.close()


def _publish_result(result: RankResult, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        result.save(temporary)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to replace existing result: {path}")
            shutil.rmtree(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def execute_runs(
    config: RunConfig,
    runs: Sequence[RunSpec],
    *,
    result_root: str | Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[RunOutcome, ...]:
    """Validate inputs, reuse matching results, and execute all remaining runs."""
    output_root = Path(result_root).resolve()
    outcomes = []
    for position, run in enumerate(runs, start=1):
        score_path = config.input_path(run.score_path)
        result_path = output_root / run.sampling / run.run_id
        scores = SampledScores.load(score_path)
        try:
            for column in (run.external_value_column, run.external_rank_column):
                if column not in scores.variants:
                    raise ValueError(f"{run.run_id}: score variants are missing {column!r}.")
            matches = _result_matches(
                result_path,
                config=config,
                run=run,
                scores=scores,
            )
            if matches and not overwrite:
                state = "reuse"
            elif result_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Existing result does not match run {run.run_id}: {result_path}. "
                    "Use --overwrite only after reviewing it."
                )
            else:
                state = "run"
            print(
                f"[{position}/{len(runs)}] {state}: {run.run_id}: "
                f"{score_path} -> {result_path}",
                flush=True,
            )
            if state == "run" and not dry_run:
                result = analyze_scores(
                    scores,
                    metadata=_analysis_metadata(config, run),
                    retain_k=config.settings.retain_k,
                    decision_k=config.settings.decision_k,
                    external_top_n=config.settings.external_top_n,
                    model_top_n=config.settings.model_top_n,
                    external_rank_column=run.external_rank_column,
                    challenger_probability=config.settings.challenger_probability,
                    higher_is_better=run.score_higher_is_better,
                    variant_chunk_size=run.variant_chunk_size,
                    min_reference_samples=run.min_reference_samples,
                )
                _publish_result(result, result_path, overwrite=overwrite)
            outcomes.append(RunOutcome(run.run_id, state, result_path))
        finally:
            scores.close()
    return tuple(outcomes)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repository_root / "analysis/runs.json")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing scores/ and features/; defaults to the configured paths.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=repository_root / "analysis/results",
    )
    parser.add_argument("--table-root", type=Path, default=None)
    parser.add_argument("--family", nargs="+", default=())
    parser.add_argument("--sampling", choices=sorted(SAMPLING_METHODS), nargs="+", default=())
    parser.add_argument("--run-id", nargs="+", default=())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run selected analyses from the command line."""
    args = _parse_args(argv)
    config = load_run_config(args.config, data_root=args.data_root)
    selected = select_runs(
        config.runs,
        families=args.family,
        sampling_methods=args.sampling,
        run_ids=args.run_id,
    )
    outcomes = execute_runs(
        config,
        selected,
        result_root=args.result_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        from .tables import build_tables_for_runs, write_tables

        table_root = args.table_root or config.root / "analysis/tables"
        tables = build_tables_for_runs(
            config,
            selected,
            result_root=args.result_root,
        )
        for output in write_tables(tables, table_root):
            print(f"Saved tables to {output}", flush=True)
    counts = Counter(outcome.state for outcome in outcomes)
    suffix = " (dry run)" if args.dry_run else ""
    print(
        f"Completed {len(outcomes)} run(s): "
        f"{counts['run']} run, {counts['reuse']} reused{suffix}.",
        flush=True,
    )
