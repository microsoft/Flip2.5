"""Run full-landscape TrpB stability analyses from diagnosed HMC fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.adapters.trpb import (
    TRPB_LIBRARIES,
    TRPB_SCORE_NAMES,
    load_trpb_landscape,
    scan_trpb_hmc,
)
from dms_fitness_stability.identity import AnalysisIdentity

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def _fit_specs() -> list[dict[str, str]]:
    manifest = json.loads((HERE / "fit_jobs.json").read_text(encoding="utf-8"))
    return manifest["fits"]


def parse_args() -> argparse.Namespace:
    specs = _fit_specs()
    models = sorted(
        {
            f"{spec['growth_rate_prior']}-{spec['growth_curve']}"
            for spec in specs
        }
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        nargs="+",
        choices=TRPB_LIBRARIES,
        default=TRPB_LIBRARIES,
    )
    parser.add_argument("--model", nargs="+", choices=models, default=models)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data-root>/trpb/results.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Posterior samples per fit. Default: all available samples.",
    )
    parser.add_argument("--retain-k", type=int, default=1000)
    parser.add_argument("--tracked-top-n", type=int, default=1000)
    parser.add_argument("--tracked-model-top-n", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=(50, 100, 1000))
    parser.add_argument(
        "--score",
        nargs="+",
        choices=TRPB_SCORE_NAMES,
        default=TRPB_SCORE_NAMES,
    )
    parser.add_argument(
        "--output-layout",
        choices=("model", "analysis-id"),
        default="model",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = set(args.model)
    specs = [
        spec
        for spec in _fit_specs()
        if spec["library"] in args.library
        and f"{spec['growth_rate_prior']}-{spec['growth_curve']}" in selected_models
    ]
    if not specs:
        raise ValueError("No available fit matches the selected libraries and models.")

    trpb_root = args.data_root / "trpb"
    output_dir = args.output_dir or trpb_root / "results"
    landscapes = {}
    for spec in specs:
        library = spec["library"]
        if library not in landscapes:
            landscapes[library] = load_trpb_landscape(
                library,
                count_path=trpb_root / "raw" / "counts" / f"{library}.csv",
                reported_fitness_path=(
                    trpb_root / "raw" / "reported_fitness" / f"{library}.csv"
                ),
            )
        model = f"{spec['growth_rate_prior']}-{spec['growth_curve']}"
        fit_path = trpb_root / "fits" / library / spec["canonical_filename"]
        if not fit_path.is_file():
            raise FileNotFoundError(f"Missing canonical TrpB fit: {fit_path}")
        fit_id = spec.get("job", Path(spec["canonical_filename"]).stem)
        identity = AnalysisIdentity(
            protein_family="trpb",
            landscape=f"trpb-{library}",
            inference_method="hmc",
            model_spec=f"{spec['growth_rate_prior']}-{spec['growth_curve']}",
            fit_id=fit_id,
        )
        result_dir = (
            output_dir / identity.analysis_id
            if args.output_layout == "analysis-id"
            else output_dir / library / model
        )
        if args.skip_existing and (result_dir / "metadata.json").is_file():
            print(f"Skipping existing result {result_dir}", flush=True)
            continue
        result = scan_trpb_hmc(
            landscapes[library],
            fit_path,
            growth_rate_prior=spec["growth_rate_prior"],
            growth_curve=spec["growth_curve"],
            top_k=tuple(args.top_k),
            retain_k=args.retain_k,
            tracked_top_n=args.tracked_top_n,
            tracked_model_top_n=args.tracked_model_top_n,
            score_names=tuple(args.score),
            n_samples=args.n_samples,
            seed=args.seed,
            fit_id=fit_id,
        )
        output = result.save(result_dir)
        print(
            f"Saved {library} {model} to {output} "
            f"({result.metadata['elapsed_seconds'] / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
