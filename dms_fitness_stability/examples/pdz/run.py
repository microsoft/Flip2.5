"""Run all PDZ stability landscapes from full SciStanPy HMC fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dms_fitness_stability.adapters.pdz import (
    PDZ_LANDSCAPES,
    PDZ_SCORE_NAMES,
    PDZLandscape,
    load_pdz_cis,
    load_pdz_cript,
    load_pdz_trans,
    scan_pdz_hmc,
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
        "--landscape",
        nargs="+",
        choices=PDZ_LANDSCAPES,
        default=PDZ_LANDSCAPES,
    )
    parser.add_argument("--model", nargs="+", choices=models, default=models)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data-root>/pdz/results.",
    )
    parser.add_argument("--trans-1-hmc", type=Path, default=None)
    parser.add_argument("--trans-2-hmc", type=Path, default=None)
    parser.add_argument("--cis-hmc", type=Path, default=None)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Posterior samples per landscape. Default: all available samples.",
    )
    parser.add_argument("--retain-k", type=int, default=1000)
    parser.add_argument("--tracked-top-n", type=int, default=1000)
    parser.add_argument("--tracked-model-top-n", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=(50, 100, 1000))
    parser.add_argument(
        "--score",
        nargs="+",
        choices=PDZ_SCORE_NAMES,
        default=PDZ_SCORE_NAMES,
    )
    parser.add_argument(
        "--output-layout",
        choices=("model", "analysis-id"),
        default="model",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def _load_landscape(
    name: str,
    *,
    data_root: Path,
) -> PDZLandscape:
    if name in {"trans-1", "trans-2"}:
        pdz_root = data_root / "pdz_trans"
        raw_root = pdz_root / "raw"
        count_paths = {
            landscape: raw_root / f"{landscape}.tsv"
            for landscape in ("trans-1", "trans-2")
        }
        return load_pdz_trans(
            name,
            count_paths=count_paths,
            reported_fitness_path=raw_root / "GSE265816_pdz3cript_sub.csv.gz",
        )

    if name == "cis":
        pdz_root = data_root / "pdz_cis"
        return load_pdz_cis(
            pdz_root / "raw" / "cis_from_bruce_azure.tsv",
            reported_fitness_path=(
                pdz_root / "raw" / "GSE265816_cript2xa3_sub.csv.gz"
            ),
        )

    if name in {"cript-c", "cript-n"}:
        raw_root = data_root / "pdz_cis" / "raw"
        fitness_name = name.replace("-", "_")
        return load_pdz_cript(
            name,
            raw_root / f"{name}.tsv",
            reported_fitness_path=(
                raw_root / f"GSE265816_{fitness_name}_sub.csv.gz"
            ),
        )

    raise ValueError(f"Unknown PDZ landscape: {name!r}.")


def main() -> None:
    args = parse_args()
    selected_models = set(args.model)
    specs = [
        spec
        for spec in _fit_specs()
        if spec["landscape"] in args.landscape
        and f"{spec['growth_rate_prior']}-{spec['growth_curve']}" in selected_models
    ]
    if not specs:
        raise ValueError("No available fit matches the selected landscapes and models.")

    pdz_root = args.data_root / "pdz"
    output_dir = args.output_dir or pdz_root / "results"
    hmc_overrides = {
        ("trans-1", "exponential-exponential"): args.trans_1_hmc,
        ("trans-2", "exponential-exponential"): args.trans_2_hmc,
        ("cis", "exponential-exponential"): args.cis_hmc,
    }

    landscapes = {}
    for spec in specs:
        name = spec["landscape"]
        if name not in landscapes:
            print(f"Loading PDZ {name} counts and reported fitness", flush=True)
            landscapes[name] = _load_landscape(name, data_root=args.data_root)
        model = f"{spec['growth_rate_prior']}-{spec['growth_curve']}"
        hmc_path = hmc_overrides.get((name, model)) or (
            pdz_root / "fits" / name / spec["canonical_filename"]
        )
        if not hmc_path.is_file():
            raise FileNotFoundError(f"Missing canonical PDZ fit: {hmc_path}")
        fit_id = spec.get("job", Path(spec["canonical_filename"]).stem)
        identity = AnalysisIdentity(
            protein_family="pdz",
            landscape=f"pdz-{name}",
            inference_method="hmc",
            model_spec=f"{spec['growth_rate_prior']}-{spec['growth_curve']}",
            fit_id=fit_id,
        )
        result_dir = (
            output_dir / identity.analysis_id
            if args.output_layout == "analysis-id"
            else output_dir / name / model
        )
        if args.skip_existing and (result_dir / "metadata.json").is_file():
            print(f"Skipping existing result {result_dir}", flush=True)
            continue

        result = scan_pdz_hmc(
            landscapes[name],
            hmc_path,
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
            f"Saved PDZ {name} {model} to {output} "
            f"({result.metadata['elapsed_seconds'] / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
