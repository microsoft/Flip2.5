"""Export a source-linked inventory of the manuscript's saved model fits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SOURCE_REVISION = "09c439f675ead70f851d814242f153ecb7712113"
SOURCE_FILES = (
    "flipv3/base_models.py",
    "flipv3/pdz3_models.py",
    "flipv3/trpb_models.py",
    "flipv3/nuclease_models.py",
    "flipv3/constants.py",
    "scistanpy/defaults.py",
    "scistanpy/model/components/parameters.py",
    "scistanpy/model/components/transformations/transformed_parameters.py",
    "scistanpy/model/nn_module.py",
    "scistanpy/model/results/hmc.py",
    "scistanpy/model/results/mle.py",
    "scistanpy/pipelines/mcmc_flip.py",
    "scistanpy/pipelines/mle_flip.py",
    "amlt_configs/cmds.txt",
    "amlt_configs/cmds-mle.txt",
)


def parse_command(command: str, defaults: dict[str, float]) -> dict[str, object]:
    """Parse one numbered job command without importing the fitting package."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("dataset", "subset", "fitness_dist", "growth_curve"):
        parser.add_argument(f"--{name}")
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=float, default=value)
    parser.add_argument("--experimental_dist", default="lognormal")
    parser.add_argument("--centered", dest="noncentered", action="store_false", default=True)
    for name, value in (("n_chains", 8), ("n_warmup", 2000), ("n_samples", 2000),
                        ("seed", 1025), ("n_epochs", 100000), ("early_stopping", 10)):
        parser.add_argument(f"--{name}", type=int, default=value)
    parser.add_argument("--lr", type=float, default=0.001)
    return vars(parser.parse_args(shlex.split(command)))


def read_defaults(path: Path) -> dict[str, float]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_HYPERPARAMS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing DEFAULT_HYPERPARAMS in source.")


def artifact_summary(path: Path, score_variable: str) -> dict[str, object]:
    """Read headers and native-score diagnostics, never a full posterior matrix."""
    result: dict[str, object] = {"artifact_available": path.is_file()}
    if not path.is_file():
        return result
    with h5py.File(path, "r") as handle:
        for name in ("num_chains", "num_samples", "num_warmup", "seed", "max_depth",
                     "stan_version_major", "stan_version_minor", "stan_version_patch"):
            if name in handle.attrs:
                result[f"artifact_{name}"] = int(np.asarray(handle.attrs[name]).item())
        target = handle.get(f"posterior/{score_variable}")
        if isinstance(target, h5py.Dataset):
            result["posterior_shape"] = list(target.shape)
            result["artifact_draw_count"] = int(np.prod(target.shape[:2]))
        if "sample_stats/divergent__" in handle:
            divergent = handle["sample_stats/divergent__"][...]
            result["divergences"] = int(np.count_nonzero(divergent))
            result["sample_stats_count"] = int(divergent.size)
        if "sample_stats/treedepth__" in handle and "artifact_max_depth" in result:
            result["tree_depth_hits"] = int(np.count_nonzero(
                handle["sample_stats/treedepth__"][...] >= result["artifact_max_depth"]
            ))
        if "sample_stats/energy__" in handle:
            energy = np.asarray(handle["sample_stats/energy__"], dtype=float)
            if energy.ndim == 2 and energy.shape[1] > 1:
                variance = np.var(energy, axis=1)
                bfmi = np.divide(
                    np.mean(np.diff(energy, axis=1) ** 2, axis=1), variance,
                    out=np.full_like(variance, np.nan), where=variance > 0,
                )
                result["bfmi_by_chain"] = bfmi.tolist()
                result["min_bfmi"] = float(np.nanmin(bfmi))
        group = handle.get("variable_diagnostic_stats")
        result["native_score_diagnostics_available"] = (
            isinstance(group, h5py.Group) and score_variable in group and "metric" in group
        )
        if result["native_score_diagnostics_available"]:
            metrics = [value.decode() if isinstance(value, bytes) else str(value)
                       for value in group["metric"][...]]
            diagnostics = np.asarray(group[score_variable], dtype=float)
            if diagnostics.shape[0] != len(metrics):
                raise ValueError(f"Unexpected diagnostic metric axis: {path}.")
            for metric, operation in (("r_hat", np.max), ("ess_bulk", np.min),
                                      ("ess_tail", np.min)):
                if metric not in metrics:
                    continue
                values = diagnostics[metrics.index(metric)].ravel()
                finite = values[np.isfinite(values)]
                result[f"{metric}_nonfinite"] = int(values.size - finite.size)
                result[f"{metric}_extreme"] = float(operation(finite)) if len(finite) else None
                if metric == "r_hat":
                    result["native_score_rhat_ge_1_01"] = int((finite >= 1.01).sum())
    return result


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3g}"
    return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _longtable(caption: str, label: str, headers: list[str], rows: list[list[object]]) -> str:
    header = " & ".join(headers) + r" \\"
    lines = [r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}[c]{" + "l" * len(headers) + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             r"\toprule", header, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    lines.extend(" & ".join(_cell(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def format_fit_inventory(fits: pd.DataFrame) -> str:
    """Render fit records without turning output formats into diagnostic status."""
    hmc = fits[fits["kind"].eq("hmc")]
    optimized = fits[fits["kind"].eq("optimized")]
    codes = {"exponential-exponential": "EE", "lomax-exponential": "LE",
             "gamma-exponential": "GE", "exponential-sigmoid": "ES",
             "lomax-sigmoid": "LS", "gamma-sigmoid": "GS", "exponential": "E", "lomax": "L"}
    hmc_rows = [[row.landscape, codes[row.model], row.job_id, row.artifact_num_chains,
                 row.artifact_num_warmup, row.artifact_num_samples,
                 row.divergences, row.tree_depth_hits,
                 f"{row.r_hat_extreme:.4f}" if np.isfinite(row.r_hat_extreme) else None,
                 row.min_bfmi]
                for row in hmc.itertuples()]
    optimized_rows = [[row.landscape, codes[row.model], row.job_id, row.command_alpha,
                       row.command_experimental_dist,
                       "C" if not row.command_noncentered else "NC",
                       "yes" if row.artifact_available else "no"]
                      for row in optimized.itertuples()]
    tex = _longtable(
        "All posterior fits used by the parameter and posterior-count analyses. "
        "All fits underwent sampler diagnostics during fitting. "
        "Iterations are per chain from artifact metadata. The maximum stored score-parameter "
        "R-hat uses all fitted variants, not only the decision cohort; -- means the value "
        "is not stored in the retained output, not that diagnostics were omitted. "
        "Depth counts samples reaching the recorded maximum tree depth. "
        "BFMI is recomputed from energy traces.",
        "tab:hmc-fits", ["Landscape", "Model", "Job", "Chains", "Warmup", "Draws",
                         "Diverg.", "Depth", r"$\max\hat R$", "BFMI"], hmc_rows,
    )
    return tex + _longtable(
        "All optimized fits used by the count-resampling analyses. "
        "Concentration, observation-family option, and centered (C) or noncentered (NC) "
        "flag come from the recorded job commands; flags may not affect every model class. "
        "Available indicates whether the original fitted artifact remains locally accessible, "
        "not whether its prepared scores are available. Model codes are defined in the text.",
        "tab:optimized-fits", ["Landscape", "Model", "Job", r"$\alpha_0$",
                               "Variation option", "Flag", "Available"], optimized_rows,
    )


def export_inventory(source_root: Path, scistanpy_root: Path, data_root: Path,
                     output_dir: Path, config: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(scistanpy_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected SciStanPy revision {SOURCE_REVISION}, found {revision}.")
    changed = subprocess.run(
        ["git", "-C", str(scistanpy_root), "diff", "--name-only", "HEAD", "--", *SOURCE_FILES],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if changed:
        raise ValueError(f"The model or fitting sources have local edits: {changed}.")
    defaults = read_defaults(scistanpy_root / "flipv3/constants.py")
    audit_path = source_root / "data/fit_audit.json"
    audit = {entry["job_id"]: entry for entry in json.loads(audit_path.read_text())}
    canonical = json.loads(config.read_text())["runs"]
    command_paths = {
        "hmc": scistanpy_root / "amlt_configs/cmds.txt",
        "optimized": scistanpy_root / "amlt_configs/cmds-mle.txt",
    }
    source_entries = {}
    for kind, filename in (("hmc", "analyses_hmc.json"), ("optimized", "analyses_mle.json")):
        entries = json.loads((source_root / "examples/paper" / filename).read_text())["analyses"]
        for entry in entries:
            if entry["protein_family"] not in {"pdz", "trpb", "nuc"}:
                continue
            key = (kind, entry["protein_family"], entry["landscape"], entry["model_spec"])
            if key in source_entries:
                raise ValueError(f"Duplicate source identity: {key}.")
            source_entries[key] = entry
    fit_rows = {}
    analysis_rows = []
    for run in canonical:
        kind = "optimized" if run["sampling"] == "mle_predictive" else "hmc"
        key = (kind, run["family"], run["landscape"], run["model"])
        entry = source_entries[key]
        if key not in fit_rows:
            job = entry["job_id"]
            command = command_paths[kind].read_text().splitlines()[job - 1]
            options = parse_command(command, defaults)
            expected_model = options["fitness_dist"]
            if options["growth_curve"]:
                expected_model += "-" + options["growth_curve"]
            if (options["dataset"] != run["family"] or
                    f"{options['dataset']}-{options['subset']}" != run["landscape"] or
                    expected_model != run["model"]):
                raise ValueError(f"Job {job} command does not match {key}.")
            artifact_kind = entry["artifact_kind"]
            records = audit[job].get(artifact_kind, [])
            validated = [record for record in records if record.get("status") == "validated"]
            if not validated:
                raise ValueError(f"No validated source record for job {job}, {artifact_kind}.")
            available = [record for record in validated if Path(record["path"]).is_file()]
            record = (available or validated)[0]
            path = Path(record["path"])
            row = {
                "kind": kind, "family": run["family"], "landscape": run["landscape"],
                "model": run["model"], "fit_id": entry["fit_id"], "job_id": job,
                "artifact_kind": artifact_kind, "artifact_path": str(path),
                "score_variable": audit[job]["score_variable"], "command": command,
                "source_effective_exp_beta": defaults["exp_beta"],
                "ignored_exp_beta_override": options["exp_beta"] != defaults["exp_beta"],
                **{f"command_{name}": value for name, value in options.items()},
                **artifact_summary(path, audit[job]["score_variable"]),
            }
            for option, attribute in (("n_chains", "num_chains"), ("n_samples", "num_samples"),
                                      ("n_warmup", "num_warmup"), ("seed", "seed")):
                actual = row.get(f"artifact_{attribute}")
                row[f"matches_command_{option}"] = (
                    None if actual is None else actual == options[option]
                )
            fit_rows[key] = row
        metadata_path = data_root / "results" / run["sampling"] / run["run_id"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["run_id"] != run["run_id"]:
            raise ValueError(f"Result identity mismatch: {metadata_path}.")
        analysis_rows.append({
            "run_id": run["run_id"], "sampling": run["sampling"],
            "fit_id": entry["fit_id"], "job_id": entry["job_id"],
            "n_candidates": metadata["n_candidates"], "n_samples": metadata["n_samples"],
            "n_source_samples": metadata["n_source_samples"],
            "n_excluded_samples": metadata["n_excluded_samples"],
        })
    if len(fit_rows) != len(source_entries):
        raise ValueError("Not every curated fit was matched to the current analysis manifest.")
    output_dir.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(fit_rows.values()).sort_values(["kind", "family", "landscape", "model"])
    fits.to_csv(output_dir / "fit_inventory.csv", index=False)
    pd.DataFrame(analysis_rows).to_csv(output_dir / "analysis_fit_mapping.csv", index=False)
    hmc = fits[fits["kind"].eq("hmc")]
    optimized = fits[fits["kind"].eq("optimized")]
    (output_dir / "fit_inventory.tex").write_text(format_fit_inventory(fits))
    hashes = {}
    for relative in SOURCE_FILES:
        with (scistanpy_root / relative).open("rb") as handle:
            hashes[relative] = hashlib.file_digest(handle, "sha256").hexdigest()
    mismatch = hmc[[column for column in hmc if column.startswith("matches_command_")]].eq(False)
    provenance = {
        "source_revision": SOURCE_REVISION, "source_file_sha256": hashes,
        "committed_loader_sha256": hashlib.sha256(subprocess.run(
            ["git", "-C", str(scistanpy_root), "show", "HEAD:flipv3/flip_dsets.py"],
            check=True, capture_output=True,
        ).stdout).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "fit_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "fit_count": len(fits), "analysis_count": len(analysis_rows),
        "hmc_count": len(hmc), "optimized_count": len(optimized),
        "hmc_command_metadata_mismatch_jobs": hmc.loc[mismatch.any(axis=1), "job_id"].tolist(),
    }
    (output_dir / "fit_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(fits)} fits and {len(analysis_rows)} analysis mappings.")
    print("Command/artifact HMC mismatches:", provenance["hmc_command_metadata_mismatch_jobs"])
    print("HMC fits with divergences:", int(hmc["divergences"].gt(0).sum()))
    print(
        "HMC fits with stored score diagnostics:",
        int(hmc["native_score_diagnostics_available"].sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scistanpy-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--config", type=Path, default=Path("analysis/runs.json"))
    arguments = parser.parse_args()
    export_inventory(arguments.source_root, arguments.scistanpy_root, arguments.data_root,
                     arguments.output_dir, arguments.config)


if __name__ == "__main__":
    main()
