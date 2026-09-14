"""Validate generated Section 4 mechanism tables before figure composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dms_fitness_stability.analysis_manifest import load_analysis_manifest
from dms_fitness_stability.identity import AnalysisIdentity

HERE = Path(__file__).resolve().parent
TABLE_NAMES = (
    "mechanism_inventory",
    "mechanism_variants",
    "mechanism_runs",
    "mechanism_effects",
    "reference_diagnostics",
    "reference_category_sensitivity",
    "mechanism_diagnostics",
)


def _manifest_identities(path: Path) -> list[AnalysisIdentity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 1:
        return [AnalysisIdentity(**entry) for entry in payload["analyses"]]
    if payload.get("schema_version") == 2:
        return [
            spec.score_identity(score)
            for spec in load_analysis_manifest(path)
            for score in spec.scores
        ]
    raise ValueError("Analysis manifest schema_version must be 1 or 2.")


def validate_mechanism_tables(
    table_dir: Path,
    *,
    manifest_path: Path = HERE / "analyses_hmc.json",
    selected_analysis_ids: set[str] | None = None,
) -> dict[str, int]:
    """Validate identities, cohorts, finite features, and observation flags."""
    tables = {
        name: pd.read_csv(table_dir / f"{name}.csv", low_memory=False)
        for name in TABLE_NAMES
    }
    inventory = tables["mechanism_inventory"]
    variants = tables["mechanism_variants"]
    runs = tables["mechanism_runs"]
    effects = tables["mechanism_effects"]
    reference = tables["reference_diagnostics"]
    categories = tables["reference_category_sensitivity"]
    diagnostics = tables["mechanism_diagnostics"]
    functionality_path = next(
        (
            path
            for path in (
                table_dir / "functionality_scores.csv.gz",
                table_dir / "functionality_scores.csv",
            )
            if path.is_file()
        ),
        None,
    )
    if functionality_path is None:
        raise FileNotFoundError("Missing functionality_scores.csv.gz or .csv.")
    functionality = pd.read_csv(functionality_path, low_memory=False)
    identities = _manifest_identities(manifest_path)
    if selected_analysis_ids is not None:
        identities = [
            identity
            for identity in identities
            if identity.analysis_id in selected_analysis_ids
        ]
        missing = selected_analysis_ids - {
            identity.analysis_id for identity in identities
        }
        if missing:
            raise ValueError(f"Unknown selected analysis IDs: {sorted(missing)}")
    expected_analysis_ids = {identity.analysis_id for identity in identities}
    expected_landscapes = {identity.landscape for identity in identities}
    expected_inference_methods = {
        identity.inference_method for identity in identities
    }
    if (
        inventory["landscape"].nunique() != len(expected_landscapes)
        or len(inventory) != len(expected_landscapes)
        or set(inventory["landscape"]) != expected_landscapes
    ):
        raise ValueError(
            "Mechanism inventory landscapes differ from the selected manifest."
        )
    if (
        runs["analysis_id"].nunique() != len(expected_analysis_ids)
        or len(runs) != len(expected_analysis_ids)
        or set(runs["analysis_id"]) != expected_analysis_ids
    ):
        raise ValueError("Mechanism runs differ from the selected manifest.")
    if set(runs["analysis_id"]) != set(variants["analysis_id"]):
        raise ValueError("Mechanism run and variant analysis IDs differ.")
    if set(runs["inference_method"]) != expected_inference_methods or set(
        variants["inference_method"]
    ) != expected_inference_methods:
        raise ValueError("Mechanism table inference methods differ from the manifest.")
    for name, table in (
        ("reference diagnostics", reference),
        ("mechanism diagnostics", diagnostics),
        ("functionality scores", functionality),
    ):
        if set(table["analysis_id"]) != expected_analysis_ids:
            raise ValueError(f"{name.title()} differ from the selected manifest.")
    model_counts = variants.groupby("analysis_id")["is_model_top_100"].sum()
    if not model_counts.eq(100).all():
        raise ValueError("Every analysis must contain exactly 100 model-top variants.")
    primary_counts = variants.groupby("analysis_id")["is_primary_cohort"].sum()
    if not primary_counts.ge(100).all():
        raise ValueError("Every primary cohort must contain at least the model top 100.")
    if (
        "is_source_control" in variants
        and variants["is_source_control"].fillna(False).astype(bool).any()
    ):
        raise ValueError("Source controls must not enter mechanism variant cohorts.")
    finite_variant_columns = (
        "mechanism_log_input_support",
        "normalized_score_interval_width",
        "score_crowding",
        "rank_interval_width",
        "top_100_inclusion_probability",
    )
    if not np.isfinite(variants[list(finite_variant_columns)]).all().all():
        raise ValueError("Primary mechanism variant features must be finite.")
    if "n_finite_samples" not in variants:
        raise ValueError("Mechanism variants must record finite sampled-score counts.")
    aligned = variants.merge(
        runs[["analysis_id", "n_samples", "n_variants"]],
        on="analysis_id",
        how="left",
        validate="many_to_one",
    )
    if (
        (aligned["n_finite_samples"] < 2).any()
        or (aligned["n_finite_samples"] > aligned["n_samples"]).any()
    ):
        raise ValueError("Mechanism finite-sample counts lie outside 2..n_samples.")
    if not np.allclose(
        aligned["normalized_score_interval_width"],
        aligned["score_interval_width"] / aligned["reference_score_iqr"],
    ):
        raise ValueError("Normalized score widths do not equal raw width / run IQR.")
    if not np.allclose(
        aligned["rank_interval_width"],
        aligned["rank_upper"] - aligned["rank_lower"],
    ):
        raise ValueError("Rank widths do not equal upper - lower quantiles.")
    if (aligned["rank_lower"] < 1).any() or (
        aligned["rank_upper"] > aligned["n_variants"]
    ).any():
        raise ValueError("Sampled-rank intervals lie outside the ranking-candidate universe.")
    finite_run_columns = (
        "boundary_score_gap",
        "boundary_local_score_spacing",
        "boundary_score_uncertainty",
        "uncertainty_to_spacing_ratio",
        "top_k_instability",
    )
    if not np.isfinite(runs[list(finite_run_columns)]).all().all():
        raise ValueError("Mechanism run features must be finite.")
    if (
        "n_reported_context_omitted_low_support" not in runs
        or (runs["n_reported_context_omitted_low_support"] < 0).any()
    ):
        raise ValueError(
            "Mechanism runs must record nonnegative low-support context omissions."
        )
    expected_missing = {
        identity.landscape
        for identity in identities
        if identity.protein_family == "nuc"
        or identity.protein_family == "k50"
        or identity.landscape in {"trpb-libA", "trpb-libB", "trpb-libC"}
    }
    missing_discordance = set(
        inventory.loc[~inventory["discordance_available"].astype(bool), "landscape"]
    )
    if missing_discordance != expected_missing:
        raise ValueError(
            "Replicate discordance must be unavailable only for NucB, K50, "
            "and TrpB libA-C, "
            f"not {sorted(missing_discordance)}."
        )
    analysis_effects = effects[effects["scope"] == "analysis_run"]
    expected_effects = {
        (analysis_id, stage)
        for analysis_id in runs["analysis_id"]
        for stage in ("score_precision", "rank_propagation")
    }
    observed_effects = set(zip(analysis_effects["analysis_id"], analysis_effects["stage"]))
    if observed_effects != expected_effects:
        raise ValueError("Missing or duplicate analysis-run effect estimates.")

    if len(reference) != len(expected_analysis_ids) or reference["analysis_id"].duplicated().any():
        raise ValueError("Reference diagnostics must contain one row per analysis.")
    if not np.array_equal(
        reference["n_excluded_from_ranking"].to_numpy(),
        reference["n_fitted_variants"].to_numpy()
        - reference["n_ranking_candidates"].to_numpy(),
    ):
        raise ValueError("Reference fitted, candidate, and excluded counts disagree.")
    if (reference["n_ranking_candidates"] < reference["top_k"]).any():
        raise ValueError("Reference top-K exceeds the ranking-candidate count.")
    for count_column, fraction_column in (
        ("model_reported_top_k_intersection", "model_reported_top_k_fraction"),
        (
            "model_reversed_reported_top_k_intersection",
            "model_reversed_reported_top_k_fraction",
        ),
    ):
        identifiable = reference.get(
            "reference_top_k_identifiable",
            pd.Series(True, index=reference.index),
        ).astype(bool)
        if not np.allclose(
            reference.loc[identifiable, fraction_column],
            reference.loc[identifiable, count_column]
            / reference.loc[identifiable, "top_k"],
        ):
            raise ValueError(f"{fraction_column} does not match its intersection count.")
        unavailable = reference.loc[
            ~identifiable,
            [count_column, fraction_column],
        ]
        if not unavailable.isna().all().all():
            raise ValueError("Non-identifiable reference top-K metrics must be missing.")
    correlation_columns = [
        column
        for column in [*reference.columns, *diagnostics.columns]
        if "spearman" in column or "kendall" in column
    ]
    for column in correlation_columns:
        source = reference[column] if column in reference else diagnostics[column]
        finite = source[np.isfinite(source)]
        if not finite.between(-1, 1).all():
            raise ValueError(f"{column} must lie between -1 and 1.")

    expected_diagnostics = {
        (analysis_id, stratum)
        for analysis_id in expected_analysis_ids
        for stratum in ("primary", "model_top_k", "challengers")
    }
    observed_diagnostics = set(zip(diagnostics["analysis_id"], diagnostics["stratum"]))
    if observed_diagnostics != expected_diagnostics:
        raise ValueError("Mechanism diagnostics must contain all three strata per run.")
    if (diagnostics["source_control_rows"] != 0).any():
        raise ValueError("Source controls must not enter mechanism diagnostics.")
    diagnostic_counts = diagnostics.pivot(
        index="analysis_id",
        columns="stratum",
        values="n_variants",
    )
    challenger_counts = runs.set_index("analysis_id")["challenger_count_q_ge_001"]
    if (
        not diagnostic_counts["model_top_k"].eq(100).all()
        or not diagnostic_counts["challengers"].eq(challenger_counts).all()
        or not diagnostic_counts["primary"].eq(
            diagnostic_counts["model_top_k"] + diagnostic_counts["challengers"]
        ).all()
    ):
        raise ValueError("Mechanism diagnostic stratum counts disagree.")

    expected_category_ids = {
        identity.analysis_id for identity in identities if identity.protein_family == "nuc"
    }
    if set(categories["analysis_id"]) != expected_category_ids:
        raise ValueError("Categorical sensitivity rows must cover exactly the NucB runs.")
    if expected_category_ids:
        category_totals = categories.groupby(["analysis_id", "cohort"], as_index=False).agg(
            count=("count", "sum"),
            fraction=("fraction", "sum"),
            top_k=("top_k", "first"),
        )
        expected_category_cohorts = {
            (analysis_id, cohort)
            for analysis_id in expected_category_ids
            for cohort in ("model_top_k", "reported_top_k")
        }
        if set(zip(category_totals["analysis_id"], category_totals["cohort"])) != (
            expected_category_cohorts
        ):
            raise ValueError("Categorical sensitivity cohorts are incomplete.")
        if (
            not category_totals["count"].eq(category_totals["top_k"]).all()
            or not np.allclose(category_totals["fraction"], 1.0)
        ):
            raise ValueError("Categorical sensitivity counts or fractions are invalid.")

    if functionality.duplicated(["analysis_id", "variant_index"]).any():
        raise ValueError("Functionality scores contain duplicate candidate rows.")
    functionality_counts = functionality.groupby("analysis_id").size()
    expected_candidate_counts = reference.set_index("analysis_id")[
        "n_score_supported_candidates"
    ]
    if not functionality_counts.eq(expected_candidate_counts).all():
        raise ValueError("Functionality score counts differ from ranking-candidate counts.")
    for analysis_id, rows in functionality.groupby("analysis_id", sort=False):
        expected_ranks = np.arange(1, len(rows) + 1)
        if not np.array_equal(
            rows.sort_values("model_reference_rank")["model_reference_rank"].to_numpy(),
            expected_ranks,
        ):
            raise ValueError(f"{analysis_id}: functionality ranks are not 1..N.")
    return {
        "landscapes": len(inventory),
        "analysis_runs": len(runs),
        "variant_rows": len(variants),
        "primary_variant_rows": int(variants["is_primary_cohort"].sum()),
        "effect_rows": len(effects),
        "reference_rows": len(reference),
        "mechanism_diagnostic_rows": len(diagnostics),
        "category_rows": len(categories),
        "functionality_rows": len(functionality),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=HERE / "output/hmc/tables",
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "analyses_hmc.json")
    parser.add_argument("--analysis-id", nargs="+", default=None)
    args = parser.parse_args()
    summary = validate_mechanism_tables(
        args.table_dir,
        manifest_path=args.manifest,
        selected_analysis_ids=(
            set(args.analysis_id) if args.analysis_id is not None else None
        ),
    )
    print("Mechanism tables valid:")
    for name, value in summary.items():
        print(f"  {name}: {value:,}")


if __name__ == "__main__":
    main()
