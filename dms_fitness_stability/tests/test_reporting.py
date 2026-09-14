import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dms_fitness_stability import StabilityResult, build_paper_tables

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt


def _result() -> StabilityResult:
    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D"],
            "input_count": [100, 80, 60, 40],
            "output_count": [400, 240, 120, 40],
            "reported_fitness": [4.0, 3.0, 2.0, 1.0],
            "reported_rank": [1.0, 2.0, 3.0, 4.0],
        }
    )
    selected = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=np.int32,
    )
    model_ranks = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ],
        dtype=np.float32,
    )
    selection = xr.Dataset(
        {
            "variant_index": (
                ("score_name", "sample", "selection_rank"),
                selected[None],
            ),
            "score": (
                ("score_name", "sample", "selection_rank"),
                np.zeros((1, 4, 3), dtype=np.float32),
            ),
            "tracked_score": (
                ("score_name", "sample", "tracked_variant"),
                np.zeros((1, 4, 4), dtype=np.float32),
            ),
            "tracked_rank": (
                ("score_name", "sample", "tracked_variant"),
                np.array(
                    [[[1, 2, 3, 4], [1, 3, 2, 4], [2, 1, 3, 4], [3, 2, 1, 4]]],
                    dtype=np.float32,
                ),
            ),
            "model_reference_variant_index": (
                ("score_name", "model_reference_rank"),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
            "model_reference_score": (
                ("score_name", "model_reference_rank"),
                np.array([[4.0, 3.0, 2.0]], dtype=np.float32),
            ),
            "model_tracked_variant_index": (
                ("score_name", "model_tracked_variant"),
                np.array([[0, 1]], dtype=np.int32),
            ),
            "model_tracked_score": (
                ("score_name", "sample", "model_tracked_variant"),
                np.zeros((1, 4, 2), dtype=np.float32),
            ),
            "model_tracked_rank": (
                ("score_name", "sample", "model_tracked_variant"),
                model_ranks[None],
            ),
        },
        coords={
            "score_name": ["log_mean_growth"],
            "sample": np.arange(4),
            "selection_rank": np.arange(1, 4),
            "tracked_variant": np.arange(4),
            "tracked_variant_index": ("tracked_variant", np.arange(4)),
            "model_reference_rank": np.arange(1, 4),
            "model_tracked_variant": np.arange(2),
        },
    )
    summary = pd.DataFrame(
        {
            "score": ["log_mean_growth"] * 4,
            "k": [2] * 4,
            "variant_index": np.arange(4),
            "variant_id": variants["variant_id"],
            "reported_rank": variants["reported_rank"],
            "reported_rank_percentile": [0.5, 0.5, 0.5, 0.5],
            "is_tracked_reported_cohort": [True] * 4,
            "is_tracked_model_cohort": [True, True, False, False],
            "model_reference_rank": [1.0, 2.0, np.nan, np.nan],
            "model_reference_value": [4.0, 3.0, np.nan, np.nan],
            "posterior_rank_lower": [1.0, 1.0, 1.0, 4.0],
            "posterior_median_rank": [1.5, 2.0, 2.5, 4.0],
            "posterior_rank_upper": [3.0, 3.0, 3.0, 4.0],
        }
    )
    metadata = {
        "schema_version": 2,
        "analysis_id": "toy__landscape__mle-bootstrap__ratio__fit-1",
        "protein_family": "toy",
        "landscape": "landscape",
        "inference_method": "mle_bootstrap",
        "model_spec": "ratio",
        "fit_id": "fit-1",
        "scores": ["log_mean_growth"],
        "n_samples": 4,
        "n_variants": 4,
        "retained_k": 3,
        "tracked_reported_top_n": 4,
        "tracked_model_top_n": 2,
        "model_reference_retained_k": 3,
        "artifact_status": "validated",
        "warnings": [],
    }
    return StabilityResult(
        metadata=metadata,
        variants=variants,
        selection=selection,
        summary=summary,
        overlaps=pd.DataFrame(),
    )


def test_build_paper_tables_uses_exact_sample_overlap_formulas():
    tables = build_paper_tables(
        _result(),
        score="log_mean_growth",
        main_k=2,
        k_values=(1, 2),
        percentile_cohorts=(2, 4),
    )

    assert set(tables) == {
        "inventory",
        "rank_percentiles",
        "rank_agreement",
        "model_rank_intervals",
        "top_k_curves",
        "top_k_candidates",
    }
    assert all(
        set(table["score"]) == {"log_mean_growth"}
        for table in tables.values()
    )
    inventory = tables["inventory"].iloc[0]
    assert inventory["n_variants"] == 4
    assert inventory["n_fitted_variants"] == 4
    assert inventory["n_excluded_from_ranking"] == 0
    k2 = tables["top_k_curves"].set_index("k").loc[2]
    assert np.isclose(k2["sample_self_consistency"], 14 / 24)
    assert np.isclose(k2["sample_vs_model_reference"], 0.75)
    assert np.isclose(k2["sample_vs_reported"], 0.75)
    assert k2["candidate_count_q_gt_0"] == 3
    assert k2["candidate_count_q_gt_0_01"] == 3
    assert k2["candidate_count_q_gt_0_05"] == 3

    intervals = tables["model_rank_intervals"].sort_values("model_reference_rank")
    assert intervals["variant_id"].tolist() == ["A", "B"]
    np.testing.assert_allclose(intervals["rank_interval_width"], [1.85, 1.7])
    assert intervals["rank_interval_leaves_top_k"].tolist() == [True, True]

    percentiles = tables["rank_percentiles"]
    assert set(percentiles["cohort_top_n"]) == {2, 4}
    areas = percentiles.groupby("cohort_top_n")["ecdf_to_uniform_area"].first()
    assert np.isclose(areas.loc[2], 0.1875)
    assert np.isclose(areas.loc[4], 0.125)
    concordance = percentiles.groupby("cohort_top_n")[
        "reported_vs_posterior_median_rank_spearman"
    ].first()
    np.testing.assert_allclose(concordance, 1.0)


def test_paper_tables_exclude_ineligible_rows_from_reported_references():
    result = _result()
    result.variants["is_ranking_candidate"] = [True, True, True, False]
    result.variants.loc[3, ["reported_fitness", "reported_rank"]] = [99.0, 0.0]
    result.metadata["n_variants"] = 3

    tables = build_paper_tables(
        result,
        score="log_mean_growth",
        main_k=2,
        k_values=(1, 2),
        percentile_cohorts=(2, 4),
    )

    assert "D" not in set(tables["rank_percentiles"]["variant_id"])
    assert tables["top_k_curves"].set_index("k").loc[1, "sample_vs_reported"] == 0.5


def test_build_paper_tables_rejects_results_without_model_tracking():
    result = _result()
    result.metadata["tracked_model_top_n"] = 0

    with np.testing.assert_raises_regex(ValueError, "tracked_model_top_n"):
        build_paper_tables(result, score="log_mean_growth", main_k=2, k_values=(2,))


def test_paper_tables_omit_non_identifiable_reported_top_k():
    result = _result()
    result.metadata["reference_top_k_identifiable"] = False
    result.metadata["tracked_reported_top_n"] = 0
    result.selection = result.selection.isel(tracked_variant=slice(0, 0))

    tables = build_paper_tables(
        result,
        score="log_mean_growth",
        main_k=2,
        k_values=(1, 2),
        percentile_cohorts=(2, 4),
    )

    assert tables["rank_percentiles"].empty
    curves = tables["top_k_curves"]
    assert curves["sample_vs_reported"].isna().all()
    assert not curves["reported_top_k_identifiable"].any()
    candidates = tables["top_k_candidates"]
    assert candidates["is_reported_top_k"].isna().all()
    assert not candidates["reported_top_k_identifiable"].any()


def _mechanism_tables():
    inventory_rows = []
    variant_rows = []
    run_rows = []
    analysis_order = 0
    for family in ("pdz", "trpb"):
        for landscape_number in (1, 2):
            landscape = f"{family}-landscape-{landscape_number}"
            observation_structure = (
                "paired_input_output_replicates"
                if family == "pdz"
                else "time_series_output_replicates_shared_input"
            )
            inventory_rows.append(
                {
                    "protein_family": family,
                    "landscape": landscape,
                    "observation_structure": observation_structure,
                    "discordance_available": True,
                }
            )
            for model_number, model_spec in enumerate(
                ("exponential-exponential", "gamma-exponential")
            ):
                analysis_id = f"{family}__{landscape}__hmc__{model_spec}__fit-{model_number}"
                for rank in range(1, 41):
                    support = np.log1p(20 * rank + 10 * landscape_number)
                    discordance = 0.05 * (rank % 7) + 0.02 * model_number
                    uncertainty = np.exp(-0.35 * support) * (1 + discordance)
                    crowding = (rank % 9) / 8 + 0.1 * model_number
                    score = 2 - 0.02 * rank
                    spacing = 0.08 if rank <= 15 else 0.01
                    if rank <= 20:
                        rank_lower = max(1.0, rank - 0.2)
                        rank_upper = rank + 0.2
                    else:
                        rank_lower = 20.0
                        rank_upper = 50.0
                    variant_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "protein_family": family,
                            "landscape": landscape,
                            "inference_method": "hmc",
                            "model_spec": model_spec,
                            "is_primary_cohort": True,
                            "is_model_top_100": True,
                            "model_reference_rank": rank,
                            "mechanism_input_count": 20 * rank + 10 * landscape_number,
                            "mechanism_output_count": 0 if rank == 1 else 30 * rank,
                            "mechanism_log_input_support": support,
                            "mechanism_reduced_heterogeneity": discordance,
                            "score_lower": score - uncertainty / 2,
                            "score_median": score,
                            "score_upper": score + uncertainty / 2,
                            "score_interval_width": uncertainty,
                            "reference_score_iqr": 1.0,
                            "normalized_score_interval_width": uncertainty,
                            "normalized_local_score_spacing": spacing,
                            "score_crowding": crowding,
                            "rank_lower": rank_lower,
                            "rank_median": float(rank),
                            "rank_upper": rank_upper,
                            "rank_interval_width": rank_upper - rank_lower,
                        }
                    )
                run_rows.append(
                    {
                        "analysis_order": analysis_order,
                        "analysis_id": analysis_id,
                        "protein_family": family,
                        "landscape": landscape,
                        "inference_method": "hmc",
                        "model_spec": model_spec,
                        "n_variants": 100,
                        "uncertainty_to_spacing_ratio": 1.5 + analysis_order,
                        "top_k_instability": 0.08 + 0.06 * analysis_order,
                    }
                )
                analysis_order += 1
    return {
        "mechanism_inventory": pd.DataFrame(inventory_rows),
        "mechanism_variants": pd.DataFrame(variant_rows),
        "mechanism_runs": pd.DataFrame(run_rows),
        "mechanism_effects": pd.DataFrame(),
    }


def _mechanism_tables_for_family(family: str) -> dict[str, pd.DataFrame]:
    tables = _mechanism_tables()
    renamed = {}
    for name, table in tables.items():
        if "protein_family" not in table:
            renamed[name] = table.copy()
            continue
        selected = table[table["protein_family"] == "pdz"].copy()
        selected["protein_family"] = family
        for column in ("analysis_id", "landscape"):
            if column in selected:
                selected[column] = selected[column].str.replace(
                    "pdz",
                    family,
                    regex=False,
                )
        renamed[name] = selected
    return renamed


def test_paper_figures_render_and_save_from_tidy_tables(tmp_path):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    results = []
    for order, family in enumerate(("pdz", "trpb")):
        result = _result()
        result.metadata.update(
            {
                "analysis_id": f"{family}__{family}-toy__mle-bootstrap__ratio__fit-1",
                "protein_family": family,
                "landscape": f"{family}-toy",
            }
        )
        tables = build_paper_tables(
            result,
            score="log_mean_growth",
            main_k=2,
            k_values=(1, 2),
            percentile_cohorts=(2,),
        )
        for table in tables.values():
            table.insert(0, "analysis_order", order)
        results.append(tables)
    combined = {
        name: pd.concat([tables[name] for tables in results], ignore_index=True)
        for name in results[0]
    }
    representative_ids = tuple(combined["inventory"]["analysis_id"])

    reported_figure = figures.figure_reported_order(combined, cohort_top_n=2)
    rank_figure = figures.figure_rank_uncertainty(
        combined,
        representative_ids=representative_ids,
        main_k=2,
    )
    stability_figure = figures.figure_top_k_stability(
        combined,
        representative_ids=representative_ids,
        main_k=2,
    )
    rendered = (reported_figure, rank_figure, stability_figure)

    reported_legend = [text.get_text() for text in reported_figure.axes[0].legend_.texts]
    assert "Uniform reference" in reported_legend
    assert all(axis.legend_._loc == 4 for axis in reported_figure.axes[:3])
    area_axis = reported_figure.axes[3]
    area_legend = area_axis.legend_
    assert area_axis.get_title() == "ECDF-to-uniform area"
    assert area_legend._loc == 9
    assert area_legend._ncols == len(area_legend.texts)
    reported_figure.canvas.draw()
    assert (
        area_legend.get_window_extent().y1
        < area_axis.get_window_extent().y0
    )

    top_rank_labels = [text.get_text() for text in rank_figure.axes[0].get_yticklabels()]
    assert top_rank_labels[0].startswith("1  (")
    rank_legend = [text.get_text() for text in rank_figure.axes[0].legend_.texts]
    assert "5th-95th percentile sampled rank" in rank_legend
    assert "Median sampled rank" in rank_legend
    assert "Published rank" in rank_legend
    rank_width_axis = rank_figure.axes[2]
    assert len(rank_width_axis.patches) == len(representative_ids)
    assert "each box = one analysis run" in rank_width_axis.get_title()
    assert "box = 25th-75th; whiskers = 5th-95th" in rank_width_axis.get_xlabel()
    median_positions = [
        collection.get_offsets()[0, 0]
        for collection in rank_width_axis.collections
    ]
    np.testing.assert_allclose(median_positions, [1.775, 1.775])
    top_k_guide = next(
        line
        for line in rank_width_axis.lines
        if line.get_label() == "K=2 reference"
    )
    np.testing.assert_allclose(top_k_guide.get_xdata(), [2, 2])
    assert "95th-percentile sampled rank > 2" in rank_figure.axes[3].get_xlabel()

    candidate_labels = [
        text.get_text() for text in stability_figure.axes[0].get_yticklabels()
    ]
    assert all("  (" in label for label in candidate_labels)
    assert candidate_labels[0].startswith("1  (")
    assert stability_figure.axes[2].get_xlabel() == "Top-set size K (log scale)"
    assert {tick.get_text() for tick in stability_figure.axes[2].get_xticklabels()} >= {
        "1",
        "2",
    }
    stability_legend = [
        text.get_text() for text in stability_figure.axes[2].legend_.texts
    ]
    assert "Two sampled top-K sets" in stability_legend
    assert "Main detailed view: K=2" in stability_legend

    for index, figure in enumerate(rendered, start=1):
        figure.canvas.draw()
        pdf, png = figures.save_figure(figure, tmp_path, f"figure-{index}")
        assert pdf.stat().st_size > 0
        assert png.stat().st_size > 0


def test_core_paper_figures_expand_to_three_hmc_families():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures_three", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    results = []
    for order, family in enumerate(("pdz", "trpb", "nuc")):
        result = _result()
        result.metadata.update(
            {
                "analysis_id": f"{family}__{family}-{family}-toy__hmc__ratio__fit-1",
                "protein_family": family,
                "landscape": f"{family}-{family}-toy",
                "inference_method": "hmc",
            }
        )
        tables = build_paper_tables(
            result,
            score="log_mean_growth",
            main_k=2,
            k_values=(1, 2),
            percentile_cohorts=(2,),
        )
        for table in tables.values():
            table.insert(0, "analysis_order", order)
        results.append(tables)
    combined = {
        name: pd.concat([tables[name] for tables in results], ignore_index=True)
        for name in results[0]
    }

    reported = figures.figure_reported_order(combined, cohort_top_n=2)
    ranks = figures.figure_rank_uncertainty(combined, main_k=2)
    stability = figures.figure_top_k_stability(combined, main_k=2)

    assert len(reported.axes) == 6
    assert "NucB analysis runs" in reported.axes[3].get_title()
    assert reported.axes[5].get_title() == "Direct reported-model rank concordance"
    assert reported.axes[5].axison is True
    assert len(ranks.axes) == 5
    assert "nuc toy" in ranks.axes[2].get_title().lower()
    assert len(stability.axes) == 6
    assert stability.axes[5].get_title().startswith("NucB stability across K")


def test_mechanism_figure_renders_all_stages(tmp_path):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    figure = figures.figure_ranking_mechanisms(_mechanism_tables())

    assert len(figure.axes) == 4
    assert "Input-count support" in figure.axes[0].get_title()
    assert "Score uncertainty to rank uncertainty" in figure.axes[1].get_title()
    assert "More-separated score neighborhood" in figure.axes[2].get_title()
    assert "Crowded score neighborhood" in figure.axes[3].get_title()
    assert figure.axes[2].get_xlim() == figure.axes[3].get_xlim()
    assert "Score relative to focal posterior median" in figure.axes[3].get_xlabel()
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, "figure-4")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0


def test_paper_figures_support_hmc_count_based_labels():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures_hmc_count", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    rows = pd.DataFrame(
        {
            "inference_method": ["hmc_count_based"],
            "protein_family": ["trpb"],
        }
    )

    assert figures._inference_method(rows) == "hmc_count_based"
    assert figures._uncertainty_terms("hmc_count_based") == (
        "HMC count-proxy median",
        "HMC count-proxy width",
    )
    assert figures.SCORE_LABELS_BY_INFERENCE["hmc_count_based"]["trpb"] == (
        "Median Johnston fitness"
    )


def test_rank_interval_overlap_fraction_handles_separation_and_mixing():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    separated = pd.DataFrame(
        {"rank_lower": [1.0, 3.0, 5.0], "rank_upper": [2.0, 4.0, 6.0]}
    )
    partial = pd.DataFrame(
        {"rank_lower": [1.0, 2.0, 5.0], "rank_upper": [3.0, 4.0, 6.0]}
    )
    mixed = pd.DataFrame(
        {"rank_lower": [1.0, 2.0, 3.0], "rank_upper": [5.0, 6.0, 7.0]}
    )

    assert figures._rank_interval_overlap_fraction(separated) == 0
    assert np.isclose(figures._rank_interval_overlap_fraction(partial), 1 / 3)
    assert figures._rank_interval_overlap_fraction(mixed) == 1


def test_run_scaling_does_not_change_score_width_deciles():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    frame = pd.DataFrame(
        {
            "analysis_id": ["run-a"] * 20 + ["run-b"] * 20,
            "raw_width": np.tile(np.arange(1, 21, dtype=float), 2),
            "scaled_width": np.r_[
                np.arange(1, 21, dtype=float) / 0.5,
                np.arange(1, 21, dtype=float) / 7.0,
            ],
        }
    )

    raw_deciles = figures._quantile_bin(frame, column="raw_width", n_bins=10)
    scaled_deciles = figures._quantile_bin(
        frame,
        column="scaled_width",
        n_bins=10,
    )
    pd.testing.assert_series_equal(raw_deciles, scaled_deciles, check_names=False)


def test_rank_window_selector_is_rank_only_and_deterministic():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    rows = []
    for rank in range(10, 40):
        mixed = rank >= 30
        rows.append(
            {
                "analysis_id": "toy-run",
                "model_reference_rank": rank,
                "rank_lower": 25.0 if mixed else rank - 0.2,
                "rank_median": float(rank),
                "rank_upper": 45.0 if mixed else rank + 0.2,
                "rank_interval_width": 20.0 if mixed else 0.4,
            }
        )
    variants = pd.DataFrame(rows)

    separated, mixed = figures._select_rank_interval_windows(
        variants,
        min_rank=10,
        max_rank=39,
    )

    assert separated["model_reference_rank"].tolist() == list(range(19, 29))
    assert mixed["model_reference_rank"].tolist() == list(range(29, 39))
    assert figures._rank_interval_overlap_fraction(separated) == 0
    assert figures._rank_interval_overlap_fraction(mixed) == 1


def test_rank_space_sensitivity_figure_renders(tmp_path):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)

    figure = figures.figure_rank_space_sensitivity(_mechanism_tables())

    assert len(figure.axes) == 4
    assert "Input-count support" in figure.axes[0].get_title()
    assert "Score uncertainty" in figure.axes[1].get_title()
    assert "More-separated sampled-rank" in figure.axes[2].get_title()
    assert "Overlapping sampled-rank" in figure.axes[3].get_title()
    assert figure.axes[2].get_xlim() == figure.axes[3].get_xlim()
    assert figure.axes[3].get_xlabel() == "Sampled full-landscape rank"
    visible_labels = " ".join(
        [
            *(axis.get_title() for axis in figure.axes),
            *(axis.get_xlabel() for axis in figure.axes),
            *(axis.get_ylabel() for axis in figure.axes),
        ]
    )
    assert "IQR" not in visible_labels
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, "figure-5")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0


def test_count_uncertainty_figures_render_and_preserve_zero_counts(tmp_path):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    tables = _mechanism_tables()

    frame = figures._count_uncertainty_frame(tables)
    landscape_colors, model_markers = figures._count_uncertainty_encodings(frame)
    zero_outputs = frame[frame["mechanism_output_count"] == 0]
    assert len(zero_outputs) > 0
    assert zero_outputs["output_log_count"].eq(0).all()
    assert len(set(landscape_colors.values())) == frame["landscape"].nunique()
    assert set(model_markers) == set(frame["model_spec"])
    assert np.allclose(
        frame["relative_rank_uncertainty"],
        frame["rank_interval_width"] / frame["n_variants"],
    )

    figure_builders = (
        figures.figure_count_uncertainty_scatter,
        figures.figure_count_uncertainty_datashader,
        figures.figure_count_uncertainty_hexbin,
    )
    for index, builder in enumerate(figure_builders, start=1):
        figure = builder(tables)
        assert len(figure.axes) == 4
        assert len(figure.legends) == 1
        assert all(axis.legend_ is None for axis in figure.axes)
        assert all(len(axis.lines) == 1 for axis in figure.axes)
        assert figure.axes[2].get_xlim()[0] < 0
        assert figure.axes[3].get_xlim()[0] < 0
        legend_labels = {text.get_text() for text in figure.legends[0].texts}
        assert "Pooled descriptive linear trend" in legend_labels
        assert {figures._short_model(model) for model in model_markers} <= legend_labels
        assert len(legend_labels) == len(landscape_colors) + len(model_markers) + 1
        assert all(text.get_fontsize() >= 8 for text in figure.legends[0].texts)
        assert figure.legends[0].get_title().get_fontsize() >= 9
        figure.canvas.draw()
        pdf, png = figures.save_figure(figure, tmp_path, f"figure-6-{index}")
        assert pdf.stat().st_size > 0
        assert png.stat().st_size > 0


def test_mle_zero_score_widths_use_plot_only_floor():
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    tables = _mechanism_tables()
    variants = tables["mechanism_variants"].copy()
    variants["inference_method"] = "mle_predictive"
    variants["score_margin_to_rank_100"] = np.where(
        variants["is_model_top_100"],
        0.5,
        -0.5,
    )
    variants["is_challenger_q_ge_001"] = ~variants["is_model_top_100"]
    variants.loc[variants.index[0], "score_interval_width"] = 0.0
    variants.loc[variants.index[0], "normalized_score_interval_width"] = 0.0
    tables["mechanism_variants"] = variants

    count_frame = figures._count_uncertainty_frame(tables)
    decision_frame = figures._decision_relative_score_frame(tables)

    assert count_frame.loc[variants.index[0], "normalized_score_interval_width"] == 0
    assert np.isfinite(count_frame["score_uncertainty_plot"]).all()
    zero_decision_rows = decision_frame[
        decision_frame["normalized_score_interval_width"] == 0
    ]
    assert len(zero_decision_rows) == 1
    assert (zero_decision_rows["normalized_score_interval_width_plot"] > 0).all()


def test_per_analysis_count_uncertainty_figures_render(tmp_path):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    tables = _mechanism_tables()

    builders = (
        figures.figure_count_score_uncertainty_by_analysis,
        figures.figure_count_rank_uncertainty_by_analysis,
    )
    for builder in builders:
        for family in ("pdz", "trpb", "nuc"):
            family_tables = (
                tables if family != "nuc" else _mechanism_tables_for_family(family)
            )
            figure = builder(family_tables, protein_family=family)
            family_variants = family_tables["mechanism_variants"]
            expected_runs = family_variants.loc[
                family_variants["protein_family"] == family,
                "analysis_id",
            ].nunique()
            assert len(figure.axes) == 2 * expected_runs
            assert figures.FAMILY_LABELS[family] in figure._suptitle.get_text()
            assert len(figure.legends) == 1
            assert all(len(axis.lines) == 1 for axis in figure.axes)
            assert figure.axes[-2].get_xlabel().startswith("Observed input count")
            assert figure.axes[-1].get_xlabel().startswith("Observed output count")
            figure.canvas.draw()
            stem = f"figure-7-{builder.__name__}-{family}"
            pdf, png = figures.save_figure(figure, tmp_path, stem)
            assert pdf.stat().st_size > 0
            assert png.stat().st_size > 0
            plt.close(figure)


@pytest.mark.parametrize("family", ["pdz", "trpb", "nuc", "k50"])
def test_count_score_corner_figure_renders_one_corner_per_analysis(tmp_path, family):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    tables = _mechanism_tables_for_family(family)
    variants = tables["mechanism_variants"].copy()
    variants["is_model_top_100"] = variants["model_reference_rank"] <= 10
    variants["is_challenger_q_ge_001"] = variants["model_reference_rank"].between(
        11,
        15,
    )
    tables["mechanism_variants"] = variants

    figure = figures.figure_count_score_corner_by_analysis(
        tables,
        protein_family=family,
        n_columns=2,
    )

    family_variants = tables["mechanism_variants"]
    expected_runs = family_variants.loc[
        family_variants["protein_family"] == family,
        "analysis_id",
    ].nunique()
    assert len(figure.axes) == 9 * expected_runs
    assert figures.FAMILY_LABELS[family] in figure._suptitle.get_text()
    assert [
        (
            subfigure._subplotspec.rowspan.start,
            subfigure._subplotspec.colspan.start,
        )
        for subfigure in figure.subfigs
    ] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert sum("\n" in subfigure._suptitle.get_text() for subfigure in figure.subfigs) == 2
    expected_status_counts = {
        "Reported top-1,000 context only": 25,
        "Median-score top 100": 10,
        "Outside challenger (q >= 1%)": 5,
    }
    for corner_axes in np.asarray(figure.axes).reshape(expected_runs, 9):
        for index in (0, 4, 8):
            assert len(corner_axes[index].patches) == 0
            assert len(corner_axes[index].lines) == 1
            assert corner_axes[index].lines[0].get_color() == "black"
            assert len(corner_axes[index].collections) == 1
        for index in (3, 6, 7):
            collections = {
                collection.get_label(): collection
                for collection in corner_axes[index].collections
            }
            assert set(collections) == set(expected_status_counts)
            assert {
                label: len(collection.get_offsets())
                for label, collection in collections.items()
            } == expected_status_counts
            assert (
                collections["Reported top-1,000 context only"].get_sizes()[0]
                < collections["Median-score top 100"].get_sizes()[0]
                < collections["Outside challenger (q >= 1%)"].get_sizes()[0]
            )
            assert len(
                collections["Outside challenger (q >= 1%)"].get_facecolors()
            ) == 0
        assert all(not corner_axes[index].axison for index in (1, 2, 5))
        assert corner_axes[6].get_xlabel() == "Input count"
        assert corner_axes[7].get_xlabel() == "Output count"
        assert corner_axes[8].get_xlabel() == "Relative score\nuncertainty"
        assert corner_axes[3].get_xlim()[0] < 0
        assert corner_axes[3].get_ylim()[0] < 0
        assert all(
            any(text.get_text().startswith("$\\rho$") for text in corner_axes[index].texts)
            for index in (1, 2, 5)
        )
    assert len(figure.legends) == 1
    assert {text.get_text() for text in figure.legends[0].texts} == set(
        expected_status_counts
    )
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, f"figure-11-{family}")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    plt.close(figure)


@pytest.mark.parametrize("family", ["pdz", "trpb", "nuc"])
def test_functionality_distributions_group_analyses_by_landscape(tmp_path, family):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    inventory = _mechanism_tables_for_family(family)["mechanism_runs"].copy()
    inventory["n_variants"] = 120
    functionality_rows = []
    for run in inventory.itertuples(index=False):
        model_offset = 0.1 if run.model_spec == "gamma-exponential" else 0.0
        landscape_number = int(run.landscape.rsplit("-", maxsplit=1)[-1])
        landscape_offset = 0.4 * (landscape_number - 1)
        rank_step = 0.025 / landscape_number
        for rank in range(1, 121):
            functionality_rows.append(
                {
                    "analysis_id": run.analysis_id,
                    "variant_index": rank - 1,
                    "score_median": (
                        2.5 + landscape_offset + model_offset - rank_step * rank
                    ),
                    "model_reference_rank": rank,
                }
            )
    functionality = pd.DataFrame(functionality_rows)

    figure = figures.figure_functionality_distributions(
        inventory,
        functionality,
        protein_family=family,
        top_k=10,
        n_columns=2,
    )

    assert figures.FAMILY_LABELS[family] in figure._suptitle.get_text()
    assert len(figure.axes) == 2
    for landscape_number, axis in enumerate(figure.axes, start=1):
        landscape_offset = 0.4 * (landscape_number - 1)
        rank_step = 0.025 / landscape_number
        expected_cutoffs = sorted(
            [
                2.5 + landscape_offset - rank_step * 10,
                2.6 + landscape_offset - rank_step * 10,
            ]
        )
        kde_lines = [line for line in axis.lines if line.get_label().startswith("KDE:")]
        cutoff_lines = [
            line
            for line in axis.lines
            if line.get_label().startswith("Rank-10 cutoff:")
        ]
        assert len(kde_lines) == 2
        assert len(cutoff_lines) == 2
        observed_cutoffs = sorted(float(line.get_xdata()[0]) for line in cutoff_lines)
        np.testing.assert_allclose(observed_cutoffs, expected_cutoffs)
        for density_line in kde_lines:
            area = np.trapezoid(density_line.get_ydata(), density_line.get_xdata())
            assert np.isclose(area, 1.0, atol=0.01)
        assert "N=120" in {text.get_text() for text in axis.texts}
    if family != "nuc":
        assert all(
            np.allclose(axis.get_xlim(), figure.axes[0].get_xlim())
            and np.allclose(axis.get_ylim(), figure.axes[0].get_ylim())
            for axis in figure.axes[1:]
        )
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, f"figure-8-{family}")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    plt.close(figure)


@pytest.mark.parametrize("family", ["pdz", "trpb", "nuc"])
def test_fitness_uncertainty_figure_uses_decision_relative_coordinates(
    tmp_path,
    family,
):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    rows = []
    for analysis_number in (1, 2):
        analysis_id = f"{family}__{family}-test-{analysis_number}__hmc__model__fit"
        for index, (margin, width, is_top) in enumerate(
            ((1.0, 0.5, True), (0.2, 1.0, True), (-0.1, 2.0, False))
        ):
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "protein_family": family,
                    "landscape": f"{family}-test-{analysis_number}",
                    "inference_method": "hmc",
                    "model_spec": "model",
                    "score_median": margin,
                    "score_interval_width": width,
                    "normalized_score_interval_width": width / 2,
                    "reference_score_iqr": 2.0,
                    "score_margin_to_rank_100": margin,
                    "is_model_top_100": is_top,
                    "is_challenger_q_ge_001": not is_top,
                    "is_primary_cohort": True,
                    "variant_index": index,
                }
            )
    tables = {"mechanism_variants": pd.DataFrame(rows)}

    figure = figures.figure_fitness_uncertainty_by_analysis(
        tables,
        protein_family=family,
        n_columns=2,
    )

    assert figures.FAMILY_LABELS[family] in figure._suptitle.get_text()
    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert axis.get_yscale() == "log"
        top_points = next(
            collection
            for collection in axis.collections
            if collection.get_label() == "Median-score top 100"
        )
        challenger_points = next(
            collection
            for collection in axis.collections
            if collection.get_label() == "Outside challenger (q >= 1%)"
        )
        np.testing.assert_allclose(
            top_points.get_offsets(),
            [[0.5, 0.25], [0.1, 0.5]],
        )
        np.testing.assert_allclose(challenger_points.get_offsets(), [[-0.05, 1.0]])
        boundary = next(
            line for line in axis.lines if line.get_label() == "Rank-100 boundary"
        )
        np.testing.assert_allclose(boundary.get_xdata(), [0, 0])
        assert any(text.get_text().startswith("Spearman rho =") for text in axis.texts)
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, f"figure-9-{family}")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    plt.close(figure)


@pytest.mark.parametrize("family", ["pdz", "trpb", "nuc"])
def test_summary_why_figure_preserves_counts_and_links_decision_panels(
    tmp_path,
    family,
):
    module_path = Path(__file__).parents[1] / "examples" / "paper" / "figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", module_path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    variant_rows = []
    run_rows = []
    for analysis_number in (1, 2):
        analysis_id = f"{family}__{family}-test-{analysis_number}__hmc__model__fit"
        for index, (margin, width, is_top, input_count, output_count, probability) in enumerate(
            (
                (1.0, 0.5, True, 0, 4, 0.9),
                (0.2, 1.0, True, 9, 0, 0.6),
                (-0.1, 2.0, False, 99, 999, 0.2),
            )
        ):
            variant_rows.append(
                {
                    "analysis_id": analysis_id,
                    "protein_family": family,
                    "landscape": f"{family}-test-{analysis_number}",
                    "inference_method": "hmc",
                    "model_spec": "model",
                    "score_median": margin,
                    "score_interval_width": width,
                    "normalized_score_interval_width": width / 2,
                    "reference_score_iqr": 2.0,
                    "score_margin_to_rank_100": margin,
                    "top_100_inclusion_probability": probability,
                    "mechanism_input_count": input_count,
                    "mechanism_output_count": output_count,
                    "is_model_top_100": is_top,
                    "is_challenger_q_ge_001": not is_top,
                    "is_primary_cohort": True,
                    "variant_index": index,
                }
            )
        run_rows.append(
            {
                "analysis_id": analysis_id,
                "sample_self_consistency": 0.75,
                "top_k_instability": 0.25,
                "challenger_count_q_ge_001": 1,
            }
        )
    tables = {
        "mechanism_variants": pd.DataFrame(variant_rows),
        "mechanism_runs": pd.DataFrame(run_rows),
    }

    figure = figures.figure_summary_why_by_analysis(
        tables,
        protein_family=family,
    )

    assert figures.FAMILY_LABELS[family] in figure._suptitle.get_text()
    assert np.asarray(figure.axes).shape == (8,)
    axes = np.asarray(figure.axes).reshape(2, 4)
    expected_uncertainty = np.log10([0.25, 0.5, 1.0])
    for row_axes in axes:
        input_offsets = np.vstack(
            [collection.get_offsets() for collection in row_axes[0].collections]
        )
        output_offsets = np.vstack(
            [collection.get_offsets() for collection in row_axes[1].collections]
        )
        np.testing.assert_allclose(
            np.sort(input_offsets[:, 0]),
            np.log10([1, 10, 100]),
        )
        np.testing.assert_allclose(
            np.sort(output_offsets[:, 0]),
            np.log10([1, 5, 1000]),
        )
        np.testing.assert_allclose(
            np.sort(input_offsets[:, 1]),
            np.sort(expected_uncertainty),
        )
        assert any(line.get_linewidth() == 1.5 for line in row_axes[0].lines)
        assert any(line.get_linewidth() == 1.5 for line in row_axes[1].lines)
        assert row_axes[0].get_xlim() == row_axes[1].get_xlim()
        assert row_axes[0].get_ylim() == row_axes[2].get_ylim()
        assert row_axes[2].get_xlim() == row_axes[3].get_xlim()
        inclusion_offsets = np.vstack(
            [collection.get_offsets() for collection in row_axes[3].collections]
        )
        np.testing.assert_allclose(
            np.sort(inclusion_offsets[:, 1]),
            [0.2, 0.6, 0.9],
        )
        assert any(
            "instability 25%; 1 challengers" in text.get_text()
            for text in row_axes[0].texts
        )
    figure.canvas.draw()
    pdf, png = figures.save_figure(figure, tmp_path, f"figure-10-{family}")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    plt.close(figure)
