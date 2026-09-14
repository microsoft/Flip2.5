from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg", force=True)
from matplotlib import image as matplotlib_image
from matplotlib import pyplot as plt

from analysis import figures, manuscript
from dms_rank_stability import LandscapeFeatures, RankResult, SampledScores, analyze_scores
from dms_rank_stability.data import FEATURE_DTYPES
from dms_rank_stability.tables import build_run_tables, combine_run_tables, write_tables


def _tables() -> dict[str, pd.DataFrame]:
    random = np.random.default_rng(23)
    candidate_count = 20
    variants = pd.DataFrame(
        {
            "variant_id": [f"V{index}" for index in range(candidate_count)]
            + ["control"],
            "is_candidate": [True] * candidate_count + [False],
            "reported_fitness": [
                float(candidate_count - index) for index in range(candidate_count)
            ]
            + [1000.0],
            "reported_rank": [
                float(index + 1) for index in range(candidate_count)
            ]
            + [np.nan],
            "reported_activity_level": [
                "high" if index < 7 else "medium" if index < 14 else "low"
                for index in range(candidate_count)
            ]
            + [np.nan],
            "is_wild_type": [True] + [False] * candidate_count,
            "is_source_control": [False] * candidate_count + [True],
        }
    )
    baseline = np.linspace(20, 1, candidate_count, dtype=np.float32)
    candidate_scores = baseline[None, :] + random.normal(
        0,
        2.2,
        size=(16, candidate_count),
    ).astype(np.float32)
    score_values = np.column_stack(
        [candidate_scores, np.full(16, 1000.0, dtype=np.float32)]
    )
    scores = SampledScores(
        variants,
        xr.Dataset(
            {"score": (("sample", "variant"), score_values)},
            coords={
                "sample": np.arange(16),
                "variant": np.arange(len(variants)),
                "source_sample_id": ("sample", np.arange(100, 116)),
            },
        ),
    )
    result = analyze_scores(
        scores,
        metadata={
            "run_id": "pdz-synthetic-parameter-posterior-example",
            "family": "pdz",
            "landscape": "pdz-synthetic",
            "sampling": "parameter_posterior",
            "model": "exponential-exponential",
            "score": "log_mean_growth",
            "score_path": "data/downloads/scores/synthetic",
            "feature_path": "data/downloads/features/synthetic",
            "k_values": [1, 2, 5, 10],
            "interval": [0.05, 0.95],
            "seed": 23,
            "external_value_column": "reported_fitness",
            "external_higher_is_better": True,
        },
        retain_k=10,
        decision_k=5,
        external_top_n=10,
        model_top_n=5,
        challenger_probability=0.05,
        variant_chunk_size=4,
    )
    variant_number = np.arange(len(variants))
    feature_arrays = {}
    for name, dtype in FEATURE_DTYPES.items():
        if name == "observation_count":
            values = (variant_number + 2).astype(dtype)
        elif name in {"zero_input_fraction", "zero_output_fraction"}:
            values = (variant_number / 100).astype(dtype)
        elif name == "input_count":
            values = (4 * variant_number + 1).astype(dtype)
        elif name == "output_count":
            values = (variant_number**2 + 2).astype(dtype)
        else:
            values = (np.log1p(variant_number) + 0.2).astype(dtype)
        feature_arrays[name] = ("variant", values)
    features = LandscapeFeatures(
        variants[["variant_id", "is_candidate"]],
        xr.Dataset(feature_arrays, coords={"variant": np.arange(len(variants))}),
    )
    table_sets = []
    for family, model in (
        ("pdz", "exponential-exponential"),
        ("trpb", "exponential-exponential"),
        ("nuc", "lomax"),
    ):
        family_result = RankResult(
            metadata={
                **result.metadata,
                "run_id": f"{family}-synthetic-parameter-posterior-{model}",
                "family": family,
                "landscape": f"{family}-synthetic",
                "model": model,
            },
            variants=result.variants.copy(),
            stability=result.stability.copy(deep=True),
        )
        table_sets.append(build_run_tables(family_result, features))
    return combine_run_tables(table_sets)


def _assert_rendered(figure, expected_axes: int) -> None:
    assert len(figure.axes) == expected_axes
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    assert pixels.shape[0] > 100
    assert pixels.shape[1] > 100
    assert np.count_nonzero(np.any(pixels < 245, axis=2)) > 1000
    plt.close(figure)


def test_all_clean_figure_builders_render_from_generated_tables():
    tables = _tables()
    figures.set_paper_style()

    _assert_rendered(
        figures.figure_reported_order(tables, cohort_top_n=5),
        6,
    )
    _assert_rendered(
        figures.figure_rank_uncertainty(tables, main_k=5, detail_top_n=5),
        5,
    )
    _assert_rendered(figures.figure_top_k_stability(tables, main_k=5), 6)
    _assert_rendered(figures.figure_ranking_mechanisms(tables), 4)
    _assert_rendered(figures.figure_rank_space_sensitivity(tables), 4)
    for renderer in ("scatter", "hexbin", "datashader"):
        _assert_rendered(
            figures.figure_count_uncertainty(tables, renderer=renderer),
            4,
        )
    _assert_rendered(
        figures.figure_count_score_uncertainty_by_analysis(tables, family="pdz"),
        2,
    )
    _assert_rendered(
        figures.figure_count_rank_uncertainty_by_analysis(tables, family="pdz"),
        2,
    )
    _assert_rendered(
        figures.figure_count_score_corner_by_analysis(tables, family="pdz"),
        9,
    )
    _assert_rendered(
        figures.figure_functionality_distributions(
            tables["runs"],
            tables["functionality_scores"],
            family="pdz",
            top_k=5,
        ),
        1,
    )
    _assert_rendered(
        figures.figure_fitness_uncertainty_by_analysis(tables, family="pdz"),
        1,
    )
    _assert_rendered(
        figures.figure_summary_why_by_analysis(tables, family="pdz"),
        4,
    )


def test_table_loaders_and_figure_export_round_trip(tmp_path: Path):
    tables = _tables()
    table_dir = write_tables({"parameter_posterior": tables}, tmp_path / "tables")[0]

    loaded = figures.load_all_tables(table_dir)
    assert set(loaded) == {
        *figures.CORE_TABLE_NAMES,
        *figures.MECHANISM_TABLE_NAMES,
        "functionality_scores",
    }
    figure = figures.figure_top_k_stability(loaded, main_k=5)
    pdf, png = figures.save_figure(figure, tmp_path / "figures", "top_k")
    plt.close(figure)

    assert pdf.stat().st_size > 1_000
    assert png.stat().st_size > 1_000
    image = matplotlib_image.imread(png)
    assert image.std() > 0.01


def test_figure_export_standardizes_typeface_and_text_sizes(tmp_path: Path):
    figures.set_paper_style()
    figure = plt.figure(figsize=(figures.MANUSCRIPT_FIGURE_WIDTH, 3), layout="constrained")
    subfigure = figure.subfigures(1, 1)
    subfigure.suptitle("TrpB libB", fontsize=18, fontfamily="serif")
    axis = subfigure.subplots()
    axis.plot([0, 1], [0, 1], label="Samples")
    axis.set_title("Score uncertainty", fontsize=16, fontfamily="serif")
    axis.set_xlabel("Input reads", fontsize=5, fontfamily="monospace")
    note = axis.text(0.5, 0.5, "100 candidates", fontsize=6, fontfamily="serif")
    legend = axis.legend(fontsize=12)
    figures._panel_label(axis, "a")
    figures.save_figure(figure, tmp_path, "typography")
    assert matplotlib_image.imread(tmp_path / "typography.png").shape[1] == 1950
    for text in (axis.xaxis.label, note, *legend.get_texts(), *axis.get_xticklabels()):
        assert text.get_fontfamily() == [figures.FIGURE_FONT]
        assert text.get_fontsize() == figures.FIGURE_TEXT_SIZE
    for text in (axis.title, subfigure._suptitle, axis.texts[-1]):
        assert text.get_fontfamily() == [figures.FIGURE_FONT]
        assert text.get_fontsize() == figures.FIGURE_HEADING_SIZE
    plt.close(figure)


def test_compact_landscape_labels_preserve_source_names():
    assert figures._short_landscape("trpb-libB", "trpb") == "libB"
    assert figures._short_landscape("nuc-G3", "nuc") == "G3"
    assert figures._short_landscape("pdz-trans-1", "pdz") == "trans-1"


def test_manuscript_figures_render_with_explicit_examples():
    base = _tables()
    collections = {}
    for sampling in manuscript.SCHEMES:
        tables = {name: frame.copy() for name, frame in base.items()}
        for frame in tables.values():
            if "sampling" in frame:
                frame["sampling"] = sampling
        collections[sampling] = tables
    examples = tuple(
        (row.landscape, row.model, row.family)
        for row in base["runs"].itertuples()
    )
    figures.set_paper_style()
    for figure, axes in (
        (manuscript.figure_reported(collections, cohort_top_n=5, examples=examples), 9),
        (manuscript.figure_ranks(collections, detail_top_n=5, examples=examples), 6),
        (manuscript.figure_selection(collections, examples=examples), 6),
        (manuscript.figure_explanation(collections, examples=examples), 12),
        (manuscript.figure_count_support(collections), 2),
    ):
        assert figure.get_figwidth() == figures.MANUSCRIPT_FIGURE_WIDTH
        for axis in figure.axes:
            assert axis.xaxis.label.get_fontsize() == figures.FIGURE_TEXT_SIZE
            assert axis.yaxis.label.get_fontfamily() == [figures.FIGURE_FONT]
        _assert_rendered(figure, axes)
    comparisons = manuscript.reference_comparison_rows(
        collections, decision_k=5, examples=examples,
    )
    assert len(comparisons) == 9
    for column in ("fixed_shared_count", "sample_shared_count", "model_shared_count",
                   "reported_shared_count"):
        assert comparisons[column].between(0, 5).all()


def test_manuscript_selection_displays_batch_budgets_without_changing_tables():
    tables = _tables()
    original = tables["top_k_curves"].copy(deep=True)
    examples = tuple(
        (row.landscape, row.model, row.family)
        for row in tables["runs"].itertuples()
    )
    figures.set_paper_style()
    figure = manuscript.figure_selection({"parameter_posterior": tables}, examples=examples)
    for axis, (landscape, model, _) in zip(figure.axes[3:], examples, strict=True):
        expected = original[
            original["landscape"].eq(landscape)
            & original["model"].eq(model)
            & original["k"].between(10, 1000)
        ].sort_values("k")
        assert not expected.empty
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), expected["k"])
        np.testing.assert_allclose(
            axis.lines[0].get_ydata(), expected["sample_self_consistency"]
        )
        np.testing.assert_allclose(axis.get_xlim(), (10, 1000))
        assert axis.get_xlabel() == "Batch size k"
    pd.testing.assert_frame_equal(tables["top_k_curves"], original)
    _assert_rendered(figure, 6)


def test_explanation_keeps_all_points_without_binning_or_regression(monkeypatch):
    tables = _tables()
    original = tables["mechanism_variants"].copy(deep=True)
    examples = tuple((row.landscape, row.model, row.family) for row in tables["runs"].itertuples())

    def reject_aggregation(*args, **kwargs):
        raise AssertionError("Figure 4 must retain points without binning or fitting lines.")

    monkeypatch.setattr(figures, "_quantile_summary", reject_aggregation)
    monkeypatch.setattr(figures, "_trend", reject_aggregation)
    figure = manuscript.figure_explanation({"parameter_posterior": tables}, examples=examples)
    for row_number, specification in enumerate(examples):
        rows = original[original.landscape.eq(specification[0]) & original.is_primary_cohort]
        for axis in figure.axes[4 * row_number:4 * row_number + 4]:
            point_count = sum(len(collection.get_offsets()) for collection in axis.collections)
            assert point_count == len(rows)
            assert all(collection.get_rasterized() for collection in axis.collections)
            for line in axis.lines:
                assert np.ptp(np.asarray(line.get_xdata(), dtype=float)) == 0
    pd.testing.assert_frame_equal(tables["mechanism_variants"], original)
    _assert_rendered(figure, 12)


def test_explanation_rejects_gamma_and_duplicate_examples():
    tables = {"parameter_posterior": _tables()}
    examples = tuple(
        (row.landscape, row.model, row.family)
        for row in tables["parameter_posterior"]["runs"].itertuples()
    )
    with pytest.raises(ValueError, match="Gamma"):
        manuscript.explanation_rows(
            tables, examples=(("pdz-synthetic", "gamma-exponential", "PDZ"), *examples[1:])
        )
    with pytest.raises(ValueError, match="three distinct"):
        manuscript.explanation_rows(tables, examples=(examples[0],) * 3)
    rows = tables["parameter_posterior"]["mechanism_variants"]
    rows.loc[rows.index[rows.is_primary_cohort][0], "mechanism_input_count"] = np.nan
    with pytest.raises(ValueError, match="finite coordinates"):
        manuscript.explanation_rows(tables, examples=examples)


def test_explanation_exports_every_variant_and_the_run_index(tmp_path: Path):
    tables = _tables()
    examples = tuple((row.landscape, row.model, row.family) for row in tables["runs"].itertuples())
    metadata = manuscript.export_explanation_tables(
        {"parameter_posterior": tables}, tmp_path, examples=examples
    )
    rows = pd.read_csv(tmp_path / "explanation_variants.csv")
    index = pd.read_csv(tmp_path / "explanation_runs.csv")
    original = tables["mechanism_variants"]
    expected = original[original.is_primary_cohort].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[expected.columns], expected, check_dtype=False)
    counts = rows.groupby("run_id").size().reindex(index.run_id)
    np.testing.assert_array_equal(counts, index.cohort_size)
    assert index.reference_members.add(index.challengers).eq(index.cohort_size).all()
    assert metadata["aggregation"] == "none"
    assert metadata["excluded_model_families"] == ["gamma"]
    assert metadata["variant_rows"] == len(expected)
    assert set(path.name for path in tmp_path.iterdir()) == {
        "explanation_variants.csv", "explanation_runs.csv"
    }


def test_generated_evidence_tables_are_centered_and_use_short_labels(tmp_path: Path, monkeypatch):
    comparison = pd.DataFrame({
        "example": ["TrpB libB", "NucB G3"],
        "sampling": ["parameter_posterior"] * 2, "k": [100] * 2,
        "fixed_shared_count": [33, 68], "sample_shared_count": [50.96, 93.49],
        "model_shared_count": [63.44, 95.36], "reported_shared_count": [29.81, 68.14],
    })
    monkeypatch.setattr(
        manuscript, "reference_comparison_rows", lambda tables, **kwargs: comparison
    )
    curves = pd.DataFrame({
        "family": ["trpb", "trpb"], "landscape": ["trpb-libB", "trpb-libB"],
        "k": [100, 1000], "sample_self_consistency": [0.51, 0.8],
    })
    manuscript.export_evidence_tables(
        {"parameter_posterior": {"top_k_curves": curves}}, tmp_path, examples=()
    )
    for stem in ("reference_comparisons", "landscape_set_sizes"):
        tex = (tmp_path / f"{stem}.tex").read_text()
        assert r"\centering" in tex
        assert "library B" not in tex and "generation 3" not in tex
    assert "TrpB libB" in (tmp_path / "reference_comparisons.tex").read_text()


def test_model_points_use_stable_vertical_offsets_and_hollow_medians():
    frame = pd.DataFrame({
        "run_id": ["pdz-third", "pdz-first", "pdz-second", "pdz-single"],
        "family": ["pdz"] * 4,
        "landscape": ["pdz-tied"] * 3 + ["pdz-single"],
        "model": ["lomax-exponential", "exponential-exponential",
                  "exponential-sigmoid", "exponential-exponential"],
        "value": [0.3, 0.3, 0.3, 0.5],
    })
    original = frame.copy(deep=True)
    examples = ("pdz-tied", "exponential-exponential", "PDZ tied")
    plotted = []
    for data in (frame, frame.sample(frac=1, random_state=42)):
        figure, axis = plt.subplots()
        manuscript._landscape_dots(
            axis, {"parameter_posterior": data}, "value", example=examples
        )
        collections = {item.get_gid(): item for item in axis.collections if item.get_gid()}
        points = collections["model-points:parameter_posterior:pdz-tied:value"]
        offsets = points.get_offsets()
        np.testing.assert_array_equal(offsets[:, 0], [0.3, 0.3, 0.3])
        assert len(np.unique(offsets[:, 1])) == 3
        assert points.get_alpha() >= 0.6
        median = collections["model-median:parameter_posterior:pdz-tied:value"]
        assert not len(median.get_facecolors())
        np.testing.assert_allclose(median.get_offsets()[0], [0.3, -0.19])
        highlighted = collections["example-point:pdz-first:value"]
        np.testing.assert_allclose(highlighted.get_offsets()[0], offsets[0])
        arrow = next(
            item for item in axis.texts if item.get_gid() == "example-arrow:pdz-first:value"
        )
        np.testing.assert_allclose(arrow.xy, offsets[0])
        single = collections["model-points:parameter_posterior:pdz-single:value"]
        np.testing.assert_allclose(single.get_offsets()[0], [0.5, 0.81])
        plotted.append(offsets.copy())
        plt.close(figure)
    np.testing.assert_array_equal(plotted[0], plotted[1])
    pd.testing.assert_frame_equal(frame, original)


def test_reported_overview_highlights_exact_example_values():
    base = _tables()
    examples = tuple((row.landscape, row.model, row.family) for row in base["runs"].itertuples())
    alternative = base["rank_percentiles"].copy()
    alternative["run_id"] += "-alternative"
    alternative["model"] = "alternative"
    area = alternative["ecdf_to_uniform_area"]
    alternative["ecdf_to_uniform_area"] = np.where(area < 0.45, area + 0.05, area - 0.05)
    correlation = alternative["external_vs_posterior_median_rank_spearman"]
    alternative["external_vs_posterior_median_rank_spearman"] = np.where(
        correlation < 0.5, 0.5, 0.0
    )
    base["rank_percentiles"] = pd.concat([base["rank_percentiles"], alternative])
    figure = manuscript.figure_reported(
        {"parameter_posterior": base}, cohort_top_n=5, examples=examples
    )
    assert all(
        handle.get_markerfacecolor() == "none" for handle in figure.legends[0].legend_handles
    )
    for column, specification in enumerate(examples):
        rows = base["rank_percentiles"]
        selected = rows[rows.landscape.eq(specification[0]) & rows.cohort_top_n.eq(5)].iloc[0]
        for row, metric in ((1, "ecdf_to_uniform_area"),
                            (2, "external_vs_posterior_median_rank_spearman")):
            axis = figure.axes[3 * row + column]
            highlight = [item for item in axis.collections
                         if (item.get_gid() or "").startswith("example-point:")]
            assert len(highlight) == 1
            landscape_rows = rows[
                rows.landscape.eq(specification[0]) & rows.cohort_top_n.eq(5)
            ].drop_duplicates("run_id")
            assert not np.isclose(landscape_rows[metric].median(), selected[metric])
            np.testing.assert_allclose(highlight[0].get_offsets()[0], [selected[metric], -0.11])
            arrows = [item for item in axis.texts
                      if (item.get_gid() or "").startswith("example-arrow:")]
            assert len(arrows) == 1
            np.testing.assert_allclose(arrows[0].xy, [selected[metric], -0.11])
    _assert_rendered(figure, 9)


def test_rank_overview_highlights_fit_instead_of_landscape_median():
    base = _tables()
    original = base["model_rank_intervals"].copy()
    examples = tuple((row.landscape, row.model, row.family) for row in base["runs"].itertuples())
    alternative = original.copy()
    alternative["run_id"] += "-alternative"
    alternative["model"] = "alternative"
    alternative["rank_interval_width"] += 100
    base["model_rank_intervals"] = pd.concat([original, alternative])
    figure = manuscript.figure_ranks(
        {"parameter_posterior": base}, detail_top_n=5, examples=examples
    )
    for axis, specification in zip(figure.axes[3:], examples, strict=True):
        expected = original.loc[
            original.landscape.eq(specification[0]), "rank_interval_width"
        ].median()
        highlight = [item for item in axis.collections
                     if (item.get_gid() or "").startswith("example-point:")]
        assert len(highlight) == 1
        np.testing.assert_allclose(highlight[0].get_offsets()[0], [expected, -0.11])
    _assert_rendered(figure, 6)


def test_manuscript_summary_combines_models_within_landscape():
    frame = pd.DataFrame({
        "family": ["pdz", "pdz", "pdz", "pdz"],
        "landscape": ["first", "first", "first", "second"],
        "value": [1.0, 2.0, 9.0, 8.0],
    })
    summary = manuscript.landscape_summary(frame, "value").set_index("landscape")
    assert summary.loc["first", "value"] == 2.0
    assert summary.loc["second", "value"] == 8.0


def test_summary_why_omits_regressions_and_preserves_scatter_data(monkeypatch):
    tables = _tables()
    original = tables["mechanism_variants"].copy(deep=True)

    def reject_regression(*args, **kwargs):
        raise AssertionError("Publication summary must not fit regression lines.")

    monkeypatch.setattr(figures, "_trend", reject_regression)
    run_id = tables["runs"].loc[tables["runs"]["family"].eq("pdz"), "run_id"].iloc[0]
    figure = figures.figure_summary_why_by_analysis(tables, family="pdz", run_ids=(run_id,))
    rows = original[original["run_id"].eq(run_id) & original["is_primary_cohort"]]
    for position, axis in enumerate(figure.axes):
        assert sum(len(collection.get_offsets()) for collection in axis.collections) == len(rows)
        assert axis.get_yscale() == ("log" if position < 3 else "linear")
        for line in axis.lines:
            assert np.ptp(np.asarray(line.get_xdata(), dtype=float)) == 0
        assert all(collection.get_rasterized() for collection in axis.collections)
    np.testing.assert_allclose(figure.axes[2].get_xlim(), figure.axes[3].get_xlim())
    pd.testing.assert_frame_equal(tables["mechanism_variants"], original)
    _assert_rendered(figure, 4)
    with pytest.raises(ValueError, match="Summary run IDs"):
        figures.figure_summary_why_by_analysis(tables, family="pdz", run_ids=("absent",))


def test_summary_why_cross_family_keeps_each_point_and_only_selected_limits():
    tables = _tables()
    run_ids = tuple(tables["runs"]["run_id"].iloc[::-1])
    variants = tables["mechanism_variants"]
    first = variants.index[variants["is_primary_cohort"]][0]
    variants.loc[first, "mechanism_input_count"] = 0
    variants.loc[first, "normalized_score_interval_width"] = 0
    extra = variants.loc[[first]].assign(
        run_id="unselected-run", mechanism_input_count=1e20,
        normalized_score_interval_width=1e20,
    )
    tables["mechanism_variants"] = pd.concat([variants, extra], ignore_index=True)
    original = tables["mechanism_variants"].copy(deep=True)
    figure = figures.figure_summary_why_by_analysis(tables, run_ids=run_ids)
    primary = variants[variants["is_primary_cohort"]]
    floor = primary.loc[primary.normalized_score_interval_width.gt(0),
                        "normalized_score_interval_width"].min() / 2
    for position, run_id in enumerate(run_ids):
        rows = primary[primary["run_id"].eq(run_id)]
        rows = pd.concat([rows[rows.is_model_top_k], rows[rows.is_challenger]])
        margin = rows.score_margin_to_top_k_boundary / rows.reference_score_iqr
        width = np.maximum(rows.normalized_score_interval_width, floor)
        expected = (
            (np.log10(rows.mechanism_input_count + 1), width),
            (np.log10(rows.mechanism_output_count + 1), width),
            (margin, width),
            (margin, rows.top_k_inclusion_probability),
        )
        axes = figure.axes[position * 4:position * 4 + 4]
        for axis, coordinates in zip(axes, expected, strict=True):
            actual = np.concatenate([collection.get_offsets() for collection in axis.collections])
            np.testing.assert_allclose(actual, np.column_stack(coordinates))
            assert len(actual) == len(rows)
        assert axes[0].get_xlim()[1] < 10
        assert axes[0].get_ylim()[1] < 1e10
        np.testing.assert_allclose(axes[2].get_xlim(), axes[3].get_xlim())
    pd.testing.assert_frame_equal(tables["mechanism_variants"], original)
    _assert_rendered(figure, 12)
    with pytest.raises(ValueError, match="explicit run IDs"):
        figures.figure_summary_why_by_analysis(tables)


def test_summary_why_pages_cover_every_run_once_and_preserve_order():
    runs = pd.DataFrame({
        "run_id": [f"pdz-{position}" for position in range(7)] + ["nuc-0"],
        "family": ["pdz"] * 7 + ["nuc"],
    })
    pages = manuscript.summary_why_pages({"parameter_posterior": {"runs": runs}})
    assert [len(page["run_ids"]) for page in pages] == [3, 3, 1, 1]
    assert [run_id for page in pages for run_id in page["run_ids"]] == runs.run_id.tolist()
    assert len({page["stem"] for page in pages}) == len(pages)
    assert len({page["label"] for page in pages}) == len(pages)
    with pytest.raises(ValueError, match="positive"):
        manuscript.summary_why_pages({}, runs_per_page=0)


def test_reported_curve_export_preserves_every_percentile(tmp_path: Path):
    tables = _tables()
    for model in ("exponential-sigmoid", "lomax-exponential", "gamma-exponential",
                  "gamma-sigmoid"):
        for name in ("runs", "rank_percentiles"):
            frame = tables[name]
            extra = frame[frame.family.eq("pdz") & frame.model.eq("exponential-exponential")]
            extra = extra.assign(run_id=extra.run_id + f"-{model}", model=model)
            tables[name] = pd.concat([frame, extra], ignore_index=True)
    original = tables["rank_percentiles"].copy(deep=True)
    figures.set_paper_style()
    run_ids = tuple(tables["runs"].run_id)
    for _, runs in tables["runs"].groupby("family"):
        figure = manuscript.figure_reported_curves(
            tables, run_ids=tuple(runs.run_id), cohorts=(5, 10)
        )
        for column, cohort in enumerate((5, 10)):
            lines = {line.get_gid(): line for line in figure.axes[column].lines
                     if line.get_gid()}
            assert len(lines) == len(runs)
            for run_id in runs.run_id:
                data = original[original.run_id.eq(run_id) & original.cohort_top_n.eq(cohort)]
                data = data.sort_values("external_rank_sample_percentile", kind="stable")
                line = lines[f"reported-curve:{run_id}:{cohort}"]
                np.testing.assert_array_equal(
                    line.get_xdata(), np.r_[0, data.external_rank_sample_percentile, 1]
                )
                np.testing.assert_array_equal(line.get_ydata(), np.r_[0, data.ecdf, 1])
        _assert_rendered(figure, 2)
    pages = manuscript.export_reported_curves(
        {"parameter_posterior": tables}, tmp_path, examples=(), cohorts=(5, 10)
    )
    assert len(pages) == 3
    assert max(len(page["run_ids"]) for page in pages) == 5
    index = pd.read_csv(tmp_path / "reported_curves_index.csv")
    assert len(index) == 14 and not index.duplicated(["run_id", "cohort_top_n"]).any()
    assert set(index.run_id) == set(run_ids)
    assert set(index.cohort_top_n) == {5, 10}
    assert set(index.panel) == {"a", "b"}
    statistics = pd.read_csv(tmp_path / "reported_curve_statistics.csv").set_index("run_id")
    assert len(statistics) == 7 and statistics.analysis_id.is_unique
    for cohort in (5, 10):
        expected = index[index.cohort_top_n.eq(cohort)].set_index("run_id")
        expected = expected.reindex(statistics.index)
        np.testing.assert_allclose(
            statistics[[f"n_{cohort}", f"area_{cohort}", f"spearman_{cohort}"]],
            expected[["supported_candidates", "area", "spearman"]], equal_nan=True,
        )
    tex = (tmp_path / "reported_curve_statistics.tex").read_text()
    assert r"\begin{longtable}" in tex and r"\endfirsthead" in tex and r"\endhead" in tex
    assert all(identifier in tex for identifier in statistics.analysis_id)
    for page in pages:
        assert (tmp_path / f"{page['stem']}.pdf").stat().st_size > 1000
        assert page["label"] in (tmp_path / "reported_curves.tex").read_text()
    pd.testing.assert_frame_equal(tables["rank_percentiles"], original)


def test_reported_curves_reject_missing_cohorts():
    tables = _tables()
    with pytest.raises(ValueError, match="Missing curve"):
        manuscript.figure_reported_curves(tables, run_ids=(tables["runs"].run_id.iloc[0],))
    with pytest.raises(ValueError, match="one protein system"):
        manuscript.figure_reported_curves(tables, run_ids=tuple(tables["runs"].run_id))
    plt.close("all")


def test_summary_why_export_includes_all_synthetic_runs(tmp_path: Path):
    tables = _tables()
    figures.set_paper_style()
    pages = manuscript.export_summary_why({"parameter_posterior": tables}, tmp_path)
    index = pd.read_csv(tmp_path / "summary_why_index.csv")
    assert len(pages) == 3
    assert set(index.run_id) == set(tables["runs"].run_id)
    assert not index.run_id.duplicated().any()
    assert (index.reference_members + index.challengers).eq(index.cohort_size).all()
    tex = (tmp_path / "summary_why.tex").read_text()
    assert tex.count(r"\begin{figure}") == 3
    for page in pages:
        assert page["label"] in tex
        assert (tmp_path / f"{page['stem']}.pdf").stat().st_size > 1000
