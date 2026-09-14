import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dms_fitness_stability import (
    plot_calibration_divergence,
    plot_calibration_overlay,
    plot_inclusion,
    plot_overlap,
    plot_rank_calibration,
    plot_rank_divergence_boxplots,
    plot_rank_intervals,
    plot_report,
    plot_value_intervals,
)


def _write_plot_result(path):
    path.mkdir()
    metadata = {
        "landscape": "toy-landscape",
        "n_variants": 100,
        "scores": ["growth"],
        "top_k": [2],
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = []
    for index in range(5):
        rows.append(
            {
                "landscape": "toy-landscape",
                "score": "growth",
                "k": 2,
                "variant_index": index,
                "variant_id": f"variant-{index}",
                "variant_label": f"V{index}",
                "reported_rank": index + 1,
                "top_k_count": 10 - index,
                "top_k_probability": 1.0 - index * 0.15,
                "is_tracked_reported_cohort": True,
                "posterior_value_mean": 5.0 - index,
                "posterior_value_sd": 0.2,
                "posterior_value_lower": 4.6 - index,
                "posterior_value_upper": 5.4 - index,
                "posterior_mean_rank": index + 1.5,
                "posterior_rank_sd": 0.5,
                "posterior_rank_lower": max(1.0, index + 0.5),
                "posterior_rank_upper": index + 2.5,
                "posterior_top_k_probability": max(0.0, 0.9 - index * 0.2),
                "reported_rank_percentile": 0.15 + index * 0.15,
            }
        )
    pd.DataFrame(rows).to_csv(path / "summary.csv", index=False)
    pd.DataFrame(
        {
            "score": ["growth"] * 8,
            "k": [2] * 8,
            "comparison": ["sample_vs_sample"] * 4 + ["sample_vs_reference"] * 4,
            "recall": [0.5, 0.6, 0.7, 0.6, 0.3, 0.4, 0.5, 0.4],
            "jaccard": [1 / 3, 3 / 7, 7 / 13, 3 / 7, 3 / 17, 0.25, 1 / 3, 0.25],
        }
    ).to_csv(path / "overlaps.csv", index=False)
    return path


def test_individual_plots_render_from_result_directory_without_netcdf(tmp_path):
    result_dir = _write_plot_result(tmp_path / "result")

    inclusion_figure, inclusion_axis, displayed = plot_inclusion(
        result_dir,
        score="growth",
        k=2,
        n_variants=3,
    )
    assert len(displayed) == 3
    assert inclusion_axis.get_xlim() == (-0.02, 1.02)
    assert any("Reported top K" == text.get_text() for text in inclusion_axis.legend_.texts)

    overlap_figure, overlap_axis = plot_overlap(result_dir, score="growth", k=2)
    assert overlap_axis.get_ylim() == (-0.02, 1.02)

    rank_figure, rank_axis, ranks = plot_rank_intervals(
        result_dir,
        score="growth",
        k=2,
    )
    assert len(ranks) == 2
    assert rank_axis.get_yticklabels()[0].get_text().endswith("V0")

    value_figure, _, values = plot_value_intervals(
        result_dir,
        score="growth",
        k=2,
    )
    assert len(values) == 2

    calibration_figure, calibration_axes, diagnostics = plot_rank_calibration(
        result_dir,
        score="growth",
        k=2,
    )
    assert len(calibration_axes) == 2
    assert diagnostics["n_variants"].tolist() == [5, 2]

    for figure in (
        inclusion_figure,
        overlap_figure,
        rank_figure,
        value_figure,
        calibration_figure,
    ):
        plt.close(figure)


def test_inclusion_plot_omits_reported_legend_when_reference_is_unavailable(
    tmp_path,
):
    result_dir = _write_plot_result(tmp_path / "result")
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["reference_top_k_identifiable"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    figure, axis, displayed = plot_inclusion(
        result_dir,
        score="growth",
        k=2,
        n_variants=3,
    )

    assert axis.legend_ is None
    assert not displayed["reported_top_k"].any()
    plt.close(figure)


def test_plot_report_builds_requested_views(tmp_path):
    result_dir = _write_plot_result(tmp_path / "result")

    figures = plot_report(
        result_dir,
        k=2,
        views=("inclusion", "overlap"),
        n_variants=3,
    )

    assert set(figures) == {"growth.inclusion", "growth.overlap"}
    for figure in figures.values():
        plt.close(figure)


def test_plotting_reports_available_score_and_k(tmp_path):
    result_dir = _write_plot_result(tmp_path / "result")

    with np.testing.assert_raises_regex(ValueError, "choose from.*growth"):
        plot_inclusion(result_dir, score="missing", k=2)
    with np.testing.assert_raises_regex(ValueError, "choose from.*2"):
        plot_inclusion(result_dir, score="growth", k=10)


def test_multi_result_calibration_and_rank_divergence_plots(tmp_path):
    result_dir = _write_plot_result(tmp_path / "result")
    results = {"Landscape A": result_dir, "Landscape B": result_dir}

    overlay_figure, overlay_axis, overlay_diagnostics = plot_calibration_overlay(
        results,
        score="growth",
        k=2,
    )
    divergence_figure, divergence_axis, divergence_diagnostics = (
        plot_calibration_divergence(
            results,
            score="growth",
            k=2,
        )
    )
    rank_figure, rank_axes, rank_data = plot_rank_divergence_boxplots(
        results,
        score="growth",
        k=2,
        reported_top_n=2,
    )

    assert len(overlay_axis.lines) == 3
    assert "Growth" in overlay_axis.get_title()
    assert overlay_diagnostics["result"].tolist() == ["Landscape A", "Landscape B"]
    assert np.isfinite(overlay_diagnostics["calibration_l1_area"]).all()
    assert divergence_axis.get_xlabel() == "Area between ECDF and uniform diagonal"
    assert "Growth" in divergence_axis.get_title()
    assert divergence_diagnostics["result"].tolist() == ["Landscape A", "Landscape B"]
    assert len(rank_axes) == 2
    assert any("Growth" in text.get_text() for text in rank_figure.texts)
    assert set(rank_data["bound"]) == {"5th percentile", "95th percentile"}
    assert len(rank_data) == 8

    for figure in (overlay_figure, divergence_figure, rank_figure):
        plt.close(figure)
