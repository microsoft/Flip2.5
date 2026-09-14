import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import image as matplotlib_image
from matplotlib import pyplot as plt

from analysis import figures
from dms_rank_stability import SampledScores
from dms_rank_stability.results import RankResult
from dms_rank_stability.runs import execute_runs, load_run_config
from dms_rank_stability.tables import TABLE_NAMES, build_tables_for_runs, write_tables


def test_checked_in_example_runs_from_data_to_nonblank_figure(example_root, tmp_path):
    config = load_run_config(example_root / "analysis/runs.json")
    score = SampledScores.load(
        example_root / "scores/toy-parameter-posterior-example"
    )
    try:
        assert score.score.shape == (12, 9)
        assert score.variants.loc[0, "variant_id"] == ""
        assert score.variants["is_candidate"].sum() == 8
        assert score.variants.loc[8, "is_source_control"]
    finally:
        score.close()

    result_root = tmp_path / "results"
    outcomes = execute_runs(config, config.runs, result_root=result_root)
    assert [outcome.state for outcome in outcomes] == ["run"]
    result_path = outcomes[0].result_path
    assert {path.name for path in result_path.iterdir()} == {
        "metadata.json",
        "variants.csv",
        "stability.nc",
    }
    result = RankResult.load(result_path)
    try:
        assert result.metadata["n_candidates"] == 8
        assert 8 not in result.stability["selection_variant_index"].values
    finally:
        result.close()

    tables_by_sampling = build_tables_for_runs(
        config,
        config.runs,
        result_root=result_root,
    )
    table_dir = write_tables(tables_by_sampling, tmp_path / "tables")[0]
    assert {path.name for path in table_dir.iterdir()} == {
        *{f"{name}.csv" for name in TABLE_NAMES if name != "functionality_scores"},
        "functionality_scores.csv.gz",
    }

    tables = figures.load_all_tables(table_dir)
    figure = figures.figure_top_k_stability(tables, main_k=2)
    pdf, png = figures.save_figure(figure, tmp_path / "figures", "top-k")
    plt.close(figure)
    assert pdf.stat().st_size > 1_000
    assert png.stat().st_size > 1_000
    image = matplotlib_image.imread(png)
    assert image.std() > 0.01
