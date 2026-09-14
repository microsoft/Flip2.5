import json
from pathlib import Path

import pytest

NOTEBOOKS = {
    "plots_parameter_posterior.ipynb": {
        "sampling": "parameter_posterior",
        "run_count": "49",
        "family_counts": '"nuc": 7, "pdz": 14, "trpb": 28',
        "representatives": (
            "pdz-cis-parameter-posterior-lomax-exponential",
            "trpb-libd-parameter-posterior-gamma-exponential",
            "nuc-g1-parameter-posterior-exponential",
        ),
    },
    "plots_mle_predictive.ipynb": {
        "sampling": "mle_predictive",
        "run_count": "60",
        "family_counts": '"nuc": 7, "pdz": 19, "trpb": 34',
        "representatives": (
            "pdz-cis-mle-predictive-lomax-exponential",
            "trpb-libd-mle-predictive-gamma-exponential",
            "nuc-g1-mle-predictive-exponential",
        ),
    },
    "plots_posterior_predictive.ipynb": {
        "sampling": "posterior_predictive",
        "run_count": "49",
        "family_counts": '"nuc": 7, "pdz": 14, "trpb": 28',
        "representatives": (
            "pdz-cis-posterior-predictive-lomax-exponential",
            "trpb-libd-posterior-predictive-gamma-exponential",
            "nuc-g1-posterior-predictive-exponential",
        ),
    },
}
FORBIDDEN_TERMS = (
    "arti" + "fact",
    "au" + "dit",
    "ca" + "che",
    "depre" + "cated",
    "fit" + "_id",
    "job" + "_id",
    "lega" + "cy",
    "migra" + "tion",
    "pro" + "xy",
    "sche" + "ma",
)
REQUIRED_SECTIONS = (
    "## 1. Configure the Python Environment",
    "## 2. Load or Generate Input Data",
    "## 3. Implement the Core Workflow",
    "## 4. Visualize Results",
    "## 5. Validate the Implementation",
)
REQUIRED_FIGURE_CALLS = (
    "figure_reported_order",
    "figure_rank_uncertainty",
    "figure_top_k_stability",
    "figure_ranking_mechanisms",
    "figure_rank_space_sensitivity",
    "figure_count_uncertainty",
    "figure_count_score_uncertainty_by_analysis",
    "figure_count_rank_uncertainty_by_analysis",
    "figure_functionality_distributions",
    "figure_fitness_uncertainty_by_analysis",
    "figure_summary_why_by_analysis",
    "figure_count_score_corner_by_analysis",
)


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_paper_notebook_is_clean_complete_and_syntactically_valid(filename):
    path = Path(__file__).parents[1] / "analysis" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 14
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{filename}:cell-{cell_number}", "exec")

    expected = NOTEBOOKS[filename]
    assert all(section in markdown for section in REQUIRED_SECTIONS)
    assert f'SAMPLING = "{expected["sampling"]}"' in code
    assert f"assert len(runs) == {expected['run_count']}" in code
    assert expected["family_counts"] in code
    assert all(run_id in code for run_id in expected["representatives"])
    assert all(name in code for name in REQUIRED_FIGURE_CALLS)
    assert "assert len(expected_stems) == 26" in code
    assert "assert len(generated_paths) == 2 * len(expected_stems)" in code

    cell_text = f"{markdown}\n{code}".lower()
    assert not {term for term in FORBIDDEN_TERMS if term in cell_text}
