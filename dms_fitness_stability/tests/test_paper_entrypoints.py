import importlib.util
import subprocess
import sys
from pathlib import Path


def _help(script: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_hmc_entrypoints_separate_matrix_and_fit_workflows():
    paper_root = Path(__file__).parents[1] / "examples/paper"

    canonical = _help(paper_root / "run_hmc.py")
    legacy = _help(paper_root / "fit_preprocessing/run_hmc_from_fits.py")

    assert "persisted posterior score matrices" in canonical
    assert "--score-root" in canonical
    assert "--audit" not in canonical
    assert "audited fit artifacts" in legacy
    assert "--audit" in legacy
    assert "--score-root" not in legacy


def test_hmc_full_result_path_preserves_canonical_layout():
    script = Path(__file__).parents[1] / "examples/paper/run_hmc_from_scores.py"
    spec = importlib.util.spec_from_file_location("run_hmc_from_scores", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = Path("results")
    assert module.result_directory(
        root,
        sample_view="full",
        analysis_id="analysis",
    ) == root / "analysis"
    assert module.result_directory(
        root,
        sample_view="release_1000",
        analysis_id="analysis",
    ) == root / "release_1000/analysis"


def test_count_based_entrypoints_expose_score_roots():
    paper_root = Path(__file__).parents[1] / "examples/paper"

    for relative_path in (
        "run_mle.py",
        "run_hmc_count_based.py",
        "audit_mle.py",
        "audit_hmc_count_based.py",
        "preflight_hmc_count_based.py",
    ):
        help_text = _help(paper_root / relative_path)
        assert "--score-root" in help_text
        assert "--proxy-root" in help_text

    mechanism_help = _help(paper_root / "summarize_mechanisms.py")
    assert "Global score root" in mechanism_help
    assert "Deprecated method-specific score root" in mechanism_help


def test_mechanism_proxy_root_alias_overrides_global_default(monkeypatch):
    script = Path(__file__).parents[1] / "examples/paper/summarize_mechanisms.py"
    spec = importlib.util.spec_from_file_location("summarize_mechanisms", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    legacy_root = Path("scores/mle_predictive/canonical")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--proxy-root", str(legacy_root)],
    )

    args = module.parse_args()

    assert args.score_root is None
    assert args.legacy_score_root == legacy_root
