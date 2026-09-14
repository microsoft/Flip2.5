import re
from pathlib import Path


def test_public_text_omits_retired_internal_terms():
    root = Path(__file__).parents[1]
    patterns = (
        "pro" + r"x(?:y|ies)",
        "au" + r"dits?",
        "migra" + r"t(?:e|ed|es|ing|ion|ions)",
        "compati" + r"bility",
        "depre" + r"cat(?:e|ed|es|ing|ion)",
        "arti" + r"facts?",
        "fit" + r"[ _-]*ids?",
        "job" + r"[ _-]*ids?",
        "lega" + r"cy",
        "sche" + r"mas?",
    )
    excluded = re.compile(
        r"(?:^|/)(?:\.git|\.venv|\.pytest_"
        + r"cache|\.ruff_"
        + r"cache|__py"
        + r"cache__)(?:/|$)"
    )
    checked_suffixes = {".json", ".md", ".py", ".ipynb"}
    failures = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or path.suffix not in checked_suffixes
            or excluded.search(relative)
            or relative.startswith(("analysis/executed/", "preprocessing/output/"))
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                failures.append(f"{relative}: {pattern}")
    assert not failures, failures
