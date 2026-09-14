"""Validate the raw data and fits for all three PDZ landscapes."""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path
from typing import Callable

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)
HERE = Path(__file__).resolve().parent


def _load_validator(path: Path) -> Callable[[Path], None]:
    namespace = runpy.run_path(str(path))
    validator = namespace["validate_data"]
    if not callable(validator):
        raise TypeError(f"Expected validate_data() in {path}")
    return validator


def validate_data(data_root: Path) -> None:
    """Run the source-specific input checks used by the unified PDZ workflow."""
    examples_root = HERE.parent
    validators = (
        _load_validator(examples_root / "pdz_trans" / "validate_data.py"),
        _load_validator(examples_root / "pdz_cis" / "validate_data.py"),
    )
    for validator in validators:
        validator(data_root)
    print("All PDZ landscape inputs are valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    validate_data(args.data_root)


if __name__ == "__main__":
    main()
