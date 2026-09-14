from pathlib import Path

import pytest


@pytest.fixture
def example_root() -> Path:
    """Return the checked-in tiny example data root."""
    return Path(__file__).parents[1] / "data/example"
