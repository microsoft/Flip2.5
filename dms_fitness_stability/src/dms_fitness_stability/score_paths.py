"""Canonical locations for persisted score bundles."""

from __future__ import annotations

from pathlib import Path

SCORE_METHODS = {"hmc", "mle_predictive", "hmc_count_based"}
SCORES_RELATIVE_ROOT = Path("paper/analysis_data/scores")


def score_collection_root(
    scores_root: str | Path,
    inference_method: str,
    *,
    sample_view: str | None = None,
) -> Path:
    """Return one method/view collection beneath a global scores root."""
    if inference_method not in SCORE_METHODS:
        raise ValueError(f"Unsupported score inference method: {inference_method!r}.")
    if sample_view is None:
        sample_view = "full" if inference_method == "hmc" else "canonical"
    if inference_method != "hmc" and sample_view != "canonical":
        raise ValueError(
            f"{inference_method} score bundles require sample_view='canonical'."
        )
    return Path(scores_root) / inference_method / sample_view


def score_bundle_root(
    data_root: str | Path,
    inference_method: str,
    *,
    sample_view: str | None = None,
) -> Path:
    """Return the canonical root for one inference method and sample view."""
    return score_collection_root(
        Path(data_root) / SCORES_RELATIVE_ROOT,
        inference_method,
        sample_view=sample_view,
    )
