from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts import pdz_trans_hmc_flip_replicates as replicates


def test_make_hmc_flip_replicates(monkeypatch, tmp_path):
    cript = "KNYKQTSV"
    variants = {
        "trans-1": [replicates.WT_PDZ[:50] + cript, "A" + replicates.WT_PDZ[1:50] + cript],
        "trans-2": [replicates.WT_PDZ[50:] + cript, "A" + replicates.WT_PDZ[51:] + cript],
    }
    counts = {
        source: {
            "variants": source_variants,
            "starting_counts": np.arange(6).reshape(3, 2),
            "timepoint_counts": np.arange(6, 12).reshape(3, 2),
        }
        for source, source_variants in variants.items()
    }
    mappings = {
        source: replicates.variant_map(source, source_variants)
        for source, source_variants in variants.items()
    }
    flip = pd.DataFrame(
        {
            "sequence": [
                mappings["trans-1"].sequence.iloc[0],
                mappings["trans-1"].sequence.iloc[1],
                mappings["trans-2"].sequence.iloc[1],
            ],
            "target": [0.1, 0.2, 0.3],
            "set": ["train", "test", "test"],
            "validation": [False, False, False],
        }
    )
    flip_path = tmp_path / "flip.csv"
    flip.to_csv(flip_path, index=False)

    def fake_load(path):
        source = Path(path).stem
        return counts[source]

    def fake_open(path, *, group, **_kwargs):
        source = Path(path).stem
        if group == "observed_data":
            return xr.Dataset(
                {
                    "starting_counts": (("a", "b"), counts[source]["starting_counts"]),
                    "timepoint_counts": (("a", "b"), counts[source]["timepoint_counts"]),
                }
            )
        if group == "posterior_predictive":
            offset = 1_000 if source == "trans-2" else 0
            starting = np.arange(1, 25, dtype=np.int32).reshape(1, 4, 3, 2) + offset
            return xr.Dataset(
                {
                    "starting_counts": (("chain", "draw", "a", "b"), starting),
                    "timepoint_counts": (("chain", "draw", "a", "b"), 2 * starting),
                }
            )
        return xr.Dataset(
            {
                "log_r_mean": (
                    ("chain", "draw", "b"),
                    np.arange(8, dtype=float).reshape(1, 4, 2)
                    + (100 if source == "trans-2" else 0),
                )
            }
        )

    monkeypatch.setattr(replicates, "load_pdz_dataset", fake_load)
    monkeypatch.setattr(replicates.xr, "open_dataset", fake_open)
    count_paths = {source: Path(f"{source}.tsv") for source in variants}
    hmc_paths = {source: Path(source) for source in variants}

    first = replicates.make_hmc_flip_replicates(
        3,
        seed=7,
        count_paths=count_paths,
        hmc_paths=hmc_paths,
        flip_path=flip_path,
    )
    second = replicates.make_hmc_flip_replicates(
        3,
        seed=7,
        count_paths=count_paths,
        hmc_paths=hmc_paths,
        flip_path=flip_path,
    )

    assert len(first) == 3
    assert all(len(dataset) == len(flip) for dataset in first)
    assert all(dataset.sequence.is_unique for dataset in first)
    assert all(dataset.published_target.tolist() == flip.target.tolist() for dataset in first)
    assert all(set(replicates.POSTERIOR_COUNT_COLUMNS) <= set(dataset) for dataset in first)
    assert all(set(replicates.PROXY_COLUMNS) <= set(dataset) for dataset in first)
    assert all(np.allclose(dataset["proxy_count_ratio"], 2.0) for dataset in first)
    assert all(np.allclose(dataset["proxy_log_r_mean"], dataset["target"]) for dataset in first)
    assert all(left.equals(right) for left, right in zip(first, second))
    assert all(
        dataset.loc[dataset.sequence == flip.sequence.iloc[0], "hmc_source"]
        .isin(["trans-1", "trans-2"])
        .all()
        for dataset in first
    )
    for dataset in first:
        shared = dataset.loc[dataset.sequence == flip.sequence.iloc[0]].iloc[0]
        expected_offset = 1_000 if shared.hmc_source == "trans-2" else 0
        assert shared.posterior_starting_count_1 >= expected_offset

    all_draws = replicates.make_hmc_flip_replicates(
        4,
        seed=7,
        count_paths=count_paths,
        hmc_paths=hmc_paths,
        flip_path=flip_path,
    )
    trans1_only = mappings["trans-1"].sequence.iloc[1]
    trans2_only = mappings["trans-2"].sequence.iloc[1]
    assert [
        dataset.loc[dataset.sequence == trans1_only, "target"].item()
        for dataset in all_draws
    ] == [1.0, 3.0, 5.0, 7.0]
    assert [
        dataset.loc[dataset.sequence == trans2_only, "target"].item()
        for dataset in all_draws
    ] == [101.0, 103.0, 105.0, 107.0]