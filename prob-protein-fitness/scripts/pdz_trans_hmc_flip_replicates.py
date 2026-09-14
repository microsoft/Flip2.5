"""Generate PDZ FLIP datasets with targets sampled from HMC posteriors."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]

from SciStanPy.flipv3.flip_dsets import load_pdz_dataset

WT_PDZ = "GEEDIPREPRRIVIHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEA"
WT_CRIPT = "KNYKQTSV"
POSTERIOR_COUNT_COLUMNS = tuple(
    f"posterior_{stage}_count_{replicate}"
    for stage in ("starting", "timepoint")
    for replicate in range(1, 4)
)
PROXY_COLUMNS = ("proxy_log_r_mean", "proxy_count_ratio", "proxy_dimsum_fitness")
DEFAULT_COUNT_PATHS = {
    source: ROOT / "data" / f"{source}.tsv" for source in ("trans-1", "trans-2")
}
DEFAULT_HMC_PATHS = {
    "trans-1": ROOT / "amlt/large-clean/large-clean_82/pdz/trans-1/pdz_trans-1_exponential_exponential-20260407030507.nc",
    "trans-2": ROOT / "amlt/large-clean/large-clean_88/pdz/trans-2/pdz_trans-2_exponential_exponential_diagnosed.nc",
}
DEFAULT_FLIP_PATH = ROOT / "data/single_to_double_from_flip_repo.csv"


def available_hmc_draws(hmc_paths: dict[str, Path] | None = None) -> int:
    """Return the maximum number of complete datasets shared by all HMC fits."""
    hmc_paths = DEFAULT_HMC_PATHS if hmc_paths is None else hmc_paths
    available = []
    for path in hmc_paths.values():
        with xr.open_dataset(path, group="posterior", engine="h5netcdf", chunks={}) as posterior:
            available.append(posterior.sizes["chain"] * posterior.sizes["draw"])
    return min(available)


def variant_map(source: str, variants: list[str]) -> pd.DataFrame:
    """Map each SciStanPy variant index to its reconstructed FLIP sequence."""
    aa = pd.Series(variants)
    pdz = aa.str[:50] + WT_PDZ[50:] if source == "trans-1" else WT_PDZ[:50] + aa.str[:50]
    sequence = (
        pdz.str.split("*", n=1, regex=False).str[0]
        + ":"
        + aa.str[-8:].str.split("*", n=1, regex=False).str[0]
    )
    return pd.DataFrame({"b": np.arange(len(aa)), "sequence": sequence})


def _select_targets(
    flip: pd.DataFrame,
    sampled: dict[str, tuple[pd.Series, dict[str, np.ndarray]]],
    replicate: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = pd.concat(
        [
            pd.DataFrame(
                {
                    "sequence": sequences,
                    "hmc_source": source,
                    **{
                        column: values[column][replicate]
                        for column in ("target", *POSTERIOR_COUNT_COLUMNS, *PROXY_COLUMNS)
                    },
                }
            )
            for source, (sequences, values) in sampled.items()
        ],
        ignore_index=True,
    )
    targets = candidates.groupby("sequence", group_keys=False).sample(
        n=1,
        random_state=int(rng.integers(2**32)),
    )
    missing = set(flip["sequence"]) - set(targets["sequence"])
    if missing:
        raise ValueError(f"HMC fits do not cover {len(missing)} FLIP sequences.")
    return targets


def _count_proxies(
    starting_counts: np.ndarray,
    timepoint_counts: np.ndarray,
    wt_starting_counts: np.ndarray,
    wt_timepoint_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate aggregate count ratios and a DiMSum-style fitness proxy."""
    starting_total = starting_counts.sum(axis=1, dtype=float)
    timepoint_total = timepoint_counts.sum(axis=1, dtype=float)
    count_ratio = np.divide(
        timepoint_total,
        starting_total,
        out=np.full_like(timepoint_total, np.nan),
        where=starting_total > 0,
    )

    wt_starting_counts = wt_starting_counts[:, :, None]
    wt_timepoint_counts = wt_timepoint_counts[:, :, None]
    valid = (
        (starting_counts > 0)
        & (timepoint_counts > 0)
        & (wt_starting_counts > 0)
        & (wt_timepoint_counts > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        replicate_fitness = (
            np.log(timepoint_counts / starting_counts)
            - np.log(wt_timepoint_counts / wt_starting_counts)
        )
        variance = (
            1 / starting_counts
            + 1 / timepoint_counts
            + 1 / wt_starting_counts
            + 1 / wt_timepoint_counts
        )
        weights = np.where(valid, 1 / variance, 0.0)
    weight_sum = weights.sum(axis=1)
    dimsum_fitness = np.divide(
        np.where(valid, replicate_fitness * weights, 0.0).sum(axis=1),
        weight_sum,
        out=np.full_like(weight_sum, np.nan),
        where=weight_sum > 0,
    )
    return count_ratio, dimsum_fitness


def make_hmc_flip_replicates(
    n: int,
    *,
    seed: int = 23,
    count_paths: dict[str, Path] | None = None,
    hmc_paths: dict[str, Path] | None = None,
    flip_path: Path = DEFAULT_FLIP_PATH,
) -> list[pd.DataFrame]:
    """Generate N FLIP datasets with targets replaced by joint HMC draws.

    One ``(chain, draw)`` sample is selected per trans fit and generated
    dataset. For sequences present in both fits, one source is selected at
    random and recorded in ``hmc_source``. The published FLIP target is retained
    in ``published_target``.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    count_paths = DEFAULT_COUNT_PATHS if count_paths is None else count_paths
    hmc_paths = DEFAULT_HMC_PATHS if hmc_paths is None else hmc_paths
    flip = pd.read_csv(flip_path)
    rng = np.random.default_rng(seed)
    sampled: dict[str, tuple[pd.Series, dict[str, np.ndarray]]] = {}

    for source, hmc_path in hmc_paths.items():
        counts = load_pdz_dataset(str(count_paths[source]))
        mapping = variant_map(source, cast(list[str], counts["variants"]))

        with xr.open_dataset(hmc_path, group="observed_data", engine="h5netcdf") as observed:
            if not (
                np.array_equal(observed["starting_counts"].values, counts["starting_counts"])
                and np.array_equal(observed["timepoint_counts"].values, counts["timepoint_counts"])
            ):
                raise ValueError(f"{source}: HMC variant order does not match the count loader.")

        rows = mapping[mapping["sequence"].isin(flip["sequence"])].reset_index(drop=True)
        wt_rows = mapping[mapping["sequence"] == f"{WT_PDZ}:{WT_CRIPT}"]
        if len(wt_rows) != 1:
            raise ValueError(f"{source}: expected exactly one wild-type PDZ:CRIPT row.")
        selected_b = np.append(rows["b"].to_numpy(), wt_rows["b"].item())
        with xr.open_dataset(
            hmc_path,
            group="posterior",
            engine="h5netcdf",
            chunks={},
        ) as posterior_dataset:
            posterior = posterior_dataset["log_r_mean"].assign_coords(
                sequence=("b", mapping["sequence"].to_numpy())
            )
            posterior = posterior.stack(sample=("chain", "draw")).transpose("sample", "b")
            if n > posterior.sizes["sample"]:
                raise ValueError(
                    f"Requested {n} datasets, but {source} has only "
                    f"{posterior.sizes['sample']} posterior draws."
                )
            draw_ids = (
                np.arange(posterior.sizes["sample"])
                if n == posterior.sizes["sample"]
                else rng.choice(posterior.sizes["sample"], n, replace=False)
            )
            selected = posterior.isel(
                sample=draw_ids,
                b=xr.DataArray(rows["b"].to_numpy(), dims="variant"),
            )
            if not np.array_equal(selected["sequence"].values, rows["sequence"].to_numpy()):
                raise ValueError(f"{source}: selected HMC rows do not match FLIP sequences.")
            log_r_mean = selected.compute().values

        with xr.open_dataset(
            hmc_path,
            group="posterior_predictive",
            engine="h5netcdf",
            chunks={},
        ) as predictive_dataset:
            predictive = predictive_dataset[["starting_counts", "timepoint_counts"]].isel(
                b=xr.DataArray(selected_b, dims="variant_with_wt")
            )
            predictive = predictive.stack(sample=("chain", "draw")).transpose(
                "sample", "a", "variant_with_wt"
            )
            predictive = predictive.isel(sample=draw_ids).compute()
            starting_counts = predictive["starting_counts"].values[:, :, :-1]
            timepoint_counts = predictive["timepoint_counts"].values[:, :, :-1]
            count_ratio, dimsum_fitness = _count_proxies(
                starting_counts,
                timepoint_counts,
                predictive["starting_counts"].values[:, :, -1],
                predictive["timepoint_counts"].values[:, :, -1],
            )

        values = {
            "target": log_r_mean,
            "proxy_log_r_mean": log_r_mean,
            "proxy_count_ratio": count_ratio,
            "proxy_dimsum_fitness": dimsum_fitness,
        }
        for replicate_index in range(3):
            values[f"posterior_starting_count_{replicate_index + 1}"] = starting_counts[
                :, replicate_index
            ]
            values[f"posterior_timepoint_count_{replicate_index + 1}"] = timepoint_counts[
                :, replicate_index
            ]
        sampled[source] = rows["sequence"], values

    base = flip.rename(columns={"target": "published_target"})
    replicates = []
    for replicate in range(n):
        targets = _select_targets(flip, sampled, replicate, rng)
        dataset = base.merge(targets, on="sequence", validate="one_to_one")
        dataset.insert(0, "replicate", replicate)
        replicates.append(
            dataset[
                [
                    "replicate",
                    "sequence",
                    "target",
                    "hmc_source",
                    *POSTERIOR_COUNT_COLUMNS,
                    *PROXY_COLUMNS,
                    "set",
                    "validation",
                    "published_target",
                ]
            ]
        )
    return replicates


if __name__ == "__main__":
    datasets = make_hmc_flip_replicates(10)
    print(f"Generated {len(datasets)} datasets with {len(datasets[0])} rows each.")