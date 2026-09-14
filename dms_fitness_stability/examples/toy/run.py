"""Run the complete stability workflow on a four-variant synthetic landscape."""

from pathlib import Path

import pandas as pd
import xarray as xr

from dms_fitness_stability import AnalysisIdentity, analyze_stability, prepare_counts

HERE = Path(__file__).resolve().parent

variants = prepare_counts(pd.read_csv(HERE / "counts.csv"))
scores = xr.DataArray(
    [
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 4.0, 3.0, 2.0],
        [4.0, 2.0, 3.0, 1.0],
        [3.0, 4.0, 2.0, 1.0],
    ],
    dims=("sample", "variant"),
    coords={"variant": variants["variant_id"].to_numpy()},
    name="growth",
)
result = analyze_stability(
    scores,
    variants,
    identity=AnalysisIdentity(
        protein_family="toy",
        landscape="toy",
        inference_method="synthetic",
        model_spec="demonstration",
        fit_id="fixed-samples",
    ),
    score_name="growth",
    top_k=[1, 2],
    n_pairs=100,
    seed=23,
)
output_dir = result.save(HERE / "results")
print(f"Saved synthetic stability result to {output_dir}")
