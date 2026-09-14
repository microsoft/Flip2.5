# Generic tabular landscape

This example starts from ordinary CSV files for a previously unknown landscape.
It does not use a family adapter, model-fit NetCDF, posterior variable name, or
raw-count layout from the paper analyses.

From the repository root:

```bash
uv run python examples/tabular/preprocess.py
uv run python examples/tabular/run.py
```

The first command reads `source.json`, validates `variants.csv`, `scores.csv`,
and `counts.csv`, then writes the two standard inputs:

```text
output/
├── scores/
│   ├── metadata.json
│   ├── variants.csv
│   ├── analysis_variants.csv
│   └── fitness_scores.nc
└── features/
    ├── metadata.json
    ├── variants.csv
    └── features.nc
```

The second command reads `analysis.json`. It ranks the lower
`measurement_error` values first, writes the five-file numerical result, builds
six core and two mechanism tables, and renders
`output/figures/inclusion.png` using only generic code. The mechanism tables
rescan only the model top K and sampled challengers, join the aligned count
features, and retain exact full-landscape sampled ranks for that cohort.

To adapt this example:

1. Replace the rows in the three CSV files while keeping their column names.
2. Give every variant a unique `variant_id` and an explicit boolean
   `is_ranking_candidate` value.
3. Update the identity and score description in `source.json`.
4. Update the score name, direction, K values, and optional external reference
   in `analysis.json`.
5. Run the same two commands.

`scores.csv` must contain one row for every
`score_name × sample_id × variant_id` combination. `counts.csv` is optional; if
present, it must contain one paired input/output row for every
`replicate_id × variant_id` combination and at least two replicates. Large
matrices should use the blockwise score writer rather than a long CSV.