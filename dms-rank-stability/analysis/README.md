# Analysis

The analysis reads only sampled scores, landscape features, and reviewed run
settings. Compact numerical results feed tidy tables, and the paper notebooks
render those tables without opening model fits or raw counts.

`runs.json` declares 158 analyses across parameter-posterior, MLE-predictive,
and posterior-predictive sampling. It records scientific identity, score and
feature paths, score direction, external-reference columns, and the shared
rank settings. It contains no model-fit locations or compute-service IDs.

## Data Location

Run commands from the repository root after following the installation steps
in [../README.md](../README.md). The analysis data directory has this layout:

```text
analysis-data/
  scores/<sampling>/<run-id>/scores.nc
  scores/<sampling>/<run-id>/variants.csv
  features/<landscape>/features.nc
  features/<landscape>/variants.csv
  results/<sampling>/<run-id>/     optional saved numerical results
  tables/<sampling>/              optional saved tables
```

Keep every bundle's files together. Pass the directory containing `scores/`
and `features/`, not either subdirectory, as `--data-root`. Relative input-root
paths are resolved from the current working directory; quote paths containing
spaces. The full scores and features occupy about 223 GB (207 GiB). Saved
results require about 19 GB and the tables about 0.5 GB, plus space for figures.

Without `--data-root`, input paths are resolved relative to the configuration's
repository root. The paper configuration uses `data/downloads/`; the tiny
example uses its own data directory. With an input root, the paper's
`data/downloads/` prefix is replaced by that directory. Configurations using
paths such as `scores/...` and `features/...` resolve directly below it.
Recorded relative paths and scientific settings do not change.

## Recompute the Analysis

Run all analyses from the prepared scores and features:

```bash
uv run python analysis/run.py --data-root "/path/to/analysis-data"
```

The runner validates each score bundle and writes a three-file result below
`analysis/results/<sampling>/<run-id>/`. It then writes fourteen numerical
tables below `analysis/tables/<sampling>/`. A valid result with matching
settings, variants, and source sample IDs is reused. A nonmatching result is
preserved unless `--overwrite` is supplied.

Changing `--data-root` does not redirect outputs. Use `--result-root` and
`--table-root` to choose output locations. To reuse supplied results and
rebuild the complete tables in the notebook's default location:

```bash
uv run python analysis/run.py --data-root "/path/to/analysis-data" --result-root "/path/to/analysis-data/results" --table-root analysis/tables
```

Scores and features are still required when reusing results. Missing results
are computed; matching results are reused. Do not use `--overwrite` unless
you intend to recompute existing results.

Use `--family`, `--sampling`, or `--run-id` to select a subset. This small
read-only check validates one score bundle and reports whether it would run:

```bash
uv run python analysis/run.py --data-root "/path/to/analysis-data" --run-id nuc-g3-parameter-posterior-exponential --dry-run
```

`--dry-run` does not compute results or build tables; feature loading occurs
when tables are built. To execute a subset, use separate output directories:

```bash
uv run python analysis/run.py --data-root "/path/to/analysis-data" --family nuc --sampling parameter_posterior --result-root analysis/results/subset --table-root analysis/tables/subset
```

Table generation replaces the selected sampling directories in `--table-root`
with tables for the selected runs. Keep subset tables separate from complete
paper tables, which the notebooks require. Scores are read in blocks rather
than loaded as one full matrix, but the full analysis can take substantial time.

## Table Contents

Each sampling directory contains:

- Six rank products: `runs`, `rank_percentiles`, `rank_agreement`,
  `model_rank_intervals`, `top_k_curves`, and `top_k_candidates`.
- Four mechanism products: `mechanism_inventory`, `mechanism_variants`,
  `mechanism_runs`, and `mechanism_effects`.
- Three supporting products: `reference_diagnostics`,
  `reference_category_sensitivity`, and `mechanism_diagnostics`.
- `functionality_scores.csv.gz`, containing all candidates with finite median
  scores and their model-reference ranks.

## Recreate Figures

Only the complete tables are required for this workflow. Place the supplied
`tables/` directory at `analysis/tables/` in a fresh checkout, keeping its
`parameter_posterior/`, `mle_predictive/`, and `posterior_predictive/`
subdirectories. The recomputation commands above also create this layout.
There is no need to run the analysis again if these tables are already present.

`figures.py` contains the table loaders and the twelve compositions used by
all three notebooks. The notebooks preserve separate parameter-posterior,
MLE-predictive, and posterior-predictive interpretations while sharing the
same plotting functions.

Execute them from the repository root:

```bash
uv run --group notebook jupyter nbconvert --execute --to notebook --output-dir analysis/executed analysis/plots_parameter_posterior.ipynb
uv run --group notebook jupyter nbconvert --execute --to notebook --output-dir analysis/executed analysis/plots_mle_predictive.ipynb
uv run --group notebook jupyter nbconvert --execute --to notebook --output-dir analysis/executed analysis/plots_posterior_predictive.ipynb
```

Each notebook writes 26 PDF/PNG pairs below
`analysis/figures/<sampling>/` and checks that every expected file is present
and nonempty. Tracked notebooks contain no saved outputs.

Generated results, tables, and figures are not tracked by Git.