# DMS Rank Stability

Tools for measuring rank stability in deep mutational scanning landscapes.
The analysis starts from sampled fitness-score matrices and aligned
count-derived landscape features; model fits and raw counts are not required.

## Install

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest
```

## Reproduce the Paper

Run commands from the repository root. The analysis data can be stored in any
local directory containing `scores/` and `features/`. Set `--data-root` to
that directory to run the 158 analyses declared in `analysis/runs.json`:

```bash
uv run python analysis/run.py --data-root "/path/to/analysis-data"
```

Without `--data-root`, the paper runs read from `data/downloads/`. Changing
the input location does not change the run definitions or output locations.
Results and tables are written under `analysis/results/` and `analysis/tables/`.

See [analysis/README.md](analysis/README.md) for the directory layout, selected
runs, reuse of supplied results, and the three figure notebooks. To recreate
figures alone, use the supplied tables; scores, features, and fits are not
needed for that workflow.

The public archive links are not assigned yet. A local copy of the analysis
data can be used now. See [data/README.md](data/README.md) for the data formats
and the download command to use after publication.

## Tiny Example

The checked-in tiny example is available now and runs the same data-to-table
path without an external download:

```bash
uv run python analysis/run.py --config data/example/analysis/runs.json --result-root data/example/output/results --table-root data/example/output/tables
```

## Optional Fit Examples

Six optional fit examples show how the prepared score matrices are produced
for PDZ, TrpB, and NucB. They are separate from the paper rerun and require the
companion fit files. See [preprocessing/README.md](preprocessing/README.md).