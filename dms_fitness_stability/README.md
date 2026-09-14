# DMS Fitness Stability

Small, reproducible tools for measuring how stable variant rankings and top-K
selections are across fitted deep mutational scanning landscapes.

## Quick start

Install the environment and run the complete generic example:

```bash
uv sync
uv run python examples/tabular/preprocess.py
uv run python examples/tabular/run.py
```

The example represents a previously unknown protein family with a
lower-is-better score. It starts from ordinary CSV files and produces validated
score and feature bundles, a numerical stability result, six core tables, two
mechanism tables, and a PNG figure under `examples/tabular/output/`. It does not
use a paper-family adapter or model-fit NetCDF. See
[examples/tabular/README.md](examples/tabular/README.md) to adapt it to another
landscape.

Run the full test suite with:

```bash
uv run pytest
```

## Two pipelines

The repository separates source interpretation from stability analysis:

1. **Source preprocessing** converts model-fit NetCDFs or family-specific count
	sources into `fitness_scores.nc` and `features.nc` bundles with aligned
	variant tables, provenance, and hashes.
2. **Stability analysis and reporting** consumes only those standard bundles
	plus a small JSON analysis configuration. It produces numerical results,
	tidy CSV tables, and figures.

Most users should start from an existing score matrix or the generic tabular
example. Source adapters are needed only to regenerate a bundle from an original
fit or a family-specific experimental layout.

The score bundle stores `fitness_score[score_name, sample, variant]`. Its
analysis variant table declares candidate eligibility and optional external
reference values. The feature bundle stores one aligned value per variant for
the count-support and replicate-agreement quantities used by current mechanism
analyses. Scores can be higher-is-better or lower-is-better, and an external
reference ranking is optional.

## Legacy toy example

```bash
uv run python examples/toy/run.py
```

The example writes:

```text
examples/toy/results/
├── metadata.json
├── variants.csv
├── stability.nc
├── summary.csv
└── overlaps.csv
```

Large raw datasets and model fits are not committed to Git. On this machine they
live under the canonical external data root:

```text
/data/data_dms_stability/
├── pdz_trans/
│   ├── raw/
│   └── fits/
├── pdz_cis/
│   ├── raw/
│   └── fits/
├── pdz/
│   └── results/
│       ├── trans-1/
│       ├── trans-2/
│       └── cis/
├── trpb/
│   ├── raw/
│   │   ├── counts/
│   │   └── reported_fitness/
│   └── fits/
├── nuc/
│   └── raw/
│       ├── counts/
│       └── reported_fitness/
└── k50/
	└── raw/
		├── NGS_count_lib1.csv
		├── NGS_count_lib2.csv
		├── NGS_count_lib3.csv
		├── NGS_count_lib4.csv
		└── reported_fitness/
```

These are physical files, not symlinks. The data root can be changed without
editing code:

```bash
export DMS_FITNESS_STABILITY_DATA=/path/to/data_dms_stability
```

The PDZ runner also accepts `--data-root`. Explicit `--trans-1-hmc`,
`--trans-2-hmc`, and `--cis-hmc` paths remain available when a fit lives outside
that root.

Validate all copied files before running an analysis:

```bash
uv run python examples/pdz/validate_data.py
```

## NucB and K50 Sources

NucB jobs use the authors' highest-sort-gate enrichment factor as continuous
reported fitness for each generation. The four selected fields and their source
notebook are documented in
[examples/nuc/README.md](examples/nuc/README.md). The merged publication
landscape's categorical activity calls are retained as separate annotations.

K50 job 63 uses the final raw NGS snapshot for model input and a compact
projection of the authors' published Dataset 2/3 `dG_ML` values as the external
reported reference. The projection is joined by exact construct identity and
retains censoring and confidence-interval metadata. The full publication
archives are not staged, and the 443.6 GB job-63 MLE output was intentionally
retired after its canonical score matrix and downstream products were
validated. The exact recovery command is in `data/retired_product_fits.json`.
See [examples/k50/README.md](examples/k50/README.md) for the alignment and
duplicate policies.

## TrpB Landscapes

The TrpB adapter keeps every fitted variant and aligns reported fitness by the
amino-acid key in the original count-table order. It scans `r`, `log_r_mean`, or
`inv_r_mean` as appropriate and expresses all three as posterior log mean growth
for comparable top-K rankings.

After downloading and materializing the fits as described in
`data/README.md`, validate and run one library/model pair with:

```bash
uv run python examples/trpb/validate_data.py
uv run python examples/trpb/run.py \
	--library libA \
	--model exponential-exponential
```

Omit `--library` and `--model` to run all 18 entries in the checked-in job
manifest. Use `--n-samples` for a smaller reproducible posterior subset. By
default, results are written below
`/data/data_dms_stability/trpb/results/<library>/<model>`.

The scanner records fit limitations and fallback use in each `metadata.json`.
See `examples/trpb/KNOWN_ISSUES.md` for the artifacts that should be regenerated.

`examples/trpb/plots.ipynb` applies the same `plot_report` rank, value, and
calibration views used by the PDZ trans notebook to three featured fits and then
all 18 TrpB fits. It also uses the shared multi-result plotting API for:

- overlaid reported-rank calibration ECDFs and their divergence from the
	uniform diagonal; and
- signed 5th/95th-percentile posterior-rank deviations from reported rank.

The compact TrpB analyses retain exact ranks for the reported top 1,000
variants. Accordingly, the notebook compares reported top 100 with stored
reported top 1,000 cohorts; it does not label the latter as the full landscape.
A true full-landscape rank-divergence figure requires rerunning with every
variant tracked.

## PDZ Landscapes

PDZ currently includes `trans-1`, `trans-2`, `cis`, `cript-c`, and `cript-n`.
Each landscape can have multiple fitted model specifications, including
different growth-rate priors and growth curves. A landscape/model/fit
combination is saved as a distinct analysis run; models fitted to the same
landscape remain grouped because they share experimental counts.

Only input preparation differs. The trans count tables encode a mutated PDZ
half plus CRIPT and use `GSE265816_pdz3cript_sub.csv.gz` for published fitness.
The fit operationally named PDZ `cis` is GEO library `cript2xa3_cis`: its count
table encodes eight-residue CRIPT pairwise mutants and it uses
`GSE265816_cript2xa3_sub.csv.gz`. GEO libraries `cript_n` and `cript_c` are two
different landscapes and have their own fit artifacts and result directories.

Some downloaded PDZ/CRIPT fits are locally repairable but remain incomplete and
should be rerun. See [examples/pdz/KNOWN_ISSUES.md](examples/pdz/KNOWN_ISSUES.md)
for artifact-level evidence and the replacement checklist.

Run every manifested PDZ landscape/model fit with all available posterior samples:

```bash
uv run python examples/pdz/validate_data.py
uv run python examples/pdz/run.py
```

Select a subset in the same way that the TrpB runner selects libraries:

```bash
uv run python examples/pdz/run.py --landscape cis
uv run python examples/pdz/run.py --landscape trans-1 trans-2
```

The full scan retains each proxy's posterior top 1,000 and exact value/rank
trajectories for the fixed reported top 1,000. Summary and overlap tables are
generated at K = 50, 100, and 1,000. Results are written to
`/data/data_dms_stability/pdz/results/<landscape>` by default.

After the scans finish, build a compact landscape-by-score overview without
reopening an HMC fit:

```bash
uv run python examples/pdz/summarize.py --k 50
```

## Plotting

Plot functions accept either a saved result directory or a loaded
`StabilityResult`. Passing a directory reads only `metadata.json`, `summary.csv`,
and `overlaps.csv`; it does not load `stability.nc` or reopen the model fit.

One command generates inclusion, overlap, rank-interval, value-interval, and
rank-calibration figures for every saved score:

```python
from dms_fitness_stability import plot_report

figures = plot_report("/data/data_dms_stability/pdz/results/trans-1", k=50)
```

Select one score or a smaller set of views when needed:

```python
figures = plot_report(
	"/data/data_dms_stability/pdz/results/trans-2",
	k=50,
	scores=("log_mean_growth",),
	views=("inclusion", "overlap", "rank", "value"),
)
```

Individual plots are equally direct:

```python
from dms_fitness_stability import plot_inclusion, plot_rank_intervals

plot_inclusion(result_dir, score="count_ratio", k=50)
plot_rank_intervals(result_dir, score="count_ratio", k=50)
```

Compare calibration and rank divergence across multiple saved results with the
same plotting module:

```python
from dms_fitness_stability import (
	plot_calibration_overlay,
	plot_rank_divergence_boxplots,
)

results = {
	"landscape A": "path/to/result-a",
	"landscape B": "path/to/result-b",
}
plot_calibration_overlay(results, score="log_mean_growth", k=50)
plot_rank_divergence_boxplots(
	results,
	score="log_mean_growth",
	k=50,
	reported_top_n=100,
)
```

See `examples/pdz/plots.ipynb` for comprehensive individual and joint reports
across the available PDZ landscape/model analyses.

## Paper Workflow

The minimal paper-facing workflow is in
[examples/paper/README.md](examples/paper/README.md). It treats every complete
downloaded landscape/model combination as an analysis run, adds exact ranks for
the model-defined top cohort, writes six inference-agnostic summary tables, and
uses one table-driven notebook to compose the paper and exploratory figures,
including all-candidate score geometry and decision-relative score uncertainty.
The same compact contract is designed to accept future MLE-bootstrap score samples.

The [figure construction note](examples/paper/README.md#how-the-figures-are-made)
maps every panel to its retained arrays, formula, summary-table producer, and
plotting function. [CODE_GUIDE.md](CODE_GUIDE.md) gives a code-reading order,
explains the compact schema and four scanner passes, and lists safe modification
paths for adding fits, landscapes, scores, statistics, figures, or a future
MLE-bootstrap adapter.

The revised manuscript outline, including the two-stage Section 4 mechanism
claim and Figures 4, 8, 9, and 10 designs, is versioned at
[examples/paper/outline.tex](examples/paper/outline.tex).