# HMC count-based fitness proxy provenance

Last updated: 2026-08-31

This file records how count-derived fitness landscapes are constructed from the
posterior-predictive counts stored in the curated HMC artifacts. The analysis
label is `hmc_count_based`.

## Statistical interpretation

One HMC sample identifies a paired posterior parameter draw and
posterior-predictive count draw. The family-specific fitness proxy is recomputed
from those counts without refitting the model. Variation across proxy landscapes
therefore combines:

1. uncertainty in the HMC posterior parameters; and
2. stochastic variation in a new count realization conditional on those
   parameters.

This is the joint posterior-predictive distribution of a count-derived score. It
is not the posterior distribution of the fitted latent fitness parameter, the
conditional-repeatability target used by `mle_predictive`, or a
resample-and-refit bootstrap.

## Tensor and sample contract

Each persisted bundle contains
`fitness_score[score_name, sample, variant]` as `float32`. The requested matrix
with variants in rows and HMC samples in columns is:

```python
matrix = fitness_score.sel(score_name=score_name).transpose("variant", "sample")
```

HMC chains and draws are flattened in chain-major order:

```text
source_sample_id = source_chain * draws_per_chain + source_draw
```

The `sample` coordinate is the contiguous row number in the persisted bundle.
`source_sample_id`, `source_chain`, and `source_draw` preserve the original HMC
coordinates. Canonical artifacts include every chain and draw; `--n-samples`
is available only for deterministic smoke tests.

## Analysis scope

`analyses_hmc_count_based.json` preserves the exact fit and artifact selections
from `analyses_hmc.json`:

| Family | Runs | Landscapes | Primary score |
| --- | ---: | ---: | --- |
| PDZ | 14 | 5 | `dimsum_fitness` |
| TrpB | 28 | 9 | `johnston_fitness` |
| NucB | 7 | 4 | `highest_gate_enrichment` |
| Total | 49 | 18 | one score per run |

The manifest resolves 40 diagnosed HMC artifacts and 9 validated raw-HMC
fallbacks. Raw fallbacks are deliberate audited selections, including jobs 56
and 78. K50 and TrpB four-site are absent because no HMC fits are available for
those landscapes.

## Score contracts

### PDZ

The implementation uses `pdz-dimsum-fitness-v1`, the same provisional
DiMSum-style approximation used by the sampled-MLE workflow. For each draw it
computes replicate log enrichment relative to the fitted wild type and combines
replicates with inverse Poisson log-ratio variance weights. Replicates with a
zero required count receive zero weight. The exact formula and source
limitations are documented in `MLE_FITNESS_PROXIES.md`.

The contract remains provisional because the exact DiMSum revision and source
configuration have not been recovered. No affine calibration to published
fitness is applied.

### TrpB

The implementation uses the verified `trpb-johnston-fitness-v1` contract. Each
HMC draw is processed in two bounded-memory passes. The first derives stop-codon
means and landscape maxima separately for every retained timepoint, replicate,
and draw. The second applies the source filters and normalization and averages
valid values over retained timepoints and replicates. This is numerically the
same function used for sampled-MLE counts.

### NucB

The implementation uses the verified generation-specific contracts
`nuc-g1-highest-gate-enrichment-v1` through
`nuc-g4-highest-gate-enrichment-v1`. Input and selected-gate frequencies are
normalized over every fitted row, including source controls. Source controls
are excluded only when variants are ranked. G1 retains the source normalization
constants recorded in the adapter.

## Persistence and storage

The canonical dense float32 score matrices contain exactly 144,283,328,000
bytes (134.37 GiB) before NetCDF compression. Eleven PDZ matrices exceed 5 GiB;
the largest are jobs 71, 72, and 73 at approximately 13.0 GiB each. Builders
therefore use one temporary disk-backed memmap per run, process source arrays in
bounded chunks, publish one completed bundle at a time, and remove temporary
files after persistence.

Each score-bundle directory contains:

```text
metadata.json
variants.csv
analysis_variants.csv
fitness_scores.nc
```

Canonical roots are:

```text
/data/data_dms_stability/paper/analysis_data/scores/hmc_count_based/canonical
/data/data_dms_stability/paper/hmc_count_based/results
/data/data_dms_stability/paper/hmc_count_based/mechanisms/cache
```

## Reproduction

```bash
uv run python examples/paper/validate_manifest.py \
  --manifest examples/paper/analyses_hmc_count_based.json
uv run python examples/paper/preflight_hmc_count_based.py
uv run python examples/paper/preprocess_hmc_count_based.py
uv run python examples/paper/preprocess_hmc_count_based.py --execute
uv run python examples/paper/run_hmc_count_based.py
uv run python examples/paper/run_hmc_count_based.py --execute
uv run python examples/paper/audit_hmc_count_based.py
```

Then generate the compact and mechanism tables:

```bash
uv run python examples/paper/summarize.py \
  --manifest examples/paper/analyses_hmc_count_based.json \
  --result-root /data/data_dms_stability/paper/hmc_count_based/results \
  --output-dir examples/paper/output/hmc_count_based/tables
uv run python examples/paper/summarize_mechanisms.py \
  --manifest examples/paper/analyses_hmc_count_based.json \
  --result-root /data/data_dms_stability/paper/hmc_count_based/results \
  --score-root /data/data_dms_stability/paper/analysis_data/scores \
  --cache-root /data/data_dms_stability/paper/hmc_count_based/mechanisms/cache \
  --output-dir examples/paper/output/hmc_count_based/tables
uv run python examples/paper/validate_mechanisms.py \
  --manifest examples/paper/analyses_hmc_count_based.json \
  --table-dir examples/paper/output/hmc_count_based/tables
```

`plots_hmc_count_based.ipynb` consumes only the generated tables and writes
figures under `examples/paper/output/hmc_count_based/figures`.

## Implementation validation

Unit tests compare multi-chain, chunked builders with dense family score
functions and verify chain/draw coordinates and temporary-file cleanup. A
real-artifact smoke test completed on 2026-08-31 for jobs 3, 21, and 59 using
four deterministic HMC samples per fit. It produced score bundles, compact
results, six core reporting tables, mechanism caches, and mechanism tables that
passed `validate_mechanisms.py`. Job 59 was also preflighted, built, scanned,
and audited with all 16,000 HMC samples.

The complete 49-run all-draw score sweep was generated and audited on
2026-09-01: 14 PDZ, 28 TrpB, and 7 NucB bundles, occupying approximately 103
GiB after compression. Compact results were generated for all 49 runs. Forty-six
runs retain every source draw. CRIPT-N jobs 78, 80, and 81 exclude 315, 82, and
161 draws, respectively, because those draws contain fewer than 1,000 finite
ranking candidates. No other run excludes a draw. The joint score/result audit,
paper-table generation, and complete mechanism pass were subsequently completed.

Model-reference candidates require at least two finite proxy scores. This rule
excluded one singleton-supported TrpB libF variant from the reference ordering
for jobs 31 and 33; those two compact results were rebuilt. All 49 results then
passed the corrected reference-support invariant and score/result audit.

The final version-4 mechanism pass produced 49 caches and 14 table files. Table
validation covers 18 landscapes, 49 analysis runs, 58,155 tracked run-variant
rows, 25,915 primary-cohort rows, and 1,627,616 functionality rows. A clean-kernel
execution of `plots_hmc_count_based.ipynb` completed all 26 code cells without
errors and wrote 23 nonblank PNG/PDF figure pairs. Final storage is approximately
103 GiB for score bundles, 18 GiB for compact results, 5.6 GiB for mechanism caches,
126 MiB for tables, and 46 MiB for figures.
