# Paper analysis

This directory is the minimal paper-facing workflow for reported-order recovery,
rank uncertainty, and top-K decision stability. The unit is an analysis run:
one landscape, inference procedure, model specification, and fit artifact.
Alternative models for the same landscape remain separate runs but are grouped
by landscape in figures because they use the same experimental counts.

`analyses_hmc.json` is the curated HMC-only manifest: 14 PDZ runs, 28 TrpB runs,
and 7 NucB runs. It resolves 40 diagnosed HMC artifacts and 9 validated raw-HMC
fallbacks through `data/fit_audit.json`. K50 is absent because job 63 is
MLE-only. The older `analyses.json` and `output/tables` directory preserve the
previous 26-run PDZ/TrpB snapshot; they are not the input to the expanded HMC
notebook.

The workspace paper outline is [`outline.tex`](outline.tex). Compile it with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/dms-outline-build examples/paper/outline.tex
```

It uses New PX when available and falls back to MathPazo in smaller TeX
installations.

## 1. Validate source files

```bash
uv run python examples/pdz/validate_data.py
uv run python examples/trpb/validate_data.py
uv run python examples/nuc/validate_data.py \
  --data-root /data/data_dms_stability
```

## 2. Build schema-2 compact results

The canonical launcher reads the validated full posterior score matrices; it
does not open model-fit NetCDFs or refit a model. First inspect the complete
plan, then add `--execute` to build the compact result bundles:

```bash
uv run python examples/paper/run_hmc.py
uv run python examples/paper/run_hmc.py --execute
```

Results are written below `/data/data_dms_stability/paper/hmc/results`. Use
`--family`, `--job`, or `--analysis-id` to process a smaller resumable subset.
The manifest identity and score contract are checked again during
summarization. The legacy fit-backed equivalence command is
`fit_preprocessing/run_hmc_from_fits.py`.

The variant table may distinguish the complete fitted posterior axis from the
candidate-ranking universe through `is_ranking_candidate`. The column is
optional and defaults to all rows for PDZ and TrpB. NucB retains the appended
premature-stop fiducials for exact HMC alignment but excludes them from every
reported cohort, sampled rank, model reference, top-K set, and mechanism
cohort. NucB result metadata therefore records both `n_fitted_variants` and the
ranking-candidate count `n_variants`.

### Build sampled-MLE fitness proxies

`analyses_mle.json` identifies the canonical 61-artifact set: 19 PDZ, 34 TrpB,
and 7 NucB fits plus K50 job 63. The K50 artifact declares separate
chymotrypsin and trypsin scores, so reporting expands the manifest to 62 score
views. `analyses_mle_k50.json` is the focused K50 subset, and
`analyses_mle_pilot.json` lists the three sampled-MLE sources retained locally.
NucB and TrpB use verified source scores; PDZ uses the approved provisional
DiMSum-style approximation; K50 uses two explicitly non-thermodynamic
sensitivity scores. Each path persists a labeled
`score_name x sample x variant` ensemble. For every analysis, select one score
and transpose it to obtain the `variant x sample` matrix used by ranking:

```python
matrix = fitness_score.sel(score_name=score_name).transpose("variant", "sample")
```

Inspect the audited resolution plan, then execute it:

```bash
uv run python examples/paper/preflight_mle.py
uv run python examples/paper/preprocess_mle.py
uv run python examples/paper/preprocess_mle.py --execute
```

Those full-manifest commands document the original build and now require the 58
retired sampled-MLE sources to be restored. The three retained preprocessing
sources can still be resolved locally with:

```bash
uv run python examples/paper/preflight_mle.py \
  --manifest examples/paper/analyses_mle_pilot.json \
  --job 3 59 64
uv run python examples/paper/preprocess_mle.py \
  --manifest examples/paper/analyses_mle_pilot.json
```

K50 job 63 follows its adapter's direct validation and score-audit path rather
than `preflight_mle.py`.

The direct preflight opens every source artifact and verifies embedded observed
counts, predictive axes, nonnegative integer counts, and fixed multinomial
totals. Its resumable report is `data/mle_count_preflight.json`; it covers all
60 PDZ, TrpB, and NucB artifacts. K50 job 63 follows the dedicated K50 adapter
validation and score-audit path. Use `--job` to select a manifest subset. PDZ
and TrpB use native variant chunks. TrpB calculates stop means and landscape
maxima in a first pass and variant scores in a second pass.

Outputs are written below
`/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical`.
Each analysis contains
`metadata.json`, `variants.csv`, `analysis_variants.csv`, and
`fitness_scores.nc`. The source formula, normalization universe, implementation
status, and exact literature/code references are maintained in
[`MLE_FITNESS_PROXIES.md`](MLE_FITNESS_PROXIES.md).

These samples describe repeated count experiments conditional on fixed MLE
parameters. They are not MLE parameter confidence intervals or a
resample-and-refit bootstrap.

After preprocessing, inspect and execute the compact rank-stability scan:

```bash
uv run python examples/paper/run_mle.py
uv run python examples/paper/run_mle.py --execute
```

The standard five-file results are written below
`/data/data_dms_stability/paper/mle_predictive/results`. The scanner reads only
the persisted score ensemble, not the original sampled-MLE artifact.

Audit all persisted score and result contracts with:

```bash
uv run python examples/paper/audit_mle.py --allow-missing-source
```

The audit recomputes finite ranking-candidate support per draw, verifies source
signatures where the source remains, shapes, compression, score variability,
and temporary-file cleanup, then loads every compact result. The canonical
summary contains 62 score views; both K50 scores have 1,000 draws and complete
finite support. A strict audit without `--allow-missing-source` is appropriate
only for a fully restored source set.

After all matrices passed a fresh strict audit on 2026-08-31, 57 sampled-MLE
source NetCDF files were intentionally retired. The K50 job-63 source fit was
retired on 2026-09-03 after its matrix and result were revalidated and the exact
recovery evidence was published. `analyses_mle_pilot.json` identifies the three
remaining preprocessing examples: jobs 3, 64, and 59. All original
experimental count/reference files and all canonical score matrices remain.
See `data/mle_source_retirement.json` and `data/retired_product_fits.json` for
the exact inventory, recovery command, and rebuild limitation.

Build the six core paper tables with:

```bash
uv run python examples/paper/summarize.py \
  --manifest examples/paper/analyses_mle.json \
  --result-root /data/data_dms_stability/paper/mle_predictive/results \
  --output-dir examples/paper/output/mle_predictive/tables
```

Only PDZ jobs 78 and 81 exclude predictive draws: 14 and 33, respectively,
because those draws contain fewer than 1,000 finite ranking candidates. All
other analyses retain 1,000 draws. Original source draw IDs are preserved.

### Build HMC count-based fitness proxies

`analyses_hmc_count_based.json` preserves the 49 fits and audited artifact
choices in `analyses_hmc.json`, but recomputes the MLE-family proxy from each
HMC posterior-predictive count draw. PDZ uses `dimsum_fitness`, TrpB uses
`johnston_fitness`, and NucB uses `highest_gate_enrichment`. K50 and TrpB
four-site are absent because they have no HMC fits.

First validate the manifest and posterior-predictive count tensors:

```bash
uv run python examples/paper/validate_manifest.py \
  --manifest examples/paper/analyses_hmc_count_based.json
uv run python examples/paper/preflight_hmc_count_based.py
```

Inspect the matrix and storage plan, then build the persisted score ensembles:

```bash
uv run python examples/paper/preprocess_hmc_count_based.py
uv run python examples/paper/preprocess_hmc_count_based.py --execute
```

Canonical artifacts use every HMC chain and draw. The dense float32 matrices
total 134.37 GiB before compression, so preprocessing is sequential,
disk-backed, resumable, and guarded by a free-space check. `--n-samples` is
available for deterministic smoke tests, not for canonical output.

Run and audit the compact analysis:

```bash
uv run python examples/paper/run_hmc_count_based.py
uv run python examples/paper/run_hmc_count_based.py --execute
uv run python examples/paper/audit_hmc_count_based.py
```

Score bundles are written below
`/data/data_dms_stability/paper/analysis_data/scores/hmc_count_based/canonical`;
result bundles remain below `/data/data_dms_stability/paper/hmc_count_based`.
Generate the paper tables and mechanism tables with:

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

Open `plots_hmc_count_based.ipynb` after the tables are complete. These score
distributions combine posterior parameter uncertainty and stochastic
posterior-predictive count variation. They are distinct from both the latent
fitness-parameter posterior in `plots.ipynb` and the fixed-MLE conditional
repeatability distribution in `plots_mle.ipynb`. Formula and tensor provenance
is recorded in
[`HMC_COUNT_BASED_PROXIES.md`](HMC_COUNT_BASED_PROXIES.md).

On 2026-09-03, all 61 MLE-predictive and 49 HMC count-based score bundles were
physically converted to the same schema-2 `ScoreEnsemble` contract used by
direct HMC. Score arrays, variant mappings, results, mechanism caches, tables,
and figures passed the frozen pre-conversion audits exactly. The activation,
manifest hash, rollback inventory, and audit links are recorded in
`data/score_schema_v2_activation.json`.

## 3. Build tidy tables

```bash
uv run python examples/paper/summarize.py
```

The command writes six CSV files below `examples/paper/output/hmc/tables/`:

- `inventory.csv`
- `rank_percentiles.csv`
- `rank_agreement.csv`
- `model_rank_intervals.csv`
- `top_k_curves.csv`
- `top_k_candidates.csv`

The tables contain `analysis_id`, landscape, inference procedure, model
specification, fit ID, sample count, and artifact status. The overlap curves are
calculated exactly from stored selections rather than estimated from random
sample pairs.

## 4. Compose figures

Open and run `plots.ipynb`. It reads the expanded tables, including the NucB
runs, from `examples/paper/output/hmc/tables/` and writes PDF and PNG figures to
`examples/paper/output/hmc/figures/`; it never opens an HMC file.

The reported-rank percentile panels are descriptive. Their scalar is named
`ECDF-to-uniform area`; it is not interpreted as calibration or as a monotone
measure of rank recovery.

## 5. Build the Section 4 mechanism analysis

Section 4 separates two linked stages:

```text
count support and replicate/observation agreement -> sampled score uncertainty
sampled score uncertainty x local crowding -> rank and top-K uncertainty
```

Generate the observational mechanism tables with:

```bash
uv run python examples/paper/summarize_mechanisms.py
uv run python examples/paper/validate_mechanisms.py
```

For the MLE-predictive tables, use:

```bash
uv run python examples/paper/summarize_mechanisms.py \
  --manifest examples/paper/analyses_mle.json \
  --result-root /data/data_dms_stability/paper/mle_predictive/results \
  --score-root /data/data_dms_stability/paper/analysis_data/scores \
  --cache-root /data/data_dms_stability/paper/mle_predictive/mechanisms/cache \
  --output-dir examples/paper/output/mle_predictive/tables
uv run python examples/paper/validate_mechanisms.py \
  --manifest examples/paper/analyses_mle.json \
  --table-dir examples/paper/output/mle_predictive/tables
```

The first run rereads each standard score bundle and can
take substantial I/O time. Per-analysis caches are resumable and keyed by
source path, size, modification time, score contract, and tracked cohort. Use
`--analysis-id <ID>` for a one-run smoke check. The migration-only
`--fit-backed` flag retains the original fit-artifact path under
`fit_preprocessing`; normal reporting does not open model fits.

This is not a second model fit or uncertainty analysis. The compact stability
bundles retain sample top-1,000 sets, the reported top 1,000, and the model top
100. Figures 4--10 additionally require posterior medians for every candidate
and complete score/rank trajectories for outside challengers whose identities
are known only after top-100 inclusion probabilities are calculated. Those
fields were not retained by the original compact-result schema, so the
mechanism builder reconstructs them in one targeted and one full-candidate HMC
pass. Figures 1--3 do not need this pass. A future result schema should retain
the all-candidate medians and post-selection challenger cohort during the
primary scan so the extra HMC read can be removed.

The implemented mechanism definitions are:

- **Count support:** `log1p(total input count)`. Output counts and zero counts
  remain sensitivity variables because output abundance partly reflects
  fitness.
- **PDZ replicate discordance:** calculate each replicate's smoothed
  output/input log enrichment from within-library proportions using a 0.5
  pseudocount, approximate its variance by `1/(input+0.5) + 1/(output+0.5)`,
  and report inverse-variance-weighted residual heterogeneity divided by its
  replicate degrees of freedom.
- **TrpB D--I output-replicate discordance:** calculate smoothed output log
  proportions for the two replicates at each timepoint, remove the weighted
  timepoint mean, and pool residual heterogeneity across timepoints. The shared
  input library cancels from within-timepoint contrasts. TrpB A--C have one
  output observation per timepoint, so replicate discordance is unavailable;
  timepoint changes are not mislabeled as replicate disagreement.
- **Score uncertainty:** 5th--95th percentile width of exact sampled latent
  `log_mean_growth`, divided by the IQR of median scores among the model-reference
  top 1,000. Raw widths are retained.
- **Local crowding:** median-score drop per rank across five positions above and
  below the focal rank, normalized by the same top-1,000 score IQR. Smaller
  spacing is greater crowding. Fixed 1% and 2% IQR score windows provide a
  sensitivity definition independent of focal posterior width.
- **Boundary difficulty:** median score uncertainty among reference ranks
  81--100 divided by the median score drop per rank across a fixed window around
  rank 100. The direct rank-100/101 gap is retained as a diagnostic. Its decision
  response is `1 - sample_self_consistency` at `K=100`.

For HMC, the command uses latent score samples: PDZ and TrpB use
`log_mean_growth`, and NucB uses modeled log fluorescence. For MLE-predictive
analysis, it uses DiMSum-style, Johnston-fitness, highest-gate-enrichment, or
protease-specific K50 enrichment proxy draws. MLE widths are conditional
predictive proxy widths, not parameter posterior widths. It writes:

- `mechanism_inventory.csv`: one row per experimental landscape, including
  verified count-array shapes, observation structure, depths, zeros, and
  artifact warnings;
- `mechanism_variants.csv`: exact sampled score/rank uncertainty, count support,
  observation discordance, local median-score crowding, and cohort flags;
- `mechanism_runs.csv`: rank-100 boundary spacing, boundary score uncertainty,
  uncertainty-to-spacing ratio, and top-100 instability;
- `mechanism_effects.csv`: standardized descriptive within-run slopes and
  landscape-level medians; and
- `reference_diagnostics.csv`: direct model-versus-reported Spearman and
  Kendall agreement, current and reversed top-100 overlap, WT ranks, fitted and
  candidate counts, and within-landscape model top-100 overlap;
- `reference_category_sensitivity.csv`: categorical composition of model and
  continuous-reference top-100 sets when a categorical annotation is
  available; for NucB this uses the Thomas et al. activity taxonomy;
- `mechanism_diagnostics.csv`: Figure 10 Spearman associations for the full
  primary cohort, model top 100, and outside challengers, plus a rank-residual
  partial Spearman adjusted for input support; and
- `functionality_scores.csv.gz`: posterior-median latent score and fixed
  median-score reference rank for every ranking-eligible candidate in every
  analysis.

The effect table is descriptive. Variants within a run share posterior draws,
and score interval width, rank interval width, and inclusion probability are
related summaries of the same fitted uncertainty. Coefficients and $R^2$ values
therefore describe the observed two-stage association; they are not
independent-variant causal tests or calibration evidence.

The primary cohort is the median-score top 100 plus outside variants with at
least 1% sampled top-100 inclusion probability. Reported top 100/1,000 variants
are retained as context when they have at least two finite score draws. Six
TrpB runs omit 10 reported-context rows under this rule; no model-top member or
sampled challenger is relaxed. Because many challengers are not in the original
exact-rank cohorts, the summarizer rereads the score source in bounded-memory
passes and caches only the required union. It does not refit a model.

Execute the parallel MLE notebook with:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace \
  examples/paper/plots_mle.ipynb
```

`plots_mle.ipynb` writes 29 nonblank PNG/PDF pairs below
`examples/paper/output/mle_predictive/figures`. Shared plots include K50 when
the statistic is comparable. Figure 3 retains its PDZ/TrpB/NucB comparison and
adds a dedicated K50 figure comparing chymotrypsin and trypsin without an
arbitrary published-top curve. Figures 7--10 add K50-specific outputs. The
representative shared runs are PDZ job 3, TrpB job 21, and NucB job 55.

Run the notebook again after the tables are generated. Figure 4 shows:

- **(a)** within-run input-support deciles versus normalized 90% sampled score
  uncertainty;
- **(b)** within-run score-uncertainty deciles versus median 90% sampled rank
  uncertainty;
- **(c)** posterior score intervals in a more-separated local score
  neighborhood; and
- **(d)** posterior score intervals in a crowded neighborhood from the same
  fit, with a focal variant of similar score uncertainty.

Panels (c) and (d) use a deterministic matched pair from the model-reference
top 100. Eligible focal variants have complete five-rank neighborhoods, ranks
10--95, are at most 15 ranks apart within one analysis run, and have normalized
score-interval widths within a factor of 1.2. The selected pair has the largest
local-spacing contrast among eligible pairs. Selection uses score uncertainty,
spacing, and rank only; it does not use sampled-rank width. The current tables
select TrpB libB exponential--sigmoid reference ranks 10 and 23, whose focal
score widths are 1.45 and 1.42 top-1,000 score-IQR units, local spacings are
0.317 and 0.014 IQR per rank, and sampled-rank widths are 58 and 142 ranks.

Replicate discordance, the uncertainty-by-crowding regression, and rank-100
uncertainty-to-spacing summaries remain in the generated tables as supporting
analyses rather than main-figure panels.

These are observational associations. They do not establish that count depth,
discordance, or crowding causes the measured uncertainty. See
[`FIT_PROVENANCE.md`](FIT_PROVENANCE.md) for the information required before
controlled downsampling or replicate-ablation refits can be implemented.

Posterior subsampling with `--n-samples` is only a Monte Carlo sensitivity
analysis. It is not read downsampling and cannot test the effect of sequencing
depth.

## Why this notebook uses different plotting code

The older [`examples/pdz/plots.ipynb`](../pdz/plots.ipynb) and
[`examples/trpb/plots.ipynb`](../trpb/plots.ipynb) are comprehensive exploratory
report books. They call the shared
[`plot_report()`](../../src/dms_fitness_stability/plotting.py#L857) interface for
each fit and score, producing separate inclusion, fixed-$K$ overlap, reported-
cohort rank interval, score interval, and reported-rank percentile plots. They
also compare `log_mean_growth` with posterior-predictive count-ratio scores and
show many individual fits. Those views remain useful for checking one result,
comparing score definitions, and diagnosing unusual models.

The paper plan asks for a smaller set of different statistical objects. The
new code in [`figures.py`](figures.py) is a composition layer for those objects;
it does not replace the old diagnostic API.

| Paper need | Why the old helper was not sufficient | Current implementation |
| --- | --- | --- |
| Separate published-versus-model disagreement from uncertainty under one fixed analysis | [`plot_rank_intervals()`](../../src/dms_fitness_stability/plotting.py#L572) selects variants through [`_reported_rows()`](../../src/dms_fitness_stability/plotting.py#L543), so the displayed cohort and vertical ordering are defined by the publication | Figure 2 uses the median-score model-reference top 100 and its exact full-landscape sampled ranks |
| Summarize uncertainty at the candidate boundary across every landscape/model run | The old rank-divergence boxplots are centered on reported rank and can be dominated by one very large landscape/model scale | Figure 2 shows model-reference ranks 81--100 directly, then reports interval width and top-100 exit fraction per analysis run |
| Show decision stability as a function of $K$ | [`plot_overlap()`](../../src/dms_fitness_stability/plotting.py#L494) is a two-box summary at one $K$, based on the saved random sample-pair table | Figure 3 uses exact dense-$K$ curves for sample self-consistency, overlap with the median-score model reference, and overlap with the reported set |
| Show incumbents and plausible challengers around the model-defined cutoff | [`plot_inclusion()`](../../src/dms_fitness_stability/plotting.py#L419) simply takes the variants with the largest inclusion probabilities and marks reported membership | Figure 3 deliberately includes model-reference ranks 91--100 plus the ten strongest outside challengers and marks both model and reported membership |
| Assemble three manuscript figures rather than hundreds of diagnostic plots | The old notebooks iterate through every fit and available score | The paper notebook reads six validated tables and calls exactly three multi-panel composition functions |

Figure 1 is closest to an old view: both use the reported-rank percentile ECDF.
The paper version keeps it as a descriptive published-versus-model diagnostic,
uses the safer name `ECDF-to-uniform area`, does not order runs as though smaller
area meant better recovery, and places PDZ, TrpB, and the scalar summary in one
figure.

The visual language has now been aligned with the older notebooks even though
the statistical content remains paper-specific. In particular,
[`figures.py`](figures.py#L15) reuses `DEFAULT_COLOR`, `REFERENCE_COLOR`, and
`INTERVAL_COLOR` from the shared plotting module. The paper style in
[`set_paper_style()`](figures.py#L91) uses the same DejaVu Sans typography,
light-gray grid, blue sample/model marks, green reported-reference marks,
pale-blue intervals, and green/gray rank regions. The extra landscape colors
and model line styles are necessary only because the paper panels show many
landscape/model runs at once.

## How the figures are made

### Shared definitions and data flow

Each posterior draw is one sample landscape: it assigns a latent
`log_mean_growth` score to every variant and therefore induces one complete
ranking. Rank 1 is highest fitness. A future MLE bootstrap will use bootstrap
samples in exactly the same role.

The fixed model reference is the full-candidate ordering induced by each
variant's median sampled `log_mean_growth`. It is computed by
[`select_reference_top_k()`](../../src/dms_fitness_stability/streaming.py#L8),
with `variant_index` used to break exact median-score ties. This is the rank of
the median score; it is not the median sampled rank. For each variant, we
calculate its median `log_mean_growth` across samples. We then sort all
candidates by these per-variant medians to obtain one fixed model-reference
ordering. This is the ordering induced by median scores, not the per-variant
median of the sampled ranks.

The large HMC files are processed in bounded-memory blocks by
[`scan_pdz_hmc()`](../../src/dms_fitness_stability/adapters/pdz.py#L1000) and
[`scan_trpb_hmc()`](../../src/dms_fitness_stability/adapters/trpb.py#L420). When
`--tracked-model-top-n 100` is requested, each scanner makes four passes:

1. collect score trajectories for the reported top 1,000;
2. scan every candidate, retain each sample's ordered top 1,000, complete exact
  ranks for the reported cohort, and construct the median-score model
  reference;
3. collect score trajectories for the model-reference top 100; and
4. scan every candidate again to calculate exact full-landscape ranks for those
   100 variants.

The resulting draw-level arrays are saved in `stability.nc`. The manifest
summarizer in [`summarize_manifest()`](summarize.py#L65) validates every
`analysis_id` and calls
[`build_paper_tables()`](../../src/dms_fitness_stability/reporting.py#L345).
That function creates the six CSV tables. Figure composition then happens in
[`figures.py`](figures.py); the notebook only selects settings, displays the
figures, and calls [`save_figure()`](figures.py#L1137).

### Figure 1: reported ordering

**Question.** If the published ordering is compared with rankings produced by
the fitted analysis, where does each reported rank fall in its sampled rank
distribution?

For reported variant $i$, let $r_i^{\mathrm{rep}}$ be its reported rank and
$R_i$ its sampled rank. The plotted percentile is

$$
u_i = P(R_i < r_i^{\mathrm{rep}})
      + \frac{1}{2}P(R_i = r_i^{\mathrm{rep}}).
$$

[`_rank_percentiles()`](../../src/dms_fitness_stability/reporting.py#L146)
computes $u_i$ directly from the exact `tracked_rank` array and writes
`rank_percentiles.csv`. It also calculates the absolute area between the
empirical CDF $F$ and the uniform diagonal,

$$
A = \int_0^1 |F(t)-t|\,dt,
$$

using
[`ecdf_to_uniform_area()`](../../src/dms_fitness_stability/reporting.py#L88).
[`figure_reported_order()`](figures.py#L192) reads only
`inventory.csv` and `rank_percentiles.csv` and creates:

- **(a)** one annotated example ECDF, currently PDZ cis under the
  Lomax-Exponential model;
- **(b)** all PDZ analysis-run ECDFs, with color identifying landscape and line
  style identifying model specification;
- **(c)** the corresponding TrpB ECDFs; and
- **(d)** the corresponding NucB ECDFs;
- **(e)** one ECDF-to-uniform area per analysis run, grouped by landscape; and
- **(f)** direct Spearman concordance between reported rank and each variant's
  posterior-median sampled rank for the same reported cohort.

This is a directional, descriptive diagnostic. An area near zero means the
reported ranks occupy approximately uniform sampled percentiles; it does not
mean that sampled ranks are close to reported ranks. In particular, exact rank
recovery puts every $u_i$ at 0.5 and gives area 0.25. Direct differences between
the reported ordering and the median-score model reference are therefore also
saved in `rank_agreement.csv` by
[`_rank_agreement()`](../../src/dms_fitness_stability/reporting.py#L191). Panel
(f) provides the more familiar monotone diagnostic: positive values indicate
concordant ordering, zero indicates little monotone relation, and negative
values would support a reversed ordering.

### Figure 2: rank uncertainty under a fixed analysis

**Question.** After fixing the landscape, inference procedure, model
specification, and score definition, how far can model-prioritized variants move
in the full ordering?

[`_model_rank_intervals()`](../../src/dms_fitness_stability/reporting.py#L216)
reads the exact `model_tracked_rank` trajectories for the model-reference top
100 and writes `model_rank_intervals.csv`. For every variant it records the 5th,
50th, and 95th sampled rank percentiles, the 90% interval width, inclusion
probability in the top 100, and whether the interval extends below the top-100
decision boundary.

[`figure_rank_uncertainty()`](figures.py#L495) creates:

- **(a)** median-score reference ranks 1--50 under PDZ
  cis/Lomax-Exponential;
- **(b)** median-score reference ranks 1--50 under TrpB
  libD/Gamma-Exponential;
- **(c)** the distribution of 90% rank-interval widths in raw rank positions
  among the model-reference top 100 for every analysis run, with 25th--75th
  percentile boxes, 5th--95th percentile whiskers, and median markers on a log
  scale; the vertical guide at width 100 marks one top-100 span; and
- **(d)** the fraction of each run's model-reference top 100 whose
  95th-percentile sampled rank is greater than 100.

In panels (a) and (b), every row label begins with its median-score reference
rank and places the variant name in parentheses. The blue circle is the median
sampled rank, the pale-blue horizontal line is the 5th--95th percentile sampled
rank interval, the green diamond is the published rank, and the green region is
the top 100. Panel (c) also retains raw rank positions so its widths are read
against the fixed top-100 decision scale. The detailed plotting logic is in
[`_rank_interval_panel()`](figures.py#L391). The published-rank marker measures
published-versus-model displacement; the interval measures finite-sample
uncertainty conditional on the fitted analysis. Panel (d) puts its exact rule
on the x-axis: `95th-percentile sampled rank > 100`.

### Figure 3: top-K decision stability

**Question.** Do plausible sampled landscapes select the same candidates, and
do those candidates agree with fixed model and reported reference sets?

For sample $s$, let $T_k^{(s)}$ be its top-$k$ set. If $c_i(k)$ is the number of
samples in which variant $i$ enters that set and $S$ is the number of samples,
[`_top_k_curves()`](../../src/dms_fitness_stability/reporting.py#L259) computes
the exact distinct-sample self-consistency

$$
C_k = \frac{\sum_i c_i(k)[c_i(k)-1]}{S(S-1)k}.
$$

It also computes

$$
C_k^\star = \frac{1}{k}\sum_{i\in T_k^\star}\frac{c_i(k)}{S},
\qquad
O_k^{\mathrm{rep}} =
\frac{1}{k}\sum_{i\in T_k^{\mathrm{rep}}}\frac{c_i(k)}{S},
$$

where $T_k^\star$ is the fixed median-score model-reference set and
$T_k^{\mathrm{rep}}$ is the reported top-$k$. These values, together with
candidate-cloud sizes at several inclusion-probability thresholds, are written
to `top_k_curves.csv`. Variant-level probabilities and reference-set flags are
written to `top_k_candidates.csv` by
[`_top_k_candidates()`](../../src/dms_fitness_stability/reporting.py#L301).

[`figure_top_k_stability()`](figures.py#L758) creates:

- **(a)** model-reference ranks 91--100 and the ten strongest outside
  challengers for PDZ cis/Lomax-Exponential;
- **(b)** the corresponding TrpB libD/Gamma-Exponential candidates;
- **(c)** the corresponding NucB G1/Exponential candidates;
- **(d)** $C_k$, $C_k^\star$, and $O_k^{\mathrm{rep}}$ for PDZ;
- **(e)** the same three curves for TrpB; and
- **(f)** the same three curves for NucB.

Every candidate label begins with its median-score reference rank; the name in
parentheses is secondary. Candidate-panel construction is in
[`_candidate_panel()`](figures.py#L605).
For the family curves, [`_stability_curve_panel()`](figures.py#L685) draws every
analysis run faintly, takes the median across model/inference runs within each
landscape, and then takes the median across landscapes for the bold family
curve. This gives each landscape equal weight rather than treating several
models fit to the same counts as independent experiments.

The evaluated grid is `K = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000`; every
value is labeled on the logarithmic x-axis and marked on the bold curves. The
vertical dotted guide marks the main detailed analysis at `K = 100`, while a
lighter guide identifies the supporting `K = 50` view. The legend spells out
the three expected shared fractions, and each panel title states that thin
curves are individual analysis runs while thick curves are landscape-weighted
medians.

The current notebook settings are `MAIN_K = 100` and one explicitly selected
example analysis ID per protein family. Change the examples or $K$ in the setup cell of
[`plots.ipynb`](plots.ipynb). If a larger $K$ is requested, the compact bundles
must already contain at least that many `model_tracked_rank` trajectories.

For NucB, the reported comparator is the generation-specific highest-sort-gate
enrichment factor: high re-sequenced gate for G1, 97.5th percentile for G2,
99th percentile for G3, and 99.5th percentile for G4. It is a continuous,
gate-specific assay proxy, while the HMC likelihood combines all available
gates. The separate categorical activity taxonomy is reported in
`reference_category_sensitivity.csv`; it is not substituted for the continuous
ranking because its few levels create extensive ties.

### Figure 4: from experimental support to ranking resolution

**Question.** Why do some fitted variants have uncertain ranks, and why can two
variants with similar score uncertainty have different rank uncertainty?

For variant $i$ in posterior sample $s$, let $z_i^{(s)}$ be transformed latent
`log_mean_growth`, with larger values meaning greater fitness. Its central 90%
posterior score width is

$$
W_i^z=Q_{0.95}(z_i)-Q_{0.05}(z_i).
$$

The model-reference ordering is fixed by sorting each candidate's posterior
median $m_i=Q_{0.50}(z_i)$ in descending order. Separately, every posterior
sample ranks all candidates by $z_i^{(s)}$ to give an exact full-landscape
sampled rank $R_i^{(s)}$. The central 90% rank width is

$$
W_i^R=Q_{0.95}(R_i)-Q_{0.05}(R_i),
$$

in rank positions. The fixed median-score rank and median sampled rank need not
agree because ranking and taking medians do not commute.

The `IQR` in Figure 4 is neither of those intervals. It is one score-scale ruler
per run:

$$
S_{\mathrm{run}}
=Q_{0.75}(m_{(1:1000)})-Q_{0.25}(m_{(1:1000)}),
$$

where $m_{(1:1000)}$ are the first 1,000 posterior-median scores in the fixed
reference ordering. The relative score uncertainty is
$U_i=W_i^z/S_{\mathrm{run}}$. Thus $U_i=1$ means that one variant's central 90%
score interval is as wide as the middle 50% cross-variant spread of median
scores among the run's top 1,000. The IQR is a unit conversion for comparing
fits with different score scales; it is not uncertainty for variant $i$.

Panels (a) and (b) use the model-reference top 100 plus outside variants with at
least 1% posterior top-100 inclusion probability. Bins are constructed within
each analysis run, so their x-axes show relative decile, not a shared numerical
threshold:

- **(a)** sorts variants by `log1p(total input count)`. Each point is the median
  $U_i$ in one input-support decile. A downward curve means that better input
  support is associated with narrower fitted score intervals.
- **(b)** sorts variants by $U_i$. Each point is the median $W_i^R$ in one
  score-uncertainty decile. An upward curve means that wider fitted score
  intervals are associated with wider sampled-rank intervals.

The connecting segments are not regression fits. Faint lines are individual
landscape/model runs. Bold curves first take a median across fitted models
within each landscape and then a median across landscapes in one protein
family, giving each landscape equal final weight. Both y-axes are logarithmic;
the underlying units remain relative score width in panel (a) and rank positions
in panel (b). No confidence band is shown.

Panels (c) and (d) hold the analysis run and approximate score uncertainty fixed
while contrasting local median-score geometry. For focal reference rank $r$,

$$
D_r=\frac{m_{(r-5)}-m_{(r+5)}}{10S_{\mathrm{run}}}
$$

is the mean median-score drop per rank across ten steps. Smaller $D_r$ means a
more crowded neighborhood. The deterministic selector considers top-100 focal
ranks 10--95, requires complete five-rank neighborhoods, no more than 15 ranks
between focal variants, and a score-width ratio no larger than 1.2. It chooses
the eligible same-run pair with the largest spacing contrast. It does not use
rank width or inclusion probability during selection.

The current pair is TrpB libB under Exponential--Sigmoid:

| Quantity | More-separated (c) | Crowded (d) |
|---|---:|---:|
| fixed reference rank | 10 | 23 |
| raw score width $W_i^z$ | 0.354 | 0.346 |
| run score IQR $S_{\mathrm{run}}$ | 0.244 | 0.244 |
| relative score width $U_i$ | 1.449 | 1.418 |
| local spacing $D_r$ | 0.317 | 0.014 |
| sampled-rank 5th, 50th, 95th percentiles | 8, 16, 66 | 13, 40, 155 |
| rank width $W_i^R$ | 58 | 142 |

Each example panel displays 11 fixed reference ranks. A point is a variant's
posterior-median score and a horizontal line is its central 90% posterior score
interval. Scores are centered on the focal median and divided by the common run
IQR; the panels share horizontal limits. The highlighted row is the focal
variant. Horizontal interval overlap is only an illustration: exact rank widths
come from joint full-landscape sample rankings, including posterior correlation
and unshown candidates. Panel (c) is more separated, not fully resolved, and
neither panel shows the rank-100 boundary.

Figure 4 gives a model-conditional, proximate explanation: input support is
associated with score precision, score precision is associated with rank
precision, and local score spacing changes how score uncertainty appears in the
ordering. It does not estimate the causal effect of adding reads. That requires
altering counts and refitting. The complete table-column and implementation
trace is in the [Figure 4 section of `CODE_GUIDE.md`](../../CODE_GUIDE.md#figure-4-exact-quantities-and-data-path).

## Figure 8: median fitness-score geometry

Figure 8.1 contains one panel for each PDZ landscape, Figure 8.2 one panel for
each TrpB landscape, and Figure 8.3 one panel for each NucB generation. Each
curve is a Scott-bandwidth Gaussian KDE of the posterior-median primary latent
score over every ranking-eligible candidate: `log_mean_growth` for PDZ and TrpB, and
modeled log fluorescence for NucB. Multiple model analyses of one landscape are
overlaid in the same panel. The model-specific vertical line marks fixed
median-score reference rank 100. All panels within one protein-family figure
share score and density limits, making distribution ranges and concentration
directly comparable.

These KDEs describe between-candidate central score geometry. They do not show
within-candidate posterior uncertainty. For CRIPT-N Gamma--Sigmoid, the first
100 median scores span about 0.085 score units, while the median central 90%
posterior score width among those variants is about 1.54 units. The latter is
nearly 18 times the former and must be shown separately. Here, *functionality*
is an operational model-conditional score distribution, not an experimentally
validated functional/nonfunctional classification.

## Figure 9: decision-relative fitness and score uncertainty

Figure 9.1 contains one panel for each PDZ analysis run, Figure 9.2 one panel for
each TrpB run, and Figure 9.3 one panel for each NucB run. The plotted cohort is
the fixed median-score top 100 plus every outside candidate with at least 1%
sampled top-100 inclusion probability. For variant $i$, define

$$
x_i=\frac{m_i-m_{(100)}}{S_{\mathrm{run}}},
\qquad
y_i=\frac{Q_{0.95}(z_i)-Q_{0.05}(z_i)}{S_{\mathrm{run}}},
$$

where $m_i=Q_{0.50}(z_i)$, $m_{(100)}$ is the score at fixed reference rank
100, and $S_{\mathrm{run}}$ is the IQR of posterior-median scores among fixed
reference ranks 1--1,000. Thus $x=0$ is the top-100 boundary, positive values
are fixed top-100 members, and negative values are outside challengers. The
log-scaled y-axis is one candidate's central 90% posterior score width in the
same run-specific score units.

Blue filled points denote fixed top-100 members and orange open points denote
outside challengers. Each panel reports the within-run Spearman correlation.
The y-limits are shared within protein family; x-limits remain panel-specific so
large challenger clouds do not compress other runs. Across the 49 runs, the
correlation ranges from -0.961 to 0.727 and is negative in 42 runs: 13 of 14 PDZ
runs, all 28 TrpB runs, and only 1 of 7 NucB runs. NucB therefore shows that
lower decision-relative score is not universally associated with wider
posterior score intervals.

This is a descriptive, model-conditional relationship. Scores and intervals
come from the same posterior samples, variants within a run are dependent, and
multiple models of one landscape share counts. Figure 9 helps localize where
score uncertainty appears, but does not establish that fitness causes
uncertainty or replace the local-spacing and joint-rank analyses.

## Figure 10: per-analysis summary of why the top 100 is uncertain

Figure 10.1 contains one row per PDZ analysis, Figure 10.2 one row per TrpB
analysis, and Figure 10.3 one row per NucB analysis. Every row uses the same
primary decision cohort as Figure 9: the fixed median-score top 100 plus outside
variants with at least 1% sampled top-100 inclusion probability. Four linked
columns summarize the proposed mechanism:

1. **Input support to score uncertainty.** Plot every candidate at
  $\log_{10}(n_i^{\mathrm{input}}+1)$ against $U_i$. Zero counts remain at
  zero. The line is an ordinary least-squares descriptive trend in
  log-count/log-uncertainty coordinates.
2. **Output abundance to score uncertainty.** Repeat the point-level display
  using $\log_{10}(n_i^{\mathrm{output}}+1)$. This is a separate sensitivity
  diagnostic because output abundance is partly determined by fitness; it is
  not interchangeable with pre-selection input support.
3. **Decision margin to score uncertainty.** Plot the Figure 9 coordinates
  $X_i=(m_i-m_{(100)})/S_{\mathrm{run}}$ and $U_i$, with $X_i=0$ marking the
  fixed rank-100 boundary.
4. **Decision margin to top-100 inclusion.** Plot $X_i$ against $q_i(100)$,
  with guides at $X_i=0$ and $q_i(100)=0.5$. A sharp transition indicates a
  resolved boundary; a broad mixed transition indicates unstable membership.

Filled blue points are fixed top-100 members and open orange points are outside
challengers. Input and output columns share one zero-preserving count scale
within protein family. The first three columns share the same log-$U_i$ scale.
The last two columns share a panel-specific $X_i$ range within each analysis
row. Row labels report top-100 instability and challenger count.

Across all 49 runs, the score-uncertainty correlation is negative for input
count (range -0.993 to -0.018) and output count (range -0.998 to -0.037). The
decision-margin correlation is more heterogeneous: it ranges from -0.961 to
0.727 and is negative in 42 runs. These are descriptive within-run
associations, not independent-variant hypothesis tests. Variants share
posterior draws, output is fitness-dependent, and multiple analyses of one
landscape share experimental counts. The figure summarizes the proximate chain;
it does not establish that support or fitness causes uncertainty.

`mechanism_diagnostics.csv` checks the displayed associations separately among
fixed model-top-100 members and outside challengers and reports a partial
rank-correlation adjusted for input support. In NucB, several positive
primary-cohort decision-margin correlations weaken or reverse within these
strata. This supports a cohort-composition interpretation of the unusual slopes
rather than a reversed score direction or a failure of the stability analysis.

## Future MLE bootstrap results

The shared analysis boundary is an aligned `sample x variant` score ensemble.
HMC chain/draw arrays and MLE-bootstrap samples therefore use the same ranking,
compact-result, reporting, and plotting code. A concrete MLE-bootstrap adapter
should be added only after its file schema is fixed; `inference_method` and
`model_spec` must remain separate metadata fields.
