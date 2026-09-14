# MLE fitness proxy provenance

Last updated: 2026-08-30

This file is the authoritative record of how count-derived fitness proxies are
computed for sampled-MLE analyses. A proxy must not be described as
paper-faithful until the calculation from embedded `observed_data` reproduces
the source score under the rules recorded here.

## Statistical interpretation

SciStanPy first optimizes the model parameters and then draws stochastic model
nodes with their parent parameters fixed at the optimized values. It explicitly
states that uncertainty in parent parameters is not propagated to child nodes.
Consequently, the `posterior_predictive` count draws in `*_samples.nc` represent
repeated experiments conditional on the fitted MLE model. They do not represent
confidence intervals for the MLE estimator and they are not a resample-and-refit
bootstrap.

The analysis target is therefore **conditional repeated-experiment stability**.
The inference label for these results is `mle_predictive`. A future workflow
that resamples counts and refits the optimizer must use a different label.

Precise implementation reference:

- `microsoft/SciStanPy@09c439f675ead70f851d814242f153ecb7712113`
- `scistanpy/model/results/mle.py::MLE.get_inference_obj`
- `scistanpy/model/results/mle.py::MLE.draw`
- `scistanpy/model/results/mle.py::MLEToNetCDFConverter._stream_draws`
- The dependency revision is pinned in `pyproject.toml`.

## Tensor contract

Each analysis stores
`fitness_score[score_name, sample, variant]` as `float32`. For one score, the
requested table with variants in rows and draws in columns is exactly
`fitness_score.sel(score_name=...).transpose("variant", "sample")`.

All count arrays for one sample must come from the same MLE draw. Library totals,
replicates, timepoints, gates, controls, and the fitted variant order must be
preserved. Variants may be excluded from ranking without being removed from the
normalization universe.

Contract statuses are:

- **verified**: observed counts reproduce the source score under the recorded
  formula and eligibility rules;
- **provisional**: the likely source procedure is identified, but the observed
  score oracle has not passed;
- **sensitivity**: the calculation is useful but is not the source-paper score;
- **deferred**: the source procedure is known but is outside the current pass.

## Analysis-to-contract map

| Dataset | Landscape | Sampled-MLE jobs | Primary contract | Status | Sensitivity contract |
| --- | --- | --- | --- | --- | --- |
| PDZ | cis | 1, 3, 4 | `pdz-dimsum-fitness-v1` | provisional | `pdz-pooled-count-ratio-v1` |
| PDZ | CRIPT-C | 70, 71, 72, 73, 74 | `pdz-dimsum-fitness-v1` | provisional | `pdz-pooled-count-ratio-v1` |
| PDZ | CRIPT-N | 76, 77, 78, 80, 81 | `pdz-dimsum-fitness-v1` | provisional | `pdz-pooled-count-ratio-v1` |
| PDZ | trans-1 | 82, 84 | `pdz-dimsum-fitness-v1` | provisional | `pdz-pooled-count-ratio-v1` |
| PDZ | trans-2 | 88, 90, 92, 93 | `pdz-dimsum-fitness-v1` | provisional | `pdz-pooled-count-ratio-v1` |
| TrpB | libA | 7, 8, 9 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libB | 11, 12, 13 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libC | 15, 16, 17 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libD | 19, 20, 21 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libE | 25, 26, 27 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libF | 31, 32, 33, 34 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libG | 37, 38, 39 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libH | 43, 44, 45 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | libI | 49, 50, 51 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| TrpB | four-site | 64, 65, 66, 67, 68, 69 | `trpb-johnston-fitness-v1` | **verified** | `trpb-pooled-count-ratio-v1` |
| NucB | G1 | 55, 56 | `nuc-g1-highest-gate-enrichment-v1` | **verified** | none |
| NucB | G2 | 57, 58 | `nuc-g2-highest-gate-enrichment-v1` | **verified** | none |
| NucB | G3 | 59 | `nuc-g3-highest-gate-enrichment-v1` | **verified** | none |
| NucB | G4 | 61, 62 | `nuc-g4-highest-gate-enrichment-v1` | **verified** | none |
| K50 | global | 63 | `k50-paper-dg-ml-v1` | deferred | `k50-chymotrypsin-mean-log-enrichment-v1`; `k50-trypsin-mean-log-enrichment-v1` |

The historical implementation pilots were jobs 3, 64, 59, and 63. The canonical
manifest now contains all 60 PDZ, TrpB, and NucB jobs plus K50 job 63. The K50
artifact declares two score views, so paper tables contain 62 analysis views.
`analyses_mle_k50.json` retains job 63 as a focused subset.

On 2026-08-31, after a fresh strict audit of all 61 score bundles and compact
results, 57 sampled-MLE source NetCDF files were intentionally retired to make
room for the HMC count-based analysis. The K50 source fit was retired on
2026-09-03 after its persisted matrix and result were revalidated. Three source
artifacts remain as preprocessing examples: PDZ job 3, TrpB four-site job 64,
and NucB G3 job 59. The original experimental count and reference-fitness roots
were not removed. Exact paths, byte counts, retired job IDs, recovery evidence,
and the post-retirement product inventory are recorded in
`data/mle_source_retirement.json` and `data/retired_product_fits.json`.

All 61 canonical score bundles remain, representing 62 score views. Compact
results, mechanism caches, paper tables, and figures can be regenerated from
those score bundles. The 58 retired matrices cannot be regenerated locally
without downloading or regenerating their sampled-MLE source artifacts.

## PDZ

### Source score

The source files contain DiMSum `fitness` and `sigma` estimates. GEO states that
the reads were demultiplexed, quality controlled, trimmed, processed, and
analysed with DiMSum. The experiment used three biological input/output
replicates. Larger fitness is better.

Exact references:

- Zarin, Hidalgo-Carcedo, and Lehner, *A complete map of specificity encoding
  enables reprogramming of a protein interaction*, bioRxiv,
  `10.1101/2024.04.25.591103`.
- NCBI GEO series `GSE265816`, including the sample-level `GSM822...` records
  and the processed files `GSE265816_pdz3cript_sub.csv.gz`,
  `GSE265816_cript2xa3_sub.csv.gz`, `GSE265816_cript_c_sub.csv.gz`, and
  `GSE265816_cript_n_sub.csv.gz`.
- Faure, Schmiedel, Baeza-Centurion, and Lehner, *DiMSum: an error model and
  pipeline for analyzing deep mutational scanning data and diagnosing common
  experimental pathologies*, Genome Biology 21, 207 (2020),
  `10.1186/s13059-020-02091-3`.
- DiMSum source repository: `https://github.com/lehner-lab/DiMSum`.

The exact DiMSum revision and command/configuration used for `GSE265816` have
not yet been recovered. Therefore `pdz-dimsum-fitness-v1` remains provisional.
Verification requires reproducing both published `fitness` and `sigma`, not only
their rank correlation.

### Current sensitivity calculations

For variant $i$, replicate $r$, and wild type $w$, the current
`pdz.py::_count_proxies` calculation is

$$
f_{ir} = \log\left(\frac{c^{out}_{ir}}{c^{in}_{ir}}\right)
       - \log\left(\frac{c^{out}_{wr}}{c^{in}_{wr}}\right),
$$

with Poisson approximation

$$
v_{ir} = \frac{1}{c^{in}_{ir}} + \frac{1}{c^{out}_{ir}}
       + \frac{1}{c^{in}_{wr}} + \frac{1}{c^{out}_{wr}},
$$

and inverse-variance aggregation

$$
f_i = \frac{\sum_r f_{ir}/v_{ir}}{\sum_r 1/v_{ir}}.
$$

Replicates with any zero in the four required counts receive zero weight. This
is a provisional DiMSum-style approximation, not yet the verified GEO
procedure. No affine calibration to the published values is applied. The
separate sensitivity score is the pooled ratio
$\sum_r c^{out}_{ir}/\sum_r c^{in}_{ir}$.

Repository implementation: `src/dms_fitness_stability/adapters/pdz.py`.

### Cis sampled-MLE pilot

Job 3 was generated from the validated sampled-MLE artifact and persisted at:

```text
/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical/
  pdz__pdz-cis__mle-predictive__lomax-exponential__mle-3/
```

The observed proxy is finite for all 9,475 published candidates but is not an
equality oracle for published DiMSum fitness. The observed comparison has
Pearson correlation 0.992934, Spearman correlation 0.993850, mean absolute
error 1.903416, and maximum absolute error 4.959765. An affine diagnostic gives
slope 0.246309, intercept 0.023926, and residual mean absolute error 0.024408;
that fit is recorded only as evidence of scale mismatch and is not applied to
the sampled scores. The contract therefore remains provisional.

Measured results:

- source artifact: 555,473,531 bytes;
- tensor shape: `score=1 x sample=1,000 x variant=10,660`;
- ranking cohort: 9,475 published candidates; 1,185 other fitted rows excluded;
- finite sampled proxy values: 10,545,256 of 10,660,000 (98.924%);
- finite variants per draw: 10,508 to 10,575;
- variants with nonzero sampled range: 10,659 of 10,660;
- output sizes: 33,616,864-byte NetCDF, 1,475,475-byte variant table, and
  12,433-byte metadata file (35,104,772 bytes total);
- preprocessing wall time: 4.70 seconds; peak resident memory: 2,004,312 KB;
- compact rank scan wall time: 13.38 seconds; peak resident memory: 342,380 KB;
- at $K=100$, mean sample-to-sample recall is 0.921980 and mean
  sample-to-published-reference recall is 0.883450.

### Full PDZ sampled-MLE sweep

The approved provisional proxy was subsequently run for all 19 mapped PDZ MLE
fits: cis jobs 1/3/4, CRIPT-C jobs 70-74, CRIPT-N jobs 76/77/78/80/81,
trans-1 jobs 82/84, and trans-2 jobs 88/90/92/93. Direct HDF5 inspection
confirmed matching observed counts and 1,000 predictive count draws in every
artifact.

Observed-proxy Spearman correlations are 0.993850 for cis, 0.869598 for
CRIPT-C, 0.888128 for CRIPT-N, 0.992489 for trans-1, and 0.996779 for trans-2.
CRIPT-N has finite observed proxies for 86,641 of 91,815 published candidates;
the other four landscapes have complete published-candidate coverage. These
statistics describe agreement and coverage, not a significance threshold or an
exact DiMSum oracle.

Native-variant chunking produced all 19 score bundles in 16 minutes 57 seconds
with 7,763,496 KB peak resident memory and no swaps. The bundles occupy
7,564,344,763 bytes. Compact scanning took 14 minutes 15 seconds with 571,680 KB
peak resident memory and produced 1,294,148,216 bytes of results. Jobs 78 and
81 exclude 14 and 33 all-undefined predictive draws, respectively; the other 17
analyses use all 1,000 draws. At $K=100$, mean sample-to-sample recall ranges
from 0.493934 to 0.958725 across the 19 fits.

## TrpB

### Four-site source score

For four-site variant $i$, timepoint $t$, and output replicate $r$, the
deposited amino-acid notebook first uses the mean of the two input-replicate
counts,

$$
\bar c_{i0} = \frac{c_{i0,1}+c_{i0,2}}{2}.
$$

It retains a variant-timepoint row only when $\bar c_{i0}\geq 10$ and both
output-replicate counts are positive. The deposited `calc_mu` helper multiplies
these count arguments by measured culture optical density:

$$
x_{i0} = OD_0\bar c_{i0}, \qquad
x_{itr} = OD_{tr}c_{itr}, \qquad
\mu_{itr} = \frac{\log(x_{itr}/x_{i0})}{t}.
$$

At each timepoint and replicate, let $\bar\mu^{stop}_{tr}$ be the mean over
stop-codon variants and $\mu^{max}_{tr}$ the maximum over the landscape. The
normalized fitness is

$$
F_{itr} =
\frac{\mu_{itr}-\bar\mu^{stop}_{tr}}
     {\mu^{max}_{tr}-\bar\mu^{stop}_{tr}}.
$$

The final score averages valid $F_{itr}$ over both replicates and the 16, 20,
24, 28, and 36 hour timepoints. The 12 hour timepoint is explicitly excluded
before aggregation. Because $OD_0$, $OD_{tr}$, and $1/t$ are constant across
variants within each timepoint-replicate normalization, their additive and
positive multiplicative effects cancel from $F_{itr}$. The repository therefore
computes the numerically equivalent normalized score from
$\log(c_{itr}/\bar c_{i0})$. Larger fitness is better.

### Three-site source score

Libraries A-I use one shared input count per variant. Libraries A-C have one
output replicate at 18 and 44 hours. Libraries D-I have two output replicates
and library-specific retained times: D/G/H/I use 16, 20, 24, and 36 hours; E
uses 20, 24, and 36 hours; F uses 20 and 36 hours.

For each retained time, the source filters to input count at least 10 and
positive output. In the two-replicate libraries, both output counts must be
positive or that variant-timepoint is omitted. Each output replicate is then
normalized by its same-time stop-variant mean and landscape maximum exactly as
in the four-site equation above. Final fitness is the mean over valid retained
timepoints and output replicates. OD and time factors again cancel under this
within-timepoint normalization, so `trpb_three_site_fitness()` uses the
equivalent log count ratio.

Exact references:

- Johnston et al., *A combinatorially complete epistatic fitness landscape in
  an enzyme active site*, PNAS 121, e2400439121 (2024),
  `10.1073/pnas.2400439121`, Methods section "Fitness Score Calculations".
- Processed data and analysis code: CaltechDATA `10.22002/h5rah-5z170`.
- Exact deposited executable path: `code.zip`,
  `code/notebooks/foursite_seq_processing/4-site_import_combine_data.ipynb`,
  cells 9, 11, 16, 17, 20, and 26, plus
  `code/scripts/growth_analysis/growth_analysis/functions.py::calc_mu`. The
  member SHA-256 values are
  `bf93ece96ae6f0882f22f0bac30594d86b80af380779c5f5cda619fdc404fb61`
  and `f3fe2fd6bb6bb7a8960389aef929fb92b2c4fe5caca5ff7f29ef294f82d7a5f3`,
  respectively.
- Exact three-site executable paths in `code.zip`:
  `code/notebooks/threesite_seq_processing/3-site_ABC_import_combine_data.ipynb`
  (cells 12, 14, 15, and 18; SHA-256
  `ebd8d2e92a15aadc553cfa30c5d1e578e38b1f2db1c3dbe8efc3e78120f9905b`)
  and `3-site_DEFGHI_import_combine_data.ipynb` (cells 4-14, 20, 22, 24,
  27, 29, and 34; SHA-256
  `0cbb467534c79937b28e843f1a5b33fa4b830c9a262c0f11af862bcd1f4f754c`).
- Exact deposited data pair: `data.zip`,
  `data/figure_data/4-site_merged_replicates/20230827/four-site_merged_AAs.csv`
  and `four-site_simplified_AA_data.csv`. Their respective SHA-256 values are
  `8feeaee38b7b9d5871f9fe03e5fea3f0661ae2fe5477a2226d45c84f333afa7a`
  and `d7e3ab4bcb721819ddf79ac0c9d322046dcdc4ce7d0df7d5ccd423fee771cb76`,
  exactly matching the staged four-site count and reported-fitness files.
- The paper cites Kowalsky et al., *High-resolution sequence-function mapping
  of full-length proteins*, PLOS ONE 10, e0118193 (2015),
  `10.1371/journal.pone.0118193`, for the specific-growth-rate theory.
- Count tensor loader: `flipv3/flip_dsets.py::load_trpb_dataset` at the FLIP
  revision pinned transitively by this repository.

The installed `flipv3/flip_data_analysis.py::TrpBDataAnalyzer.recalculate_fitness`
is not the oracle for this contract: it retains the 12 hour timepoint and masks
each input replicate separately. The repository implementation follows the
deposited notebook instead.

Observed counts reproduce every reported score with identical finite coverage
for all ten landscapes at `rtol=atol=1e-12`. Maximum absolute errors are
$6.66\times10^{-16}$ (A), $3.42\times10^{-16}$ (B),
$5.48\times10^{-16}$ (C), $1.94\times10^{-16}$ (D),
$1.44\times10^{-16}$ (E), $1.67\times10^{-16}$ (F/G),
$2.22\times10^{-16}$ (H/I), and $4.44\times10^{-16}$ (four-site).
`trpb-johnston-fitness-v1` is therefore verified for libA-I and four-site. The
pooled timepoint/input count ratio remains a sensitivity score and must not be
called Johnston fitness. Sampled-MLE preprocessing now implements both flat
libA--C and hierarchical libD--I geometries, plus the four-site geometry.

Repository implementation: `src/dms_fitness_stability/adapters/trpb.py`.

### Four-site sampled-MLE pilot

Job 64 was generated from the validated sampled-MLE artifact and persisted at:

```text
/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical/
  trpb__trpb-four-site__mle-predictive__exponential-exponential__mle-64/
```

Measured results:

- source artifact: 14,514,442,100 bytes;
- tensor shape: `score=1 x sample=1,000 x variant=194,361`;
- NetCDF dtype/chunks/compression: `float32`, `(1, 64, 4,096)`, gzip level 4;
- finite sampled proxy values: 193,341,995 of 194,361,000 (99.476%);
- finite variants per draw: 193,278 to 193,405;
- variants with at least two finite draws and nonzero sampled range: 194,242;
- variants with no finite draws: 109; variants with one finite draw: 10;
- output sizes: 667,362,940-byte NetCDF, 22,955,559-byte variant table, and
  2,065-byte metadata file (690,320,564 bytes total);
- preprocessing wall time: 2 minutes 11 seconds;
- preprocessing peak resident memory: 5,015,228 KB;
- compact rank scan wall time: 1 minute 6 seconds;
- compact rank scan peak resident memory: 414,588 KB.

### Complete non-K50 execution

Direct HDF5 preflight verifies all 60 sampled-MLE artifacts against staged
observed counts, including one chain, 1,000 draws, exact count-array geometry,
nonnegative integer values, and fixed library totals. The persisted report is
`data/mle_count_preflight.json`.

The 41 refreshed or newly generated TrpB/NucB proxies took 16 minutes 56
seconds; peak resident memory was 5,847,192 KB during four-site processing.
Their compact scans took 13 minutes 23 seconds with 554,536 KB peak resident
memory. Across all 60 analyses, proxies occupy 19,003,194,040 bytes and compact
results occupy 3,875,623,477 bytes. `data/mle_proxy_audit.json` records the
validated pre-retirement source and persistence contract for every run. After
source retirement, run `audit_mle.py --allow-missing-source` to validate every
persisted score and result while still checking the three retained source
signatures.

Mechanism output contains 60 run rows, 19 landscape rows, 66,876 tracked
run-variant rows, and 3,372,475 candidate median-score rows. Nineteen
candidate-run combinations across 10 runs have no finite proxy draw. Six TrpB
runs consequently omit 10 published-context rows with fewer than two finite
draws. Model-top members and sampled challengers always require at least two
finite draws. Mechanism caches occupy 478,302,540 bytes and final tables occupy
160,130,947 bytes.

The clean-kernel `plots_mle.ipynb` execution took 4 minutes 22 seconds with
8,473,632 KB peak resident memory. It produced 23 PNG/PDF pairs totaling
53,516,397 bytes. Exact-zero predictive widths are retained in tables; log-scale
figures place them at a plot-only floor equal to half the smallest positive
width.

## NucB

### Source score

For variant $i$ in one selected fluorescence gate,

$$
EF_i =
\frac{c^{sort}_i / \sum_j c^{sort}_j}
     {c^{input}_i / \sum_j c^{input}_j}.
$$

The sums include the full fitted count universe, including the appended
negative-control rows. Negative controls are subsequently excluded from
candidate ranking. There is no pseudocount: a zero selected-gate count gives
zero enrichment, and a zero input count is undefined. Larger enrichment is
better.

The source-paper highest gates and fitted-array mappings are:

| Generation | Source column | Fitted input | Fitted output |
| --- | --- | --- | --- |
| G1 | `ef_2_high_reseq_g1` | `ic1[0]+ic1[1]+ic2[0]+ic2[1]+ic3` | `hc2[1]` |
| G2 | `ef_1_97.5_g2` | `ic1[0]+ic1[1]+ic2` | `c975` |
| G3 | `ef_2_99_g3` | `ic1` | `oc1[3]` |
| G4 | `ef_4_99.5_g4` | `ic1` | `oc1[3]` |

Exact references:

- Thomas et al., *Engineering highly active nuclease enzymes with machine
  learning and high-throughput screening*, Cell Systems (2025),
  `10.1016/j.cels.2025.101236`.
- Data and analysis repository:
  `google-deepmind/nuclease_design@702284847c4558567d8288169a86dc147cfaab32`.
- Highest-gate selection reference:
  `notebooks/plot_liquid_culture_vs_ef_correlation.ipynb` at that commit.
- Enrichment producer and exact source logic at that commit:
  `notebooks/get_enrichment_factor_data.ipynb` (SHA-256
  `5880c7af3e643d4bcb5d3d3ac7679128c1332cdf5c4d0b036aa4c696181a40f7`),
  `nuclease_design/constants.py`
  (`d988e254ea7f437f2ee66071647b22237a4bd93e8158783caf3bb2ee06d1a6b8`),
  `preprocessing.py`
  (`024fe04d1c12dada731e85d3dd9a032efac727e1c71842b4c9c4cb3b31a69e3d`),
  and `preprocessing_utils.py`
  (`1379bf755b27f15262552756d10faa3205f326cbcd89a7926a84c17b27bc1320`).
- Exact count-array construction:
  `microsoft/SciStanPy@09c439f675ead70f851d814242f153ecb7712113`,
  `flipv3/flip_dsets.py::load_nuclease_data`.
- Frozen source identities, score columns, and shapes:
  `examples/nuc/data_manifest.json`.

### Generation verification

`nuc-g3-highest-gate-enrichment-v1` is implemented by
`normalized_count_enrichment` and `nuclease_g3_highest_gate_enrichment` in
`src/dms_fitness_stability/adapters/nuclease.py`.

On 2026-08-28, the source validator recomputed all 18,618 candidate values from
the staged G3 input and 99th-percentile gate counts. Including all 18,790 fitted
rows in the normalization gave maximum absolute error
$7.11\times10^{-15}$ and agreement at `rtol=atol=1e-12`. Excluding the 172
negative controls produced a constant scale error and maximum absolute error
0.389864, confirming that controls belong to the normalization universe.

The same full-table check verifies G4. Recomputing `ef_4_99.5_g4` from `ic1`
and `oc1[3]`, with all 15,554 fitted rows in the frequency totals, reproduces
all 15,404 candidate values with maximum absolute error
$1.42\times10^{-14}$ at `rtol=atol=1e-12`. The 150 source controls are excluded
only from ranking. `nuc-g4-highest-gate-enrichment-v1` is therefore verified.

G2 initially failed because only the deep input `ic1[0]` was used. The source
configuration pools three inputs: deep replicate 1, shallow replicate 1, and
replicate 2, represented by `ic1[0] + ic1[1] + ic2`. This reproduces all 14,755
main and control values with maximum absolute error $1.42\times10^{-14}$.

G1 pools five inputs: the original and resequenced inputs for sorts 1 and 2,
plus sort 3, represented by `ic1[0] + ic1[1] + ic2[0] + ic2[1] + ic3`. The
source computes abundance totals before removing variants that mutate the
wild-type stop codon. The staged fitted arrays omit 132 such DNA variants that
passed the source threshold; they contributed 23,322 pooled-input reads and 310
selected-output reads. Therefore the exact source normalization totals are
6,000,424 input reads and 1,937,171 output reads. Using

$$
EF_i = \frac{c^{sort}_i}{c^{input}_i}
  \frac{6{,}000{,}424}{1{,}937{,}171}
$$

reproduces all 10,760 main and control values with maximum absolute error
$2.84\times10^{-14}$. The omitted normalization mass changes every finite G1
score by the same factor, so it never changes ranks. G1 and G2 are now verified;
their jobs 55-58 have not yet been executed.

Validation command:

```bash
uv run python examples/nuc/validate_data.py \
  --data-root /data/data_dms_stability
```

The first complete sampled-MLE score bundle was generated on 2026-08-28 from
job 59 at:

```text
/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical/
  nuc__nuc-g3__mle-predictive__exponential__mle-59/
```

Measured results:

- source artifact: 526,991,806 bytes;
- tensor shape: `score=1 x sample=1,000 x variant=18,790`;
- NetCDF dtype/chunks/compression: `float32`, `(1, 64, 4,096)`, gzip level 4;
- finite sampled proxy values: 100%;
- variants with nonzero sample standard deviation: 6,313 of 18,790;
- median and maximum variant sample standard deviation: 0 and 12.846668;
- sampled draws with zero input counts: 0 values;
- output sizes: 26,791,239-byte NetCDF, 7,204,289-byte variant table, and
  1,494-byte metadata file (33,997,022 bytes total);
- measured wall time: 9.68 seconds;
- measured peak resident memory: 343,756 KB.

The zero median sample standard deviation is not evidence that the whole
ensemble collapsed. Many low-activity variants have zero highest-gate counts in
every conditional repeat, while 6,313 variants have nonzero draw variability.

G4 jobs 61 and 62 were generated after their full-table oracle passed. Both
store `score=1 x sample=1,000 x variant=15,554`; job 61 has 15,553,955 finite
values and 3,685 variants with nonzero sampled range, while job 62 has
15,553,997 finite values and 3,600 varying variants. Their score bundles occupy
21,742,423 and 21,279,082 bytes. Combined preprocessing took 17.93 seconds with
405,432 KB peak resident memory; compact scanning took 28.99 seconds with
420,472 KB peak resident memory. At $K=100$, job 61 sample-to-sample and
sample-to-reference recalls are 0.844534 and 0.385780; job 62 values are
0.362051 and 0.246600.

All four generation contracts now pass full-table observed-count oracles. G3
job 59 and G4 jobs 61/62 have completed proxy and rank analyses. G1 jobs 55/56
and G2 jobs 57/58 are verified and ready, but intentionally remain unprocessed
until a separate execution step is approved.

## K50

### Deferred paper score

The paper score `dG_ML` is a quality-filtered absolute folding stability in
kcal/mol, with larger values indicating greater stability. It is not a direct
count ratio. The source workflow:

1. calibrates effective protease concentrations;
2. infers sequence-specific K50 values from untreated and treated counts;
3. fits an unfolded-state sequence model using 64,238 scrambled sequences;
4. converts K50, K50,U, and K50,F to folding $\Delta G$; and
5. applies confidence, domain, mutant, censoring, and assay-range rules.

The published dynamic range is $[-1,5]$ kcal/mol. Published `<-1` and `>5`
values are censored at those limits for the current reference ranking, and `-`
is unavailable. There is no global wild type. The upper censoring limit contains
41,517 aligned sequences, so reported top-50, top-100, and top-1,000 membership
is not uniquely identifiable.

Exact references:

- Tsuboyama et al., *Mega-scale experimental analysis of protein folding
  stability in biology and design*, Nature 620, 434-444 (2023),
  `10.1038/s41586-023-06328-6`.
- Data release: Zenodo `10.5281/zenodo.7992926`.
- Source pipeline:
  `Rocklin-Lab/cdna-display-proteolysis-pipeline@597442a33f931a5e86003134c279b6f2a58ffd33`.
- Exact processing stages:
  `Pipeline_K50_dG/STEP1_run.ipynb` through `STEP5_run.ipynb`, with their
  corresponding module notebooks.
- Frozen source identities, axis conventions, and censoring policy:
  `examples/k50/data_manifest.json`.

The source repository reports about 13 GPU-hours for one complete five-stage
dataset pass. Repeating that workflow for 1,000 count draws is deferred.

### Protease-specific sensitivity scores

The canonical analysis keeps chymotrypsin and trypsin enrichment separate. It
does not average the two proteases. These scores summarize protease resistance
but are not `dG_ML`. For library $l$, protease $p$, replicate $r$,
concentration $q$, and variant $i$, define the smoothed log enrichment

$$
E_{lprqi} =
\log\left(
\frac{c^{treated}_{lprqi}+0.5}
     {\sum_j c^{treated}_{lprqj}+0.5A_l}
\right)
-
\log\left(
\frac{c^{untreated}_{lpri}+0.5}
     {\sum_j c^{untreated}_{lprj}+0.5A_l}
\right),
$$

where $A_l$ is the number of fitted variants in library $l$. The proxy averages
$E_{lprqi}$ over the 2 replicates, 11 treated concentrations, and all library
occurrences of a global sequence, separately for each protease $p$. Larger
values indicate greater resistance to that protease. All fitted rows, including
scrambled controls, remain in each library-frequency denominator. The 64,238
scrambled controls remain in the persisted tensor but are excluded from
ranking. The ranking cohort contains 1,760,285 non-scrambled sequences.

Both sensitivity scores ignore qPCR calibration, concentration spacing, K50
curve fitting, the unfolded-state model, and the thermodynamic K50-to-$\Delta
G$ transformation. Their status remains sensitivity even when they correlate
strongly with `dG_ML`.

Repository adapter: `src/dms_fitness_stability/adapters/k50.py`.

### Sampled-MLE sensitivity analysis

Job 63 was generated from the validated sampled-MLE artifact and persisted at:

```text
/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical/
  k50__k50__mle-predictive__massive-only-mle__k50-mle-lr0-01/
```

The artifact stores `fitness_score[score_name, sample, variant]`. Selecting one
of `mean_log_chymotrypsin_enrichment` or `mean_log_trypsin_enrichment` and then
transposing produces the analysis input matrix:

```python
matrix = fitness_score.sel(score_name=score_name).transpose("variant", "sample")
```

Each resulting matrix has 1,824,523 variant rows and 1,000 conditional count-
draw columns. Both observed scores are finite for all fitted sequences. Among
the 524,836 non-scrambled sequences with published `dG_ML`, chymotrypsin has
Pearson correlation 0.886343 and Spearman correlation 0.892327; trypsin has
Pearson correlation 0.911285 and Spearman correlation 0.919945. These values
check direction and alignment; they do not promote either score to a `dG_ML`
oracle.

Measured results:

- source artifact: 443,627,674,689 bytes;
- tensor shape: `score=2 x sample=1,000 x variant=1,824,523`;
- NetCDF dtype/chunks/compression: `float32`, `(1, 64, 4,096)`, gzip level 4;
- finite sampled proxy values: 3,649,046,000 of 3,649,046,000 (100%);
- library occurrence counts: 1,752,477 sequences occur once, 71,850 twice,
  and 196 three times;
- output sizes: 11,660,634,936-byte NetCDF, 416,789,219-byte variant table, and
  15,967-byte metadata file (12,077,440,122 bytes total);
- score construction time recorded by the builder: 1,330.26 seconds;
- at $K=100$, chymotrypsin sample self-consistency is 0.114902 and sample-to-
  model-reference recall is 0.112710;
- at $K=100$, trypsin sample self-consistency is 0.745967 and sample-to-model-
  reference recall is 0.821710.

### Interpretation of K50 stability

K50 stability results are valid for the two count-derived protease-enrichment
rankings. Across the 1,000 conditional count draws, they quantify sample-to-
sample top-K overlap, per-variant inclusion probabilities, rank intervals, and
overlap with each ranking defined by its median proxy. They do not quantify
stability of a unique published `dG_ML` top-K set.

Published `dG_ML` is censored at 5 kcal/mol, where 41,517 sequences are tied.
The publication therefore defines a top plateau, not a unique top 50, 100, or
1,000. Breaking that tie by file or variant order would create an arbitrary
reference set. In addition, the enrichment proxy omits K50 curve fitting, qPCR
calibration, the unfolded-state model, thermodynamic conversion, and quality
filtering, so it must not be described as reconstructed folding stability.

### Role of Spearman correlation

For the 524,836 aligned non-scrambled sequences, the two observed proxies and
published `dG_ML` have Spearman correlations

$$
\rho_s(F^{chymotrypsin}, dG_{ML}) = 0.892327,
\qquad
\rho_s(F^{trypsin}, dG_{ML}) = 0.919945.
$$

These are alignment and direction diagnostics: they show that larger proxy
values generally correspond to larger published stability values and that the
count-derived scores capture substantial related signal. They are not pass/fail
tests of proxy adequacy. With this sample size, a null-hypothesis p-value is not
informative, and a high global correlation does not establish score equality,
kcal/mol calibration, top-tail agreement, ordering within the censored plateau,
or equivalence to the published K50-to-$\Delta G$ pipeline.

> The proxies support conditional repeated-experiment stability of two
> protease-enrichment rankings. They do not support claims about stability of the
> published `dG_ML` ranking, especially its censored top-K region.

No sample-to-published-top-K statistic is emitted. The published upper censoring
plateau contains 41,517 sequences, so selecting 50, 100, or 1,000 members would
be arbitrary. Paper tables retain self-consistency and model-reference results,
set `sample_vs_reported` to missing, leave `is_reported_top_k` unknown, and emit
no K50 `rank_percentiles` rows.

## Maintenance rule

Every implemented proxy must write its contract ID and status into output
metadata. Any change to an equation, normalization universe, zero handling,
eligibility rule, source revision, or score direction requires a new contract
version and an update to this file. Runtime, peak memory, output size, finite
coverage, and observed-count oracle results will be appended after each pilot
run.