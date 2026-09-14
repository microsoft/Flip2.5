# FLIP2 Zero-Shot Scores and PDZ Baseline Plan

## What zero-shot means

A zero-shot protein language model (pLM) does not use FLIP fitness targets when
it scores a sequence. The model was pretrained on large collections of
unlabeled protein sequences and outputs amino-acid probabilities from sequence
context.

For a masked language model and mutated positions $M$, FLIP2 describes the
variant score as a sum of mutant-versus-wild-type log likelihood ratios:

$$
S(x) = \sum_{i \in M}
\left[
\log p_\theta(x_i^{\mathrm{mut}} \mid x_{\setminus M})
-
\log p_\theta(x_i^{\mathrm{WT}} \mid x_{\setminus M})
\right].
$$

A higher score means the mutant residues are more plausible under the model.
The score is not experimental fitness, is not calibrated to fitness units, and
does not require assay labels.

For autoregressive Dayhoff models, FLIP2 averages N-to-C and C-to-N likelihoods.
For PDZ3, it scores the PDZ and CRIPT sequences separately and averages their
scores:

$$
S_{\mathrm{PDZ3}} = \frac{S(\mathrm{PDZ}) + S(\mathrm{CRIPT})}{2}.
$$

This does not model the PDZ-CRIPT interaction directly. The paper reports that
zero-shot likelihoods are less effective for two-protein landscapes such as
PDZ3 than for variants of a single wild-type protein.

## How the score is evaluated

FLIP2 treats the scalar zero-shot score as a sequence-only ranking prediction
and evaluates it against fitness with Spearman correlation:

$$
\rho = \operatorname{Spearman}(S(x), y).
$$

For HMC-resampled fitness dataset $r$, the pLM score remains fixed while the
target changes:

$$
\rho^{(r)} = \operatorname{Spearman}(S, y^{(r)}).
$$

The distribution of $\rho^{(r)}$ measures how target uncertainty affects the
zero-shot baseline conclusion. The pLM is not rerun for each HMC replicate.

## Contrast with supervised baselines

| Method | Uses sequence | Uses training fitness | Refit per HMC replicate |
|---|---:|---:|---:|
| Zero-shot pLM | Yes | No | No |
| Ridge one-hot | Yes | Yes | Yes |
| Ridge one-hot + pLM scores | Yes | Yes | Yes |

In Ridge + pLM, fixed zero-shot scores become additional features. Ridge is
then trained separately against each HMC-resampled target dataset.

## Where to read in FLIP2

Manuscript: [FLIP2: Expanding Protein Fitness Landscape Benchmarks for
Real-World Machine Learning Applications](https://www.biorxiv.org/content/10.64898/2026.02.23.707496v4)

- Section 4.1, **Zero-Shot Likelihood Scores**, PDF pages 6-7: score definition,
  model choices, and PDZ3 component averaging.
- Figure 2, PDF page 6: zero-shot Spearman correlations on full landscapes and
  test splits.
- Section 4.2, **Linear Models**, PDF page 7: one-hot Ridge and Ridge augmented
  with ESM2, CARP, and Dayhoff scores.
- Figure 3, PDF page 7: Ridge, Ridge + likelihoods, and best zero-shot results.
- Figure 4B-C, PDF page 8: zero-shot versus fine-tuning and Ridge + likelihoods.
- Discussion, PDF page 8: limitations on two-protein and multi-backbone tasks.

Useful cited background:

- Hsu et al. (2022), *Learning protein fitness models from evolutionary and
  assay-labeled data*.
- Lin et al. (2023), ESM2.
- Yang et al. (2024), *Convolutions are competitive with transformers for
  protein sequence pre-training* (CARP).
- Yang et al. (2025), *The Dayhoff Atlas*.

## Reproduction plan for HMC-resampled PDZ3

1. Generate $N$ FLIP-shaped target datasets with
   `scripts/pdz_trans_hmc_flip_replicates.py`.
2. Compute one-hot sequence features once and reuse them across replicates.
3. Fit one-hot Ridge separately on each replicate's non-validation train rows.
4. Evaluate each fitted model on that replicate's test targets.
5. Compute zero-shot scores once for the 734 unique FLIP sequences.
6. Reuse the fixed scores to evaluate zero-shot Spearman on every replicate.
7. Fit Ridge with one-hot plus the available zero-shot scores for every replicate.
8. Report the full posterior distribution, median, and interval for each metric.

## Published PDZ3 reference

FLIP2 Table A16 reports the following Spearman correlations on the PDZ3
single-to-double test set:

| Model | Spearman |
|---|---:|
| Ridge (one-hot) | 0.476 |
| Ridge (one-hot + likelihoods) | 0.478 |
| Dayhoff likelihood | -0.007 |
| ESM2-650M likelihood | 0.048 |
| CARP-640M likelihood | 0.043 |

Our current one-hot Ridge implementation gives `0.479650` on the original
published target, a difference of `0.003650` from the rounded paper value. This
is the reference check to pass before comparing the HMC-resampled target
distribution.

## Figure 3G learning-curve protocol

The grey PDZ3 curves in Figure 3G do not use the prescribed FLIP split. The
FLIP2 code ignores `set` and `validation`, randomly shuffles all 734 rows, and
uses:

- 10 logarithmically spaced training sizes from 50 to $0.8 \times 734$;
- 50 total random partitions, cycling through the 10 sizes;
- therefore five random train/test partitions per size;
- random seed 23;
- all rows not selected for training as test rows;
- Ridge alpha 10.

The colored point uses the prescribed single-to-double split: 124
non-validation training rows, 31 validation rows, and 579 test rows.

For the HMC adaptation, the random partitions are generated once and held fixed
across all target draws. Correlations are first averaged over the five partitions
within each HMC draw. Quantiles across those per-draw means therefore describe
uncertainty from the inferred targets rather than from repeatedly changing the
train/test split. The published-target band separately shows variation across
the five random partitions.

## Missing artifacts

The one-hot Ridge baseline is runnable now. The following sequence-only score
caches still need to be generated or downloaded:

- ESM2-650M masked-mutant likelihood-ratio scores;
- CARP-640M masked-mutant likelihood-ratio scores;
- Dayhoff bidirectional autoregressive scores.

See `data/zero_shot/README.md` for the cache schema. The current FLIP repository
`esm2_zeroshot.py` computes whole-sequence negative cross-entropy, while the
FLIP2 manuscript specifies masked-mutant log likelihood ratios for PDZ3. Reproduce
the manuscript definition rather than assuming those two calculations are
equivalent.

## Metric and split conventions

The current implementation uses **Spearman rank correlation**, not Pearson
correlation.

For the prescribed PDZ3 single-to-double benchmark:

- target scaling is fit on every row with `set == "train"`, matching the FLIP2
  script;
- Ridge is fit only on `set == "train"` and `validation == False`;
- validation rows are excluded from model fitting;
- Spearman is computed only on `set == "test"` (579 rows).

For the random-split learning curves, FLIP2 ignores the prescribed `set` and
`validation` labels. At each training size, randomly selected rows form the
training set and every remaining row forms the test set. Spearman is computed
on that random test complement.

Zero-shot models can be evaluated on either the full landscape or the prescribed
test set because they do not use training targets. Both are retained as
separately labeled results.

## Why the Ridge representation is appropriate for PDZ3

The one-hot Ridge input follows the FLIP2 code:

1. remove the `:` separator from `PDZ:CRIPT`;
2. concatenate the PDZ and CRIPT components;
3. one-hot encode each position over the 20 amino acids;
4. zero-pad stopped/truncated sequences to the maximum length;
5. flatten the sequence-position array;
6. predict one scalar fitness target for each PDZ-CRIPT pair.

The rule “score PDZ and CRIPT separately, then average” applies to pLM
likelihood scores, not to one-hot Ridge. Ridge sees the paired sequence as one
fixed-position feature vector and learns additive position effects.

## Recommended pLM execution order

Start with ESM2-650M because it has the clearest public interface and an exact
paper reference: PDZ3 test Spearman `0.048` in Table A16.

For each FLIP sequence:

1. split `PDZ:CRIPT`;
2. compare each component with its wild type to identify mutated positions;
3. mask all mutated positions simultaneously;
4. run the pLM on the masked component;
5. sum mutant-versus-wild-type log likelihood ratios over mutated positions;
6. score PDZ and CRIPT independently and average the two scores;
7. cache one scalar per FLIP sequence.

The score cache is generated once and reused for the published target and every
HMC target draw. Validate the original-FLIP test Spearman against Table A16
before interpreting HMC uncertainty.

Three published rows have stopped/truncated CRIPT components of lengths 0, 1,
and 2; all three are training rows. Their scoring policy must be explicit before
using pLM scores as supervised Ridge features, although they do not affect the
prescribed test-only zero-shot Spearman.

After ESM2 works, port the same masked-mutant score to CARP-640M (paper reference
`0.043`). Run Dayhoff last because it requires two 3B autoregressive models and
bidirectional scoring (paper reference `-0.007`).

## Practical GCR/AMLT workflow

The first cluster job should generate sequence-only pLM scores. It does not need
the large HMC NetCDF files.

1. Implement and test an ESM2 scorer locally on 5-10 sequences or with a small
  ESM2 checkpoint.
2. Add invariants: wild-type score is zero, sequence order is preserved, and all
  734 keys receive a score.
3. Submit one GPU job for ESM2-650M through AMLT/GCR.
4. Write only a small sequence-keyed CSV to `$AMLT_OUTPUT_DIR`.
5. Download that CSV and validate published-target test Spearman against `0.048`.
6. Place the validated scores in `data/zero_shot/pdz3_scores.csv`.

Use `amlt target list manifold` to discover available GCR targets, then create a
config with `amlt template`; do not hand-write target or storage settings.

For scaling to additional landscapes, define one adapter per landscape with:

- published split loader;
- sequence reconstruction and wild-type sequence(s);
- pLM scoring policy;
- HMC variant-index mapping;
- posterior target variable;
- split definitions.

Cache sequence-only features independently of target draws. Run GPU pLM scoring
once per landscape/model, and run CPU Ridge jobs in chunks of HMC draws. Store
large model/HMC artifacts in Azure storage and return only compact score,
prediction, and metric tables through AMLT outputs.

## Reported-versus-proxy fitness diagnostics

Every generated HMC dataset now retains three proxy columns from one aligned
joint `(chain, draw)` sample and one trans source:

- `proxy_log_r_mean`: the posterior `log_r_mean` value already used as the HMC
  target;
- `proxy_count_ratio`: the sum of the three posterior-predictive output counts
  divided by the sum of the three posterior-predictive input counts;
- `proxy_dimsum_fitness`: replicate-wise natural-log output/input enrichment
  relative to the wild-type PDZ3:CRIPT row, merged using inverse count-variance
  weights.

The six generated counts are retained as
`posterior_starting_count_1` through `posterior_starting_count_3` and
`posterior_timepoint_count_1` through `posterior_timepoint_count_3`. For
sequences represented in both trans fits, the counts and every proxy are kept
with the same randomly selected `hmc_source`; columns are never selected
independently.

### Count array dimensions

For a selected set of HMC draws, the posterior-predictive count arrays have
shape:

```text
(number of HMC draws, 3 experimental replicates, number of variants)
```

Using indices $s$, $a$, and $b$ for HMC draw, experimental replicate, and
variant, respectively, define:

- $C^{\mathrm{in}}_{sab}$: generated input count for variant $b$;
- $C^{\mathrm{out}}_{sab}$: generated output count for variant $b$;
- $C^{\mathrm{in}}_{sa,\mathrm{WT}}$: generated input count for wild type;
- $C^{\mathrm{out}}_{sa,\mathrm{WT}}$: generated output count for wild type.

The wild-type arrays initially have shape `(number of HMC draws, 3)`. The code
adds a length-one variant dimension so NumPy compares every variant with wild
type from the **same HMC draw and same experimental replicate**. No averaging
over HMC draws occurs when constructing a proxy dataset.

### Simple pooled count-ratio proxy

The first count proxy pools the three experimental replicates before taking the
ratio:

$$
R_{sb}
=
\frac{\sum_{a=1}^{3} C^{\mathrm{out}}_{sab}}
  {\sum_{a=1}^{3} C^{\mathrm{in}}_{sab}}.
$$

This is `proxy_count_ratio`. It is intentionally simple: it is not normalized
to wild type and does not assign explicit uncertainty weights to the three
replicates. If the pooled input count is zero, the result is missing (`NaN`).

### Replicate-specific wild-type-normalized fitness

For the DiMSum-style proxy, each experimental replicate first produces its own
fitness estimate:

$$
f_{sab}
=
\log\left(\frac{C^{\mathrm{out}}_{sab}}{C^{\mathrm{in}}_{sab}}\right)
-
\log\left(
\frac{C^{\mathrm{out}}_{sa,\mathrm{WT}}}
  {C^{\mathrm{in}}_{sa,\mathrm{WT}}}
\right).
$$

Equivalently, this is the logarithm of the variant enrichment divided by the
wild-type enrichment. A value of zero means that the variant and wild type
enriched equally in that experimental replicate; positive and negative values
mean stronger and weaker enrichment than wild type, respectively.

All four counts must be positive. A replicate with a zero variant or wild-type
input/output count is omitted for that draw and variant rather than receiving a
pseudocount.

### What the variance means

The quantity called `variance` in the implementation is **not the empirical
variance of the three replicate fitness estimates**. In particular, the code
does not calculate `replicate_fitness.var(axis=1)`.

Instead, it assigns a separate approximate sequencing-count variance to every
fitness estimate $f_{sab}$:

$$
v_{sab}
\approx
\frac{1}{C^{\mathrm{in}}_{sab}}
+
\frac{1}{C^{\mathrm{out}}_{sab}}
+
\frac{1}{C^{\mathrm{in}}_{sa,\mathrm{WT}}}
+
\frac{1}{C^{\mathrm{out}}_{sa,\mathrm{WT}}}.
$$

This expression follows from a Poisson count approximation. If
$C \sim \operatorname{Poisson}(\lambda)$, then
$\operatorname{Var}(C)=\lambda$. A first-order delta-method approximation gives

$$
\operatorname{Var}(\log C)
\approx
\left(\frac{1}{\lambda}\right)^2\operatorname{Var}(C)
=
\frac{1}{\lambda}
\approx
\frac{1}{C}.
$$

The replicate fitness is a sum and difference of four log counts. Assuming
their count noise is approximately independent, the four variances add. Thus,
low-count measurements have larger estimated variance, while high-count
measurements have smaller estimated variance.

At this point, `replicate_fitness`, `variance`, and `weights` all still have
shape:

```text
(number of HMC draws, 3 experimental replicates, number of variants)
```

There are therefore three separate fitness estimates and three separate
uncertainty estimates for each draw and variant.

### Why use inverse-variance weights?

The code converts variance to precision:

$$
w_{sab}=\frac{1}{v_{sab}}.
$$

A replicate supported by many reads receives a larger weight because its log
enrichment is expected to have less count-sampling noise. A low-depth replicate
receives a smaller weight. For example, output/input counts of $2/1$ and
$2000/1000$ have the same ratio, but the latter ratio is much more precisely
estimated. Giving those measurements equal influence would discard that
information.

Under independent, approximately unbiased measurements with known variances,
the inverse-variance weighted mean is the minimum-variance linear estimator of
their shared underlying fitness. This is the statistical motivation for the
weights; weighting is not mathematically required to define enrichment.

### Where the three replicates are combined

Only after calculating one fitness, variance, and weight per replicate does the
implementation reduce the replicate axis:

$$
F_{sb}
=
\frac{\sum_{a=1}^{3}w_{sab}f_{sab}}
  {\sum_{a=1}^{3}w_{sab}}.
$$

This is the operation implemented by `sum(axis=1)`, because axis 1 is the
three-replicate dimension. The transformation is:

```text
replicate_fitness: (draws, 3, variants)
variance:          (draws, 3, variants)
weights:           (draws, 3, variants)
          |
          | weighted sum over axis 1
          v
combined fitness:  (draws, variants)
```

An invalid replicate receives weight zero. If all three replicates are invalid
for one draw and variant, the combined proxy is `NaN`.

An unweighted alternative would be the mean of the valid $f_{sab}$ values. That
would be a useful sensitivity analysis: comparing it with
`proxy_dimsum_fitness` would isolate the effect of count-depth weighting from
the effect of wild-type normalization.

The proxy diagnostic computes one Spearman correlation per HMC dataset between
the published FLIP fitness and each proxy over all variants with finite values.
These correlations measure agreement between target definitions and are not
Ridge model performance.

The Zarin et al. paper states that trans fitness and error were calculated with
DiMSum after requiring at least 10 input reads in one replicate. DiMSum's core
calculation is replicate-wise log enrichment relative to wild type, followed by
replicate normalization and inverse-variance merging. The study then applied a
separate linear transformation across libraries using shared variants. The
fitted DiMSum error-model parameters, replicate normalization parameters, and
cross-library transformation coefficients are not available in this repository.
Therefore `proxy_dimsum_fitness` is deliberately labeled **DiMSum-style**: it
reproduces the directly count-derived, wild-type-centered, count-error-weighted
part, but it is not claimed to exactly reconstruct the final reported score.

More precisely, this implementation uses a first-order Poisson count variance.
It does not reproduce DiMSum's fitted empirical error model, fitted replicate
shift/scale normalization, dropout-pseudocount settings, or the paper's final
cross-library linear transformation. The scientifically precise description of
this proxy is therefore **wild-type-normalized log enrichment combined across
experimental replicates using count-derived inverse-variance weights**.

## Rank-space compatibility with reported FLIP fitness

The per-draw Spearman plot asks how strongly each HMC-derived vector agrees with
FLIP. The rank-space compatibility analysis asks a different, calibrated
question: is the complete FLIP ranking about as far from the HMC rank center as
a held-out HMC draw normally is?

For each proxy, all fitness vectors are converted to normalized within-vector
ranks. The 16,000 HMC draws are split reproducibly (`seed=23`) into 8,000
reference and 8,000 calibration draws. Reference draws define each variant's
mean rank $\mu_b$ and rank standard deviation $\sigma_b$. The standardized
discrepancy for rank vector $r$ is

$$
T(r)=\frac{1}{B}\sum_{b=1}^{B}
\left(\frac{r_b-\mu_b}{\max(\sigma_b,\epsilon)}\right)^2,
$$

where $\epsilon=1/(B-1)$ is one normalized rank step. This is not a covariance-
aware Mahalanobis distance; dependencies among variants are instead represented
implicitly by calibrating the complete score against held-out HMC draws.

The analysis also compares every held-out draw and FLIP with the consensus rank
learned from the reference draws using Spearman correlation. Thus:

- a high FLIP discrepancy percentile means FLIP is farther from the HMC rank
  center than most held-out HMC draws;
- a low FLIP consensus-correlation percentile means FLIP is less consensus-like
  than most held-out HMC draws;
- the upper-tail fraction is the fraction of held-out draws at least as
  discrepant as FLIP, with the finite-sample correction $(k+1)/(N+1)$.

One fixed complete-case sequence set is used per proxy so all discrepancy scores
within a proxy are comparable. `log_r_mean` and count ratio retain all 734
variants; DiMSum-style fitness retains 729 and excludes five variants with at
least one missing value across the 16,000 draws.

Current results:

| Proxy | Variants | FLIP-consensus Spearman | Consensus percentile | FLIP discrepancy | Discrepancy percentile | Upper tail |
|---|---:|---:|---:|---:|---:|---:|
| Posterior `log_r_mean` | 734 | 0.939 | 0.050% | 3.090 | 100% | 0.0125% |
| Posterior-predictive count ratio | 734 | 0.899 | 0.0125% | 3.277 | 100% | 0.0125% |
| DiMSum-style count fitness | 729 | 0.907 | 0.0125% | 3.991 | 100% | 0.0125% |

FLIP therefore has high absolute rank agreement with every proxy, but its rank
vector is still more discrepant from the HMC center than every held-out HMC draw
in this split. This is best described as a strong **estimator-compatibility
signal**, not a formal posterior predictive rejection: FLIP and HMC are derived
from the same experiment, and the reported fitness estimator is not an
independent draw from the HMC posterior. Variant-level squared contributions are
retained to identify which sequences drive the global difference.

### Concise explanation to share with the figure

- We convert each HMC fitness draw and the reported FLIP fitness vector to ranks
  across variants, so the comparison is insensitive to differences in scale and
  monotonic calibration.
- We split the 16,000 HMC draws into 8,000 reference and 8,000 held-out draws.
  The reference draws define the HMC mean rank and rank uncertainty for every
  variant. The histograms show results for the held-out draws; the dashed line
  shows the reported FLIP vector.
- **Left column:** for each held-out draw, we calculate the mean squared distance
  from the HMC mean ranks, standardized by each variant's HMC rank SD. This asks
  how far the whole ranking is from the HMC center while giving more weight to
  disagreements at variants whose HMC ranks are tightly determined. The dashed
  line is the same distance calculated for FLIP.
- **Right column:** we average each variant's rank over the reference draws to
  obtain an HMC consensus ordering. We then calculate Spearman correlation with
  that consensus for every held-out draw and for FLIP. The dashed line is the
  FLIP-consensus correlation.
- FLIP has high absolute agreement with the HMC consensus
  ($\rho=0.939$, $0.899$, and $0.907$ for `log_r_mean`, count ratio, and
  DiMSum-style fitness), but it is less consensus-like and farther from the HMC
  center than every held-out HMC draw in this split.
- The interpretation is that FLIP and HMC broadly agree on variant ordering,
  but their remaining systematic differences are larger than the variation
  among HMC draws. This may reflect differences between fitness estimators,
  model assumptions, or posterior uncertainty. It is a compatibility
  diagnostic, not a formal posterior predictive rejection, because FLIP and HMC
  are derived from the same underlying experiment.

  ## Ridge comparisons across target proxies

  The one-hot Ridge analysis now runs independently for all three HMC-derived
  target definitions. All proxies use the same sequences, prescribed FLIP split,
  200-draw learning-curve subset, and random learning-curve partitions. Missing
  proxy targets are removed within each train/test split and the finite sample
  counts are recorded.

  Prescribed single-to-double test-split results across all 16,000 draws:

  | HMC target proxy | Median Spearman | 95% interval |
  |---|---:|---:|
  | Posterior `log_r_mean` | 0.538 | [0.509, 0.572] |
  | Posterior-predictive count ratio | 0.562 | [0.521, 0.613] |
  | DiMSum-style count fitness | 0.421 | [0.394, 0.447] |

  The published-target reference is unchanged at Spearman 0.480. The count-ratio
  target gives the strongest prescribed-split Ridge performance, followed by
  `log_r_mean`; DiMSum-style targets give substantially lower prescribed-split
  performance. The learning-curve figure uses one row per target proxy and two
  columns: target uncertainty after averaging five split repeats, and pooled
  target-plus-partition variation.
