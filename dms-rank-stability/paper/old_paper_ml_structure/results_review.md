# Evidence and proposed paper story

Framing update, 2026-09-09: these notes retain the earlier story and numerical
review. The author's current direction centers batch membership and internal
rank resolution, not single-candidate selection. Resampling supplies noisy
realizations; it is not the paper's subject. Earlier winner-based arguments
below are superseded as motivation, while their numerical records are retained.
The current message and contract are in [README.md](README.md) and
[review.md](review.md).

Reviewed 2026-09-08. These are research and drafting notes, not manuscript prose.
Subsequent source-based revision: [supplementary_details.tex](supplementary_details.tex)
now specifies the model equations and fit inventory. The routine called `mle`
includes parameter-density terms, so the current manuscript uses "optimized-fit
count resampling" while keeping the `mle_predictive` data identifier. Earlier
MLE wording below records the terminology at review time; it is not the
current definition of the fitting objective. See [README.md](README.md) for
the source findings, available diagnostics, and remaining checks.
Numerical statements below come from saved companion-project tables. No models
were refitted, no experimental counts were changed, and no FLIP2 prediction
models were evaluated during this review. The manuscript has not yet been
rewritten at the time of the original review. A first illustrated draft is now
in [main.tex](main.tex), with current build instructions and the claim ledger
in [README.md](README.md).

Scope decision after the whiteboard review: the author confirmed a focus on
leading cohorts, not a full-landscape uncertainty analysis. The draft uses
reported top-100/top-1,000 comparisons, central top-100 rank intervals, plausible
challengers, and selection sizes up to 1,000. Earlier suggestions to add
all-variant uncertainty summaries are superseded and are not a drafting gate.

The author's whiteboard from the discussion with Bruce supplies the organizing
structure below. The three analysis sections take precedence over the earlier
proposal to organize the paper around the TrpB/NucB selection contrast. That
contrast remains useful evidence within the second analysis section.

## Recommended message

An ordering released with a deep mutational scan reflects both a choice of
fitness construction and measurements with finite precision. First ask how the
ordering changes when fitness is constructed differently. Then, within a
specified construction, ask how accounting for noise changes what we can say
about ranks. Finally, explain the observed patterns using count support and
the separation of competing scores. The paper is about the ranking information
in the experiments, with candidate selection as one important consequence.

The task-dependent resolution result supports this broader story: a scan can
resolve a winner without resolving a top-100 set, or resolve a top-100 set
without resolving the winner. It should illustrate the impact of noise, not
replace the original three-part analysis.

This is an empirical paper about the experiments and their resulting labels,
with two equal motivations: protein engineers select variants for follow-up;
machine-learning researchers train and benchmark predictors against the same
fitness measurements. Neither audience should be an afterthought.

The author confirms that the landscapes considered here contribute to FLIP2,
which defines particular variant subsets and evaluation tasks. That makes the
benchmark motivation direct. It does not turn top-focused rank summaries
into a completed FLIP2 evaluation: ranking within its subsets and propagating
label uncertainty into predictor comparisons remain future work. Add the
appropriate FLIP2 citation and dataset mapping when drafting the introduction.

## Reconstructing the discussion with Bruce

This is a reading of the supplied photograph, not a verbatim meeting record.
The three section headings and a top-versus-all comparison were interpreted
from the photograph. The author subsequently clarified the chosen top-focused
scope, which takes precedence over that reconstruction. Some smaller annotations and numerical examples are
less clear; no exact values or finalized statistical definitions are inferred
from those marks.

| Analysis section on the board | Scientific question | Displays discussed on the board |
| --- | --- | --- |
| 1. Changing the definition or construction of fitness | How does reported fitness compare with what we derive by fitting the counts? | Reported-versus-derived correlation; an example "calibration curve"; curves or summaries of their differences across experiments. |
| 2. Impacts of including noise | Once a fitness construction is fixed, how much uncertainty is there in the ordering? | A top-100 rank example; summaries by landscape and model of the magnitude of rank changes; top-100 and all-variant comparisons; lower and upper rank bounds or deviations summarized over the dataset. |
| 3. Drivers of those impacts | Why do some ranks or regions have more uncertainty than others? | The explicit conjecture is low counts; examine rank distributions or interval widths and how they vary across variants. |

The board also asks to use the data broadly and reduce arbitrary choices. My
reading is that top 100 was a useful decision-scale illustration, not the
definition of the whole scientific problem. Fitting and the three resampling
recipes belong before these analysis sections, so each question can be asked
using the same workflow.

### Analysis 1: Does the fitness construction change the ordering?

Start with the reported landscape and compare it with the fitted analyses.
Use the current reported-rank percentile curves, direct rank correlations,
and central-order comparisons. Explain an example curve before summarizing
its behavior across landscapes and model specifications, as on the board.
Rank comparisons do not require the raw fitness scores to share a numerical
scale. On a comparable scale, a reported score's position within its resampled
distribution can also be a useful diagnostic.

The current percentile curve is an agreement diagnostic, not a calibration
test with known ground truth. Its area from the uniform diagonal is descriptive:
exact rank recovery puts every percentile at 0.5, not on a uniform distribution.
Use an accurate name in the paper without losing the comparison Bruce's sketch
was intended to show. A reported rank's position within its resampled
distribution combines central disagreement and spread; the central-order
comparison helps distinguish them.

The conclusion is about dependence on the fitness construction, not a proof
that the original study is wrong or that one fitted model recovers universal
biological fitness. The PDZ example in the evidence below shows why broad
correlation and detailed ordering need separate examination.

### Analysis 2: What is the impact of including noise?

Keep the central ordering fixed within each analysis and show the sampled
rank distributions. The board's lower/upper annotations suggest summarizing
movement relative to a reference rank rather than mixing candidates at very
different rank positions. A concrete implementation, already used for the
tracked candidates, is to report the 5th and 95th sampled-rank percentiles
minus the rank induced by median scores. Those are signed deviations; the
central 90% interval width is a separate summary. The reference rank need not
be inside that interval.

Begin with individual rank intervals, then compare their distributions across
landscapes and models for the central top 100. Keep the distinction between
sampled ranks and the median-score reference. Reported-order diagnostics for
the top 1,000 and selection sizes up to 1,000 provide the broader leading-cohort
context. The existing outputs are sufficient for this initial draft.

Then translate rank uncertainty into top-k inclusion and overlap. Place the
TrpB libB/NucB G3 examples here, alongside the majority improvement from 100
to 1,000 and its repeated exceptions. These are consequences of the rank
analysis and remain within the agreed top-focused scope.

### Analysis 3: Why do we see these impacts?

Begin with the board's low-count conjecture. The available results support
negative input-support versus score-width associations in all 158 analyses
within the current decision cohort. Use simplified Figures 10 and 11 to show
count support, score precision, and the consequences for ranks and membership.
Local score separation extends the original conjecture: precise scores alone
do not determine ranking resolution when nearby competitors are close.

Use the full per-analysis displays in the supplement. Keep the main text
explanatory and show the across-landscape evidence alongside a few examples.
The current mechanism cohort is selected around the top-100 decision, so the
explanation is explicitly about that cohort. These are
observational, model-conditional associations, not read-depth interventions.

### What is ready for the initial draft

Existing curves, rank intervals, top-k results, and mechanism displays already
support the three questions. The first illustrated draft uses compact versions
of Figures 1--3 and combines the explanatory messages from Figures 9--11.
No all-variant rank-uncertainty extension is requested.

A focused check of the saved parameter-resampling tables found 100 rows per
analysis in the model-rank interval table and reported-percentile cohorts of
100 and 1,000. The broader mechanism table covers 0.7% to 27.0% of candidates
per analysis (median 11.3%); none of the 49 analyses covers every candidate.
The full-candidate functionality table contains median scores and reference
ranks, not sampled-rank intervals.

This restriction also appears in the current
[rank analysis](../src/dms_rank_stability/ranking.py#L222) and
[interval table](../src/dms_rank_stability/tables.py#L282): exact full-landscape
ranks are stored for a selected cohort, not for every candidate. Computing a
tracked candidate's rank against all competitors does not mean every candidate
has a stored rank distribution.

This coverage check clarifies what the stored summaries represent; it does not
create a missing requirement. Do not label the selected mechanism cohort as
"all variants." The initial draft retains the existing candidate and cohort
definitions without changing the ranking computation.

## Sources and aggregation

The reviewed numerical tables are under this companion-project output root:

```text
/home/t-renatob/RenatoBerlinghieri/dms-fitness-stability/examples/paper/output/
    hmc/tables/
    mle_predictive/tables/
    hmc_count_based/tables/
```

The corresponding current labels are `parameter_posterior`, `mle_predictive`,
and `posterior_predictive`. After excluding the two K50 score views, the
family-landscape-model identities match [../analysis/runs.json](../analysis/runs.json):
49, 60, and 49 analyses, respectively. Their landscape counts are 18, 19, and
18. TrpB four-site is present only in the MLE collection.

The review used the inventory, top-k curves, model-rank intervals, reference
diagnostics, and mechanism run/effect/diagnostic tables, together with the
existing figure exports and their [plotting implementation](../analysis/figures.py).
This checks saved results and their interpretation, not the correctness of
every upstream fit or the full score-to-table pipeline. During drafting,
canonical tables at `/data/data_dms_rank_stability/tables` were also located
and used to regenerate the manuscript figures; headline and cohort-level
statistics were checked against them. The generated figure provenance records
the canonical table checksums.

For statements about landscapes, take the median across model specifications
within each landscape at each k. Do not count different fitted models or the
three resampling schemes as independent experiments. Current Figure 3's bold
curves instead take the median over runs within each family, so they weight
landscapes according to their number of fitted models. State or revise this
aggregation when composing the main-text figure.

## What the tables support

### 1. Substantial candidate turnover appears under all three schemes

For TrpB libB with the Exponential-Exponential specification:

| Resampling scheme | Same top-1 variant across resamples | Expected shared variants between two top-100 sets | Median central 90% rank-interval width among reference top 100 | Reference top-100 intervals extending below rank 100 |
| --- | ---: | ---: | ---: | ---: |
| Parameter resampling | 100% | 50.96 of 100 (50.96%) | 281.00 positions | 84 of 100 |
| MLE-based count resampling | 100% | 32.12 of 100 (32.12%) | 624.23 positions | 91 of 100 |
| Posterior-based count resampling | 100% | 25.87 of 100 (25.87%) | 859.50 positions | 92 of 100 |

This is a useful example for the second analysis section: agreement on the
winner does not establish agreement on a longer candidate list. Depending on the resampling scheme, a
top-100 set has about 49 to 74 members absent from another resampled set, on
average. Those are changes in selection, not demonstrated biological errors.

In the same order, 516, 977, and 1,186 variants each have more than 1% top-100
inclusion probability. The candidate pool contains 9,261 variants for parameter
resampling and 9,257 for the count-based schemes. The comparison establishes
qualitative persistence across procedures, not an additive decomposition of
noise with an identical candidate universe.

### 2. A stable group need not have a resolved winner

NucB G3, Exponential specification, gives the complementary example:

| Resampling scheme | Fraction of resample pairs sharing the winner | Expected shared fraction of two top-100 sets | Expected shared fraction of two top-1,000 sets |
| --- | ---: | ---: | ---: |
| Parameter resampling | 2.14% | 93.49% | 97.75% |
| MLE-based count resampling | 4.19% | 87.02% | 97.51% |
| Posterior-based count resampling | 4.99% | 87.79% | 96.46% |

There are 18,618 eligible variants. Uniformly chosen sets of 100 would share
only 0.54% of their members on average, so the observed overlap is not simply
a consequence of selecting a large fraction of the candidate pool.

Use this contrast with TrpB libB to show that resolution is local and
task-dependent. Avoid a blanket claim that the scans are uninformative or
that every experiment resolves coarse rankings but not fine ones.

### 3. Larger sets are usually more stable over the checked range

Let C_k denote the mean shared fraction between two distinct retained
resampled top-k sets, as stored in `sample_self_consistency`. This is not the
probability that the entire sets are identical. For k = 100, multiplying C_k
by 100 gives the expected number of shared variants.

| Scheme | Landscapes with C_1000 > C_100 | Analysis runs with C_1000 > C_100 |
| --- | ---: | ---: |
| Parameter resampling | 13 of 18 (72.2%) | 36 of 49 |
| MLE-based count resampling | 14 of 19 (73.7%) | 42 of 60 |
| Posterior-based count resampling | 13 of 18 (72.2%) | 33 of 49 |

The same five landscapes decrease under all three schemes: TrpB libraries D,
E, F, G, and H. This repeated pattern is more useful scientifically than a
generic warning about exceptions. It motivates examining the competing scores
at each cutoff rather than assigning one reliability label to a landscape.
The fixed-k mechanism analysis alone does not establish why every curve turns.

As a descriptive sensitivity check, subtract the uniform-subset baseline k/N
and divide by 1 - k/N, where N is the run's candidate count. The counts of
increasing landscapes and the identities of the five exceptions are unchanged.
This is a check on interpretation, not a proposed new main-text metric.

Do not strengthen the endpoint comparison into monotonicity at every tested k:
only 6/18, 4/19, and 6/18 landscape-median curves are nondecreasing at every
tested step. A supported sentence is: "Increasing the candidate-set size from
100 to 1,000 improves stability in most landscapes under all three schemes."

### 4. High overall agreement can coexist with uncertain top selections

PDZ trans-1, Exponential-Exponential parameter resampling, has Spearman
correlation 0.975 between central model scores and reported scores across
the landscape. Yet two resampled top-100 sets share only 59.80 members on
average; 78 of the central model's top-100 variants have central 90% rank
intervals extending below rank 100. The median interval width is 178.5 ranks.

This supports keeping published-order recovery as an early result, but a high
whole-landscape correlation must not stand in for decision-level resolution.
It does not establish that either ordering is biological ground truth.

### 5. The explanatory section has consistent count-support evidence

The primary mechanism cohort is the central model top 100 plus outside
challengers with top-100 inclusion probability at least 0.01. It is not an
all-variant population analysis.

| Pattern within the primary cohort | Parameter resampling | MLE-based count resampling | Posterior-based count resampling |
| --- | ---: | ---: | ---: |
| Negative input-count versus score-width correlation | 49/49 runs | 60/60 runs | 49/49 runs |
| Negative output-count versus score-width correlation | 49/49 runs | 59/60 runs | 49/49 runs |
| Positive adjusted score-uncertainty coefficient for rank width | 17/18 landscapes | 18/19 landscapes | 17/18 landscapes |
| Positive adjusted crowding coefficient for rank width | 15/18 landscapes | 17/19 landscapes | 14/18 landscapes |

The input-count association is negative in all landscape summaries as well.
The median landscape correlation is -0.770, -0.951, and -0.950 for the three
schemes. Boundary uncertainty relative to score spacing correlates with
top-100 instability at 0.969, 0.882, and 0.955 across landscape medians.
The latter quantities summarize the same score samples, so their relationship
is descriptive, not independent validation of an experimental mechanism.

This supports an accessible explanation: uneven count support produces uneven
score precision under the fitted models; score uncertainty changes ranks and
membership when competing scores are hard to separate. The analysis does not
estimate how much additional sequencing would improve an experiment.

Keep weaker patterns secondary. Replicate-discordance effects are small and
less consistent; the uncertainty-by-crowding interaction is positive in only
3/18, 7/19, and 6/18 landscapes. Higher central fitness is not universally
associated with narrower score intervals, particularly in NucB. Do not make
either interaction or a universal fitness-precision trend the main explanation.

## Fitting and resampling

Give model fitting an explicit section before the ranking results. Describe
the observed counts and the actual replicate, timepoint, or gate structure;
the assay-level score; the selection and observation models; the candidate
model specifications; and how maximum likelihood and Hamiltonian Monte Carlo
were run and checked. State sample retention and candidate eligibility.

The companion provenance notes identify fit choices and score constructions,
but do not provide the complete original model source and submission settings.
Recover the likelihoods, priors, optimizer/sampler settings, and fit diagnostics
before writing a complete fitting section. A validated artifact is not by
itself a convergence or model-adequacy guarantee.

Then present one common workflow: generate a resampled fitness landscape,
rank the candidates, and calculate the same summaries.

| Scheme | Generation recipe |
| --- | --- |
| Parameter resampling | Draw fitted parameters from their posterior and calculate each variant's assay-level score. |
| MLE-based count resampling | Fit by maximum likelihood, simulate counts at the fitted parameters, and recompute the count-based fitness score. |
| Posterior-based count resampling | Draw parameters from their posterior, simulate counts at those parameters, and recompute the count-based fitness score. |

These are three noise-aware resampling schemes. Let the recipes communicate
what varies, without making repeated terminology disclaimers the paper's main
theme. For the MLE procedure, the code recomputes fitness from simulated counts
but does not refit the probabilistic model. PDZ's count-based score is a
provisional DiMSum-style approximation; the TrpB and NucB formulas have source
verification in the companion provenance notes. State this where the scores
are defined, rather than implying exact reproduction of every published score.

## Proposed main-text progression

1. Introduction: two audiences, one experimental ordering, and a question about
   fitness construction and ranking resolution. Introduce FLIP2 as a concrete
   user of these landscapes, not as an evaluation already completed here.
2. Fitting assay models and resampling fitness: data, models, inference, three
   generation recipes, and shared ranking summaries.
3. Analysis 1: Does the fitness construction change the ordering? Explain the
   reported-versus-derived comparisons, including the percentile curve and
   direct correlation, then summarize across analyses.
4. Analysis 2: What is the impact of including noise? Show individual rank
   intervals, their variation within central top-100 cohorts and across fits,
   and consequences for selection budgets up to 1,000. Use the existing
   top-focused outputs without adding a full-landscape analysis.
5. Analysis 3: Why do we see these impacts? Start with the low-count conjecture
   and use simplified Figures 10 and 11 to connect support, score precision,
   competing scores, and inclusion.
6. Discussion: implications for variant selection and benchmark labels in equal
   measure; model dependence and scope; future FLIP2 subset and predictor analyses.

Keep the three analysis sections intact, using rank and top-k subsections
inside Analysis 2 if needed. Further analysis sections can follow these three.
Compare the resampling schemes alongside each question, with full model-by-model
sensitivity in the supplement, rather than making two schemes an afterthought.

## Main-text explanation figure

Use two deliberately contrasting examples already introduced in the results,
such as TrpB libB and NucB G3. A compact display can retain three relationships
from Figure 10: count support versus score width, distance from the top-100
score cutoff versus score width, and that distance versus inclusion probability.
Explain each relationship before moving to the next.

Use Figure 11's joint input/output view to show how count support is distributed
and correlated, rather than presenting two marginal trends as independent
mechanisms. A small joint-count panel can replace duplicated count panels;
keep the full corner plots and complete per-analysis Figure 10 panels in the
supplement. Retain a local-score-neighborhood illustration where necessary to
make the role of competing scores concrete.

Use a consistent candidate cohort, label score-width normalization and log
transforms explicitly, and retain the distinction between central top-100
members and challengers. Do not extrapolate a straight-line fit through
inclusion probabilities beyond their range. Put the across-landscape support
for the explanation in the text or a compact summary, not another wall of fits.

## What not to use as the lead claim

The most extreme parameter-resampling PDZ cases are strongly specification
dependent. For example, trans-1 top-100 overlap is about 59.80% under the
Exponential-Exponential model but 0.27% under Gamma-Exponential. Those fits
deserve explanation and diagnostic review before an abstract uses the most
extreme value as a statement about the experiment. The TrpB/NucB contrast is
a stronger illustration of the second section because its qualitative message
persists across all three resampling schemes; it does not replace the board's
first question about fitness construction.

Do not claim that a predictor leaderboard changes, that a particular FLIP2
performance gap is unresolvable, or that adding reads has a demonstrated
causal effect. None of those analyses was part of this evidence pass.