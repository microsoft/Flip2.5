# Manuscript review

## Typography and alignment pass: 2026-09-09

The manuscript and supplementary figures now share one typeface and size
hierarchy: DejaVu Sans, 8-point labels/ticks/legends, and 9-point headings
and panel letters. All exports are 6.5 inches wide and placed at that width
in the manuscript, so LaTeX does not introduce differences in apparent font
size. Longer axis labels wrap instead of shrinking.

Labels and captions use the compact names `libB`, `G3`, and `trans-1`.
The generated evidence tables now contain an explicit centering command;
the manuscript also centers figure/table floats and long tables. PDF checks
confirm consistent text sizes and centered placement for all 64 figures,
and centered horizontal rules for all nine table placements. The scientific
quantities, candidate sets, and fit diagnostics are unchanged.

## Three-example point display: 2026-09-09

The user approved the Figure 10-style design with three differing examples
and no Gamma parameter fits. Figure 4 now shows PDZ trans-1 and TrpB
library B under Exponential--Exponential, and NucB generation 3 under
Exponential. These are the existing main-text examples, not a new search
for extreme fits. The full supplementary collection, including Gamma
specifications, is retained.

Each row follows the same reference members and challengers through input
count versus score width, output count versus score width, score margin
versus width, and score margin versus inclusion frequency. All 1,063
candidate rows are retained. The rejected decile and neighborhood display
and its active manuscript references have been replaced; earlier descriptions
below are historical.

The explanation now proceeds one example at a time. PDZ has closely related
count support and score width, but many plausible replacements. TrpB has
a broader width range and challengers whose median scores lie farther
below the cutoff. NucB has relatively broad intervals for many reference
members while retaining high inclusion, with only 28 challengers and 77
reference members included in at least 99% of samples. These patterns
support looking at candidate-level counts, scores, and joint-sample
selection together; they do not identify a causal mechanism.

The caption defines all four columns and the cohort, and the supplementary
methods state that margin ranges differ by row. The exporter keeps original
variant values and plotted coordinates, plus a three-run index and example
provenance. No ranks, scores, fits, or selection frequencies were recomputed.

## General argument and style review: 2026-09-09

This records the general prose review. It follows sections A--H of the skill's
[review checklist](../.github/skills/paper-writing-renato/references/review-checklist.md)
and compares the draft with the applied evaluation and sensitivity examples
in the skill. The earlier entries below record previous versions.

### Main findings

1. **The paper kept explaining its intended message instead of progressing
   through the evidence.** Variants of "what the measurements support,"
   "these distinctions," and "not just scores" appeared in the opening,
   background, methods, results, and conclusion. Each repetition added a
   qualification or a new name. The revisions retain the two motivating
   questions in the introduction, then use concrete quantities: fixed-list
   intersections, sampled ranks, inclusion frequencies, and batch overlap.

2. **The methods were organized around the implementation rather than the
   reader's dependencies.** Candidate masks and an inventory of sampling
   schemes appeared before the assay descriptions or recipes. Fitting
   preceded the count model, and the count-score function was used before
   its calculation was described. The order is now data and assays, count
   models, score definitions, fitting, sampling, ranks, and selections.
   The inventory follows the sampling recipes. Full model equations and
   provenance remain in the supplement.

3. **The explanation was stronger in tone than in evidence.** The count
   correlations are consistent across the analyzed cohorts. The spacing
   examples and interaction regressions do not establish what causes the
   observed turnover, and the two selected neighborhoods are not a comparison
   at the batch cutoff. The revised section separates the count result,
   descriptive score/rank relationships, and unresolved model sensitivity.
   Regression coefficient counts have moved to the supplementary methods.

4. **Caveats interrupted results repeatedly, yet the principal validation
   gap remained easy to miss.** The distinction between parameter and count
   samples now lives in the sampling paragraph. The first comparison explains
   score-definition differences before interpreting rank disagreement. The
   discussion states missing predictive checks immediately after its opening
   result, then treats engineering and benchmark consequences separately.
   No validation gap has been removed by changing the prose.

5. **Figure 4's summaries discard the variation the user wants to inspect.**
   Decile medians and family medians cannot show the full range of candidates
   at comparable counts or the separation between reference members and
   challengers. The new direction is point-level, drawing on Figures 10 and
   11. Figure assets and plotting code were deliberately left unchanged
   while that design is discussed. The decile figure is a retained working
   trial, not an endorsed final figure.

### Contract audit

| Promise in the opening | Evidence in the paper | Assessment |
| --- | --- | --- |
| Study protein fitness for batch selection and prediction | Introduction; assay descriptions; Discussion | Both audiences are explicit. Predictor training/evaluation is motivation, not an experiment performed here. |
| Fitness calculation can change the selected batch | First analysis section; fixed-reference table | Delivered by fixed-list intersections. Figure 1 also incorporates sampled-rank variability and is not a pure construction comparison. |
| Noise within a fixed analysis can replace batch members | Second analysis section; Figure 3; overlap table | Delivered for specified models and score constructions. These are model samples, not independent physical repeats. |
| Leading ranks are uncertain | Figure 2; sampled-rank interval definitions | Delivered for the reference top 100. Full-landscape rank-distribution coverage is neither promised nor required. |
| Batch size changes the extent of turnover | Figure 3; supplementary 100/1,000 endpoint table | Delivered without claiming monotonicity or a universal benefit from a larger batch. |
| Input counts are associated with score uncertainty | Third analysis section; Figure S2; all per-analysis point displays | Negative correlation in every analyzed decision cohort; multiple runs share experimental data. |
| Competing scores help interpret rank uncertainty | Third analysis section; current Figure 4; supplementary regressions | Descriptive evidence, not an identified causal mechanism. The main visual argument remains under discussion. |
| Sampling assesses plausible alternative scores and count outcomes | Methods; model supplement | Three recipes documented without presenting resampling as a new contribution. |

The body also reports model sensitivity and the available fitting diagnostics.
These limit the interpretation of the headline result and belong beside it;
they are not separate methodological contributions. No theorem, calibrated
coverage guarantee, optimal batch policy, or changed ML leaderboard is
promised. Theory-specific checklist items are therefore not applicable.

### Targeted edits implemented

| Before | After | Reason |
| --- | --- | --- |
| "These results motivate reporting ... which distinctions the measurements support." | "The extent of this turnover differs across landscapes and selection budgets ..." | End the abstract with the finding, not a generic instruction to the reader. |
| "Eligibility is fixed within a run by the aligned candidate table." at the start of the methods | Data, assays, and model outputs first; candidate conventions beside the score definitions | Remove implementation dependencies from the reader's first encounter with the data. |
| "These fits supply alternative noisy realizations ... not competing sequence-to-fitness predictors." | Define the likelihood, scores, fitting, and sampling directly | Do not tell the reader what kind of paper this is not after the introduction has established the goal. |
| "Figure ... makes the variability within these runs concrete." | State the rank-interval result, then identify the blue medians and black reference ticks | Replace narration about the exhibit with a finding and its visual evidence. |
| "Inclusion probabilities identify ... plausible replacements" repeated twice in one paragraph | "The cutoff therefore separates candidates with similar selection frequencies." | State what the displayed probabilities imply in the example. |
| Main-text inventory of positive uncertainty, crowding, and interaction coefficients | Brief result and limitation in the main text; exact counts in the supplement | Keep the third analysis about the scientific question rather than a list of regression outputs. |
| "The presence of a numerical result does not erase those limitations." | State the PDZ sensitivity and point to the missing model checks | A concrete evidence gap is more useful than an admonition. |
| "Central reference," "central top 100," and "median-score reference" | "Median-score reference" and the defined shorthand "reference top 100" | Prevent readers from treating one reference as several different objects. |

The reported-rank percentile formula moved to the computational appendix;
the main text retains the interpretation needed to read Figure 1. The
equations defining scores, ranks, inclusion, and overlap are preserved.
The supplement now distinguishes completed TrpB/NucB score reconstructions
from missing displays and the unrecovered DiMSum configuration.

### Figure 4 discussion, not implementation

The question I would put above a candidate figure is: **Which variants can
replace members of the proposed batch, and how do their count support and
score uncertainty differ?** This connects the explanation to the selection
result without promising that a scatterplot identifies a causal mechanism.

The most direct candidate is two or three Figure 10-style rows, one per
specified analysis. Each row retains every reference member and challenger
and displays score width against input count, output count, and score margin,
then inclusion frequency against the same margin. Separating input and output
columns allows color and fill to identify the same candidate groups throughout.
No deciles, density replacement, fitted curves, or pooling across runs is
needed. The existing top-focused cohort remains explicit; these are points
for each retained variant, not all variants in the original landscape or
all individual sampled scores.

Figure 11 suggests a second option: retain its three pairwise point panels
for input count, output count, and score width, then add a batch-inclusion
panel. This is useful if joint input/output support is central to the claim.
Its density diagonals and upper-triangle correlations would not replace
the point displays. The current Figure 11 also includes retained non-primary
context rows, unlike Figure 10's primary cohort; any adaptation must state
and consistently use its cohort. Its count axes are log10(count+1), and its
score-width variable is log-transformed, which must be explicit in labels.

A compact alternative would put score margin and width on the two axes and
encode inclusion by color. It retains individual points and three quantities,
but exact inclusion frequencies become harder to compare than when they
have their own axis. This is a layout tradeoff, not an argument for binning.

For any version, the scientific examples should be chosen after agreeing on
the question. Preserve all points, including zeros and outliers under stated
plotting conventions. Use small markers, open/filled group encodings, shared
axes where comparable, and a full view plus a labeled zoom if needed.
Differences between assays or model specifications are descriptive contrasts,
not interventions. The complete supplement remains the check against relying
on only the displayed examples.

### Questions still open

1. Should the new Figure 4 primarily locate plausible batch replacements,
   or explain the joint input/output count pattern? The first is closer to
   the paper's selection result.
2. Which model checks are necessary before interpreting the largest turnover
   estimates as experimental uncertainty? The strongly specification-sensitive
   PDZ results make this an evidence question, not a wording question.
3. For a downstream demonstration, is the relevant consequence changed batch
   activity, retrieval scores, or a comparison between predictors? Turnover
   alone does not establish any of these effects.

## Protein fitness and Figure 4 trial: 2026-09-09

The abstract now defines protein fitness in its first sentence and identifies
fitness as the target of ML training and evaluation. Its final two sentences
state one result and consequence: changing the fitness calculation or adding
noise within a fixed analysis can change the selected batch, so selection
uncertainty belongs alongside the fitness estimates. The secondary contrast
about stable membership and uncertain internal ranks remains in the results,
not as a competing abstract conclusion. The abstract has 170 words and eight
sentences, without numerical result claims.

The user approved trying the proposed four-panel Figure 4. It now shows
input-count and output-count decile summaries in blue and orange, respectively;
score-width versus rank-width decile summaries; and two matched score
neighborhoods in one fit. The run-level lines remain visible, while the
family medians give each landscape equal final weight. They are descriptive
connections between bins, not fitted regressions.

The matched comparison is the verified TrpB library-B Exponential--Sigmoid
pair at reference ranks 10 and 23. Its similar normalized score widths
(1.45 and 1.42) accompany different local spacing and sampled-rank intervals
(8--66 and 13--155). Matching excludes rank and inclusion outcomes. The
comparison illustrates local rank uncertainty within the top batch; it is
not a matched test at the top-100 cutoff and does not isolate a causal effect.
A check restricted to ranks 75--95 found a much smaller spacing contrast
in this fit, so that restriction was not adopted for this trial.

The caption, main explanation, and supplementary selection rule have been
updated. Figure-data CSVs preserve the deciles, family medians, and complete
neighborhoods. Earlier notes below saying Figure 4 was awaiting approval are
historical; the design is now implemented for review. No fits, underlying
rank/selection results, or Figure 10 pages were changed.

## Opening accessibility pass: 2026-09-09

The user-supplied earlier outline is the reference for this exposition pass.
Its useful progression is the practical use of DMS measurements, the
sequence/score table, and the two things that table hides: choices in score
calculation and uncertainty from finite counts. The manuscript now follows
that progression before introducing technical summaries.

| Previous obstacle | Implemented revision |
| --- | --- |
| Abstract ends with specific overlap percentages and unnamed examples | General result: score calculation can change a batch, and noise can change membership or internal ordering differently. Numerical evidence remains in the results. |
| Introduction uses marginal errors, joint realizations, cohorts, and a median-score reference early | Explain scores, counts, the reported table, and batch selection in ordinary language; define the technical summaries in the methods. |
| Background introduces a multinomial variance and a shared-offset equation before the modeling section | Explain finite sampling with the count-ratio example and shared score variation in words. The formal count models and joint sampling definitions remain in the methods and supplement. |
| Generalized results risk sounding like all-variant conclusions | State that the input-count association concerns the leading candidates examined in each analysis. |

The contract is unchanged: distinguish score-construction effects from noise
within a fixed analysis, quantify leading-rank and batch uncertainty, and
interpret it using count support and competing scores. The first results
section and its fixed-reference table deliver the construction claim; the
next section and Figures 2--3 deliver the general abstract findings. No new
benchmark result, model check, or exclusive novelty claim is introduced.

The abstract has 169 words and eight sentences; the introduction has five
paragraphs. No mathematical notation appears before the methods. These
structural checks accompany the definition-order review, not replace it.

Figure 4, its builder, and its caption are unchanged. The current comparison
provides continuity with the TrpB/NucB batch examples but does not isolate a
mechanism: assay, fitted score, counts, and competitor structure differ
together. The next design should be discussed before implementation. The
outline's score-neighborhood displays are a possible way to make competitor
spacing visible, with candidate inclusion calculated from joint samples
rather than inferred from overlap of marginal intervals. Numerical examples
and associations in the older outline need verification before reuse.

## Batch-first framing pass: 2026-09-09

This pass follows the skill's contract audit and definition-first reading.
The revisions are implemented in [main.tex](main.tex) and the main selection
figure. The historical review below is retained as a record, not as the
current statement of the contribution.

### Main changes

1. Resampling is the way we generate joint noisy realizations and ask which
   batches the fitted experiment could also support. Its recipes are
   documented for reproducibility and sensitivity, not compared as candidate
   methodological contributions.
2. Batch composition is the practical outcome. The title, abstract,
   introduction, results, and discussion no longer motivate single-candidate
   selection. Figure 3 displays batch sizes 10--1,000; the numerical source
   tables retain their smaller cutoffs.
3. The contribution is stated positively: experimental uncertainty can change
   the batch that guides engineering and defines a predictor's retrieval
   target. Prior score-error estimation is relevant background, not evidence
   that this empirical batch-level question has already been answered.

### Contract audit

| Contract | Deliverer | Boundary |
| --- | --- | --- |
| Who wants what: engineers choose batches; ML researchers learn and evaluate their retrieval | Introduction and Discussion | Neither audience is an afterthought; no claim that all engineering objectives reduce to top-k membership. |
| A reported ordering supplies exact batch membership without displaying its uncertainty | Background; reported/fitted comparison | Changing the score definition and adding noise within one definition are distinct. |
| Fitness construction changes batch membership | Section 4, fixed-reference table, Fig. 1 | The table isolates fixed-set differences; Fig. 1 also depends on resampled ranks. |
| Noise changes batch members, not just their internal ordering | Section 5.2, Fig. 3, overlap table | TrpB example: 26--51 shared members per 100 on average under the stated models. |
| A more stable batch can contain uncertain ranks | Sections 5.1--5.2, Figs. 2--3 | NucB example: 87--93 shared members, alongside overlapping rank intervals. |
| Count support and competing scores help interpret resolution | Section 6, Fig. 4, supporting figures | Descriptive relationships, not a causal estimate of additional sequencing. |
| The experimental target used for ML has uncertainty | Introduction and Discussion | No demonstrated predictor-order reversal or training improvement. |

### Targeted framing edits

| Previous emphasis | Revised emphasis | Reason |
| --- | --- | --- |
| "same winner in every retained sample" versus uncertain longer lists | "uncertainty can replace a substantial fraction of a proposed batch" | The application concerns batches, and the retained evidence directly measures their turnover. |
| "three resampling schemes" as a contribution-level inventory | "joint fitness realizations" as the means of measuring rank and batch uncertainty | The scientific question should remain understandable without knowing the three recipes. |
| "Existing methods already address several parts of this workflow" | Name what the cited methods estimate, then distinguish joint batch membership from marginal score precision | Related work contextualizes the contribution without substituting a vague similarity claim for a comparison. |
| Selection curves starting at one | Batch curves starting at ten | The visual story now agrees with the stated practical unit. |

### Positioning and evidence

The stronger message is not that DMS measurements are noisy. It is that the
membership of an experimentally defined target batch can itself be uncertain,
with substantial turnover in the analyzed data. This changes how one should
interpret an exact top-ranked list before attributing a disagreement to a
protein-design or prediction method. It does not establish that the alternative
batches are worse, or that a predictor comparison actually reverses.

For a future Nature Communications submission, the empirical message should
carry the paper. The important evidence gaps are (1) showing that the inferred
turnover reflects the experiment rather than model inadequacy, using predictive
checks and suitable independent or held-out observations, and (2) demonstrating
a consequence for a concrete batch objective or benchmark comparison. High
turnover among nearly equivalent variants and turnover involving materially
different activity have different practical implications. These are research
priorities, not additional analyses claimed or performed in this writing pass.

A Nature Methods direction would need a demonstrated general methodological
advance and validation, not simply more detail about existing resampling
procedures. Nothing in the current framing requires that direction.

The current references establish that marginal fitness uncertainty and
benchmark design have substantial prior literature. They do not establish
either that this exact batch-resolution contribution exists elsewhere or that
it is absent. No exhaustive novelty search was performed in this pass.
An exclusive "first" or "to our knowledge, no previous study" statement
should follow a focused comparison with the nearest relevant methods and
empirical studies. Strong quantitative results do not require that statement
to be described clearly.

## Historical review: 2026-09-08

Reviewed 2026-09-08 against the paper-writing-renato skill and review checklist.
Revision status: the manuscript has since received definition-order repairs,
expanded background, explicit count features, fixed-reference and reference-overlap
tables, and a source-grounded model/fitting supplement. The review below is a
historical list of findings with original line references, not a claim that
each remains unresolved. Current gaps and validation are listed in
[README.md](README.md).
This is feedback, not an edited manuscript. The top-focused scope and the three
agreed analysis sections are retained. No new full-landscape or benchmark
evaluation is requested. Locations refer to the draft at review time.

## Overall assessment

The draft follows the visible house structure more closely than its
definition-first discipline. The two-audience motivation, named reported
ordering, five-paragraph introduction, and three analysis questions are in
place. But the body still depends on knowledge of the code and prior project
discussions. A reader cannot reconstruct the fitted scores, identify all the
analysis populations, or interpret every reported summary from the paper alone.

The most important revisions are:

1. Define the actual fitted models and latent fitness scores, not just names of
   functions and saved quantities. The fitting and diagnostic gap requires
   source information, not more generic statistical prose.
2. Repair the definition order: one analysis run, joint score samples, central
   reference, sampled ranks, sets and inclusion frequencies, then cohorts and
   summaries. Use consistent names for those objects.
3. Bring the claims into alignment with the displayed statistics. Section 3's
   median sampled ranks do not isolate a changed central fitness construction;
   Section 5's main-effect coefficient signs do not establish positive slopes
   throughout an interaction model.

## Contract audit

Repeated promises in the abstract and introduction are grouped below. A
deliverer can be present while its method definition or interpretation remains
incomplete.

| Promise and location | Deliverer | Assessment |
| --- | --- | --- |
| Study leading rankings in three protein systems; [main.tex](main.tex#L41), [main.tex](main.tex#L90) | Section 2; Table 1; examples throughout Sections 3--5 | Scope is explicit. The identities of the individual PDZ landscapes, TrpB libraries, and NucB generations still need a compact scientific description. |
| Fit assay models and use three resampling schemes; [main.tex](main.tex#L42), [main.tex](main.tex#L91) | Section 2; Appendix A | Recipes are present, but likelihoods, priors, fitted score maps, and fit-quality evidence are not. Partial delivery, explicitly marked TODO. |
| Examine fitness-construction dependence separately from within-analysis uncertainty; [main.tex](main.tex#L39), [main.tex](main.tex#L78) | Section 3; Fig. 1 and Fig. S1; Section 4 | Only partly delivered: Section 3 compares reported ranks with median sampled ranks, not the median-score reference defined in Section 2. Keep the diagnostics, but do not treat them as an isolated construction effect. |
| Quantify rank intervals; [main.tex](main.tex#L44), [main.tex](main.tex#L100) | Section 4.1; Fig. 2 | Delivered as a sample summary. Missing definition links and finite-score conventions make the estimand harder to recover than necessary. |
| Assess candidate-set stability; [main.tex](main.tex#L44), [main.tex](main.tex#L97) | Section 4.2; Fig. 3 | Delivered for sample-to-sample overlap and inclusion. The two additional reference-overlap measures defined in Section 2 are not displayed in this manuscript. |
| TrpB has a stable winner but 26--51 shared top-100 members; NucB has 2--5% winner agreement but 87--93% top-100 overlap; [main.tex](main.tex#L46), [main.tex](main.tex#L103) | Section 4.2; Fig. 3 | Numerical deliverers are present. The abstract must say that shared counts are averages and that the ranges are across schemes, not uncertainty intervals. |
| Relate resolution to count support and local score separation; [main.tex](main.tex#L44), [main.tex](main.tex#L98) | Section 5; Fig. 4; Fig. S2; Appendix C | Count-support evidence is displayed, but its predictor needs a per-assay definition. The crowding argument relies on conditional regression coefficients and is currently overstated in the section opener. |
| Negative input-support versus score-width association in all 158 analyses; [main.tex](main.tex#L107) | Section 5; Fig. S2 | Delivered as within-run Spearman correlations in selected cohorts. Define the input-count feature and cohort before the result, and retain the distinction between analyses and independent experiments. |
| Relevance to engineering and predictor evaluation; [main.tex](main.tex#L110) | Discussion; cited motivation | An implication, not a completed benchmark experiment. The draft correctly keeps FLIP2 subset and predictor evaluation as future work. |

The body also reports the majority increase in overlap from 100 to 1,000 and
its chance-adjusted check. These support the decision-scale story without
needing to become new headline promises. There is no missing theorem or
coverage guarantee to add: the draft does not promise either.

## Targeted edits: abstract and introduction

### A1. Clarify what the headline range measures

[main.tex](main.tex#L46)

**Quote:** "two resampled top-100 lists share only 26--51 candidates"

**Replacement:** "Across the three schemes, two resampled top-100 lists share
26--51 candidates on average."

**Reason:** The range combines three mean-overlap estimates. As written, it
could be read as a range across pairs of lists or an uncertainty interval.
Identify the range's source for the NucB percentages as well.

### A2. Remove an unsupported contrast

[main.tex](main.tex#L44)

**Quote:** "We relate the observed differences in resolution to count support
and local score separation, rather than treating every part of a landscape as
equally informative."

**Replacement:** "We relate score and rank uncertainty among leading candidates
to count support and separation between competing scores."

**Reason:** The original adds a contrast that is not assigned to a named
baseline and can sound like a claim about every region of the landscape.
The replacement states the actual analysis and its agreed scope.

### A3. Introduce one name for the central reference

[main.tex](main.tex#L85), [main.tex](main.tex#L100)

**Quote:** "central scores"; "central top 100"

**Replacement at first use:** "the ordering obtained by ranking each variant's
median resampled score, which we call the median-score reference"

**Reason:** Later text alternates among central, reference, model-reference,
and median-score ordering. Choose one principal name and define it once. Keep
the reported ordering and median sampled ranks explicitly distinct. The
introduction can use a short prose definition; the equations belong in Section 2.

### A4. Name the biological objects before relying on their labels

[main.tex](main.tex#L90), [main.tex](main.tex#L131)

**Quote:** "scans of PDZ, TrpB, and NucB"

**Replacement direction:** Introduce the PDZ protein-interaction system, the
tryptophan-synthase enzyme TrpB, and the nuclease NucB in plain language. Add a
compact assay inventory explaining what cis, trans-1/trans-2, CRIPT-C/CRIPT-N,
TrpB library B, three-site/four-site, and NucB generations denote.

**Reason:** Citations establish provenance but do not identify these objects
for a reader trying to interpret the figure labels. Do not invent residue sets
or assay mappings; obtain the exact descriptions from the study records.

## Targeted edits: setup and methods

### M1. A generic likelihood is not a model definition

[main.tex](main.tex#L162)

**Quote:** "Write $\theta$ for the full parameter vector" and
"NucB specifications use the recorded activity-family label."

**Replacement direction:** For each assay, define the observed count arrays and
their indices, the random quantity receiving the named distribution, how its
parameters determine observed counts, and which component is called fitness.
Then specify the likelihood and prior before explaining optimization or
posterior sampling. Replace the NucB label sentence with its actual model.

**Reason:** The present expressions for maximum likelihood and Bayes' rule are
correct generic identities, but do not tell the reader what is fitted. The
Exponential--Exponential explanation merely repeats two labels. This needs
more work, not more prose: the original model and inference records are needed.

### M2. Define the latent score map, not its storage name

[main.tex](main.tex#L183), [main.tex](main.tex#L200)

**Quote:** "The parameter ensembles store log-mean-growth scores for PDZ and
TrpB and model-fluorescence scores for NucB."

**Replacement direction:** Write the actual $h_i(\theta)$ for each family,
define the growth or fluorescence quantity and its units/transformation, and
state why larger values are better. Explain whether a mean is across
replicate-level parameters or refers to a distribution parameter.

**Reason:** Calling $h_i$ a score function defines its role, not its value.
Appendix A defines count-derived scores only; it does not supply the missing
latent-score definition. Leave a specific TODO until the parameterization is
verified, rather than guessing that a stored label means $\log E[r_i]$.

### M3. Define the analysis context and empirical notation explicitly

[main.tex](main.tex#L182), [main.tex](main.tex#L213)

**Quote:** "$\widetilde\phi_i:=Q_{0.50}(\phi_i)$" and
"$[Q_{0.05}(R_i),Q_{0.95}(R_i)]$"

**Replacement:** "Fix one landscape, model specification, and resampling
scheme, and let $S$ be its number of retained samples. We write
$Q_a(\phi_i)$ and $Q_a(R_i)$ for the empirical quantiles of the finite score
samples $\{\phi_i^{(s)}\}$ and rank samples $\{R_i^{(s)}\}$, respectively."

**Reason:** Only the superscripted samples have been introduced; the
unsuperscripted $\phi_i$ and $R_i$ appear without a stated interpretation.
This convention repairs the notation without adding a new distribution or
separate indices for every scheme. Preserve the manuscript's existing macros
when applying the edit.

### M4. Define inclusion before using it to define a cohort

[main.tex](main.tex#L222), [main.tex](main.tex#L228)

**Quote:** "outside candidates with top-100 inclusion probability at least 0.01"

**Replacement after defining sets and $\hat q_i(k)$:** "Our decision cohort
contains the median-score reference top 100 and every other candidate selected
in at least 1% of retained top-100 sets. We call the latter candidates
challengers. Separately, reported-order comparisons use the reported top 100
and top 1,000."

**Reason:** The current text uses inclusion probability before defining it and
switches among several top cohorts. Move this paragraph after candidate
selection. Do not suggest that challengers must themselves lie in the central
top 1,000; that is not the current cohort definition.

### M5. State eligibility and finite-score rules at the definition of rank

[main.tex](main.tex#L129), [main.tex](main.tex#L214), [main.tex](main.tex#L661)

**Quote:** "against all eligible competitors in sample $s$"

**Replacement:** "among ranking-eligible candidates with finite scores in
sample $s$; the rank is undefined when the focal candidate's score is undefined"

**Add nearby:** "Rank intervals omit samples in which the candidate's rank is
undefined. Inclusion frequencies use all retained samples, counting an
undefined score as no selection."

**Reason:** A fixed candidate eligibility list and a sample-specific finite
competitor set are different. Appendix B partly explains the convention, but
too late to interpret the primary definitions. Also replace "original candidate
exclusions" with the actual per-assay rules or a precise table reference.

### M6. Fix the singleton-set exception

[main.tex](main.tex#L245)

**Quote:** "None is the probability that two whole sets are identical."

**Replacement:** "For $k>1$, these shared fractions generally differ from the
frequency of exact set agreement. For $k=1$, $\widehat C_1$ is the fraction
of distinct retained sample pairs selecting the same winner."

**Reason:** The blanket statement is false for singleton sets, precisely the
case used in the headline result. The empirical identity does not require
assuming that Monte Carlo samples are independent.

### M7. Avoid defining measures that have no deliverer

[main.tex](main.tex#L233)

**Quote:** "We report three shared fractions"

**Replacement direction:** Either display $\widehat C_k^\star$ and
$\widehat O_k^{\mathrm{rep}}$ in a named supporting exhibit, or introduce only
$\widehat C_k$ here and move the other two definitions beside their eventual
use. Do not change the underlying analysis simply to simplify the prose.

**Reason:** The manuscript figure exporter uses sample-to-sample consistency
for the selection curves. Reference-overlap measures are computed elsewhere,
but their definitions currently create reader effort without a result to inspect.

## Targeted edits: analysis sections

### R1. Section 3 must distinguish a changed reference from a changed summary

[main.tex](main.tex#L264), [main.tex](main.tex#L287)

**Quote:** "Does the fitness construction change the ordering?" followed by
correlations with "median sampled ranks."

**Replacement direction:** Keep the descriptive percentile and median-rank
diagnostics, but add or foreground the comparison of reported ranks with the
median-score reference $\widetilde r_i$. Label the median sampled-rank
comparison as agreement under the resampling distribution, not as an isolated
effect of changing the central fitness construction.

**Reason:** Ranking and taking medians do not commute, as Section 2 correctly
states. Even unchanged median scores can give different median sampled ranks.
A constructed check with score samples $(3,2,1)$, $(3,4,1)$, and $(-100,2,5)$
has median scores $(3,2,1)$ and central ranks $(1,2,3)$, but median sampled
ranks $(2,2,3)$. These are illustrative numbers, not paper results. This
does not invalidate Fig. 1; it limits what its particular summary isolates.

### R2. Give the reported-rank percentile a reusable definition

[main.tex](main.tex#L273)

**Quote:** "the fraction of finite sampled ranks better than
$r_i^{\mathrm{rep}}$, assigning half weight to ties"

**Replacement:** "For candidate $i$, let $u_i$ be the mean of
$\mathbf{1}\{R_i^{(s)}<r_i^{\mathrm{rep}}\}+
\tfrac12\mathbf{1}\{R_i^{(s)}=r_i^{\mathrm{rep}}\}$ over its finite sampled
ranks. We plot the empirical cumulative distribution of $u_i$ in the reported
cohort."

**Reason:** This connects the verbal definition, direction of the diagnostic,
and figure-axis label. The existing observation that exact recovery gives
$u_i=0.5$ is precise and should stay.

### R3. Define count support before making it the explanation

[main.tex](main.tex#L443), [main.tex](main.tex#L458)

**Quote:** "Within each run, we relate $U_i$ to input and output count support."

**Replacement direction:** Define the input and output count features for
each assay, including aggregation over replicates, timepoints, or gates;
specify how shared input observations are counted. State that the reported
associations are within-run Spearman correlations over the decision cohort.

**Reason:** The axes identify read counts but not which observations enter
each total. "Count support" is central to the scientific explanation and
cannot remain a feature-table label. The within-run computation itself is
clear in [tables.py](../src/dms_rank_stability/tables.py#L730).

### R4. Define crowding before using its coefficients, and qualify the opener

[main.tex](main.tex#L445), [main.tex](main.tex#L491), [main.tex](main.tex#L702)

**Quote:** "Score uncertainty and local crowding are also positively associated
with rank uncertainty in most landscape summaries."

**Replacement:** "We also examine how score uncertainty and the spacing of
competing scores relate jointly to rank uncertainty."

**Add before the regression:** "We use local crowding to mean the negative
logarithm of normalized median-score spacing around a candidate's reference
rank; larger values indicate closer competing scores. Appendix C gives the
window and normalization."

**Reason:** The current definition appears only after the results, in Appendix
C. More importantly, positive main-effect coefficients do not establish a
positive relationship throughout the observed covariate range when an
interaction is included. In a model with terms
$\beta_U U+\beta_G G+\beta_{UG}UG$, the slope in $U$ is
$\beta_U+\beta_{UG}G$, not just $\beta_U$. The implementation forms the
product before standardizing the columns, so separate standardization does
not fix this issue. The coefficient-sign counts can be reported literally;
a broader positive-association conclusion needs conditional-slope summaries
or another directly interpretable analysis. This review does not assert that
the numerical coefficient counts are wrong.

### R5. Replace vague interpretation with the actual comparison

[main.tex](main.tex#L447)

**Quote:** "how well its uncertainty separates it from competing scores"

**Replacement:** "how the score's uncertainty compares with the gaps between
competing scores"

**Reason:** Uncertainty does not separate a score from its competitors; the
relationship between uncertainty and separation determines resolution.

### R6. Give the across-landscape set-size result an exhibit

[main.tex](main.tex#L426)

**Quote:** "We therefore report the full set-size curves"

**Replacement direction:** Point to a supporting figure or table covering
the landscape-level comparison, including the five TrpB exceptions. If only
the current figure is retained, say that Fig. 3 displays curves for the three
specified examples, and give the aggregate count in a compact supporting table.

**Reason:** The values were checked previously, but the current Fig. 3 has
only three examples. The five decreasing landscapes and the majority increase
have no full-cohort exhibit in the compiled manuscript. A notebook reference
without a named location is not a substitute for a paper exhibit.

## Targeted edits: appendices and captions

### P1. Make valid-replicate restrictions part of the PDZ equation

[main.tex](main.tex#L607)

**Quote:** sums over all $r$ in the inverse-variance average, followed by
"Replicates with a zero required count receive zero weight."

**Replacement direction:** Define the set of replicates with all four required
counts positive and restrict both sums to that set. State that the score is
undefined when the set is empty. Give $f_{ir}$ and $v_{ir}$ their plain-language
glosses next to their equations.

**Reason:** Multiplying an undefined log ratio by zero does not make the
displayed mathematical expression defined. The implementation masks invalid
observations, so the prose can be aligned with it without changing the method.

### P2. Replace references to unnamed records with explicit rules

[main.tex](main.tex#L625), [main.tex](main.tex#L641), [main.tex](main.tex#L675)

**Quote:** "Library-specific timepoints follow the deposited notebooks";
"The selected gates are generation-specific"; "the deterministic selection order."

**Replacement direction:** Give the retained times, selected gates and any
normalization correction in a small assay table. Specify or cite a versioned
algorithm for exact-tie selection, and separately state the central-reference
tie rule. Explain what a source block is if it remains in the prose.

**Reason:** These sentences point to implementation context without supplying
a rule. Deterministic is a property of a rule, not its definition. Do not
replace the tie policy without checking the consequences for existing results.

### P3. Define the additional regression predictor or remove it from the account

[main.tex](main.tex#L713)

**Quote:** "adding the count-aware observation discordance feature where available"

**Replacement direction:** Give the statistic, the verified observation
structures where it is defined, and the missing-value rule; otherwise report
that this part of the model description is still incomplete.

**Reason:** No definition of this predictor appears anywhere in the manuscript.
The same paragraph should say whether "log input support" means log count,
log(1 + count), or the stored feature's transformation.

### P4. Make captions self-contained without repeating every general caveat

[main.tex](main.tex#L362), [main.tex](main.tex#L411), [main.tex](main.tex#L729)

**Quote:** "same three specifications as Fig. 1"; "Models match Fig. 1";
"This repeats Fig. 1."

**Replacement direction:** Name the examples and specifications in each
caption and expand maximum-likelihood estimate (MLE) and interquartile range
(IQR) where the shortened figure labels are introduced. Keep the cohort,
interval level, sample count, and visual encodings. Add the decision-cohort
sizes to Fig. 4 and identify the individual candidates behind anonymous
"Outside" rows in a supporting table.

**Reason:** A caption should not require a second figure to identify the
plotted experiment. Conversely, the generic no-coverage/no-ground-truth warning
need not be repeated in every paragraph; state the interpretation once in the
methods and retain only the figure-specific qualifications.

## What already follows the skill

- The two-audience opening gives a concrete use case immediately. Keep it.
- The reported ordering has a name, and prior uncertainty-aware methods receive
  a fair description rather than being presented as ignoring uncertainty.
- "Ranking median scores need not give the median sampled ranks" at
  [main.tex](main.tex#L219) is an important, correct distinction. The rest of
  the manuscript needs to apply it consistently.
- The paragraph distinguishing between-candidate spread from within-candidate
  uncertainty at [main.tex](main.tex#L451) is useful and largely defined.
  This is not another missing object; it needs only consistent cohort naming.
- Rank intervals, set membership, and winner agreement are connected to
  different practical questions. Keep the TrpB/NucB contrast within Analysis 2.
- The draft avoids a claim of causal benefit from more reads and correctly
  reserves FLIP2 predictor evaluation for future work.

## Revision order and open questions

First repair the method and definition chain; then revise the interpretation
of Section 3 and the interaction results; finally shorten repeated caveats,
move numeric triples to exhibits where possible, and tighten the captions.
The three-analysis structure itself does not need another reorganization.

Questions that determine the substantive edits:

1. Where are the authoritative model equations and latent-score definitions
   for each assay, together with the retained fits' diagnostic records?
2. Should Section 3 foreground the median-score reference to distinguish the
   construction comparison, keeping median sampled-rank diagnostics alongside it?
3. Should the two reference-overlap measures remain in the paper with a named
   supplement, or be removed from the main methods until used?
4. Is the crowding claim intended as a descriptive illustration, or should it
   be supported by conditional-slope summaries across the observed covariate range?

## Review validation

The source has no editor-reported LaTeX errors; that does not check scientific
definitions. I inspected the relevant rank, figure, and regression computations
and ran small constructed checks for median-rank and interaction interpretation.
I did not refit models, rerun the complete data analysis, edit the manuscript,
or verify its missing convergence and predictive diagnostics in this review.
