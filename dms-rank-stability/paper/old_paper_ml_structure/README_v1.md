# Paper draft

[main.tex](main.tex) is the expanded illustrated bioRxiv-oriented draft,
revised 2026-09-09. It includes a background section, repaired definition
order, explicit latent-score maps, fixed-reference and resampled-overlap
comparisons, the three agreed analysis sections, and a source-grounded model
supplement in [supplementary_details.tex](supplementary_details.tex). The draft
now explains each assay from protein activity through selection and sequencing
to the reported score, and cites twelve sources. It has four main figures,
two aggregate supporting figures, 58 per-analysis supplementary figures, and
complete tables of the 49 posterior and 60 optimized fits underlying the
158 analysis runs. The 58 per-analysis figures occupy one page each.
Predictive checks, some historical provenance, and publication metadata remain
explicit `[TODO: ...]` items. Reserved supplementary subsections distinguish
unperformed checks from the evidence already supplied.

The agreed analysis structure follows the discussion with Bruce: (1) does the
fitness construction change the ordering, (2) what is the impact of including
noise, and (3) why do we see those patterns? Engineering and machine-learning
benchmarking are equal motivations. The author explicitly chose top-focused
analysis: reported top 100/1,000, central top 100, plausible challengers, and
batch sizes from 10 to 1,000. Full-landscape uncertainty analysis is not planned
or required for this draft.

The central message is that uncertainty can change the membership of the
experimental target batch, not merely perturb scores or reorder its members.
Resampling supplies joint noisy realizations with which to measure that
effect; the paper does not propose or benchmark resampling strategies.
TrpB library B and NucB generation 3 illustrate different amounts of top-100
batch turnover, not a contrast involving single-candidate selection.
Both engineering and ML benchmarking remain central motivations. Changing
batch identities is demonstrated under the fitted analyses; a loss of
engineering utility or a change in predictor comparisons is not yet established.

The latest opening revision follows the user-supplied outline's explanation
of the released sequence/score table, assay-defined fitness, and finite counts.
Protein fitness is defined in the abstract's first sentence and named as
the ML prediction target in the introduction. The abstract now has one
general batch-selection conclusion without numerical case-study results
or a competing closing contrast; the quantitative evidence remains in the
results. The introduction and background define experimental terms before
the mathematical methods. The latest general review removed repeated
motivation, reordered the methods by definition dependencies, made the
first two analysis sections result-led, and separated descriptive explanation
from model validation. The reported-rank percentile formula and detailed
regression coefficient counts now sit in the computational supplement.

Figure 4 now uses the notebook Figure 10's four point-level relationships
for three examples: PDZ trans-1 and TrpB library B under
Exponential--Exponential, and NucB generation 3 under Exponential. These
are the same parameter-resampling examples used in the earlier main figures.
No Gamma fit enters Figure 4. Every reference member and challenger is
retained, with no deciles, fitted lines, or pooling across runs. The caption,
main explanation, and supplementary display conventions describe the
differences between the three examples.
The outline informed the exposition, not the current modeling or validation
claims.

[results_review.md](results_review.md) retains the evidence review and planning
history. The current contract audit and positioning notes lead
[review.md](review.md). These current notes take precedence over the older
single-candidate framing and proposals for whole-landscape summaries.

## Build

From the repository root, after installing the project dependencies, supply
the directory containing the three canonical table collections:

```bash
uv run python -m analysis.manuscript --table-root "/path/to/analysis-data/tables"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
```

The local drafting run used `/data/data_dms_rank_stability/tables`. The exporter
in [../analysis/manuscript.py](../analysis/manuscript.py) checks the inventory
against [../analysis/runs.json](../analysis/runs.json), writes PDF/PNG pairs
under `paper/figures/`, and records source-table SHA-256 checksums in its
generated provenance JSON. It neither loads raw fits nor modifies the tables.
It also exports the fixed-reference/overlap comparison table and the
all-landscape top-100/top-1,000 endpoint table into the same figure directory.
The Figure 10 supplement is generated as 58 pages with at most three runs per
page. [figures/summary_why.tex](figures/summary_why.tex) supplies the captions
and scheme/family index; [figures/summary_why_index.csv](figures/summary_why_index.csv)
maps every run to its figure, row, cohort counts, and plot-only width floor.
The provenance JSON records all 58 page specifications and checksums for
nine source tables per scheme.

Figure 4 exports [figures/explanation_variants.csv](figures/explanation_variants.csv)
and [figures/explanation_runs.csv](figures/explanation_runs.csv). The first
retains all 1,063 plotted candidates, their original values, and the plotted
coordinates. The second records the three run identities, cohort sizes,
sample counts, and mean shared fractions. Provenance records the specified
examples, Gamma exclusion, no aggregation, and axis conventions. The old
decile and neighborhood tables are no longer generated or listed in the
export manifest.

The included text snapshot under `paper/generated/` contains the fit tables
and complete machine-readable mappings. To regenerate it from the companion
data repository and the pinned SciStanPy checkout, run:

```bash
uv run python -m analysis.fit_inventory \
	--source-root ../dms-fitness-stability \
	--scistanpy-root ../prob-protein-fitness/SciStanPy \
	--data-root /data/data_dms_rank_stability
```

The exporter verifies SciStanPy revision `09c439f675ead70f851d814242f153ecb7712113`
and unchanged model/launcher files, reads the historical command lists, joins
all 158 analysis identities, and reads only metadata and diagnostic arrays
from available fits. It does not refit models or read full posterior score
matrices. The committed loader is hashed separately because the local loader
has unrelated user modifications; those modifications are neither reverted
nor treated as the historical fitting source.

The snapshot products are [generated/fit_inventory.csv](generated/fit_inventory.csv),
[generated/analysis_fit_mapping.csv](generated/analysis_fit_mapping.csv),
[generated/fit_inventory.tex](generated/fit_inventory.tex), and
[generated/fit_provenance.json](generated/fit_provenance.json).

Run both TeX passes from the repository root. The second resolves references;
run another pass if the log still asks for updated labels or table widths.
The manuscript uses standard TeX packages, including `natbib`, `booktabs`, and `longtable`;
[references.tex](references.tex) supplies the bibliography without a separate
BibTeX step. Figure exports and TeX build products are ignored by Git.

## Presentation

All manuscript and supplementary figure PDFs use a fixed width of 6.5 inches,
matching the manuscript text width. The shared export style uses DejaVu Sans:
8-point axis labels, ticks, legends, and annotations, with 9-point headings
and panel letters. Mathematical superscripts follow the same font's normal
scaling. Builder-specific font overrides are normalized at export. Fixed
canvas bounds and full-width LaTeX placement prevent different crops or
inclusion widths from changing the text size between figures.

Figure and table labels use the source shorthand, including `libB`, `G3`,
and `trans-1`. Explanatory body prose can still use the full names. Float
centering is set in the manuscript preamble; generated evidence tables also
emit an explicit centering command, and fit inventories use centered long
tables. Cell alignment remains appropriate to the data, with numerical
columns right-aligned where specified.

## Evidence status

The canonical tables match all 158 declared analyses: 49 parameter-resampling
runs over 18 landscapes, 60 optimized-fit count runs over 19, and 49 posterior-count runs
over 18. The headline examples, top-cohort agreement statistics, sample counts,
and count-support associations were checked against these tables. Earlier
checks used the companion project's equivalent saved tables, excluding K50.

The new figures were generated from canonical tables, not copied from the
older outline. No models were refitted, no read-depth interventions were run,
and no FLIP2 predictor evaluation was performed. These table-level checks do
not establish model adequacy or interval coverage. The checked-in toy example
and rendering tests are implementation checks, not biological evidence.

The main-text comparisons use named, matched landscape/model examples and
show every model run in summary panels. Larger summary markers take medians
across models within each landscape; runs do not become independent
experimental replications. Candidate universes can differ slightly across
score constructions, which is stated in the methods.

The main selection figure displays the stored batch budgets 10, 20, 50, 100,
200, 500, and 1,000. Smaller cutoffs remain in the original numerical tables;
they are not part of the main-text batch argument. The figure change filters
only its displayed rows and does not recompute overlaps, refit models, or
change the supplementary Figure 10 pages.

The supplementary Figure 10 pages retain the reference top 100 and every
outside candidate with top-100 inclusion frequency at least 0.01. All 158 runs
appear exactly once. The four columns display score width against input count,
output count, and score margin, followed by inclusion against margin. There
are no fitted regression lines. Count and width limits are shared within each
family and scheme; margin limits are shared within a row. Score widths use
logarithmic axes. The 49 exact-zero widths are displayed at half the smallest
positive normalized width in their family/scheme collection, without changing
the numerical tables. Existing regression summaries are separate and unchanged.

The introduction now cites DMS measurement, experimental design,
machine-learning-guided engineering, FLIP, ProteinGym, FLIP2, Enrich2, DiMSum,
and model-based enrichment estimation. Each source is attached to its own
contribution; prior uncertainty estimation is not presented as absent. The
assay descriptions distinguish PDZ binding-dependent yeast growth, TrpB
tryptophan-dependent bacterial growth, and NucB fluorescence sorting.

## Source findings

- The source routine named `mle` optimizes latent parameter densities as well
	as count likelihood terms. The prose and manuscript figures now say
	**optimized-fit count resampling**. The existing `mle_predictive` file and
	manifest identifiers are unchanged. No scores or fits were modified.
- PDZ and TrpB use multinomial count likelihoods and explicit exponential or
	sigmoid growth functions. The supplement distinguishes flat rate priors,
	normal/lognormal replicate hierarchies, and the Gamma inverse-rate model.
	NucB uses normalized threshold-survival probabilities; the actual stored
	parameter score is on the log-fluorescence scale.
- The source forwards `exp_beta`, but the relevant Exponential constructor
	methods accept `beta`. The requested NucB overrides therefore do not replace
	the default rate under the inspected entry point. Both requested settings
	and this source-level behavior are documented; original models were not fixed
	or rerun. An isolated execution of the exact mixin with a recording stub
	confirmed the argument handling.
- All 49 HMC artifacts have zero recorded divergences; six have tree-depth
	saturation. Forty have stored native-score R-hat/ESS summaries and nine do
	not. The supplement reports these limits and chain-energy BFMI estimates
	instead of treating absent diagnostics or a valid file as a pass.
- The stored source flag `low_ebfmi` compares individual energy values with a
	threshold, not a chain-level BFMI estimate. The inventory computes the
	explicitly documented energy-difference/variance ratio directly from traces.

The fitting environment could not import its installed CmdStanPy helper
`scan_config`. No packages or upstream sources were changed to work around
that incompatibility. Model documentation was grounded in source inspection,
saved artifacts, and isolated formula/keyword checks, not a successful new fit.

## Promise ledger

| Abstract or introduction claim | Deliverer | Limit |
| --- | --- | --- |
| Compare reported and fitted orderings | Section 4; fixed-reference comparison table; Fig. 1 and Fig. S1 | Fixed-order differences are distinguished from median-sampled-rank agreement; neither establishes biological ground truth. |
| Quantify rank uncertainty within leading batches | Section 5.1; Fig. 2; `model_rank_intervals` | Median-score reference top 100, not every variant; intervals are sample summaries. |
| Noise can replace batch members, not merely reorder them | Section 5.2; Fig. 3; overlap comparison table | TrpB library B shares 26--51 and NucB G3 87--93 of 100 on average across the specified constructions; these are not physical assay repeats. |
| Batch membership and internal rank resolution are distinct | Sections 5.1--5.2; Figs. 2--3 | NucB's relatively stable top-100 batch still contains uncertain individual ranks. |
| Batch size affects overlap without a universal monotone trend | Section 5.2; supplementary landscape endpoint table | Comparison 100 to 1,000; chance baseline checked. |
| Lower input support accompanies wider score intervals | Section 6; Fig. 4, Fig. S2, and per-analysis Figs. S3--S60; defined count-feature aggregation | Decision-cohort associations are not read-depth interventions. |
| Locate plausible replacements using counts, score margins, and inclusion | Section 6; Figure 4; individual-variant export | Three illustrative non-Gamma analyses, not a causal comparison between assays; local-spacing regressions remain supplementary. |
| Use joint noisy realizations to measure batch resolution | Section 3; Appendix A model and fitting details; all-fit inventory | Resampling is the measurement procedure, not a claimed new method; model validation remains incomplete. |
| Quantify uncertainty in experimental targets used for engineering and ML | Introduction and Discussion; source studies and FLIP2 citation | Direct downstream engineering and predictor effects remain unmeasured. |

## Figure map

Manuscript figure numbers are distinct from the original notebook inventory.

| Draft figure | Content | Relation to notebook figures |
| --- | --- | --- |
| 1 | Three example reported-rank curves and all-landscape top-100 correlations across schemes | Compact adaptation of 1 |
| 2 | Rank intervals within example batches and top-100 width summaries | Compact adaptation of 2 |
| 3 | Top-100 boundary candidates and batch overlap for budgets 10--1,000 | Compact adaptation of 3, without the single-candidate endpoint |
| 4 | Three example rows: input/width, output/width, margin/width, and margin/inclusion | Figure 10 point displays with every cohort candidate retained and no Gamma example |
| S1 | Reported top-1,000 comparisons | Larger-cohort version of draft Fig. 1 |
| S2 | Input/output count versus score-width correlations for all analyzed landscapes | Compact cross-scheme support for draft Fig. 4 |
| S3--S20 | Every parameter-resampling analysis, grouped by family | Figure 10, paginated with no fitted regression lines |
| S21--S42 | Every optimized-fit count-resampling analysis, grouped by family | Figure 10, paginated with no fitted regression lines |
| S43--S60 | Every posterior-count analysis, grouped by family | Figure 10, paginated with no fitted regression lines |

Figure 4 has 419 PDZ, 516 TrpB, and 128 NucB points per column. Each cohort
contains 100 median-score reference members (blue, filled), with 319, 416,
and 28 outside challengers (orange, open), respectively. Input and output
counts occupy separate columns, retaining the same group encoding throughout.
Count and width limits are shared across the three examples; the two margin
panels share limits within each row. Margin ranges differ across rows, so
the apparent steepness of the inclusion transition is not directly comparable.
All displayed widths are positive; no plotting floor affects these examples.

PDZ has a tight count/width relationship over a relatively narrow width
range. TrpB has a much broader range, with some uncertain challengers well
below the median-score cutoff. NucB has wider intervals for many reference
members than for challengers, yet 77 reference members have inclusion at
least 0.99. The contrast shows why count associations, score margins, and
joint-sample selection need to be inspected together. It is illustrative
and model-conditional. Excluding Gamma from this figure does not remove
those runs from the full inventory or supplementary displays.

The main "why?" section and supplementary descriptive-association methods both
reference the complete per-analysis supplement. The shared Figure 10 builder
in [../analysis/figures.py](../analysis/figures.py) also renders Figure 4,
using explicit cross-family run selection. Its family-specific behavior and
all supplementary analyses are retained. Notebook source cells and
their saved outputs were not modified or rerun; a future notebook execution
will use the revised builder. Figures 5--8 from the notebook inventory are
not silently reinterpreted as different analyses in this draft.

## Style comparison

The applied PM2.5 evaluation in the supplied
[writing exemplars](../.github/skills/paper-writing-renato/references/exemplars.md#L1)
is the closest structural example: scientific workflow, concrete decisions,
evaluation, and an explicit verdict. The body uses definition-first methods
and result-first analysis sections; contributions remain prose.

| Exemplar excerpt | Corresponding draft excerpt |
| --- | --- |
| "We focus our evaluation on individual decisions" | "an engineer may select a hundred variants for validation, while a machine-learning researcher asks whether a predictor identifies the same candidates" |

## Outstanding items

- Supply fit-specific observed-versus-predictive checks and explain the
	strongly specification-sensitive PDZ results.
- Recover or compute the nine missing native-score diagnostic summaries,
	rank/inclusion Monte Carlo precision, and the complete optimization loss and
	restart records. Review the six fits with tree-depth saturation.
- Confirm historical effective hyperparameters and environments beyond the
	command/metadata matches, including the `exp_beta` argument discrepancy.
- Complete source-study library/site descriptions, exact DiMSum configuration,
	and the per-task FLIP2 correspondence. The remaining physical-assay details
	are the PDZ construct distinctions and TrpB three-site position lists, plus
	NucB's fluorescent substrate, compartment protocol, and incubation conditions.
- Add versioned fitting/analysis code and a public prepared-data archive.
- Confirm authorship, affiliations, contributions, funding, and competing interests.
- Check bioRxiv's current submission requirements when preparing the submission.
- Before making an exclusive novelty claim, check the closest work on joint
	fitness uncertainty, rank uncertainty, and candidate-set stability. The
	current introduction distinguishes the contribution without asserting an
	exhaustively verified absence of prior work.

FLIP2 subset/predictor evaluation, controlled read-depth changes, and
whole-landscape uncertainty are outside this initial draft. No theory or
frequentist-coverage guarantee is claimed.

## Checks

```bash
uv run pytest tests/test_figures.py tests/test_fit_inventory.py
uv run ruff check analysis/figures.py analysis/manuscript.py analysis/fit_inventory.py tests/test_figures.py tests/test_fit_inventory.py
```

Focused tests check that Figure 10 never calls the trend fitter, preserves
its plotted data, and exports every run exactly once. The batch-focused
Figure 3 test checks its displayed budgets, agreement values, and unchanged
source tables.
Figure 4 tests check every plotted coordinate, point counts, no binning or
regression, Gamma rejection, zero handling, selected-run axis limits, and
the individual-variant and run-index exports. Unselected fits cannot alter
the main figure's axis ranges.
Presentation tests check font normalization, fixed figure width, short labels,
and centering in both table generators. The current compiled PDF was also
checked directly: all 64 figures retain their exported text sizes and are
centered, and all nine detected table-rule groups are centered on the page.
The exporter additionally checks nonblank rendered figures and records source
checksums. The current PDF builds without unresolved references, duplicate
links, or overfull boxes. All 58 supplementary pages were checked for rendered
point images and text within page bounds; representative PDZ and NucB pages
were visually inspected in the compiled PDF, alongside a TrpB figure export.
These checks concern reproducibility and presentation, not biological validity.