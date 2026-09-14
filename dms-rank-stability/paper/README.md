# Paper V2

[main.pdf](main.pdf) is the current illustrated draft, revised 2026-09-10
for a Cell Systems or Nature Communications readership. The narrative
slightly prioritizes machine-learning readers unfamiliar with experimental
protein fitness while retaining batch selection as the shared engineering
and prediction task. This is a working draft, not a submission-ready paper.
The main text now includes a complete mean absolute rank displacement
table alongside the unchanged figures. The compact curve supplement is
unchanged; the new Results discussion adds displacement to the existing
area, correlation, rank-interval, and overlap comparisons.

The structure is **Introduction, Results, Discussion, References, Materials
and Methods, and Supplementary**, with declarations before References.
Materials and Methods begins after the bibliography, not before it.
The introduction merges the previous background with
the motivation: protein function and engineering, sequence search and ML,
assay-defined fitness, pooled measurements and counts, then ranking and
batch uncertainty. It now develops the distinction between experimental
label uncertainty and predictor uncertainty, and between estimating a
fitness score and selecting a batch. Results explain each summary before
using it, then connect the displayed patterns to those decisions.
There is no separate setup or technical methods section before Results.

The expanded Results interpret the percentile curves and overview points,
individual rank intervals and cutoff crossings, all four columns of the
main overlap table, batch-budget responses, and the candidate-level count,
margin, and inclusion displays. The displacement table additionally compares
how far reported and median sampled ranks differ. New numerical statements are summaries of
the existing saved results, not new experiments. The writing review and
the comparison with Bruce's available author manuscript are recorded in
[review.md](review.md).

## Sources and Archive

- [main.tex](main.tex): abstract, merged introduction, results, and discussion.
- [materials_methods.tex](materials_methods.tex): post-References methods,
  sampling recipes, cohorts, summaries, and running-example policy.
- [supplementary_details.tex](supplementary_details.tex): assay explanations,
  full count models, fitting settings, and diagnostics.
- [supplementary_analyses.tex](supplementary_analyses.tex): score calculations,
  computational conventions, descriptive regressions, sensitivity analyses,
  and complete supplementary figure collections.
- [references.tex](references.tex): thirteen references, including Wittmann,
  Yue, and Arnold (2021) for ML-assisted engineering and training-set context.
- [review.md](review.md): current promise-to-evidence ledger, definition-order
  review, example-selection tradeoffs, and paper-specific journal adaptation.

The independently buildable v1 is frozen in
[old_paper_ml_structure/README.md](old_paper_ml_structure/README.md), including
[its PDF](old_paper_ml_structure/main.pdf), TeX dependencies, and 64 figure
PDFs. The archive checksum manifest covers 90 files. V2 exports never write
there. The original source-byte manifest and the archive-local build manifest
are distinct because the archived TeX paths were made self-contained.

To verify the frozen archive from the repository root:

```bash
pushd paper/old_paper_ml_structure
sha256sum --quiet --check ARCHIVE_SHA256SUMS
popd
```

The archive was already compiled twice. For another build, send TeX output
outside the archive so that PDF timestamps and build products do not alter
its frozen checksums:

```bash
mkdir -p /tmp/dms-paper-v1-build
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/dms-paper-v1-build paper/old_paper_ml_structure/main.tex
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/dms-paper-v1-build paper/old_paper_ml_structure/main.tex
```

## Running Examples

All four main figures and the overlap table use the same identities:

| System | Landscape | Specification | Figure 4 candidates |
| --- | --- | --- | ---: |
| PDZ | trans-1 | Exponential--Exponential | 419 |
| TrpB | libB | Exponential--Exponential | 516 |
| NucB | G3 | Exponential | 128 |

These are explicit illustrative choices in
[../analysis/manuscript.py](../analysis/manuscript.py), not nearest-median
selections. PDZ trans-1 shows many plausible replacements across a narrow
score-width range; TrpB libB shows low reported-rank agreement and substantial
batch turnover; NucB G3 shows uncertain internal ranks but a comparatively
stable top-100 group. The same identities feed every main figure and table.

[../analysis/examples.py](../analysis/examples.py) retains the centrality
audit for context. It compares area, Spearman correlation, reference-top-100
median rank width, and mean shared fraction with equally weighted non-Gamma
landscape medians within each system and sampling recipe. None of the
selected values lies outside the 1.5-IQR fences of those landscape summaries.
This descriptive check does not make every case typical: TrpB libB is near
the low-agreement end, with similar batch instability in libraries A--C.
The per-run values, positions, shortlist, and explicit selection reasons are in
[figures/example_selection_runs.csv](figures/example_selection_runs.csv) and
[figures/example_selection_candidates.csv](figures/example_selection_candidates.csv).

All retained posterior fits underwent sampler diagnostics during fitting.
Some retained outputs do not include the numerical summaries; that is not
an unchecked-fit category and does not exclude a running example. Stored
values remain unchanged, including explicit recorded warnings, and absent
values are not imputed. Gamma remains excluded from the running examples
but is retained in all complete results.

## Figures and Evidence

In the overview panels of Figures 1, 2, S1, and S2, a small filled dot
represents one model specification within a landscape and sampling scheme.
It is one summary of that analysis, not a posterior draw, chain, candidate,
or experimental replicate. Larger hollow circles, squares, and triangles
mark medians across model specifications for parameter, optimized-fit count,
and posterior count resampling, respectively. Small deterministic vertical
offsets distinguish nearby model points without changing their horizontal
metric values. Black example outlines and arrows follow the model points,
including their display offsets, rather than the median across models.
With one model, as for NucB G3, the filled dot lies inside its hollow median.

The main-text table in
[figures/rank_displacement.tex](figures/rank_displacement.tex) has 60
landscape/model rows and three columns: parameter posterior, optimized-fit
counts, and posterior counts. Each value is the mean absolute difference
between reported rank and median sampled rank among the supported reported
top 100. Ranks use the full competitor pool; no reranking within the cohort
or averaging across model specifications is performed. Missing analyses
are dashes, not zero displacements. The table has repeated headers when
it continues across pages.

[figures/rank_displacement.csv](figures/rank_displacement.csv) retains
all 158 run-level values at full precision, with run IDs and supported
candidate counts (99 or 100). The calculation takes the median sampled
rank first, then the absolute difference, then the mean over candidates.
It is not an average absolute difference over individual samples.
Displacement measures central disagreement with reported ranks, not rank
uncertainty or error against known biological truth. The table and new
Results paragraphs complement rather than replace Figure 1's metrics.

| Figures | Contents |
| --- | --- |
| 1 | Three example percentile curves; all-run area and Spearman overview rows; exact-fit outlines and arrows |
| 2 | Example sampled-rank intervals and all-run reference-top-100 width summaries; exact parameter-fit highlights |
| 3 | Candidates near the top-100 cutoff and batch overlaps for budgets 10--1,000 |
| 4 | All 1,063 reference members and challengers in the three parameter examples, without bins or point subsampling |
| S1--S2 | Reported-top-1,000 overview and all-run count/width correlations |
| S3--S11 | All 316 curves overlaid in nine system/scheme figures, with top 100 and top 1,000 in separate panels |
| S12--S69 | All 158 analyses' candidate-level count, width, margin, and inclusion displays on 58 pages |

The new curve supplement uses nine figures instead of 58. Each figure
overlays the individual models and landscapes for one protein system and
sampling scheme. Color identifies the landscape, line style identifies the
model, and a thicker line marks the running example. Curves are not averaged
or pooled, and no percentile values are dropped. Similar curves can overlap;
their numeric differences remain available in the complete statistics table.

[figures/reported_curve_statistics.tex](figures/reported_curve_statistics.tex)
provides the four-page table, with one row per analysis and both cohorts
side by side. It includes area, Spearman correlation, retained samples,
and supported candidate counts. IDs distinguish parameter, optimized-fit
count, and posterior count analyses. Display rounding does not affect
[figures/reported_curve_statistics.csv](figures/reported_curve_statistics.csv),
which retains full-precision statistics and complete run identities.

The run/cohort-to-figure index is
[figures/reported_curves_index.csv](figures/reported_curves_index.csv);
[figures/reported_curves.tex](figures/reported_curves.tex) provides captions
and a scheme/system index. Every curve has matching statistics in the table.
The exporter checks the exact step-ECDF integral against the stored area.
Area is a descriptive departure from a uniform reference, not rank error,
ROC-AUC, or a proven calibration target. Smaller area need not mean better
rank agreement; the curve, correlation, rank intervals, and batch overlap
retain different information.

The existing point-level collection is indexed by
[figures/summary_why_index.csv](figures/summary_why_index.csv) and assembled
by [figures/summary_why.tex](figures/summary_why.tex). It retains Gamma fits
alongside every other run. Count and width limits
are common within system/recipe collections; margins share limits within
a row. Plot-only floors for zero widths are indexed without altering data.

[figures/explanation_variants.csv](figures/explanation_variants.csv) retains
all Figure 4 candidates, source values, and plotted coordinates.
[figures/explanation_runs.csv](figures/explanation_runs.csv) records cohort
counts and run identities. Fixed-list intersections, sample-pair overlap,
and overlap with each fixed reference are distinguished in
[figures/reference_comparisons.csv](figures/reference_comparisons.csv).

All 73 figure PDFs used by the current manuscript have 6.5-inch canvases
and full-width TeX placement.
The common typeface is DejaVu Sans: 8-point labels, ticks, legends, and
annotations; 9-point headings and panel letters. Mathematical superscripts
follow normal font scaling. Figures and tables are centered. Short labels
such as `libB`, `G3`, and `trans-1` are retained.

## Build and Checks

All files in the active figures directory are versioned, including PDFs,
PNGs, generated tables, and metadata. A fresh checkout can compile the
manuscript without the analysis data or Python environment. From the
repository root, with pdfLaTeX and the manuscript's TeX packages installed:

```bash
pdflatex -interaction=batchmode -halt-on-error -output-directory=paper paper/main.tex
pdflatex -interaction=batchmode -halt-on-error -output-directory=paper paper/main.tex
```

To regenerate the figures and tables from the canonical analysis data,
use the project environment separately:

```bash
.venv/bin/python -m analysis.manuscript --table-root /data/data_dms_rank_stability/tables
```

The figure exporter requires the included
[generated/fit_inventory.csv](generated/fit_inventory.csv) for illustrative
selection and checks all run identities against
[../analysis/runs.json](../analysis/runs.json). It reads nine prepared tables
per sampling recipe, not full sampled-score matrices or raw fits. Outputs
go to the active figure directory; source hashes and page specifications
are recorded in [figures/provenance.json](figures/provenance.json). Canonical
tables are not modified. Figure exports are tracked by Git; the compiled
manuscript PDF and TeX cache files remain ignored.

The existing fitting snapshot can be regenerated separately:

```bash
.venv/bin/python -m analysis.fit_inventory \
  --source-root ../dms-fitness-stability \
  --scistanpy-root ../prob-protein-fitness/SciStanPy \
  --data-root /data/data_dms_rank_stability
```

That command verifies SciStanPy revision
`09c439f675ead70f851d814242f153ecb7712113`, reads command and artifact
metadata, and preserves unrelated local loader changes. The snapshot
includes [generated/analysis_fit_mapping.csv](generated/analysis_fit_mapping.csv),
[generated/fit_inventory.tex](generated/fit_inventory.tex), and
[generated/fit_provenance.json](generated/fit_provenance.json).

The manuscript test gate is:

```bash
.venv/bin/python -m pytest tests/test_displacement.py tests/test_figures.py tests/test_tables.py tests/test_examples.py tests/test_fit_inventory.py tests/test_runs.py -q
.venv/bin/ruff check analysis/manuscript.py analysis/figures.py analysis/examples.py analysis/fit_inventory.py tests/test_displacement.py tests/test_figures.py tests/test_tables.py tests/test_examples.py tests/test_fit_inventory.py
```

All 54 tests pass. Checks cover absolute displacement without signed
cancellation or cohort reranking, finite support counts, cross-scheme
alignment, missing analyses, exact area semantics, centrality audits,
explicit examples with omitted diagnostic summaries, exact highlighted fits
rather than landscape medians, full
curve/cohort coverage, batch budgets, and preservation of every Figure 4
point. The compact-export tests also verify every overlaid step, group
coverage, repeated table headers, and CSV statistic values. The real-data
audit confirms 109 fits, 158 runs, 316 matching curve
summaries, unchanged hashes for all 27 source tables, and the unchanged
90-file v1 archive. The PDF builds without unresolved references or layout
warnings. These are implementation and presentation checks, not biological
model validation.

The overview visibility test checks coincident metric values, stable
vertical offsets under shuffled input, hollow medians, singleton models,
and exact example-arrow locations. The real-data check verifies every
plotted model value and all example highlights in the four affected figures.
Only their PDF/PNG pairs were regenerated; all other figure-folder files,
including the displacement table and candidate-level plots, are unchanged.

The preceding revision restored the contrasting examples and corrected the
interpretation of omitted diagnostic output. Regression tests ensure that
missing stored summaries do not exclude an already-checked fit, and that
the printed fit inventory does not classify fits as diagnosed versus raw.
The percentile, rank-interval, boundary-inclusion, and batch-overlap
statements were checked against saved tables. The PDF was built twice and
inspected for section order and page bounds. The percentile tie convention
has not changed: exact rank matches still receive half weight.

For the displacement addition, all 158 means were independently checked
against the stored per-candidate rank differences. The table was generated
without regenerating any figures; existing PDF and PNG bytes are unchanged.
The added numerical comparisons were checked against saved tables, and
the main-text table was inspected for pagination, centering, and page bounds.

## Evidence Limits

No inference, read-depth intervention, or predictor evaluation was
added in v2. All original numerical results and 158 analyses are retained.
The source routine called `mle` optimizes prior terms as well as count
likelihoods, so the paper says **optimized-fit count resampling**.
The documented `exp_beta`/`beta` discrepancy, six fits with tree-depth
saturation, and the
incorrect upstream `low_ebfmi` flag remain disclosed in the model supplement.
The printed inventory reports numerical values present in the files;
an omitted value is not a missing diagnostic check. The diagnosis of all
retained fits was confirmed by the author. Sampler checks remain distinct
from predictive checks of the experimental model.

FLIP2 is a brief contextual citation for the reuse of these landscapes,
not the subject of a benchmark analysis or an additional provenance task.

Outstanding work includes predictive checks; unresolved historical fit
settings and optimization records; PDZ construct distinctions and
TrpB three-site position lists; NucB substrate, compartment, and incubation
details; exact DiMSum configuration; and public data/code links and author
metadata. Changed batch identities do not establish lost engineering
activity or changed predictor rankings. Whole-landscape uncertainty and
an optimal batch-selection policy are not promised by this draft.