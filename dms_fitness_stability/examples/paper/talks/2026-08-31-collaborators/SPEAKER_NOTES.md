# Speaker notes

Target: 19.5 minutes for slides 1-12, leaving about 10 minutes for questions.
Slides 13-15 are backup.

## 1. Title (0:30; cumulative 0:30)

Opening:

> A DMS paper usually gives us a table of sequences and fitness scores. We want
> to know how much of the ordering in that table is actually supported by the
> underlying counts.

State that the talk focuses on the scientific question and current results, not
the implementation history.

## 2. A DMS measures counts (1:30; cumulative 2:00)

- Walk left to right through the experiment.
- Emphasize that counts are observed; normalization, filtering, aggregation,
  score, and rank are constructed.
- Do not imply that analysis choices are arbitrary. They are necessary, but
  they should remain visible when the result is interpreted.

Transition: "This matters even when two reported scores look equally precise."

## 3. Same point estimate, different support (1:30; cumulative 3:30)

- Use the 1-to-10 versus 1,000-to-10,000 example.
- Separate the two questions explicitly:
  1. Would another reasonable score definition change the ordering?
  2. After fixing the definition, how much uncertainty remains from finite
     counts?

Main sentence: "The project is about ordering information, not a search for an
assay-independent true fitness."

## 4. Why ranking resolution matters (1:30; cumulative 5:00)

- Protein engineering: candidate selection and mutation combination.
- ML evaluation: Spearman and top-K metrics inherit uncertainty from labels.
- Use one restrained example: a two-point top-K improvement may be difficult to
  interpret if the experimental top-K set changes much more than two points.
- Do not claim a universal benchmark significance threshold; that is future
  work.

## 5. What we quantify (2:00; cumulative 7:00)

- HMC posterior: uncertainty in latent assay scores under the count model.
- MLE predictive: repeated count experiments conditional on fixed optimized
  parameters, followed by the source score calculation.
- Both produce sampled score landscapes, which induce sampled rankings.
- Say clearly: "MLE predictive spread is not MLE estimator uncertainty and is
  not a bootstrap."

Transition: "We apply this common ranking analysis to three quite different
experimental systems."

## 6. Experimental systems (2:00; cumulative 9:00)

- Introduce PDZ, TrpB, and NucB by assay, not by implementation details.
- Explain that a run is one landscape, model specification, and fit artifact;
  several runs can share the same experimental counts.
- Mention the three validation facts at the bottom.
- Retain the limitation: PDZ MLE uses a provisional DiMSum-style proxy.

Optional sentence for Bruce and Eric: "The new part since our last detailed
discussion is complete MLE-predictive coverage of all non-K50 PDZ, TrpB, and
NucB fits."

## 7. Published ordering is not always recovered (1:30; cumulative 10:30)

- Each point is an analysis run; the bar is the family median.
- Correlation near one means concordant ordering; zero means little monotone
  relation; slightly negative means reversal.
- The point is heterogeneity, not that one family or procedure "wins."
- End by separating model/published disagreement from within-analysis
  uncertainty.

## 8. Concrete rank intervals (1:30; cumulative 12:00)

- This is one PDZ cis HMC analysis and ten variants selected by median fitted
  score.
- Blue points are median sampled ranks; lines are central 90% rank intervals;
  green diamonds are published ranks.
- Point to rows 3-10: several intervals overlap and extend beyond rank 10.
- Avoid saying that the posterior median rank must equal the reference rank;
  ranking and taking medians do not commute.

Main sentence: "A promising region can be well identified while its exact
internal order is not."

## 9. Top-100 resolution (2:00; cumulative 14:00)

- Left: median interval width among each run's reference top 100.
- Right: fraction whose 95th-percentile sampled rank falls below the top-100
  cutoff.
- Read only the median exit fractions in the callout; do not enumerate every
  point.
- Stress the wide run-to-run variation within each family.
- Do not compare HMC and MLE-predictive values as estimates of one variance.

## 10. Top-K decision stability (2:00; cumulative 16:00)

- Define the y-axis verbally: expected fraction shared by two sampled top-K
  sets.
- Explain why stability generally rises with K: larger sets are less sensitive
  to local swaps near the boundary.
- Highlight that PDZ HMC is substantially less stable than its MLE-predictive
  counterpart in these summaries, while NucB becomes highly stable for larger
  K. Treat this as a procedure- and system-specific result.
- The scientific decision should determine K.

## 11. What is associated with resolution? (2:00; cumulative 18:00)

- Left: within-run input-support deciles versus normalized score width.
- Right: within-run score-uncertainty deciles versus rank width.
- Solid is HMC; dashed is MLE predictive; color is family.
- State the qualitative pattern, not every curve.
- Mention local crowding verbally: the same score width causes more rank
  movement when many neighboring variants have similar central scores.
- Keep the caveat: these are descriptive associations, not downsampling
  experiments.

## 12. Conclusions and discussion (1:30; cumulative 19:30)

Read the recommendation, not every bullet:

> Report rank intervals, top-K inclusion probabilities, and the assay and
> analysis definition with point scores.

Then invite discussion with the two questions at the bottom. Pause here rather
than moving automatically into backup.

## Backup slides

### 13. Estimands

Use if someone asks exactly what the three overlap curves mean. Emphasize that
sample-to-sample reproducibility and published-set agreement answer different
questions.

### 14. HMC versus MLE predictive

Use if someone asks why the procedures differ. The central answer is that they
sample different objects; agreement in qualitative conclusions is useful, but
their interval magnitudes are not estimates of one common variance.

### 15. Validation and limitations

Use for questions about data provenance, source-score reproduction, PDZ's
provisional status, or why the mechanism analysis is not causal.

## Likely questions

**Does uncertainty mean the published landscape is wrong?**

No. The published table is one defensible analysis output. We distinguish
disagreement between analyses from uncertainty conditional on one analysis.

**Why not just compare score error bars?**

Downstream decisions depend on ordering. A score interval has different rank
consequences in a sparse versus crowded score neighborhood.

**Why do HMC and MLE predictive results differ?**

They sample different objects. HMC propagates fitted-score parameter
uncertainty; MLE predictive samples repeated counts at fixed parameters and
recomputes a count-derived score.

**Can this tell us how many additional reads are needed?**

Not causally. That requires controlled read downsampling or prospective depth
changes followed by refitting. The current count-support associations motivate
that experiment but do not replace it.

**How should K be chosen?**

From the downstream decision: experimental capacity, desired diversity, and
cost of false inclusion/exclusion. Showing a curve over K avoids treating one
arbitrary cutoff as universal.

**Are multiple model runs independent experiments?**

No. Runs fitted to one landscape share the same counts. Figures aggregate
models within landscape before comparing families where appropriate.
