# Collaborator talk notes

Target: 18-20 minutes for slides 1-12, leaving about 10 minutes for questions.
Slides 13-15 are backup.

## 1. Title (0:30)

- The project asks how much ordering information a DMS actually contains.
- The emphasis is the downstream ordering, not a new universal definition of fitness.

## 2. A DMS measures counts (1:30)

- Start from the experiment for collaborators who have not followed the project.
- The direct observations are finite counts before and after selection.
- Normalization, filtering, and replicate aggregation turn those counts into scores and ranks.
- Therefore the published table is useful, but it is not a direct measurement with infinite precision.

Transition: there are two reasons not to read the table as an exact ground-truth ranking.

## 3. Same estimate, different support (1:30)

- Use the ratio example. Both variants have enrichment 10, but one is much less supported.
- Separate definition dependence from uncertainty after fixing a definition.
- State the main question verbatim from the callout.

## 4. Why ranking resolution matters (1:30)

- Protein engineering consumes candidate lists, not isolated score values.
- ML evaluation also consumes order through Spearman, enrichment, and top-K retrieval.
- If the labels do not resolve a difference at the relevant scale, the benchmark cannot establish that difference.
- The intended output is a probabilistic candidate set and a resolution scale.

## 5. What we quantify (2:00)

- HMC and MLE predictive analyses must not be described as two estimates of the same variance.
- HMC samples latent score landscapes and includes fitted-parameter uncertainty.
- MLE predictive fixes optimized parameters, samples repeated counts, and recomputes the source score.
- Both end at the same decision objects: sampled ranks, rank intervals, inclusion probabilities, and set overlap.
- This is why the approach is useful: uncertainty is propagated to the object people act on.

## 6. Experimental systems (1:45)

- Three assay families provide different sizes and observation structures.
- The HMC analysis has 49 runs across 18 landscapes; MLE predictive adds the TrpB four-site landscape, giving 60 runs across 19 landscapes.
- Briefly name the checks. Do not dwell on infrastructure unless asked.
- State the PDZ MLE limitation directly: it is a provisional DiMSum-style proxy.

Transition: with the scope and checks established, move from agreement with the publication to internal resolution.

## 7. Published ordering (1:30)

- Each point is one analysis run; horizontal bars are medians within each procedure and family.
- Agreement is usually positive but highly variable, including near-zero or slightly negative cases.
- This slide is a disagreement diagnostic. It does not by itself measure uncertainty under the fixed analysis.

## 8. Concrete rank example (1:30)

- Rows are variants ordered by the fixed median-score reference.
- Blue points and intervals summarize sampled full-landscape ranks; green diamonds are published ranks.
- The top variants occupy a promising region, but their intervals overlap and several cross the top-ten boundary.
- Avoid saying the median-score rank is the median sampled rank; ranking and taking medians do not commute.

## 9. Top-100 resolution across runs (1:45)

- Left: run median width of the 90% rank intervals among the model top 100.
- Right: fraction of those 100 variants whose upper interval endpoint is below the decision boundary.
- Read the six medians from the callout rather than listing individual outliers.
- Main message: finite resolution is common but heterogeneous; the result is not one universal noise level.

## 10. Stability depends on K (1:45)

- The metric is the expected overlap between two independently sampled top-K sets.
- Curves are landscape-weighted medians; shaded areas are landscape interquartile ranges.
- PDZ, TrpB, and NucB behave differently, and changing K changes the scientific answer.
- HMC and MLE predictive have different targets, but neither supports treating all exact orderings as resolved.

## 11. What is associated with resolution? (1:45)

- Left: greater input support is generally associated with narrower score intervals.
- Right: wider score intervals tend to produce wider rank intervals.
- Add verbally that local score crowding changes the conversion from score uncertainty to rank uncertainty.
- Keep the causal limitation explicit: these are descriptive associations; read downsampling and refitting would be needed for a causal depth experiment.

## 12. Conclusions and discussion (1:15)

- Read the four supported statements as the project summary.
- Emphasize the recommendation: publish decision-level uncertainty with point scores.
- Invite discussion around the decision-specific K and how benchmark comparisons should account for label resolution.

## Likely questions

### Are HMC and MLE predictive directly comparable?

No. HMC includes uncertainty in latent fitted scores under the Bayesian model. MLE predictive measures repeated-count variability conditional on fixed optimized parameters. Similar qualitative conclusions are informative, but numerical intervals are not estimates of one common variance.

### Does this show that the published rankings are wrong?

No. It separates disagreement between analyses from uncertainty within a fixed analysis. Fitness remains assay- and analysis-defined.

### Is the reported-rank percentile curve a calibration test?

No. It is a descriptive location diagnostic. The direct Spearman summary is included because the ECDF-to-uniform area is not a monotone rank-error measure.

### Can we conclude that deeper sequencing would improve resolution by a known amount?

No. Input support is associated with score precision, but the present analysis is observational. A causal estimate requires controlled count downsampling or additional sequencing followed by refitting.

### Why is PDZ marked provisional for MLE predictive analysis?

The count-derived proxy closely follows the published ordering but the exact original DiMSum command and configuration have not been recovered. It is therefore labeled DiMSum-style rather than paper-faithful.

### What should a benchmark report?

At minimum: the assay and score definition, rank intervals or pairwise ordering uncertainty, top-K inclusion probabilities at decision-relevant K, and sensitivity to reasonable analysis choices.
