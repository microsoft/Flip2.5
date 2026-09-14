# Paper concerns and proposed checks

This note records interpretation and figure-design issues found while comparing the August 2026 paper outline with the current PDZ and CRIPT analyses. It is a planning document, not a conclusion that the current results are invalid.

## 1. Highest-priority concern: the reported-rank percentile ECDF

### What is currently calculated

For variant $i$, let $r_i^{\mathrm{rep}}$ be its published rank and let $R_i$ be its posterior rank. Rank 1 is best. The implementation calculates the midpoint percentile

$$
u_i = P(R_i < r_i^{\mathrm{rep}} \mid \mathcal D)
+ \frac{1}{2}P(R_i = r_i^{\mathrm{rep}} \mid \mathcal D).
$$

The quantity is called `reported_rank_percentile` in the saved tables. Its direction is:

- $u_i$ near 0: posterior ranks are usually numerically larger, and therefore worse, than the published rank.
- $u_i$ near 1: posterior ranks are usually better than the published rank.
- $u_i$ near 0.5: the published rank is near the center of the marginal posterior rank distribution.

The plots form the empirical CDF $F$ of the $u_i$ values and calculate

$$
A = \int_0^1 |F(t)-t|\,dt.
$$

The diagonal is the CDF of a Uniform$(0,1)$ distribution.

### Why exact rank recovery does not give area zero

Suppose every posterior draw exactly reproduces every published rank:

$$
R_i = r_i^{\mathrm{rep}}
$$

for all variants and all draws. The midpoint definition then gives $u_i=0.5$ for every variant. The ECDF is a step at 0.5, not the uniform diagonal. Its area is

$$
\int_0^{0.5} t\,dt + \int_{0.5}^{1}(1-t)\,dt = 0.25.
$$

This is not a numerical defect in the implementation. It follows from comparing a point mass at 0.5 with a uniform distribution.

Conversely, $A\approx0$ means that the published ranks occupy all parts of their marginal posterior rank distributions in approximately equal proportions. It does not mean that posterior ranks are close to published ranks. The posterior distributions could be very wide while their published ranks happen to fall at a roughly uniform set of posterior quantiles.

Some useful toy cases are:

| Scenario | Distribution of $u_i$ | Approximate $A$ | What it shows |
|---|---:|---:|---|
| Every posterior rank exactly equals the published rank | all 0.5 | 0.25 | Perfect recovery does not minimize $A$ |
| Published ranks occupy evenly spaced posterior percentiles | approximately uniform | 0 | Small $A$ does not imply small rank error |
| Posterior ranks are almost always worse than published | all 0.05 | 0.4525 | $A$ detects a strong one-sided shift |
| Half the variants are at percentile 0 and half at percentile 1 | half 0, half 1 | 0.25 | Strong symmetric disagreement can equal perfect recovery |

The scalar area is therefore a distance from a uniform percentile distribution. It is not a distance between the published ordering and the posterior ordering, and it is not monotone in rank agreement.

### When a uniform percentile reference would be justified

A probability integral transform is uniform when the observed quantity is generated from the predictive distribution used to evaluate it, subject to the usual discrete-data adjustment. That argument does not apply automatically here:

1. The published rank is a deterministic estimate computed from the same counts on which the posterior is conditioned.
2. It is not an independent draw from the posterior rank distribution.
3. Published fitness and posterior latent fitness are produced by different analysis procedures.
4. Ranks across variants are dependent because every posterior draw is a joint ordering, not a collection of independent rank observations.

Uniformity could still be a useful reference under a carefully defined repeated-experiment procedure, but that procedure must be derived or simulated. It should not be assumed from the ordinary probability integral transform result.

### Why selection by published rank matters

The current `tracked` cohort is the reported top 1,000. The smaller panel is the reported top 50. Both cohorts are selected using the same published ranks that enter $u_i$.

This creates a winner's-curse effect. A variant can enter the published top 50 partly because its published estimate had favorable sampling noise. A posterior model that shrinks that estimate can assign it a worse rank even when the model and published estimator target the same underlying quantity. The selected cohort can therefore have many small $u_i$ values under a reasonable null model.

The amount of selection also differs across landscapes. The reported top 1,000 is about 11% of the cis landscape but less than 1% of each trans landscape. An ECDF-area comparison across landscapes can consequently reflect different selection severity as well as different model behavior.

The correct null for a selected cohort would need to reproduce the complete sequence:

1. generate counts;
2. compute the published estimator and ranks;
3. select the reported top $k$;
4. fit or otherwise construct the posterior rank distribution;
5. calculate the selected-cohort percentile ECDF and area.

### What the current curves can still show

The curves remain useful descriptively:

- An ECDF concentrated above the diagonal contains many small $u_i$ values, meaning that the posterior tends to move selected variants to worse ranks.
- An ECDF below the diagonal contains many large $u_i$ values, meaning that the posterior tends to improve their ranks.
- The curve shape can reveal mixtures or strong directional shifts.
- Comparing `log_mean_growth` with posterior-predictive `count_ratio` exposes score-definition sensitivity. CRIPT-N is the clearest example.

The safe current names are **reported-rank posterior-percentile ECDF** and **ECDF-to-uniform area**. Until a null is established, `rank-agreement divergence` and `calibration divergence` overstate what the scalar area measures.

### Proposed decision process for Section 1

1. **Keep the curves as exploratory diagnostics now.** Do not interpret zero area as exact agreement or rank analyses by the area as though smaller always means better recovery.
2. **Define the target of Section 1.** Decide whether it concerns center disagreement between two estimators, posterior uncertainty in ordering, or reproducibility under a repeated sequencing experiment. These are different targets.
3. **Test the selected-cohort reference by simulation.** A full check would simulate counts, rerun the published ranking procedure, apply the same top-$k$ selection, refit the posterior model, and calculate the statistic. This is computationally expensive but directly answers whether the diagonal and area have the claimed meaning.
4. **Use null envelopes if the statistic is retained.** Compare observed curves and areas with simulation-derived distributions specific to landscape size and cohort selection.
5. **Otherwise use a direct agreement summary.** Candidate summaries include posterior-median rank minus published rank, expected absolute rank displacement, and posterior distributions of Kendall or Spearman agreement. No one summary is sufficient by itself, but each has a direct ordering interpretation.
6. **Keep Section 2 separate.** Posterior interval widths and the signed 5th/95th percentile deviations already provide the variant-level uncertainty analysis.

A practical near-term recommendation is to keep the percentile ECDF in the notebook, treat it as descriptive, and avoid making it Figure 1's central scalar until Step 3 has been completed.

## 2. Latent rank uncertainty and repeated-experiment variability are different

The notebook currently shows two scores:

- `log_mean_growth` is a posterior draw of the latent assay-level growth quantity.
- `count_ratio` is calculated from posterior-predictive input and output counts. It includes the sampling variation of a replicated sequencing experiment.

The paper currently defines $R_i$ from the posterior over the latent fitness $\phi_i$. That definition directly matches `log_mean_growth`, but not posterior-predictive `count_ratio`.

This distinction matters. A posterior-predictive ranking can be unstable even when the latent ordering is well resolved, because it includes new sequencing noise. Conversely, the CRIPT-N predictive count ratio can resemble the reported ordering while its latent Gamma-Sigmoid growth ranking does not.

Plan:

1. Use the latent score as the primary Sections 1-3 estimand if the paper continues to define ranks through $\phi_i$.
2. Present posterior-predictive count-ratio results as repeated-experiment stability, posterior predictive evidence, or a sensitivity analysis.
3. Do not call the saved posterior-predictive count ratio the observed naive count ratio. The observed pooled ratio is a separate column in `variants.csv`.
4. Decide later whether the saved DiMSum-style posterior-predictive score belongs in the main sensitivity analysis.

## 3. `Tracked` is not the whole landscape

The compact results retain full-candidate posterior ranks for the reported top 1,000 variants. They do not retain posterior rank distributions for every published candidate. The current Section 1 and Section 2 plots are therefore top-cohort analyses even when a title says `tracked`.

This affects statements such as "whole-landscape agreement" and the proposed comparison between top-$k$ uncertainty and full-landscape uncertainty.

Plan:

1. Replace `whole landscape` with `reported top-1,000 cohort` wherever that is what was analyzed.
2. Decide whether a true all-variant analysis is scientifically necessary. The top-focused question may be more relevant and much cheaper.
3. If required, design specialized postprocessing of the existing HMC files. The exact files still exist, so HMC need not be rerun, but retaining draw-level ranks for more than 100,000 variants would require a memory-aware summary rather than simply setting `tracked_top_n` to the full landscape size.
4. State that the ranking universe contains variants with published fitness that were aligned to the fitted count table. It is not every fitted HMC column.

## 4. Section 2 boxplots align with the outline, with presentation caveats

The posterior-rank divergence boxplots implement the proposed Figure 2(b):

$$
Q_{0.05}(R_i)-r_i^{\mathrm{rep}}
\quad\text{and}\quad
Q_{0.95}(R_i)-r_i^{\mathrm{rep}}.
$$

They are not an obsolete analysis. They are a good compact summary and should be retained.

Presentation concerns:

- Outliers are hidden in the current boxplots.
- CRIPT-N has a much larger scale than the other analyses and can compress their boxes.
- Absolute rank differences have direct decision meaning, but their comparability depends on landscape size and cohort definition.
- The top-100 panel matches the proposed main figure. Top 1,000 is better treated as a supporting analysis.

Plan:

1. Keep the boxplots.
2. Decide whether to show outliers, add a supplementary table, or report robust quantiles explicitly.
3. Consider separating CRIPT-N, adding an inset, or using landscape-normalized deviations as a secondary view. Do not replace the absolute-rank view, because absolute movement matters for candidate selection.
4. Add the median interval width and fraction whose interval leaves the top-$k$ region, as proposed in the outline.

## 5. The current Section 3 plots are useful but do not yet match every proposed panel

The added inclusion plot displays the 50 variants with the largest posterior top-50 inclusion probabilities. It can include variants outside the published top 50 and omit unstable published top-50 variants. This makes it useful for showing incumbents and challengers, but it is not exactly the proposed panel containing all and only reported top-$k$ variants.

The overlap boxplot contains two different quantities:

- posterior sample versus posterior sample: reproducibility among plausible landscapes;
- posterior sample versus reported set: the overlap $|T_k^{(s)}\cap T_k^{\mathrm{rep}}|/k$ in the outline.

The saved `stability.nc` files retain each draw's ordered top 1,000. They support exact inclusion probabilities, candidate-cloud sizes, and overlap calculations for any $1\leq k\leq1000$, even though `overlaps.csv` contains only $k=50,100,1000$.

Plan:

1. Keep the current fixed-$k$ landscape plots as exploratory views.
2. For Figure 3(a), show every reported top-$k$ variant and its inclusion probability. Show outside challengers separately or in an adjacent panel.
3. For Figure 3(b), calculate the mean sample-versus-reported overlap $O_k$ over a dense grid of $k$ values and include posterior intervals.
4. For Figure 3(c), overlay or summarize those curves across landscapes using a consistent score definition.
5. For Figure 3(d), define `non-negligible` before plotting the candidate cloud, for example $q_i(k)>0.01$, and examine sensitivity to that threshold.
6. Label sample-versus-sample overlap as reproducibility, not as $O_k$.

No new HMC runs are required for these steps up to $k=1000$.

## 6. Model disagreement and count uncertainty should not be conflated

A difference between published ranks and posterior latent ranks can arise from at least two sources:

1. finite-count uncertainty under the posterior model;
2. different modeling and score definitions in the published and posterior analyses.

Section 1 is currently sensitive to both. Section 2 more directly describes posterior uncertainty conditional on the chosen model. The paper should avoid attributing all Section 1 disagreement to finite counts.

Plan:

- Describe Section 1 as recovery under a specified alternative analysis procedure.
- Use Section 5's alternative estimators and model families to assess procedure sensitivity.
- Reserve causal language about finite counts for analyses that vary count information while holding the procedure fixed, such as downsampling or a controlled bootstrap.

## 7. Cross-landscape and cross-model comparisons are partly confounded

The eight analyses are not a complete landscape-by-model design:

- trans landscapes have one model each;
- cis has three models;
- CRIPT-C has two models;
- CRIPT-N has one model.

A cross-analysis plot can describe the eight fitted results, but it cannot estimate a general model-family effect independently of landscape.

Plan:

- Use one prespecified model per landscape for the main cross-landscape comparison.
- Use within-cis and within-CRIPT-C comparisons for model sensitivity.
- Avoid statements that attribute an across-row difference to model family when landscape changes at the same time.

## 8. Fit checks promised in the outline are not in the compact metadata

The result bundles record fit provenance, sample count, model, score definitions, and whether embedded observed counts match the raw count table. They do not record R-hat, effective sample size, divergent transitions, or prior/posterior predictive-check results.

All eight original HMC files still exist and match their saved file signatures exactly. The missing evidence can therefore be generated without refitting, assuming the needed diagnostic variables are available in those files.

Plan:

1. Produce a fit-diagnostics table for every analysis.
2. Save diagnostic thresholds and pass/fail decisions with the result bundle or in a linked sidecar file.
3. Add prior and posterior predictive figures or numerical summaries.
4. Distinguish posterior-predictive rank analyses from posterior predictive checks of whether the model reproduces observed count structure.

## 9. Counts needed for representative variants and later sections

`variants.csv` contains pooled input and output counts for every published candidate. This is enough for a first version of Section 2's representative-variant panel.

Replicate-level counts are not retained in the compact variant table. They remain in the raw source files and, where embedded, in the HMC files. Section 4's replicate-disagreement analysis must return to those sources or save a new replicate-level summary.

Plan:

- Use pooled counts only when the panel is explicitly labeled pooled.
- Use replicate-level counts for claims about replicate disagreement.
- Save zero counts, replicate totals, and a declared disagreement statistic before building Section 4.

## Recommended order of work

1. Freeze the primary estimand and score terminology.
2. Simulate or replace the Section 1 percentile reference before treating its area as a main result.
3. Keep and refine the Section 2 rank-interval plots and signed-deviation boxplots.
4. Build the exact Section 3 reported-top-$k$ inclusion panel and dense $O_k$ curves from existing `stability.nc` files.
5. Generate and save fit diagnostics and predictive checks.
6. Decide whether full-landscape posterior ranks are needed; do not run expensive postprocessing until that decision is made.
7. Only then select representative landscapes and assemble final multi-panel paper figures.
