# Fit provenance required for controlled Section 4 experiments

The observational Section 4 analysis can be reproduced from the current raw
counts and saved HMC fits. Controlled read downsampling and replicate ablation
cannot be completed correctly in this repository alone because they require new
inference after the counts are perturbed.

Before implementing those experiments, recover and record:

1. the repository and commit containing the Stan/SciStanPy model definitions;
2. the data-preparation entry point that produced the `flip3_data` count inputs;
3. the Amulet configurations for the `small-clean` and `large-clean` jobs;
4. the exact software environment and pinned SciStanPy/Stan versions;
5. the command used to submit one PDZ and one TrpB fit;
6. the convention for input, output, timepoint, and replicate axes;
7. the diagnostic and posterior-predictive steps that created `*diagnosed.nc`;
8. the storage paths and naming convention for new fit artifacts; and
9. a small synthetic or reduced-data fit that can be run before a full sweep.

Current evidence in this workspace:

- SciStanPy is pinned in `pyproject.toml`.
- `examples/pdz/fit_jobs.json` and `examples/trpb/fit_jobs.json` map fit IDs to
  prior families, growth curves, posterior variables, and canonical filenames.
- TrpB fits came from Amulet experiment `small-clean`; several PDZ fits came
  from `small-clean` and `large-clean` jobs.
- `examples/trpb/materialize_fits.py` records the remote result convention.
- The Stan model source and job-submission YAML are not present here.

## Controlled pilot after provenance recovery

Use one PDZ landscape with paired input/output replicates and one TrpB landscape
with verified output replicates. For each, thin input only, output only, and both
at fractions 1, 0.5, 0.25, and 0.125 with at least five independent seeds. Refit
the same model specification to every thinned dataset. Compare score uncertainty,
rank intervals, candidate-cloud size, and top-100 overlap with the full-depth
analysis reference. The full-depth result is a reference, not ground truth.

Leave-one-replicate-out fits are appropriate only where the axis is a verified
replicate. Removing a TrpB timepoint is a timepoint-ablation experiment, not a
replicate-ablation experiment.

`--n-samples` on the current runners subsamples posterior draws. It tests Monte
Carlo precision only and must not be described as read downsampling.

## HMC count-based analysis

The HMC count-based workflow does not perturb observed counts or refit a model.
It uses the `posterior_predictive` counts already paired with each HMC posterior
draw and applies the same family-specific proxy used in the sampled-MLE
analysis. Its uncertainty target is the joint posterior-predictive distribution
of that count-derived score, combining posterior parameter uncertainty and a
new count realization.

The exact 49-run fit selection is frozen in
`analyses_hmc_count_based.json`. Artifact paths and fallback choices continue
to come from `data/fit_audit.json`; the new analysis does not make a damaged
diagnosed artifact usable. In particular, jobs such as 56 and 78 use the
validated raw-HMC artifacts selected by the manifest.

See `HMC_COUNT_BASED_PROXIES.md` for score formulas, tensor coordinates,
storage requirements, and reproduction commands. This workflow remains
distinct from controlled read downsampling or replicate ablation, both of
which still require new fits.
