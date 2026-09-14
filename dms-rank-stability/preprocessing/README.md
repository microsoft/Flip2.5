# Preprocessing

These six optional examples convert representative model fits into the same
two-file sampled-score inputs used by the paper analysis. They are not required
to reproduce the paper from the prepared matrices.

Preprocessing assumes existing fit files and variant tables aligned to the fit
columns; for another landscape, prepare those inputs first and adapt the
example's landscape-specific settings.

The example command does not read raw experimental files, fit statistical
models, or build landscape features from counts. See
[Experimental Data](#experimental-data) for the raw file layout and
[New Datasets](#new-datasets) for the extension boundary.

Each HMC example writes two matrices:

- posterior parameter scores;
- scores calculated from paired posterior-predictive count samples.

Each sampled-MLE example writes one matrix calculated from predictive count
samples. The family-specific calculations are explicit in `fitness.py`: PDZ
uses inverse-variance weighted log fitness relative to wild type, TrpB uses the
deposited stop/max normalization, and NucB G3 uses highest-gate enrichment.

## Inputs

First obtain the prepared paper data. The fit files contain numeric matrix
columns but not biological sequence IDs, so the examples use the prepared
variant tables to align those columns and to check the generated scores.

If the paper data are already local, no download is needed. By default the
examples read scores from `data/downloads/scores/`. For this preprocessing
command, `--data-root` points directly to the score directory containing the
three sampling subdirectories:

```bash
uv run python preprocessing/run.py pdz-hmc --fit "/path/to/fits/pdz-hmc.nc" --data-root "/path/to/analysis-data/scores"
```

See [../data/README.md](../data/README.md) for the data format and the download
command to use after publication.

The six companion fit files will be published separately from the required
paper data. Until their public URLs are assigned, pass an existing local fit
file with `--fit`.

| Example | Fit contents | Landscape |
| --- | --- | --- |
| `pdz-hmc` | HMC parameter and predictive counts | PDZ cis |
| `pdz-mle` | sampled-MLE predictive counts | PDZ cis |
| `trpb-hmc` | HMC parameter and predictive counts | TrpB libA |
| `trpb-mle` | sampled-MLE predictive counts | TrpB four-site |
| `nuc-hmc` | HMC parameter and predictive counts | NucB G3 |
| `nuc-mle` | sampled-MLE predictive counts | NucB G3 |

TrpB uses two landscapes because the retained HMC fit is for libA, whereas the
retained sampled-MLE fit is for the combined four-site landscape.

## Run

Run one example from the repository root:

```bash
uv run python preprocessing/run.py pdz-hmc --fit /path/to/pdz-hmc.nc
```

The default selects 128 fit samples with seed 23 while retaining the complete
variant axis. Use `--all-samples` to reproduce the complete prepared matrix:

```bash
uv run python preprocessing/run.py pdz-hmc \
	--fit /path/to/pdz-hmc.nc \
	--all-samples
```

Generated matrices are written below `preprocessing/output/`. Every output is
checked exactly against the corresponding prepared matrix using stable sample
IDs, variant identities, sample coordinates, block order, and float32 score
values. `--no-check` skips this comparison only. The command still requires the
example's prepared score directories for variant alignment and retains its
fixed landscape settings; this flag does not enable arbitrary new datasets.

Use the same command with `pdz-mle`, `trpb-hmc`, `trpb-mle`, `nuc-hmc`, or
`nuc-mle`. `--sample-chunk-size` bounds working memory; it does not change the
selected samples or numerical results.

## Experimental Data

Store the experimental inputs separately from prepared scores and features:

```text
analysis-data/
	raw/
		pdz/
			counts/
			reported_fitness/
		trpb/
			counts/
			reported_fitness/
		nuc/
			counts/
			reported_fitness/
	scores/
	features/
```

For the paper data, this contains 38 files:

| Family | Counts | Reported fitness |
| --- | --- | --- |
| PDZ | Five TSVs for cis, trans-1, trans-2, CRIPT-C, and CRIPT-N | Four compressed GEO reference tables |
| TrpB | Ten CSVs for libA through libI and four-site | Ten corresponding CSVs |
| NucB | Four generation tables and four negative-control tables | One cross-generation activity table |

PDZ cis, trans, and CRIPT files share the `pdz/` directory but remain separate
landscapes. Preserve the existing filenames and file contents, including the
PDZ cis filename `cis_from_bruce_azure.tsv`. The two trans landscapes share a
published reference table. Keep counts separate from reported fitness: TrpB
uses the same filenames for different data in these subdirectories. NucB's
generation tables also contain the continuous enrichment references, so retain
them as well as the cross-generation activity table and negative controls.

This is the experimental-data storage layout, not an input accepted by the
current example command. Neither the analysis command's `--data-root` nor the
fit example's `--data-root` should point to `raw/`. Fits and generated results
belong outside `raw/`.

## New Datasets

For another landscape, use a small preparation script that calls the extraction
functions directly, rather than changing the six examples in [run.py](run.py):

1. **Fit and sample upstream.** Interpret the experiment's counts, replicates or
	gates, and controls, then fit the chosen model with the appropriate fitting
	software. Parameter scores require posterior draws; predictive scores
	require sampled predictive counts. Observed counts alone are not either
	kind of sample.
2. **Prepare the variant tables.** Export `fitted_variants` as a pandas table
	with one unique `variant_id` per fitted column, in the exact order used to
	fit the model, including controls. Prepare `analysis_variants` in the same
	order or as an ordered subset, with boolean `is_candidate` and the reported
	fitness and rank columns used by the analysis configuration. Join reference
	measurements by identity without changing that order; do not infer the
	fitted order from a sorted list of sequences. The tables can be identical
	when every fitted variant is retained.
3. **Extract scores.** Use `extract_parameter_scores` from [scores.py](scores.py)
	for a posterior parameter stored as `chain x draw x variant`. Choose the
	actual posterior variable and its scientifically appropriate transform:
	`identity` leaves values unchanged, `positive` applies the natural logarithm,
	and `inverse` applies its negative. For predictive scores, adapt the matching
	function in [predictive.py](predictive.py), checking the count variables,
	axes, wild-type identity, timepoints or gates, and normalization in
	[fitness.py](fitness.py). The NucB example is specific to G3; other layouts
	need their own count interpretation.
4. **Validate a small extraction first.** Check variant order, source sample
	IDs, and selected score values against the fit and the intended calculation.
	Then increase the sample count. The parameter extractor uses
	`sample_count=None` for all available draws; this is a Python argument, not
	an option to the fixed-example command.

For example, this parameter-score call assumes the model stores an already
log-scaled score in `posterior/log_r_mean`. Set `fit_path` and `score_directory`
to your input and output paths, and supply the two tables from step 2:

```python
from preprocessing.scores import extract_parameter_scores

extract_parameter_scores(
	 fit_path,
	 source_variants=fitted_variants,
	 target_variants=analysis_variants,
	 output=score_directory,
	 posterior_variable="log_r_mean",
	 transform="identity",
	 sample_count=128,
)
```

Run the script from the repository root in the installed environment. This
call writes a `SampledScores` bundle without requiring a prepared reference
matrix. Use at least two samples and no more than the fit contains. Other
variable names or array layouts need an appropriate extraction function.

To run the complete analysis, also compute the count-derived features from
the new experiment and save a `LandscapeFeatures` bundle with matching variant
identities. The required feature names and types are defined by `FEATURE_DTYPES`
in [../src/dms_rank_stability/data.py](../src/dms_rank_stability/data.py).
Validate both bundles; see [../data/README.md](../data/README.md) for their
formats. Score extraction does not create the feature bundle.

Use the [tiny example](../data/example/README.md) and its
[run configuration](../data/example/analysis/runs.json) as a starting point for
a separate analysis configuration. Declare the score direction, reference
columns, paths, and analysis settings explicitly rather than modifying the
paper's 158 runs.

The repository does not yet provide a raw-count-to-bundle command. The six fit
examples therefore demonstrate fit-to-score conversion for their declared
landscapes, not a complete raw-data workflow for new experiments.