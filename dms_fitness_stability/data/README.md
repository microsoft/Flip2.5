# Local Data

The large datasets are stored outside this Git repository on the machine-level
`/data` volume:

```text
/data/data_dms_stability/
├── pdz_trans/
│   ├── raw/
│   │   ├── trans-1.tsv
│   │   ├── trans-2.tsv
│   │   └── GSE265816_pdz3cript_sub.csv.gz
│   └── fits/
│       ├── trans-1_exponential_exponential.nc
│       └── trans-2_exponential_exponential.nc
├── pdz_cis/
│   ├── raw/
│   │   ├── cis_from_bruce_azure.tsv
│   │   └── GSE265816_cript2xa3_sub.csv.gz
│   └── fits/
│       └── pdz_cis_exponential_exponential_diagnosed.nc
├── pdz/
│   └── results/
│       ├── trans-1/
│       ├── trans-2/
│       └── cis/
├── trpb/
│   ├── TrpBFitness.zip
│   ├── raw/
│   │   ├── counts/
│   │   │   └── libA.csv ... libI.csv
│   │   └── reported_fitness/
│   │       └── libA.csv ... libI.csv, four-site.csv
│   ├── fits/
│   │   └── libA/ ... libI/
│   └── results/
│       └── libA/ ... libI/
├── nuc/
│   └── raw/
│       ├── counts/
│       │   ├── g1.csv ... g4.csv
│       │   └── g1_neg_control.csv ... g4_neg_control.csv
│       └── reported_fitness/
│           └── landscape.csv
└── k50/
    ├── Raw_NGS_count_tables.zip
    └── raw/
        ├── NGS_count_lib1.csv ... NGS_count_lib4.csv
        ├── K50_scrambles_for_STEP3.csv
        ├── Raw_qPCR_data_FigS1.csv
        └── reported_fitness/
            └── Tsuboyama2023_Dataset2_Dataset3_20230416.csv
```

Paper analysis data are materialized separately under:

```text
/data/data_dms_stability/paper/analysis_data/
├── scores/
│   ├── hmc/
│   │   ├── release_1000/<analysis-id>/
│   │   │   ├── metadata.json
│   │   │   ├── variants.csv
│   │   │   ├── analysis_variants.csv
│   │   │   └── fitness_scores.nc
│   │   └── full/<analysis-id>/       # canonical, source-exact and equivalent
│   ├── mle_predictive/canonical/<analysis-id>/
│   │   ├── metadata.json
│   │   ├── variants.csv
│   │   ├── analysis_variants.csv
│   │   └── fitness_scores.nc
│   └── hmc_count_based/canonical/<analysis-id>/
│       ├── metadata.json
│       ├── variants.csv
│       ├── analysis_variants.csv
│       └── fitness_scores.nc
└── landscape_features/<feature-id>/
    ├── metadata.json
    ├── variants.csv
    └── features.nc
```

`release_1000` contains deterministic subsets for all 49 direct-HMC runs and
is intended for lighter reuse, not canonical paper reproduction. The validated
`full` view retains all draws. Landscape feature bundles contain exact derived
inputs for current count-support figures; source count files remain necessary
to audit or change those transformations.

The tracked release and reproduction checks are:

- `analysis_data_manifest.json`: checksums for 208 score bundles and 21 feature
    bundles.
- `score_root_relocation.json`: the approved and completed relocation of 61
    MLE-predictive and 49 HMC count-based bundles into the unified score
    hierarchy, including the manifest hash chain and 111 cache provenance
    updates.
- `mle_score_audit.json` and `hmc_count_based_score_audit.json`: current
    score/result contract audits at the unified roots. The proxy-named audits
    remain immutable pre-conversion records.
- `hmc_score_matrix_full_audit.json`: exact comparison of all full HMC matrix
    values with their source posterior targets.
- `hmc_full_matrix_equivalence_audit.json` and
    `hmc_full_mechanism_equivalence_audit.json`: full-view result, cache, and table
    equivalence.
- `landscape_feature_audit.json`: exact source recomputation for all 21 feature
    bundles.
- `notebook_reproduction_audit.json`: successful notebook execution and
    nonblank PNG/PDF validation for direct HMC, MLE-predictive, and HMC
    count-based figures.

`hmc_matrix_migration_audit.json` remains the immutable pre-migration baseline.
The separate `hmc_post_matrix_artifact_audit.json` records the current outputs;
the two differ in refreshed figure files and additive mechanism-cache support
fields, while the dedicated equivalence audits require all legacy scientific
arrays and table columns to match.

## PDZ inputs and results

The unified PDZ workflow treats `trans-1`, `trans-2`, and `cis` as three
landscapes under one analysis. Source-specific directories preserve the raw
file provenance, while all outputs share `/data/data_dms_stability/pdz/results`.

The fit operationally named PDZ `cis` is GEO library `cript2xa3_cis`. Its TSV
contains three input and three selected-output count columns but no
reported-fitness column. Published fitness and uncertainty come from
`GSE265816_cript2xa3_sub.csv.gz`; all 9,475 peptide rows are an exact subset of
the 10,660 fitted variants. `GSE265816_cript_c_sub.csv.gz` and
`GSE265816_cript_n_sub.csv.gz` describe separate CRIPT-C and CRIPT-N landscapes,
not this cis landscape.

See `examples/pdz/KNOWN_ISSUES.md` before analyzing newly downloaded fits. It
records locally repaired, incomplete artifacts and the jobs that should be
rerun or reconverted.

Validate all source files, run every currently manifested PDZ landscape/model
fit with common defaults, and write a joint overview:

```bash
uv run python examples/pdz/validate_data.py
uv run python examples/pdz/run.py
uv run python examples/pdz/summarize.py --k 50
```

## TrpB inputs

`TrpBFitness.zip` supplies the reported-fitness landscapes. The count tables
that define the fitted HMC column order come from the exact Amulet data snapshot
mounted by `small-clean`:

```bash
~/venvs/amulet/bin/amlt storage download \
    data/dms_stan/flip3_data/counts/trpb \
    /data/data_dms_stability/trpb/raw/counts \
    --storage-id default \
    --include 'lib*.csv'
```

Choose `update` if Amulet asks how to handle an existing destination. The
uploaded reported rows are validated as an ordered subsequence of the fitted
variant order; variants without reported fitness remain in the full HMC scan.

## NucB source inputs

Jobs 55, 56, 57, 58, 59, 61, and 62 use the four NucB generations from
Thomas et al. (2025), DOI `10.1016/j.cels.2025.101236`. The source repository is
`google-deepmind/nuclease_design`; its public GCS bucket contains compact
generation tables with the exact input and sorted-output counts consumed by the
SciStanPy `flipv3.load_nuclease_data()` loader.

Retrieve and validate the source-only bundle:

```bash
uv run python examples/nuc/retrieve_data.py \
    --data-root /data/data_dms_stability
uv run python examples/nuc/validate_data.py \
    --data-root /data/data_dms_stability
```

The fitted variant order is the generation table followed by its negative
controls. Exact checksums, GCS object generations, count axes, the 142-residue
*B. licheniformis* NucB reference, fitted cardinalities, and expected posterior
variables are recorded in `examples/nuc/data_manifest.json` and
`examples/nuc/fit_jobs.json`. The generation tables also contain continuous
enrichment factors. Following the authors' liquid-culture comparison notebook,
the adapter uses the highest sort gate per generation as reported fitness:
`ef_2_high_reseq_g1`, `ef_1_97.5_g2`, `ef_2_99_g3`, and `ef_4_99.5_g4`.
The 55,760-row merged landscape separately provides cross-generation categorical
activity relative to assay fiducials.

The much larger `raw_count_data/G1` through `G4` GCS objects are not retrieved:
the compact tables already preserve the fitted count arrays. Job 60 and every
fit artifact remain outside the canonical NucB source directory; selected fits
are staged separately below `/data/amlt` by `examples/fits`.

## K50 source inputs

MLE-only job 63 uses the Tsuboyama et al. (2023) cDNA-display proteolysis data,
DOI `10.1038/s41586-023-06328-6`. The four NGS count tables come from Zenodo
record `7992926`; qPCR calibration is pinned to the authors' GitHub commit. The
scramble table is remotely extracted from first-version Zenodo record `7401275`
because later releases document but accidentally omit it. Retrieval retains
only its loader-required `aa_seq_experimental` identity column, not its fitted
K50 columns.

```bash
uv run --with remotezip python examples/k50/retrieve_data.py \
    --data-root /data/data_dms_stability
uv run python examples/k50/validate_data.py \
    --data-root /data/data_dms_stability
```

The loader produces 1,824,523 unique global sequences and 64,238 scramble
indices. Counts retain the historical
`protease × replicate × concentration × variant` ordering. There is no single
global wild type because the libraries contain many natural and designed domain
backgrounds. The authors' published Dataset 2/3 table supplies absolute
quality-filtered `dG_ML` as the reported reference. Retrieval range-streams only
the needed member and joins exact `(name, dna_seq)` constructs to the job-63 raw
snapshot, yielding 524,836 quality-eligible unique sequence references. Job 63
fits `mean_dg[a]`, with `log_K50[protease,a]` as a secondary transformed
quantity.

The canonical K50 source directory deliberately excludes the full
`K50_dG_tables.zip`, full `Processed_K50_dG_datasets.zip`, and fit NetCDF files.
Only a seven-column reported-fitness projection is retained there; selected Amulet
fits are staged separately below `/data/amlt`. Exact source/archive/member
checksums, alignment policy, axes, cardinalities, and the job-63 MLE contract
are in `examples/k50/data_manifest.json` and `examples/k50/fit_jobs.json`.

## Selected fit download and audit

The cross-dataset fit workflow lives in `examples/fits`. It derives the exact
`Perfect!` selection from `HMC Results Summaries.csv`, preflights HMC and MLE
from their actual Amulet experiments, rejects downloads that breach the `/data`
free-space reserve, validates artifact contents, and writes the central
`KNOWN_ISSUES.md` inventory. It also supports exact-file raw-HMC fallback after
a diagnosed artifact fails validation.

See `examples/fits/README.md` for the phased commands and remote naming rules.

## Sampled-MLE source retention

The canonical 61 sampled-MLE artifacts were converted into persisted
count-derived score matrices and strictly audited before source retirement on
2026-08-31.
Three source NetCDF files remain for preprocessing examples: PDZ job 3, TrpB
four-site job 64, and NucB G3 job 59. The other 58 sampled-MLE sources were
removed locally. All 61 score bundles, 61 compact results, 62
mechanism caches, paper tables, and figures remain.

`mle_source_retirement.json` is the authoritative retention record. It includes
the exact retained paths, retired job IDs, byte totals, validated product
counts, and the limitation that retired matrices need a restored sampled-MLE
source before they can be rebuilt. The experimental raw-data roots shown above
remain available for validating every landscape, including K50.

## TrpB fits

The TrpB fits come from the `small-clean` Amulet experiment. Download only the
diagnosed NetCDF output for the selected jobs to the large staging volume:

```bash
~/venvs/amulet/bin/amlt results download small-clean \
    :small-clean_7 :small-clean_9 \
    :small-clean_11 :small-clean_13 \
    :small-clean_15 :small-clean_17 \
    :small-clean_19 :small-clean_21 \
    :small-clean_25 :small-clean_27 \
    :small-clean_31 :small-clean_34 \
    :small-clean_37 :small-clean_39 \
    :small-clean_43 :small-clean_45 \
    :small-clean_49 :small-clean_51 \
    --include '*diagnosed.nc' \
    --output /data/amlt \
    --n-workers 4
```

Amulet preserves the experiment, job, and remote result directories below
`/data/amlt/small-clean`. Materialize the staged files at canonical paths without
duplicating their data blocks, then validate all inputs and fit structures:

```bash
uv run python examples/trpb/materialize_fits.py
uv run python examples/trpb/validate_data.py
```

The supplied job mapping and posterior variables are recorded in
`examples/trpb/fit_jobs.json`. The four-site reported landscape is retained, but
is not analyzed until a corresponding fit is added to that manifest.

Four downloaded artifacts have known NetCDF or embedded-data limitations. See
`examples/trpb/KNOWN_ISSUES.md`; the same issues are tagged per job in
`examples/trpb/fit_jobs.json` for machine-readable tracking.

## TrpB stability results

Run every manifest entry with all 16,000 posterior samples:

```bash
uv run python examples/trpb/run.py
```

The default output is `/data/data_dms_stability/trpb/results` (or the equivalent
path below `DMS_FITNESS_STABILITY_DATA`). Each library/model directory contains
`metadata.json`, `variants.csv`, `stability.nc`, `summary.csv`, and
`overlaps.csv`. The production run currently occupies about 4.5 GB.

Limited artifacts are analyzed through the explicit `h5py` fallback and remain
marked `artifact_status: limited` with warnings in their result metadata. They
should be regenerated before publication-quality downstream reporting.

This repository uses `/data/data_dms_stability` by default. To inspect it:

```bash
ls -lh /data/data_dms_stability/pdz_trans/raw
ls -lh /data/data_dms_stability/pdz_trans/fits
ls -lh /data/data_dms_stability/pdz_cis/raw
ls -lh /data/data_dms_stability/pdz_cis/fits
ls -lh /data/data_dms_stability/pdz/results
```

To validate the copied data from the repository root:

```bash
uv run python examples/pdz/validate_data.py
uv run python examples/trpb/validate_data.py
```

Set `DMS_FITNESS_STABILITY_DATA` to use another external data root. The large
files are deliberately not copied into this directory because the repository is
on the root filesystem, which does not have enough free space for both HMC fits.
