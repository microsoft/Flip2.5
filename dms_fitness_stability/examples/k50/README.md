# Tsuboyama K50 source data for job 63

This directory records the source observations consumed by MLE-only K50 job 63
and a minimal projection of the authors' published folding-stability reference.
The full published score archives and the SciStanPy job-63 MLE output are not
retained. The canonical two-score matrix, compact result, mechanism caches, and
all seven raw/reference source files remain. Exact fit-retirement and recovery
evidence is recorded in `data/retired_product_fits.json`.

## Authoritative sources

- Paper: Tsuboyama et al., *Nature* (2023),
  <https://doi.org/10.1038/s41586-023-06328-6>.
- Final data release: <https://doi.org/10.5281/zenodo.7992926>.
- Code: `Rocklin-Lab/cdna-display-proteolysis-pipeline`, pinned at
  `597442a33f931a5e86003134c279b6f2a58ffd33`.
- Loader and job model:
  `microsoft/SciStanPy@09c439f675ead70f851d814242f153ecb7712113`, paths
  `flipv3/flip_dsets.py`, `flipv3/k50_models.py`, and
  `scistanpy/pipelines/mle_flip.py`.

The final Zenodo release documents `K50_scrambles_for_STEP3.csv` but omits it
from `Pipeline_K50_dG.zip`. This is upstream issue 7. The author-published first
release, <https://doi.org/10.5281/zenodo.7401275>, contains the exact member.
The retrieval script uses HTTP range requests to stream only that member, then
writes only its loader-required `aa_seq_experimental` column. The original
member's size, CRC32, and SHA-256 are verified, but its fitted `log10_K50_*`
columns and the rest of the pipeline archive are not retained.

## Retrieval and paths

```bash
uv run --with remotezip python examples/k50/retrieve_data.py \
  --data-root /data/data_dms_stability
uv run python examples/k50/validate_data.py \
  --data-root /data/data_dms_stability
```

```text
/data/data_dms_stability/k50/
├── Raw_NGS_count_tables.zip
└── raw/
    ├── NGS_count_lib1.csv
    ├── NGS_count_lib2.csv
    ├── NGS_count_lib3.csv
    ├── NGS_count_lib4.csv
    ├── K50_scrambles_for_STEP3.csv
    ├── Raw_qPCR_data_FigS1.csv
    └── reported_fitness/
        └── Tsuboyama2023_Dataset2_Dataset3_20230416.csv
```

The reported-fitness file is streamed by HTTP range access from
`Processed_K50_dG_datasets.zip` in Zenodo record 7992926. Retrieval verifies
the source archive and member identities but writes only the fields needed for
variant identity, absolute stability, confidence-interval width, censoring, and
domain provenance. The 1.01 GB parent archive is not retained. Archive MD5,
member CRC32/SHA-256, projected SHA-256, and exact row counts are in
`data_manifest.json`.

## Variant ordering and axes

For each library, `load_k50_data()` drops `name` and `dna_seq`, groups duplicate
rows by `aa_seq`, removes the dummy sequence `-`, and retains pandas' default
lexicographic group order. Global IDs are assigned on first appearance while
visiting v1, v2, v3, then v4. This produces 1,824,523 unique global sequences.

The source loader emits:

```text
counts_c0[protease, replicate, variant]
counts_cg0[protease, replicate, concentration, variant]
```

Protease order is chymotrypsin (`C`) then trypsin (`T`). Replicate 1 uses source
columns `01`-`12`; replicate 2 uses `13`-`24`. Concentration index 0 is the
untreated count and indices 1-11 are the treated series. Some historical model
comments label the first two axes in the opposite order, but the loader and job
63 arrays are unambiguously `protease × replicate`; the adapter preserves that
behavior.

Per-library fitted counts are v1 297,699, v2 573,650, v3 709,823, and v4
315,593. Shared arrays are `ordinal_seqs[1,824,523,86]`, 64,238 unique scramble
indices, `qpcr_log_protease_conc[11]`, `qpcr_log2_survival[2,11,8]`, and
`log_expected_protease_conc[2,11]`.

## Reported and fitted scores

There is no global K50 wild type. The four libraries combine hundreds of
natural and designed domain backgrounds, so WT-relative `ddG_ML` cannot define
one global ranking. The authors do publish absolute folding stability as
`dG_ML`, in kcal/mol, with larger values indicating greater stability. This is
the external reported-fitness reference.

The publication and final job-63 count snapshot do not use identical construct
sets or always use identical padding strings. Retrieval therefore joins rows by
exact `(name, dna_seq)`, assigns the raw table's `aa_seq` as the job-63 variant
ID, and retains publication `aa_seq_full` for audit. Of 776,298 publication
measurements, 537,021 exactly match the final raw snapshot. After excluding
2,085 matched rows marked unavailable, 524,836 unique job-63 sequences have
quality-eligible reported fitness.

The adapter follows the paper's analysis convention:

- `-` is unavailable and remains missing;
- `<-1` and `>5` map to the assay limits;
- numeric `dG_ML` is clipped to $[-1, 5]$ kcal/mol; and
- duplicate measurements of one padded sequence are median-collapsed.

Raw combined `deltaG`, 95% interval width, censoring status, and measurement
count remain in the aligned variant table. `reported_rank` is the average rank;
`reported_rank_min` and `reported_rank_max` give the best and worst rank allowed
by score ties. After duplicate aggregation, 41,517 sequences share the upper
limit of 5 kcal/mol. K = 50, 100, and 1,000 therefore identify possible members
within that plateau, not unique reported top-K sets. Downstream analyses must
not resolve the plateau by variant order and report the result as an exact
author-defined top-K.

For job 63, the primary model variable is `mean_dg[a]`, one fitted mean folding
free energy per global sequence. The validated sampled-MLE artifact exposes
`mean_dg[chain,draw,a]` with shape `[1,1,000,1,824,523]` and predictive count
draws for all four libraries. It is resolved from the external Amulet artifact
root through `data/fit_audit.json`; it is not copied into this source-only K50
directory. The paper workflow also derives the explicitly non-thermodynamic
`k50-chymotrypsin-mean-log-enrichment-v1` and
`k50-trypsin-mean-log-enrichment-v1` sensitivity scores from those count draws.
The proteases are not averaged. The artifact stores
`fitness_score[score_name, sample, variant]`; selecting either score and calling
`.transpose("variant", "sample")` produces the required 1,824,523 by 1,000
variant-by-draw analysis matrix. The 64,238 scrambled controls remain in the
matrix normalization universe but are excluded from ranking. Formulas, measured
results, interpretation, and Spearman-correlation limitations are maintained in
[`MLE_FITNESS_PROXIES.md`](../paper/MLE_FITNESS_PROXIES.md#interpretation-of-k50-stability).

The full `K50_dG_tables.zip` and `Processed_K50_dG_datasets.zip` archives remain
deferred; only the projected publication member above is retained. The
validator explicitly rejects those full archives, `.nc` files, and `fits/`
directories from this source-only tree. The generated score bundle lives below
`/data/data_dms_stability/paper/analysis_data/scores/mle_predictive/canonical`;
its result bundle remains below `/data/data_dms_stability/paper/mle_predictive`.
