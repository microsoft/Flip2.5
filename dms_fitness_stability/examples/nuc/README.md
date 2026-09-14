# NucB source data for jobs 55-62

This directory records the source-only inputs for the requested NucB HMC jobs.
No HMC output is downloaded or required by this workflow. Job 60 (G3 Lomax) is
not included because it was not part of the requested job set.

## Authoritative sources

- Paper: Thomas et al., *Cell Systems* (2025),
  <https://doi.org/10.1016/j.cels.2025.101236>.
- Data and analysis repository:
  <https://github.com/google-deepmind/nuclease_design>, pinned in the manifest
  at commit `702284847c4558567d8288169a86dc147cfaab32`.
- Public object store: `gs://nuclease_design`.
- Loader and model source:
  `microsoft/SciStanPy@09c439f675ead70f851d814242f153ecb7712113`, paths
  `flipv3/flip_dsets.py`, `flipv3/nuclease_models.py`, and
  `scistanpy/pipelines/mcmc_flip.py`. This is the same SciStanPy commit pinned
  by the workspace.

The compact `processed_data/gN.csv` objects contain the exact input and sorted
output count columns consumed by `flipv3.load_nuclease_data()`. The approximately
8 GB `raw_count_data/G1` through `G4` instrument tables are therefore not needed
to reconstruct jobs 55-62 and remain deferred.

## Retrieval and paths

Run from the repository root:

```bash
uv run python examples/nuc/retrieve_data.py \
  --data-root /data/data_dms_stability
uv run python examples/nuc/validate_data.py \
  --data-root /data/data_dms_stability
```

The retrieval script uses generation-qualified GCS media URLs and validates
publisher byte counts and MD5 values before atomically installing each file.
Local SHA-256 values are frozen in `data_manifest.json`.

```text
/data/data_dms_stability/nuc/
└── raw/
    ├── counts/
    │   ├── g1.csv ... g4.csv
    │   └── g1_neg_control.csv ... g4_neg_control.csv
    └── reported_fitness/
        └── landscape.csv
```

## Variant identity and ordering

The fitted order for each generation is all rows of `gN.csv`, followed by all
rows of `gN_neg_control.csv`. Mutation tuples are sorted by their one-indexed
position and serialized as signatures such as `D64S_A73R`. The empty signature
is the unique wild type and occurs at fitted index 0 in all four generations.

The two source tables have different experimental roles. Rows from `gN.csv`
are full-length variants eligible for candidate ranking. Rows from
`gN_neg_control.csv` are premature-stop fiducials used to characterize the
inactive background. SciStanPy includes both groups on the fitted posterior
axis, so the adapter preserves every row and `hmc_variant_index`, but marks the
fiducials with `is_source_control=True` and
`is_ranking_candidate=False`. Top-K sets and ranks use only the main-generation
rows. This is an eligibility distinction; it does not alter the likelihood,
posterior samples, or fitted score of any row.

The reference is the 142-residue *Bacillus licheniformis* NucB sequence:

```text
MIKKWAVHLLFSALVLLGLSGGAAYSPQHAEGAARYDDVLYFPASRYPETGAHISDAIKAGHADVCTIERSGADKRRQESLKGIPTKPGFDRDEWPMAMCEEGGKGASVRYVSSSDNRGAGSWVGNRLNGYADGTRILFIVQ
```

Each `gN.csv` table already includes author-computed continuous enrichment
factors. For variant $i$ in one sort gate,

$$
EF_i =
\frac{c_{i,\mathrm{sort}} / \sum_j c_{j,\mathrm{sort}}}
  {c_{i,\mathrm{input}} / \sum_j c_{j,\mathrm{input}}}.
$$

The authors' liquid-culture comparison notebook explicitly selects the highest
sort gate for each generation. The adapter uses those same fields as continuous
reported fitness, with larger enrichment indicating greater activity:

| Generation | Reported fitness | Reported variants |
| --- | --- | ---: |
| G1 | `ef_2_high_reseq_g1` | 9,441 |
| G2 | `ef_1_97.5_g2` | 14,678 |
| G3 | `ef_2_99_g3` | 18,618 |
| G4 | `ef_4_99.5_g4` | 15,404 |

Observed-count oracles exactly reproduce all four reported enrichment tables at
`rtol=atol=1e-12`. G2 pools all three inputs. G1 pools all five inputs and uses
the original source abundance totals, which include 132 DNA variants removed
from the staged fitted rows only after enrichment was calculated. Sampled-MLE
proxy analyses are complete for jobs 59, 61, and 62; verified G1/G2 jobs 55-58
have not yet been executed.

These fields define unique boundaries at K = 50, 100, and 1,000. The appended
negative controls remain fitted by SciStanPy but have no reported-fitness value.

`reported_fitness/landscape.csv` is the separate 55,760-variant merged
cross-generation landscape. It reduces the generation-specific measurements to
categorical activity relative to fiducials (`non-functional`, above zero, above
WT, or above A73R). The adapter retains these labels alongside the continuous
generation-specific enrichment reference.

## Count axes

The last axis is always the fitted variant axis.

| Generation | Loader arrays |
| --- | --- |
| G1 | `ic1[2,a]`, `ic2[2,a]`, `ic3[a]`, `lc[3,a]`, `hc1[a]`, `hc2[2,a]`, `hc3[a]`; `lt[3]`, `ht[3]` |
| G2 | `ic1[2,a]`, `ic2[a]`, `c86[a]`, `c93[a]`, `c975[a]`; `ft[3]` |
| G3 | `ic1[a]`, `oc1[4,a]`; `ft[4]` |
| G4 | `ic1[a]`, `oc1[4,a]`; `ft[4]` |

Exact fitted counts are G1 10,760, G2 14,755, G3 18,790, and G4 15,554.
The corresponding ranking-candidate counts are 9,441, 14,678, 18,618, and
15,404; the remaining 1,319, 77, 172, and 150 rows are source controls.

## Requested fit contracts

`fit_jobs.json` records the original CLI arguments and dimensions. Jobs 55 and
56 fit `absolute_mean_log_fluorescence[chain,draw,a]`; jobs 57, 58, 59, 61,
and 62 fit `log_fluorescence[chain,draw,a]`. Expected chain/draw counts are
stored per job. The validator is source-only and rejects any `.nc` or `fits/`
artifact below the canonical NucB root.