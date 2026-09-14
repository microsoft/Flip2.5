# Known PDZ Fit Issues

Recorded on 2026-08-12 while downloading the selected `small-clean` and
`large-clean` HMC artifacts. This file distinguishes structurally complete fits
from artifacts that can be read only after local repair or that should be
regenerated.

## Rerun candidates

| Experiment job | Landscape | Model | Status |
| --- | --- | --- | --- |
| `large-clean_72` | CRIPT-C | Gamma-Exponential | Repaired locally, but the diagnosed artifact is incomplete. Rerun recommended. |
| `large-clean_76` | CRIPT-N | Exponential-Exponential | Repaired locally, but the diagnosed artifact is incomplete. Rerun recommended. |
| `large-clean_78` | CRIPT-N | Gamma-Exponential | Repaired superblock, but posterior-predictive storage is corrupt/incomplete. Rerun required for the common analysis. |

## Failure evidence

All three downloaded files matched their Azure blob sizes and AzCopy completed with
zero failed transfers. Before repair, their HDF5 version-0 superblocks retained
the "open for write" consistency flag and stored an end-of-file address smaller
than the physical file:

| Job | Physical/blob bytes | Stored EOF before repair | Unaddressed trailing bytes |
| --- | ---: | ---: | ---: |
| `large-clean_72` | 184,913,381,184 | 156,502,489,929 | 28,410,891,255 |
| `large-clean_76` | 109,285,683,020 | 90,841,447,068 | 18,444,235,952 |
| `large-clean_78` | 143,656,232,729 | 135,518,058,404 | 8,138,174,325 |

The first 96 bytes were subsequently downloaded directly from each Azure blob
with a byte-range request. All three remote headers matched the saved local
pre-repair superblocks exactly, including the stale EOF and consistency flag.
This confirms that the diagnosed source artifacts are malformed in Azure; the
defect was not introduced during download.

With the original superblocks, the `posterior` group was readable, but opening
`posterior_predictive` failed with:

```text
KeyError: Unable to synchronously open object
(len not positive after adjustment for EOA)
```

The root contained no readable `sample_stats`, `observed_data`, or diagnostic
groups.

## Local repair

For each local downloaded file, the first 96 bytes were saved beside the fit as
`<fit>.superblock-before-repair.bin`. The local superblock was then changed only
to:

1. clear the consistency flag from `1` to `0`; and
2. set the stored EOF to the verified physical/blob size.

The Azure blobs were not modified. After repair, jobs 72 and 76 open with
`h5py`, `h5netcdf`, and xarray. First and last values are readable from the
posterior target and both posterior-predictive count arrays.

The repaired jobs 72 and 76 remain incomplete. Their only root groups are:

```text
posterior
posterior_predictive
```

They still lack:

```text
sample_stats
observed_data
variable_diagnostic_stats
sample_diagnostic_tests
variable_diagnostic_tests
```

Therefore these files can support a qualified posterior-ranking scan if raw
counts independently establish alignment, but they must not be treated as fully
diagnosed or publication-ready artifacts.

Job 78 is more severely damaged. Its posterior target is readable at first,
middle, and last indices, but its posterior-predictive arrays cannot support the
common three-score analysis:

- `starting_counts` has a corrupt chunk B-tree; all sampled reads and chunk
   metadata queries fail with `wrong B-tree signature`;
- `timepoint_counts` has zero allocated chunks and returns only fill-value
   zeros; and
- the same sample-stat, observed-data, and diagnostic groups are absent.

Job 78 may support a qualified posterior-target-only ranking, but count-based
scores cannot be recovered from this diagnosed artifact.

## Available remote alternatives

The same jobs retain larger pre-diagnosis HMC files that may be useful for
recovery or reconversion:

| Job | Remote artifact | Size |
| --- | --- | ---: |
| `large-clean_72` | `pdz/cript-c/pdz_cript-c_gamma_exponential-20260407031255.nc` | 265.2 GB |
| `large-clean_76` | `pdz/cript-n/pdz_cript-n_exponential_exponential-20260407031157.nc` | 153.0 GB |
| `large-clean_78` | `pdz/cript-n/pdz_cript-n_gamma_exponential-20260407031232.nc` | 226.9 GB |

These alternatives have not yet been downloaded or validated.

## Verified controls

The following artifacts downloaded in the same batch are structurally complete
and provide controls for the checks above:

- `large-clean_70`: CRIPT-C Exponential-Exponential;
- `large-clean_73`: CRIPT-C Exponential-Sigmoid;
- `large-clean_81`: CRIPT-N Gamma-Sigmoid;
- `small-clean_3`: cis/`cript2xa3_cis` Lomax-Exponential; and
- `small-clean_4`: cis/`cript2xa3_cis` Exponential-Sigmoid.

## Follow-up

Rerun jobs 72, 76, and 78, or reconvert their original HMC files, then require each
replacement to:

1. open without superblock modification in both `h5py` and `h5netcdf`;
2. contain all posterior, posterior-predictive, sample-stat, observed-data, and
   diagnostic groups;
3. match the source count table exactly through `observed_data`;
4. retain the expected posterior target and variant dimensions; and
5. pass first/last boundary reads for every array used by the stability scanner.

Do not remove an entry from this file until the replacement artifact passes all
five checks.
