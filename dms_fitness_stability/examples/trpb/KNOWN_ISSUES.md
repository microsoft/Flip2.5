# Known TrpB Fit Issues

Recorded on 2026-08-11 after downloading the 18 selected `small-clean` HMC
artifacts. These are properties of the downloaded artifacts; they have not been
repaired in this repository.

## Affected artifacts

| Job | Canonical fit | Issue |
| --- | --- | --- |
| `small-clean_15` | `libC/exponential_exponential.nc` | Missing the `observed_data` group. |
| `small-clean_25` | `libE/exponential_exponential.nc` | `h5netcdf` raises `KeyError: Unable to synchronously open object (len not positive after adjustment for EOA)`; also missing `observed_data`. |
| `small-clean_37` | `libG/exponential_exponential.nc` | Missing the `observed_data` group. |
| `small-clean_43` | `libH/exponential_exponential.nc` | Missing the `observed_data` group. |

## What was verified

- All 18 files open with `h5py`.
- Every file contains its expected posterior target with shape
  `(8, 2000, n_variants)` and the expected library width.
- The paired fit for each affected library embeds observed counts that match the
  source count table exactly: jobs 17, 27, 39, and 45 cover libC, libE, libG,
  and libH, respectively.
- The issue in job 25 is reader-specific in the current environment: its
  posterior is accessible with `h5py`, but standard `h5netcdf`/xarray loading
  should be considered unsupported until the artifact is regenerated or
  repaired.

## Current analysis behavior

The stability scanner reads posterior arrays directly with `h5py`, so all 18
fits can currently be analyzed. It does not silently treat the affected files as
healthy. Every saved result records:

- `artifact_status`: `validated` or `limited`;
- `artifact_issues`: stable issue identifiers;
- `observed_counts_status` and `count_alignment_evidence`;
- `h5netcdf_compatible` and the reader error, when present; and
- human-readable `warnings` describing the fallback and regeneration need.

The full 16,000-sample stability run completed for all 18 fits on 2026-08-11.
Results for the four affected artifacts are usable for posterior ranking but
retain `artifact_status: limited` and should carry that qualification into any
downstream report.

## Follow-up experiment

Regenerate or reconvert jobs 15, 25, 37, and 43 with a consistent NetCDF
conversion pipeline. Confirm that each replacement:

1. opens with both `h5netcdf` and `h5py`;
2. contains `observed_data/starting_counts` and
   `observed_data/timepoint_counts`;
3. matches the source count table exactly; and
4. retains the expected posterior target and dimensions.

After replacement, rerun `examples/trpb/validate_data.py`. Its observed issue
set must match `fit_jobs.json`, which forces stale issue tags to be reviewed and
removed rather than silently retained.