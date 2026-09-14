# Selected fit download and validation

This workflow selects the exact `Perfect!` rows in
`data/HMC Results Summaries.csv` plus the approved MLE-only exceptions: K50 at
ID 63 and the six TrpB four-site models at IDs 64-69. The analytical families
remain `small_clean` for IDs below 62 and `large_clean` for IDs 62 and above.

Those family labels are not always the remote Amulet experiment names:

- HMC IDs 1-62 are in `small-clean`; later HMC IDs are in `large-clean`.
- Standard sampled-MLE jobs are in `mle` as `mle_<ID>`.
- K50 MLE is `k50-mle:k50_mle_lr0-01`.

The scripts resolve those references automatically. Do not construct job names
from the family label. The 443.6 GB K50 sampled-MLE file was intentionally
retired locally on 2026-09-03; `data/retired_product_fits.json` records its
verified SHA-256 and exact Amulet recovery command.

## Source validation

Validate source ordering and reported-fitness alignment before accepting model
fits:

```bash
uv run python examples/trpb/validate_data.py --raw-only
uv run python examples/nuc/validate_data.py \
    --data-root /data/data_dms_stability
uv run python examples/k50/validate_data.py \
    --data-root /data/data_dms_stability
```

The Nuc validator checks generation-specific enrichment factors and exact fitted
axes. The K50 validator checks the 1,824,523-sequence fitted order and 524,836
quality-eligible aligned `dG_ML` references. The TrpB validator pins the
194,361-sequence four-site fitted count order and its 193,170 aligned reported
fitness values. No four-site HMC artifact is selected.

The publication sources, pinned releases, score definitions, and alignment
policies are recorded in [the NucB source notes](../nuc/README.md) and
[the K50 source notes](../k50/README.md).

## Preferred artifacts

Inventory local files, then refresh the read-only remote preflight:

```bash
uv run python examples/fits/inventory.py \
    --output data/fit_inventory.json
uv run python examples/fits/preflight.py --refresh
```

`data/fit_preflight.json` contains exact remote paths and human-readable sizes.
SAS URLs returned by Amulet are intentionally discarded. Preflight calls one
job at a time with `--json-tables` and checkpoints after every job.

The downloader is a dry run unless `--execute` is passed. It consumes only exact
paths from the preflight file and preserves 100 GB by default:

```bash
# Review one partition.
uv run python examples/fits/download.py \
    --family large_clean --kind diagnosed_hmc

# Execute only after reviewing paths and capacity.
uv run python examples/fits/download.py \
    --family large_clean --kind diagnosed_hmc --execute
```

Use `--job <ID> [...]` for smaller partitions and `--reserve-gb` to increase the
free-space reserve. A nonzero Amulet exit stops the batch, and each successful
call must create its exact expected destination.

The four-site MLE-only batch uses a dedicated preflight so refreshing it cannot
discard the broader preferred-artifact history:

```bash
uv run python examples/fits/preflight.py \
    --job 64 65 66 67 68 69 \
    --output data/four_site_mle_preflight.json --refresh
uv run python examples/fits/download.py \
    --preflight data/four_site_mle_preflight.json \
    --kind sampled_mle --job 64 65 66 67 68 69
```

The second command is still a dry run until `--execute` is added.

As of 2026-08-26, all 61 selected sampled-MLE artifacts are local and validated,
including the six TrpB four-site models. Diagnosed-HMC failures are archived in
`data/retired_diagnosed_hmc.json` before local removal. Unusable raw-HMC
validations are archived in `data/retired_raw_hmc.json` before local removal;
current HMC and raw-HMC outcomes are generated in `data/KNOWN_ISSUES.md`.

## Space constraints

After removing failed diagnosed and raw files, jobs 77 and 80 were downloaded
as a 218.2 GB batch. Job 80 validates; job 77 is unusable, so its validation was
archived and its local diagnosed file removed. Job 93 was then downloaded and
validates. Job 82's earlier diagnosed download is recorded as a historical
`h5netcdf` open failure; its original local file is unavailable for re-audit,
but its timestamped raw HMC validates and is retained.

`/data` now has 123.52 GB free. The downloader still reserves 100 GB, leaving
23.52 GB available for new artifacts. Two HMC downloads remain:

- `large_clean_77` raw fallback:
  `pdz/cript-n/pdz_cript-n_lomax_exponential-20260407031328.nc` (153.0 GB).
  This needs another 129.48 GB free to preserve the reserve.
- `large_clean_92` diagnosed HMC:
  `pdz/trans-2/pdz_trans-2_lomax_sigmoid_diagnosed.nc` (188.6 GB).
  This needs another 165.08 GB free to preserve the reserve.

No retained HMC artifact is currently classified `limited` or `unusable`, so
there is no additional failed HMC file to remove for either download.

## Audit and raw fallback

Audit every local artifact after each download phase:

```bash
uv run python examples/fits/audit.py
```

The audit opens HMC files with `h5py` and `h5netcdf`, checks score variables and
source-grounded dimensions, reads first and last score/predictive values, compares
embedded observed counts where available, inventories diagnostic groups, and
validates sampled-MLE point estimates and posterior samples. It writes:

- `data/fit_audit.json`, the machine-readable record; and
- `data/KNOWN_ISSUES.md`, the authoritative batch inventory and issue log.

Statuses are `validated`, `limited`, `unusable`, or `missing`. A missing local
diagnosed file must receive its preferred download first. Only a downloaded
diagnosed artifact classified `limited` or `unusable` triggers raw fallback.

Preflight those conditional raw candidates after refreshing the audit:

```bash
uv run python examples/fits/preflight_raw.py --refresh
uv run python examples/fits/download.py \
    --preflight data/raw_fit_preflight.json \
    --kind raw_hmc --n-workers 1
```

Raw downloads require `--n-workers 1` and are still dry-run by default. Execute
one exact candidate at a time, then rerun the audit:

```bash
uv run python examples/fits/download.py \
    --preflight data/raw_fit_preflight.json \
    --kind raw_hmc --job 56 --n-workers 1 --execute
uv run python examples/fits/audit.py
```

If a complete raw file is `unusable`, review and then execute its guarded
retirement:

```bash
uv run python examples/fits/retire_raw.py --job 74
uv run python examples/fits/retire_raw.py --job 74 --execute
uv run python examples/fits/inventory.py \
    --output data/fit_inventory.json
uv run python examples/fits/audit.py
```

The retirement command requires one unusable raw validation, an exact match to
the preflighted destination under `/data/amlt`, and an unambiguous remote raw
source. It writes the validation record atomically before deleting the file.

Never use a broad `*.nc` include, rename a raw file over a diagnosed file, or
repair an original during this audit. Preserve both outcomes in
`data/KNOWN_ISSUES.md`.

The initial staged pass audited 11 raw-fallback candidates. Eight are usable for
analysis: jobs 15, 25, 37, 43, 56, 71, 72, and 78. Jobs 74, 76, and 90 reached
their exact preflighted byte sizes but failed to open with both `h5py` and
`h5netcdf` because of `bad object header version number`. Their validation
evidence is retained in `data/retired_raw_hmc.json`, and their local files were
removed. The separately present raw file for job 82 also validates. Job 77 is a
new raw-fallback candidate after its diagnosed artifact failed validation; its
exact raw source is preflighted but deferred for storage.