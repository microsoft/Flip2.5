# Data

Each sampled-score directory contains:

- `variants.csv`: one row per matrix column, including unique `variant_id` and
  boolean `is_candidate` fields.
- `scores.nc`: one float32 `score[sample, variant]` array.

Every score file preserves stable source-sample IDs. HMC-derived files also
preserve chain and draw coordinates. A file may contain an integer
`selection_block[variant]` coordinate when exact score ties require the
original streamed block order to reproduce retained top-K membership. This
coordinate affects only deterministic selection among exactly equal scores;
it does not change scores, candidate identities, or rank definitions.

Each landscape-feature directory contains an aligned `variants.csv` and a
`features.nc` file. Variant identity, rather than row position, joins features
to sampled scores.

The default paper-data directory is `data/downloads/`, which is not tracked by
Git. To use data stored elsewhere, pass `--data-root` to the analysis command;
see [../analysis/README.md](../analysis/README.md) for the directory layout and
rerun instructions. No download is needed when those files are already local.

## Prepared Paper Data

`files.json` defines 20 planned archive parts covering 158 score matrices and
19 feature sets. The parts are split by sampling construction and family;
large PDZ collections are split again by landscape. `download.py` verifies
archive size and SHA-256, rejects unsafe members, validates each extracted
bundle through the public data classes, and incrementally publishes complete
bundle directories.

The archive DOI, URLs, byte counts, and checksums are not assigned yet. List
the planned files without network access:

```bash
uv run python data/download.py --paper --list
```

After publication, the full download command will be:

```bash
uv run python data/download.py --paper
```

Use `--family` and `--sampling` to select a smaller set. Required feature parts
are included automatically.

## Experimental Inputs

Experimental count and reference tables are stored separately under
`raw/<family>/counts/` and `raw/<family>/reported_fitness/`. All PDZ landscapes,
including cis, trans, and CRIPT, share the `pdz` family directory. See
[../preprocessing/README.md](../preprocessing/README.md#experimental-data) for
the complete layout, required files, and current limits on preparing new data.
The prepared-matrix analysis does not read these raw files.

## Tiny Example

`data/example/` contains one tracked score bundle and one feature bundle in the
same formats as the paper data. It exercises the complete workflow without a
download; see `data/example/README.md` for the command.