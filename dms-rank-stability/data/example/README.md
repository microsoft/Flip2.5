# Tiny Example

This deterministic example exercises the same score, feature, result, table,
and figure interfaces as the paper workflow without downloading external data.
It contains eight ranking candidates, one excluded source control, twelve
sampled landscapes, and all nine count-derived features.

Run it from the repository root:

```bash
uv run python analysis/run.py \
  --config data/example/analysis/runs.json \
  --result-root data/example/output/results \
  --table-root data/example/output/tables
```

The generated output directory is not tracked by Git.