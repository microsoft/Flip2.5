---
name: matrix-first-refactor
description: "Simplify or refactor this repository's matrix-first preprocessing, stability analysis, mechanism reporting, contracts, runners, or compatibility code while preserving audited scientific outputs. Use for staged architectural changes, consolidation, deletion, and new-landscape support."
argument-hint: "Describe the refactoring slice"
user-invocable: true
---

# Matrix-First Refactor

Use this workflow to simplify the repository without changing its scientific
definitions or reproducibility guarantees.

## Goal

Maintain two explicit pipelines:

1. Source preprocessing converts model fits or source counts into validated
   score matrices and landscape-feature bundles.
2. Stability analysis and reporting consume only those two contracts plus
   analysis configuration.

Optimize the public path for researchers. Prefer a documented data shape and a
direct function over a class hierarchy, registry, plugin system, or wrapper.

## Required Context

Before editing:

1. Read `/memories/repo/two_layer_matrix_first_analysis_plan.md` and the current
   approved session plan when available.
2. Read `CODE_GUIDE.md` and the nearest owning implementation and test.
3. Read only the audit artifacts relevant to the proposed slice.
4. Confirm the pre-refactor Azure checkpoint or a newer verified checkpoint.

The verified 2026-09-02 pre-refactor commits are:

- Nested project: `3430a664f4af4cd3f71220ceae33253f2b5ce956`
- Azure root integration: `bae05cca33a1d78bf64fba2a5b58fab8d6468b86`

## Boundaries

Classify each behavior before moving it:

- Metadata describes the data that were produced.
- Analysis configuration states reviewed analytical choices.
- Generic algorithms operate on validated arrays and roles.
- Source adapters interpret family-specific files and model variables.
- Reporting constructs tidy numerical tables.
- Rendering consumes tidy tables and presentation configuration.
- Audits compare sources and frozen outputs.

Downstream analysis, reporting, and rendering must not branch on:

- PDZ, TrpB, NucB, K50, or another family name
- Fit job IDs or artifact kinds
- Inference-method names
- Posterior variable names or transforms
- Raw count layouts

Family-specific behavior is allowed in source adapters only when scientifically
necessary.

## Workflow

1. Identify the owning function, its nearest golden test, and the relevant
   audit surface.
2. State one falsifiable preservation hypothesis and the cheapest check that
   could disprove it.
3. Confirm a rollback point. Stage generated outputs outside canonical roots.
4. Make one small, reversible change.
5. Immediately run the narrowest executable validation.
6. Test one representative real system when the change reaches source data.
7. Run the complete applicable score, result, mechanism, or table equivalence
   audit before switching consumers.
8. Delete the old path only after its replacement passes the complete gate.
9. Run the full test, Ruff, diagnostics, manifest, and notebook gates required
   by the affected surface.
10. Update one authoritative document; link to it instead of duplicating prose.

## Simplicity Test

Add an abstraction only if it does at least one of these:

- enforces a stable public data contract
- removes meaningful duplicated implementation
- removes downstream family or source-format branching
- creates a boundary that an executable test can enforce

Reject it if it only moves constants, renames indirection, or anticipates
hypothetical extension points.

The normal researcher workflow should expose at most:

- `ScoreEnsemble`
- `LandscapeFeatureBundle`
- `StabilityResult`
- one analysis function
- one table-building function
- one plotting entry point

A researcher adding a landscape should provide standard bundles directly or
use the generic tabular producer, then use the same analysis and reporting
commands without editing package code.

## Mandatory Preservation Rules

- Do not change score transforms, candidate universes, external-reference
  definitions, cohort rules, finite-sample thresholds, tie handling, NaN
  handling, or feature definitions without explicit scientific approval.
- Preserve bounded-memory block processing for large matrices.
- Preserve source IDs, variant mappings, schema versions, content hashes,
  random seeds, and deterministic replay metadata.
- Do not overwrite canonical artifacts or immutable migration audits.
- Do not widen tolerances to hide unexplained differences.
- Do not retire fit files during the refactor.

## Fit Exemplars

The later, separately approved fit-retirement workflow protects exactly one
original fit per applicable family and modality:

- MLE: PDZ job 3, TrpB job 64, NucB job 59
- HMC: PDZ job 3, TrpB job 7, NucB job 59

The K50 job-63 sampled-MLE fit was separately retired on 2026-09-03 with its
exact recovery evidence in `data/retired_product_fits.json`. Its score bundle,
counts, features, results, caches, tables, and audits remain.

Before any non-exemplar retirement, synthetic fixtures must cover source layouts
and transforms not exercised by those seven files.

## Required Checks

Choose the checks relevant to the slice; all are required at the final gate:

- contract round trips and exact variant-axis/hash validation
- unknown lower-is-better tabular landscape end to end
- streamed versus dense numerical equality
- downstream replay with fit and source-count roots unavailable
- seven-exemplar preprocessing smoke suite and synthetic layout coverage
- full applicable source, result, mechanism, and table audits
- table-only figure regeneration and notebook validation
- README commands in a clean temporary workspace
- matrix-only installation and import
- `uv run pytest`
- `uv run ruff check .`
- clean workspace diagnostics and `git diff --check`

## Anti-Patterns

- universal raw-count tensors or abstract count loaders
- adapter, method, or schema registries for the current fixed set
- executable expressions in configuration
- public class hierarchies for simple JSON records
- full-matrix loading where a block reader is available
- path/mtime-only provenance or cache identity
- numerical analysis mixed with table construction or rendering
- permanent migration aliases and duplicate runners
- duplicated workflow prose across READMEs
- source deletion justified only by the existence of a derived artifact

## Stop Conditions

Stop and request scientific review when:

- an audited candidate or reference universe cannot be represented exactly
- a result differs outside an already documented serialization tolerance
- a proposed simplification changes a scientific definition
- staging storage cannot preserve the canonical and rollback copies
- a required source or audit baseline is unavailable