# Archived ML-Structure Draft

Snapshot of the working paper before the v2 biology-journal restructure,
2026-09-09, based on local commit `cce8949`. The scientific text and numerical
exhibits are frozen. Only input and graphics paths in the two TeX sources
were changed to make this archive independent of the active paper.

The archive includes the three source TeX files, generated evidence and fit
tables, their provenance, and all 64 referenced figure PDFs. The previous
README and review records are retained. `SOURCE_SHA256SUMS` records the files
before path changes; `ARCHIVE_SHA256SUMS` records the buildable archive.

Build from the repository root, without running an exporter:

```bash
pdflatex -interaction=batchmode -halt-on-error -output-directory=paper/old_paper_ml_structure paper/old_paper_ml_structure/main.tex
pdflatex -interaction=batchmode -halt-on-error -output-directory=paper/old_paper_ml_structure paper/old_paper_ml_structure/main.tex
```

The active v2 uses the sources and generated outputs directly under `paper/`.
Its exporters must not write into this directory. No raw fits, sampled-score
matrices, or incidental image previews are needed to build this archive.