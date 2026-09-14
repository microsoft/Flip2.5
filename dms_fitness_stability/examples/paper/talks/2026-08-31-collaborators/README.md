# Collaborator presentation, 31 August 2026

- `slides.pdf`: presentation-ready 16:9 deck.
- `slides.tex`: Beamer source using the paper outline's Palatino-style font and
  blue/gray palette.
- `SPEAKER_NOTES.md`: 19.5-minute main talk route and likely questions.
- `build_assets.py`: reproducibly generates the five compact result panels from
  the final HMC and MLE-predictive tables.
- `assets/`: generated PDF/PNG panels used by the deck.

The main talk is slides 1-12. Slides 13-15 are backup.

From the repository root, rebuild with:

```bash
uv run python \
  examples/paper/talks/2026-08-31-collaborators/build_assets.py
cd examples/paper/talks/2026-08-31-collaborators
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
```

The deck reads only generated paper tables. It does not open fit artifacts or
modify the paper notebooks.# Collaborator presentation, 31 August 2026

The main talk is slides 1-12 and is paced for 18-20 minutes. Slides 13-15 are backup material.

Build the table-derived assets and presentation from this directory:

```bash
uv run python build_assets.py
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
```

Files:

- `slides.pdf`: compiled 16:9 presentation;
- `slides.tex`: Beamer source using the paper-outline font and color palette;
- `notes.md`: timed talk track and likely-question answers;
- `build_assets.py`: reproducible producer for the five concise result panels;
- `assets/`: vector and PNG versions of those panels.

The slide-specific plots read the final HMC and MLE-predictive tables under `examples/paper/output`. They do not modify the paper notebooks or tables.