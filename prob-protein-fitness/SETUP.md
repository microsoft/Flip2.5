# UV Python Environment Setup for prob-protein-fitness

This project uses **UV** — a fast, modern Python package installer and resolver.

## Quick Start

### 1. Install Python (if you don't have it)

Download Python 3.10+ from [python.org](https://www.python.org/downloads/) or use:
```powershell
winget install Python.Python.3.11
```

### 2. Install UV

```powershell
pip install uv
```

Or download directly from [astral-sh/uv](https://github.com/astral-sh/uv/releases).

### 3. Create the Virtual Environment

Navigate to the repo and sync all dependencies:

```powershell
cd prob-protein-fitness
uv sync --all-extras
```

This will:
- Create a `.venv/` virtual environment
- Install all core dependencies (numpy, pandas, xarray, arviz, etc.)
- Install optional extras for notebooks, analysis tools, and development

### 4. Activate the Environment

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1

# Or CMD
.venv\Scripts\activate.bat
```

### 5. Run Jupyter

```powershell
jupyter lab notebooks/
```

Then open `pdz_cis_eda.ipynb` and run the cells.

## What's Installed

### Core Dependencies
- `numpy`, `pandas`, `xarray` — Data handling
- `netcdf4` — Read/write NetCDF files
- `arviz` — Bayesian analysis & inference diagnostics
- `scipy` — Scientific computing
- `matplotlib` — Plotting

### Optional Extras

#### Notebooks (install with `--all-extras`)
- `jupyter`, `jupyterlab` — Interactive notebooks
- `ipykernel` — Jupyter kernel

#### Analysis
- `scikit-learn` — Machine learning utilities
- `seaborn`, `plotly` — Advanced plotting
- `holoviews` — Interactive visualizations

#### Development
- `pytest` — Testing
- `black` — Code formatting
- `ruff` — Linting
- `mypy` — Type checking

### SciStanPy (Optional, from GitHub)

The collaborator's analysis uses **SciStanPy** (FLIP branch). To install:

```powershell
uv pip install git+https://github.com/microsoft/SciStanPy.git@flip
```

This is optional; the notebook will work with just `arviz` and `xarray`.

## Project Structure

```
prob-protein-fitness/
├── pyproject.toml          # Project metadata & dependencies (UV config)
├── .python-version         # Python version pinning (3.10)
├── .venv/                  # Virtual environment (created by uv sync)
├── data/                   # NC files from SciStanPy runs
├── notebooks/              # Jupyter notebooks for analysis
│   └── pdz_cis_eda.ipynb   # HMC/MLE exploratory analysis
├── scripts/                # Python scripts
├── results/                # Analysis outputs
├── figures/                # Generated plots
└── notes/                  # Documentation & notes
```

## Common UV Commands

```powershell
# Sync all dependencies
uv sync

# Sync with specific extras
uv sync --extra notebooks --extra analysis

# Add a new dependency
uv add numpy

# Add a dev dependency
uv add --dev pytest

# Run a script in the venv
uv run python script.py

# Run Jupyter
uv run jupyter lab

# Update lock file
uv lock
```

## Troubleshooting

**ModuleNotFoundError: No module named 'jupyter'**
```powershell
uv sync --all-extras
```

**Python not found**
Install Python 3.10+ and make sure it's in your PATH.

**Virtual environment issues**
Delete `.venv/` and run `uv sync` again.

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [SciStanPy Docs](https://microsoft.github.io/SciStanPy/)
- [ArviZ Documentation](https://python.arviz.org/)
