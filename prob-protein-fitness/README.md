# Probabilistic Analysis of Protein Fitness

Analysis of PDZ domain protein fitness using Bayesian inference with SciStanPy.

## Data

- `pdz_cis_exponential_exponential_diagnosed.nc`: HMC posterior samples with MCMC diagnostics
- `pdz_cis_exponential_exponential_samples.nc`: MLE results with parametric bootstrap samples

## Quick Start

```bash
uv sync --all-extras
source .venv/Scripts/activate  # or .\.venv\Scripts\Activate.ps1 on PowerShell
jupyter lab notebooks/pdz_cis_eda.ipynb
```

## Project Structure

```
prob-protein-fitness/
├── data/              # Input data files (NetCDF)
├── notebooks/         # Jupyter notebooks for analysis
├── results/           # Output results
├── figures/           # Generated plots and figures
├── notes/             # Notes with ideas/explanations
├── scripts/           # Standalone Python scripts
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

## Dependencies

Core: numpy, pandas, xarray, netcdf4, arviz, scipy, matplotlib

Optional:
- **notebooks**: jupyter, jupyterlab, ipykernel
- **analysis**: scikit-learn, seaborn, plotly, holoviews
- **dev**: pytest, black, ruff, mypy

## Notebooks

### `pdz_cis_eda.ipynb`
Exploratory data analysis of PDZ cis exponential-exponential model:
- Load HMC posterior or MLE bootstrap samples
- Inspect dataset structure and dimensions
- Display key variables and statistics
- MCMC health checks (divergences, E-BFMI, tree depth)

### `pdz_linear_baseline_results.ipynb`
Visualizes Ridge baseline correlations produced by the linear baseline scripts.

Bruce's `cis_from_bruce_azure.tsv` is nucleotide/barcode-level data for the 10,660-row NetCDF cis run. The minimal target-resample script groups it by sorted `aa_seq`, checks that summed observed counts exactly match the MLE NetCDF observed data, then fits Kevin/FLIP-style Ridge regression on flattened one-hot encodings. Change `TARGET_VARIABLE` in the script to `log_r_mean`, `r`, or `log_r`.

```powershell
.\.venv312\Scripts\python.exe scripts/run_pdz_target_resample_baseline.py `
	--resample-source both `
	--n-resamples 100 `
	--output-dir results\pdz_cis_target_resample_baseline
```

Outputs are written to `correlations.csv` and `summary.csv` in the selected output directory.

## License

MIT
