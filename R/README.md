# R Integration Directory

This folder supports the optional R MaxEnt backend (`dismo` / `maxnet` / `terra` / `raster`).

- `library/`: Project-local R package library directory (populated automatically if packages are installed via `app/R/install_packages.R`).
- Core R scripts live in `app/R/`:
  - `check_r_env.R` — Checks R version, installed packages, and `maxent.jar`.
  - `install_packages.R` — Installs required CRAN packages (`dismo`, `raster`, `terra`, `sp`, `sf`, `rJava`, `maxnet`, `jsonlite`, `pROC`).
  - `run_maxent.R` — Runs `dismo::maxent()` (or `maxnet` fallback) and predicts suitability GeoTIFFs.
