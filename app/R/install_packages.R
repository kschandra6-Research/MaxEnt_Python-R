# Install required R packages for MaxEnt Presence Prediction App
# Usage: Rscript install_packages.R [project_r_lib_dir]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && nzchar(args[1])) {
  local_lib <- normalizePath(args[1], winslash = "/", mustWork = FALSE)
  if (!dir.exists(local_lib)) {
    dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
  }
  .libPaths(c(local_lib, .libPaths()))
}

required_pkgs <- c(
  "dismo",
  "raster",
  "terra",
  "sp",
  "sf",
  "rJava",
  "maxnet",
  "jsonlite",
  "pROC"
)

options(repos = c(CRAN = "https://cloud.r-project.org"))

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_pkgs, installed)

if (length(missing_pkgs) == 0) {
  cat("All required R packages are already installed.\n")
} else {
  cat("Installing missing R packages:", paste(missing_pkgs, collapse = ", "), "\n")
  target_lib <- .libPaths()[1]
  install.packages(missing_pkgs, lib = target_lib, dependencies = TRUE)
}

# Verify maxent.jar availability for dismo
jar_path <- system.file("java", "maxent.jar", package = "dismo")
if (nzchar(jar_path) && file.exists(jar_path)) {
  cat("Found maxent.jar at:", jar_path, "\n")
} else {
  cat("Note: maxent.jar not found in dismo/java. The R backend can still run via maxnet, or place maxent.jar in dismo/java.\n")
}
