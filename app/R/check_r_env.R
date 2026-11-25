# Check R environment and required MaxEnt packages
# Usage: Rscript check_r_env.R [project_r_lib_dir]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && nzchar(args[1]) && dir.exists(args[1])) {
  .libPaths(c(normalizePath(args[1], winslash = "/"), .libPaths()))
}

pkgs <- c("dismo", "raster", "terra", "sp", "sf", "rJava", "maxnet", "jsonlite", "pROC")
pkg_status <- setNames(
  lapply(pkgs, function(p) requireNamespace(p, quietly = TRUE)),
  pkgs
)

maxent_jar_path <- ""
maxent_jar_exists <- FALSE
dismo_ready <- FALSE

if (isTRUE(pkg_status$dismo)) {
  jar <- system.file("java", "maxent.jar", package = "dismo")
  if (nzchar(jar) && file.exists(jar)) {
    maxent_jar_path <- jar
    maxent_jar_exists <- TRUE
  }
  if (maxent_jar_exists && isTRUE(pkg_status$rJava)) {
    dismo_ready <- tryCatch({
      suppressWarnings(suppressMessages(library(dismo)))
      isTRUE(maxent())
    }, error = function(e) FALSE)
  }
}

out <- list(
  r_version = R.version.string,
  lib_paths = .libPaths(),
  packages = pkg_status,
  maxent_jar_exists = maxent_jar_exists,
  maxent_jar_path = maxent_jar_path,
  dismo_ready = dismo_ready,
  maxnet_ready = isTRUE(pkg_status$maxnet)
)

if (isTRUE(pkg_status$jsonlite)) {
  cat(jsonlite::toJSON(out, auto_unbox = TRUE))
} else {
  cat(sprintf(
    '{"r_version":"%s","dismo_ready":%s,"maxnet_ready":%s,"maxent_jar_exists":%s,"jsonlite":false}',
    R.version.string,
    tolower(as.character(dismo_ready)),
    tolower(as.character(isTRUE(pkg_status$maxnet))),
    tolower(as.character(maxent_jar_exists))
  ))
}
