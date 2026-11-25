# Run MaxEnt modeling using R (dismo::maxent with maxnet fallback)
# Usage: Rscript --vanilla run_maxent.R <config.json>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript run_maxent.R <config.json>")
}

config_path <- args[1]
if (!file.exists(config_path)) {
  stop(paste("Config file not found:", config_path))
}

# Read JSON config (ensure jsonlite is available)
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("R package 'jsonlite' is required.")
}
cfg <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)

if (!is.null(cfg$project_r_lib) && nzchar(cfg$project_r_lib) && dir.exists(cfg$project_r_lib)) {
  .libPaths(c(normalizePath(cfg$project_r_lib, winslash = "/"), .libPaths()))
}

suppressWarnings(suppressMessages({
  library(jsonlite)
  library(raster)
  library(pROC)
}))

seed <- ifelse(is.null(cfg$random_seed), 42L, as.integer(cfg$random_seed))
set.seed(seed)

train_df <- read.csv(cfg$train_csv, check.names = FALSE, stringsAsFactors = FALSE)
env_cols <- as.character(cfg$env_cols)
safe_cols <- make.names(env_cols, unique = TRUE)

# Rename columns safely for R modeling formulas/dismo
x_train <- train_df[, env_cols, drop = FALSE]
colnames(x_train) <- safe_cols
y_train <- as.integer(train_df$pa)

has_test <- !is.null(cfg$test_csv) && nzchar(cfg$test_csv) && file.exists(cfg$test_csv)
if (has_test) {
  test_df <- read.csv(cfg$test_csv, check.names = FALSE, stringsAsFactors = FALSE)
  x_test <- test_df[, env_cols, drop = FALSE]
  colnames(x_test) <- safe_cols
  y_test <- as.integer(test_df$pa)
} else {
  x_test <- x_train
  y_test <- y_train
}

has_pres_all <- !is.null(cfg$presence_csv) && nzchar(cfg$presence_csv) && file.exists(cfg$presence_csv)
if (has_pres_all) {
  pres_all_df <- read.csv(cfg$presence_csv, check.names = FALSE, stringsAsFactors = FALSE)
  x_pres_all <- pres_all_df[, env_cols, drop = FALSE]
  colnames(x_pres_all) <- safe_cols
} else {
  x_pres_all <- NULL
}

feat_classes <- tolower(as.character(cfg$feature_classes))
if (length(feat_classes) == 0) {
  feat_classes <- c("linear")
}
reg_mult <- ifelse(is.null(cfg$regularization), 1.0, as.numeric(cfg$regularization))
max_iter <- ifelse(is.null(cfg$max_iterations), 500L, as.integer(cfg$max_iterations))
conv_thr <- ifelse(is.null(cfg$convergence_threshold), 1e-5, as.numeric(cfg$convergence_threshold))
do_jackknife <- isTRUE(cfg$do_jackknife)

compute_auc <- function(y_true, y_pred) {
  if (length(unique(y_true)) < 2 || length(unique(y_pred)) < 2) {
    return(0.5)
  }
  tryCatch({
    as.numeric(pROC::auc(pROC::roc(y_true, y_pred, quiet = TRUE)))
  }, error = function(e) 0.5)
}

compute_corr <- function(y_true, y_pred) {
  if (length(unique(y_true)) < 2 || stats::sd(y_pred, na.rm = TRUE) == 0) {
    return(0.0)
  }
  val <- suppressWarnings(stats::cor(y_true, y_pred))
  if (is.na(val)) 0.0 else as.numeric(val)
}

backend_used <- NULL
model_obj <- NULL

# Try dismo::maxent first if maxent.jar and rJava work
dismo_available <- requireNamespace("dismo", quietly = TRUE) &&
  requireNamespace("rJava", quietly = TRUE) &&
  nzchar(system.file("java", "maxent.jar", package = "dismo")) &&
  file.exists(system.file("java", "maxent.jar", package = "dismo"))

if (dismo_available) {
  dismo_ok <- tryCatch({
    suppressWarnings(suppressMessages(library(dismo)))
    isTRUE(dismo::maxent())
  }, error = function(e) FALSE)

  if (dismo_ok) {
    maxent_args <- c(
      "autofeature=false",
      paste0("linear=", ifelse("linear" %in% feat_classes, "true", "false")),
      paste0("quadratic=", ifelse("quadratic" %in% feat_classes, "true", "false")),
      paste0("product=", ifelse("product" %in% feat_classes, "true", "false")),
      paste0("threshold=", ifelse("threshold" %in% feat_classes, "true", "false")),
      paste0("hinge=", ifelse("hinge" %in% feat_classes, "true", "false")),
      paste0("betamultiplier=", reg_mult),
      paste0("maximumiterations=", max_iter),
      paste0("convergencethreshold=", format(conv_thr, scientific = FALSE)),
      "randomseed=true",
      paste0("jackknife=", ifelse(do_jackknife && length(safe_cols) > 1, "true", "false")),
      "outputformat=cloglog"
    )
    model_path <- file.path(cfg$work_dir, "dismo_maxent_out")
    dir.create(model_path, recursive = TRUE, showWarnings = FALSE)
    model_obj <- dismo::maxent(x = x_train, p = y_train, args = maxent_args, path = model_path)
    backend_used <- "R (dismo::maxent)"
  }
}

# Fallback to maxnet if dismo/Java is unavailable
if (is.null(model_obj)) {
  if (!requireNamespace("maxnet", quietly = TRUE)) {
    stop("Neither dismo::maxent (with rJava/maxent.jar) nor maxnet is available in R.")
  }
  suppressWarnings(suppressMessages(library(maxnet)))
  class_map <- c(linear = "l", quadratic = "q", product = "p", threshold = "t", hinge = "h")
  m_classes <- paste(unname(class_map[intersect(feat_classes, names(class_map))]), collapse = "")
  if (!nzchar(m_classes)) m_classes <- "l"
  # Avoid product with single variable
  if (length(safe_cols) < 2) {
    m_classes <- gsub("p", "", m_classes)
    if (!nzchar(m_classes)) m_classes <- "l"
  }
  fml <- maxnet::maxnet.formula(y_train, x_train, classes = m_classes)
  model_obj <- maxnet::maxnet(p = y_train, data = x_train, f = fml, regmult = reg_mult)
  backend_used <- "R (maxnet)"
}

predict_df <- function(mod, newdata, backend_name) {
  if (grepl("dismo", backend_name, fixed = TRUE)) {
    as.numeric(dismo::predict(mod, newdata))
  } else {
    as.numeric(stats::predict(mod, newdata, type = "cloglog"))
  }
}

train_pred <- predict_df(model_obj, x_train, backend_used)
test_pred <- if (has_test) predict_df(model_obj, x_test, backend_used) else train_pred
pres_all_pred <- if (!is.null(x_pres_all)) predict_df(model_obj, x_pres_all, backend_used) else numeric(0)

# Extract full R model parameters/results table
r_results_list <- list()
if (grepl("dismo", backend_used, fixed = TRUE)) {
  res_mat <- model_obj@results
  for (rn in rownames(res_mat)) {
    val <- as.numeric(res_mat[rn, 1])
    if (is.finite(val)) {
      r_results_list[[rn]] <- val
    }
  }
} else {
  if (!is.null(model_obj$entropy)) r_results_list[["Entropy"]] <- as.numeric(model_obj$entropy)
  if (!is.null(model_obj$alpha)) r_results_list[["Alpha"]] <- as.numeric(model_obj$alpha)
  if (!is.null(model_obj$betas)) {
    for (bn in names(model_obj$betas)) {
      r_results_list[[paste0("beta.", bn)]] <- as.numeric(model_obj$betas[[bn]])
    }
  }
}

train_auc <- compute_auc(y_train, train_pred)
test_auc <- compute_auc(y_test, test_pred)
train_corr <- compute_corr(y_train, train_pred)
test_corr <- compute_corr(y_test, test_pred)

# Variable importance
var_importance <- setNames(rep(0.0, length(env_cols)), env_cols)
if (grepl("dismo", backend_used, fixed = TRUE)) {
  res_mat <- model_obj@results
  rnames <- rownames(res_mat)
  for (idx in seq_along(env_cols)) {
    key_perm <- paste0(safe_cols[idx], ".permutation.importance")
    key_contrib <- paste0(safe_cols[idx], ".contribution")
    if (key_perm %in% rnames) {
      var_importance[[env_cols[idx]]] <- as.numeric(res_mat[key_perm, 1])
    } else if (key_contrib %in% rnames) {
      var_importance[[env_cols[idx]]] <- as.numeric(res_mat[key_contrib, 1])
    }
  }
}
# If permutation importance is all zero (e.g., maxnet or edge case), compute permutation AUC importance
if (sum(unlist(var_importance)) <= 0) {
  base_auc <- test_auc
  if (base_auc > 0) {
    for (idx in seq_along(env_cols)) {
      sc <- safe_cols[idx]
      x_perm <- x_test
      x_perm[[sc]] <- sample(x_perm[[sc]])
      perm_pred <- predict_df(model_obj, x_perm, backend_used)
      perm_auc <- compute_auc(y_test, perm_pred)
      var_importance[[env_cols[idx]]] <- max(0.0, (base_auc - perm_auc) / base_auc * 100.0)
    }
  }
}
tot_imp <- sum(unlist(var_importance))
if (tot_imp > 0) {
  for (nm in names(var_importance)) {
    var_importance[[nm]] <- var_importance[[nm]] / tot_imp * 100.0
  }
}

# Predict suitability raster if raster_stack_tif is provided
prediction_tif <- NULL
if (!is.null(cfg$raster_stack_tif) && nzchar(cfg$raster_stack_tif) && file.exists(cfg$raster_stack_tif)) {
  r_stack <- raster::stack(cfg$raster_stack_tif)
  names(r_stack) <- safe_cols
  prediction_tif <- cfg$prediction_tif
  if (grepl("dismo", backend_used, fixed = TRUE)) {
    raster::predict(r_stack, model_obj, filename = prediction_tif, overwrite = TRUE)
  } else {
    raster::predict(r_stack, model_obj, type = "cloglog", filename = prediction_tif, overwrite = TRUE)
  }
}

# Response curves
response_curves <- list()
if (isTRUE(cfg$do_response_curves)) {
  means <- lapply(x_train, function(col) mean(col, na.rm = TRUE))
  n_pts <- 100L
  for (idx in seq_along(env_cols)) {
    sc <- safe_cols[idx]
    orig <- env_cols[idx]
    vals <- x_train[[sc]]
    vals <- vals[is.finite(vals)]
    if (length(vals) == 0) next
    x_seq <- seq(min(vals), max(vals), length.out = n_pts)
    synth <- as.data.frame(lapply(means, function(m) rep(m, n_pts)))
    colnames(synth) <- safe_cols
    synth[[sc]] <- x_seq
    y_seq <- predict_df(model_obj, synth, backend_used)
    response_curves[[orig]] <- list(x = as.numeric(x_seq), y = as.numeric(y_seq))
  }
}

# Jackknife results
jackknife_results <- list()
if (do_jackknife && length(env_cols) >= 1) {
  if (grepl("dismo", backend_used, fixed = TRUE)) {
    res_mat <- model_obj@results
    rnames <- rownames(res_mat)
    full_gain <- if ("Regularized.training.gain" %in% rnames) as.numeric(res_mat["Regularized.training.gain", 1]) else train_auc
    if (length(env_cols) == 1) {
      jackknife_results[[1]] <- list(
        variable = env_cols[1],
        gain_without = 0.0,
        gain_with_only = full_gain,
        full_gain = full_gain,
        metric = "Regularized training gain (single-layer baseline)"
      )
    } else {
      for (idx in seq_along(env_cols)) {
        sc <- safe_cols[idx]
        orig <- env_cols[idx]
        wo_key <- paste0("Training.gain.without.", sc)
        only_key <- paste0("Training.gain.with.only.", sc)
        gain_wo <- if (wo_key %in% rnames) as.numeric(res_mat[wo_key, 1]) else 0.0
        gain_only <- if (only_key %in% rnames) as.numeric(res_mat[only_key, 1]) else 0.0
        jackknife_results[[length(jackknife_results) + 1]] <- list(
          variable = orig,
          gain_without = gain_wo,
          gain_with_only = gain_only,
          full_gain = full_gain,
          metric = "Regularized training gain"
        )
      }
    }
  } else {
    full_gain <- train_auc
    if (length(env_cols) == 1) {
      jackknife_results[[1]] <- list(
        variable = env_cols[1],
        gain_without = 0.5,
        gain_with_only = full_gain,
        full_gain = full_gain,
        metric = "Training AUC (single-layer baseline)"
      )
    } else {
      for (idx in seq_along(env_cols)) {
        sc <- safe_cols[idx]
        orig <- env_cols[idx]
        other_cols <- setdiff(safe_cols, sc)
        gain_wo <- tryCatch({
          fml_wo <- maxnet::maxnet.formula(y_train, x_train[, other_cols, drop = FALSE], classes = "lq")
          m_wo <- maxnet::maxnet(p = y_train, data = x_train[, other_cols, drop = FALSE], f = fml_wo, regmult = reg_mult)
          compute_auc(y_train, as.numeric(stats::predict(m_wo, x_train[, other_cols, drop = FALSE], type = "cloglog")))
        }, error = function(e) 0.0)
        gain_only <- tryCatch({
          fml_only <- maxnet::maxnet.formula(y_train, x_train[, sc, drop = FALSE], classes = "lq")
          m_only <- maxnet::maxnet(p = y_train, data = x_train[, sc, drop = FALSE], f = fml_only, regmult = reg_mult)
          compute_auc(y_train, as.numeric(stats::predict(m_only, x_train[, sc, drop = FALSE], type = "cloglog")))
        }, error = function(e) 0.0)
        jackknife_results[[length(jackknife_results) + 1]] <- list(
          variable = orig,
          gain_without = gain_wo,
          gain_with_only = gain_only,
          full_gain = full_gain,
          metric = "Training AUC"
        )
      }
    }
  }
}

output <- list(
  backend = backend_used,
  train_auc = train_auc,
  test_auc = test_auc,
  train_correlation = train_corr,
  test_correlation = test_corr,
  y_train = as.integer(y_train),
  y_test = as.integer(y_test),
  train_predictions = as.numeric(train_pred),
  test_predictions = as.numeric(test_pred),
  presence_train_predictions = as.numeric(train_pred[y_train == 1]),
  presence_test_predictions = as.numeric(test_pred[y_test == 1]),
  presence_all_predictions = as.numeric(pres_all_pred),
  r_results = r_results_list,
  variable_importance = as.list(var_importance),
  response_curves = response_curves,
  jackknife_results = jackknife_results,
  prediction_tif = prediction_tif
)

jsonlite::write_json(output, cfg$result_json, auto_unbox = TRUE, digits = 8)
cat("R MaxEnt completed successfully using:", backend_used, "\n")
