# ======================================================================
# RESULTS STORAGE
# ======================================================================

results_summary_full <- data.frame(
  Dataset                = character(),
  Thiessen_Uniform_Score = numeric(),
  Thiessen_Dirichlet_Score = numeric(),
  Thiessen_Adaptive_Score = numeric(),
  Uniform_BART_Score     = numeric(),
  Dirichlet_BART_Score   = numeric(),
  RF_Score               = numeric(),
  # SVM_Score              = numeric(),
  XGB_Score              = numeric(),
  baseline_score         = numeric(),
  stringsAsFactors = FALSE
)

# ======================================================================
# GLOBAL SETTINGS & HYPERPARAMETERS
# ======================================================================
library(glmnet)

n_runs    <- 20
num_cores <- 20L

### PERFORMANCE FIX: SPLIT MCMC ITERATIONS ###
MCMC_ITER_CV        <- 50000
MCMC_BURNIN_CV      <- 40000
DIRICHLET_WARMUP_CV <- 2000

MOMENTUM_SETTLE_CV      <- 10000L
MOMENTUM_WINDOW_CV      <- 20000

MCMC_ITER_FINAL        <- 100000
MCMC_BURNIN_FINAL      <- 80000
DIRICHLET_WARMUP_FINAL <- 2000
MOMENTUM_SETTLE_FINAL  <- 20000
MOMENTUM_WINDOW_FINAL  <- 40000
# Keep CV tuning lightweight while final fits retain multiple chains.
ADAPTIVE_CV_CHAINS     <- 1L
ADAPTIVE_FINAL_CHAINS  <- 5L
ADAPTIVE_FINAL_THREADS <- 5L

ADAPTIVE_BURNIN_FINAL_small <- 80000L
ADAPTIVE_BURNIN_FINAL_medium <- 80000L
ADAPTIVE_BURNIN_FINAL_large <- 80000L
ADAPTIVE_DRAWS <- 20000L


# ======================================================================
# PARALLEL SETUP
# ======================================================================

library(doParallel)
library(doRNG)

reproducible <- TRUE
EXPECTED_R_VERSION <- "4.6.1"
EXPECTED_PACKAGE_VERSIONS <- c(
  glmnet = "5.0",
  doParallel = "1.0.17",
  doRNG = "1.8.6.3",
  devtools = "2.5.2",
  flexBART = "2.0.6",
  dbarts = "0.9.34",
  ranger = "0.18.0",
  # e1071 = "1.7.17",
  xgboost = "3.2.1.1"
)
EXPECTED_THIESSEN_VERSION <- "0.0.0.9000"

assert_benchmark_versions <- function() {
  actual_r_version <- as.character(getRversion())
  if (actual_r_version != EXPECTED_R_VERSION) {
    stop("Expected R ", EXPECTED_R_VERSION, "; found R ", actual_r_version, ".")
  }

  actual_package_versions <- vapply(names(EXPECTED_PACKAGE_VERSIONS), function(package) {
    if (!requireNamespace(package, quietly = TRUE)) return(NA_character_)
    as.character(utils::packageVersion(package))
  }, character(1))
  mismatched <- is.na(actual_package_versions) |
    actual_package_versions != EXPECTED_PACKAGE_VERSIONS
  if (any(mismatched)) {
    details <- sprintf(
      "%s: expected %s, found %s",
      names(EXPECTED_PACKAGE_VERSIONS)[mismatched],
      EXPECTED_PACKAGE_VERSIONS[mismatched],
      ifelse(is.na(actual_package_versions[mismatched]), "not installed",
             actual_package_versions[mismatched])
    )
    stop("Benchmark package version mismatch: ", paste(details, collapse = "; "))
  }
}

if (reproducible) assert_benchmark_versions()

SERIAL_SEED <- 142L
FOREACH_SEED <- 901200L
RNGkind(kind = "L'Ecuyer-CMRG", normal.kind = "Inversion", sample.kind = "Rejection")
set.seed(SERIAL_SEED)

dataset_seed_for <- function(data_name) {
  seed <- as.double(SERIAL_SEED)
  for (code_point in utf8ToInt(enc2utf8(data_name))) {
    seed <- (seed * 31 + code_point) %% (.Machine$integer.max - 1)
  }
  as.integer(seed + 1)
}

# Fix: Force PSOCK even on Unix to prevent C++/Rust memory deadlocks and OOM kills.
cluster_type <- "PSOCK"
#Sys.setenv(THIESSEN_EXPERIMENTAL = "1")
#devtools::load_all("/home/jd191252/thiessen/r", recompile = TRUE)
#stopifnot(core_experimental())
cl <- makeCluster(num_cores, type = cluster_type)
registerDoParallel(cl)
registerDoRNG(seed = FOREACH_SEED)

clusterExport(
  cl,
  varlist = c(
    "reproducible",
    "EXPECTED_R_VERSION", "EXPECTED_PACKAGE_VERSIONS",
    "EXPECTED_THIESSEN_VERSION", "assert_benchmark_versions"
  ),
  envir = environment()
)

clusterEvalQ(cl, {
  Sys.setenv(
    THIESSEN_EXPERIMENTAL = "1",
    OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    VECLIB_MAXIMUM_THREADS = "1"
  )
  options(mc.cores = 1L)
  devtools::load_all("/home/jd191252/thiessen/r", recompile = FALSE)
  stopifnot(core_experimental())
  library(flexBART)
  library(dbarts)
  library(ranger)
  # library(e1071)
  library(xgboost)
  if (reproducible) {
    assert_benchmark_versions()
    actual_thiessen_version <- as.character(getNamespaceVersion("thiessen"))
    if (actual_thiessen_version != EXPECTED_THIESSEN_VERSION) {
      stop("Expected thiessen ", EXPECTED_THIESSEN_VERSION,
           "; found thiessen ", actual_thiessen_version, ".")
    }
  }
})

# Export global MCMC constants once — never re-exported by foreach
clusterExport(cl,
              varlist = c("MCMC_ITER_CV",    "MCMC_BURNIN_CV",    "DIRICHLET_WARMUP_CV",
                         "MOMENTUM_SETTLE_CV", "MOMENTUM_WINDOW_CV",
                          "ADAPTIVE_CV_CHAINS", "ADAPTIVE_FINAL_CHAINS", "ADAPTIVE_FINAL_THREADS",
                          "MCMC_ITER_FINAL", "MCMC_BURNIN_FINAL", "DIRICHLET_WARMUP_FINAL",
                          "MOMENTUM_SETTLE_FINAL", "MOMENTUM_WINDOW_FINAL"),
              envir = environment())

# ======================================================================
# CV HYPERPARAMETER GRIDS
# ======================================================================
{
  # --- Thiessen: Uniform inclusion ---
  base_grid_0 <- expand.grid(
    m          = c(50,200),
    k          = c(1, 3),
    Omega      = c(1.5,3),
    LambdaRate = c(2,4)
  )
  nu_q_cases  <- data.frame(nu = c(3, 6), q = c(0.99, 0.85))
  grid_mode0  <- merge(base_grid_0, nu_q_cases, by = NULL)
  
  # --- Thiessen: Dirichlet inclusion ---
  base_grid_1 <- expand.grid(
    m = c(200), k = c(1, 3), Omega = c(1.5),
    LambdaRate = c(2,4)
  )
  grid_mode_alpha <- data.frame(
    alpha = c(1), a_alpha = c(0.5,1), b_alpha = c(1)
  )
  #grid_mode1 <- merge(merge(base_grid_1,nu_q_cases), grid_mode_alpha, by = NULL)
  grid_mode1 <- merge(merge(base_grid_1,nu_q_cases, by = NULL), grid_mode_alpha, by = NULL)

  # Adaptive Dirichlet reuses the selected Dirichlet hyperparameters; its
  # adaptive-only settings are tuned separately below.
  adaptive_grid <-merge(data.frame(adapt_boost = c(0.05,0.1,0.6,1.2),
      adapt_penalty = c(0.04, 0.08, 0.5, 1)),
  
      expand.grid(
        momentum_decay = c(0.01,0.1,0.2,0.3,0.4)
      )
    )
  
  # --- Uniform BART ---
  grid_uBART_base <- expand.grid(
    ntree = c(50,200), base = c(0.95, 0.99), power = c(2.0, 3.0),
    k = c(1.0, 3.0)
  )
  uBART_prior_cases <- data.frame(
    sigdf = c(3, 6), sigquant = c(0.99, 0.85)
  )
  grid_uBART <- merge(grid_uBART_base, uBART_prior_cases, by = NULL)
  
  # --- Dirichlet BART ---
  grid_dBART_base <- expand.grid(
    ntree = c(200), base = c(0.95, 0.99), power = c(2.0, 3.0),
    k = c(1.0, 3.0), a_u = c(0.5, 1.0), b_u = 1.0
  )
  dBART_prior_cases <- data.frame(
    nu = c(3, 6), sigquant = c(0.99, 0.85)
  )
  grid_dBART <- merge(grid_dBART_base, dBART_prior_cases, by = NULL)
  
  # --- Random Forest ---
  grid_rf <- expand.grid(
    mtry_prop       = c(0.2, 0.4, 0.6, 0.8),
    min.node.size   = c(1, 3, 5, 10),
    sample.fraction = c(0.632, 0.8),
    num.trees       = c(250, 500),
    replace         = c(TRUE, FALSE),
    use_extratrees  = c(TRUE, FALSE)
  )
  
  # --- SVM (disabled) ---
  # grid_svm_radial <- expand.grid(
  #   kernel = "radial", cost = c(0.1, 1, 10, 50),
  #   gamma = c(0.001, 0.01, 0.1, 0.5), degree = 3, coef0 = 0,
  #   epsilon = c(0.01, 0.1)
  # )
  # grid_svm_poly <- expand.grid(
  #   kernel = "polynomial", cost = c(0.1, 1, 10, 50),
  #   gamma = c(0.01, 0.1), degree = c(2, 3), coef0 = c(0, 1),
  #   epsilon = c(0.01, 0.1)
  # )
  # grid_svm <- rbind(grid_svm_radial, grid_svm_poly)
  
  # --- XGBoost ---
  grid_xgb <- expand.grid(
    nrounds = c(100, 300), max_depth = c(3, 5, 7),
    learning_rate = c(0.01, 0.05, 0.1),
    subsample = c(0.7, 1.0), colsample_bytree = c(0.7, 1.0)
  )
}

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
createFolds <- function(y, k = 5, returnTrain = TRUE) {
  n                <- length(y)
  shuffled_indices <- sample(1:n)
  folds            <- split(shuffled_indices,
                            cut(seq_along(shuffled_indices), breaks = k, labels = FALSE))
  if (returnTrain) {
    train_folds        <- lapply(folds, function(test_idx) setdiff(1:n, test_idx))
    names(train_folds) <- paste0("Fold", 1:k)
    return(train_folds)
  } else {
    names(folds) <- paste0("Fold", 1:k)
    return(folds)
  }
}

get_cv_data <- function(X_train, Y_train, train_idx) {
  X_cv_train <- X_train[train_idx, , drop = FALSE]
  X_cv_val   <- X_train[-train_idx, , drop = FALSE]
  Y_cv_train <- Y_train[train_idx]
  Y_cv_val   <- Y_train[-train_idx]
  non_const  <- apply(X_cv_train, 2, function(col) length(unique(col)) > 1)
  list(
    X_train = X_cv_train[, non_const, drop = FALSE],
    X_val   = X_cv_val[,  non_const, drop = FALSE],
    Y_train = Y_cv_train,
    Y_val   = Y_cv_val
  )
}

thiessen_control_for <- function(params, prior, is_classification,
                                 burn_in, draws, dirichlet_warmup, momentum_settle, momentum_window, lasso_sigma = NULL) {
  value_or <- function(name, default = NULL) {
    value <- params[[name]]
    if (is.null(value) || length(value) == 0L || is.na(value)) default else value
  }

  inclusion <- switch(
    prior,
    uniform = "uniform",
    dirichlet = dart_inclusion(
      alpha = value_or("alpha", 1.0),
      a_alpha = value_or("a_alpha", 0.5),
      b_alpha = value_or("b_alpha", 1.0),
      dirichlet_warmup = dirichlet_warmup
    ),
    adaptive = dart_inclusion(
      alpha = value_or("alpha", 1.0),
      a_alpha = value_or("a_alpha", 0.5),
      b_alpha = value_or("b_alpha", 1.0),
      adapt_boost = value_or("adapt_boost", 0),
      adapt_penalty = value_or("adapt_penalty", 0),
      momentum_decay = value_or("momentum_decay", 0.3),
      dirichlet_warmup = dirichlet_warmup,
      momentum_settle = momentum_settle,
      momentum_window = momentum_window
    ),
    stop("Unknown Thiessen prior: ", prior)
  )

  outcome <- if (is_classification) {
    probit_outcome()
  } else {
    gaussian_outcome(nu = value_or("nu", 6), q = value_or("q", 0.85), sigma_hat = lasso_sigma)
  }

  thiessen_control(
    outcome = outcome,
    mean_params = term_params(
      tessellations = value_or("m", 200),
      k = value_or("k", 3),
      lambda_c = value_or("LambdaRate", 4),
      structure = structure_params(
        omega = value_or("Omega"),
        inclusion = inclusion
      )
    ),
    general_params = general_params(
      burn_in = burn_in, draws = draws, thinning = 1
    )
  )
}

fit_thiessen_cv <- function(cv, params, prior, is_classification,
                            burn_in, draws, dirichlet_warmup = 0L,
                            momentum_settle = 0L, momentum_window = 1L,
                            lasso_sigma = NULL,
                            chains = ADAPTIVE_FINAL_CHAINS, threads = 1L) {
  
  # Compute Lasso Sigma for regression
  if (!is_classification && is.null(lasso_sigma)) {
    fit_lasso <- cv.glmnet(as.matrix(cv$X_train), cv$Y_train)
    best_idx <- which(fit_lasso$lambda == fit_lasso$lambda.min)
    lasso_sigma <- sqrt(fit_lasso$cvm[best_idx])
  }
  model <- thiessen(
    cv$X_train, cv$Y_train, xtest = cv$X_val,
    seed = sample.int(.Machine$integer.max, 1L),
    control = thiessen_control_for(
      params, prior, is_classification, burn_in, draws, dirichlet_warmup, momentum_settle, momentum_window, lasso_sigma
    ),
    chains = chains,
    threads = threads
  )  
  score <- if (is_classification) mean(round(model) == cv$Y_val) else
    sqrt(mean((model - cv$Y_val)^2))

  # Delete model to free memory inside the worker
  rm(model)
  score
}

fit_flexbart_chains <- function(x_train, y_train, x_test, is_classification,
                                ntree, base, power, ndpost, nskip,
                                a_u, b_u, k, nu, sigquant,
                                chains = ADAPTIVE_FINAL_CHAINS) {
  response <- if (is_classification) as.integer(y_train) else as.numeric(y_train)
  y_range <- if (is_classification) 1 else diff(range(response)) / stats::sd(response)
  tau_vec <- y_range / (2 * k * sqrt(ntree))
  train_data <- as.data.frame(x_train, check.names = FALSE)
  train_data$Y <- response
  test_data <- as.data.frame(x_test, check.names = FALSE)
  rm(x_train, x_test, y_train, response)
  family <- if (is_classification) stats::binomial(link = "probit") else stats::gaussian()

  model <- flexBART::flexBART(
    formula = Y ~ bart(.),
    train_data = train_data,
    test_data = test_data,
    family = family,
    M_vec = as.integer(ntree),
    alpha_vec = base,
    beta_vec = power,
    sparse = TRUE,
    nest_v = FALSE,
    a_u = a_u,
    b_u = b_u,
    tau_vec = tau_vec,
    nu = nu,
    sigquant = sigquant,
    n.chains = as.integer(chains),
    nd = as.integer(ndpost),
    burn = as.integer(nskip),
    thin = 1L,
    save_samples = FALSE,
    save_trees = FALSE,
    verbose = FALSE
  )
  predictions <- if (is_classification) model$prob.test.mean else model$yhat.test.mean
  rm(model)
  as.vector(predictions)
}

fit_uniform_dbarts <- function(x_train, y_train, x_test, is_classification,
                               ntree, base, power, k, ndpost, nskip,
                               chains = ADAPTIVE_FINAL_CHAINS, threads = 1L,
                               sigma_estimate = NA_real_,
                               sigdf = 3, sigquant = 0.90,
                               prediction_batch_size = 128L) {
  sampler_seed <- sample.int(.Machine$integer.max, 1L)
  sampler <- dbarts::bart(
    x.train = as.matrix(x_train),
    y.train = as.numeric(y_train),
    x.test = as.matrix(x_test),
    ntree = ntree,
    base = base,
    power = power,
    k = k,
    sigdf = sigdf,
    sigquant = sigquant,
    seed = sampler_seed,
    sigest = sigma_estimate,
    ndpost = ndpost,
    nskip = nskip,
    nchain = as.integer(chains),
    nthread = as.integer(threads),
    keeptrainfits = FALSE,
    keeptrees = FALSE,
    sampleronly = TRUE,
    verbose = FALSE
  )
  sampler$run(numBurnIn = as.integer(nskip), numSamples = 0L)

  prediction_sum <- numeric(nrow(x_test))
  draws_remaining <- as.integer(ndpost)
  while (draws_remaining > 0L) {
    batch_size <- min(as.integer(prediction_batch_size), draws_remaining)
    samples <- sampler$run(numBurnIn = 0L, numSamples = batch_size)
    test_draws <- matrix(samples$test, nrow = nrow(x_test))
    if (is_classification) test_draws <- stats::pnorm(test_draws)
    prediction_sum <- prediction_sum + rowSums(test_draws)
    draws_remaining <- draws_remaining - batch_size
    rm(samples, test_draws)
  }

  rm(sampler)
  gc()
  prediction_sum / (as.integer(ndpost) * as.integer(chains))
}

format_elapsed <- function(seconds) {
  hours <- floor(seconds / 3600)
  minutes <- floor((seconds %% 3600) / 60)
  remaining <- seconds %% 60
  sprintf("%02d:%02d:%05.2f", hours, minutes, remaining)
}

clusterExport(
  cl,
  varlist = c(
    "thiessen_control_for", "fit_thiessen_cv", "fit_flexbart_chains",
    "fit_uniform_dbarts"
  ),
  envir = environment()
)

# ======================================================================
# MAIN BENCHMARK LOOP
# ======================================================================

for (data_name in names(benchmark_datasets)) {
  dataset_seed <- dataset_seed_for(data_name)
  set.seed(dataset_seed)
  registerDoParallel(cl)
  registerDoRNG(seed = dataset_seed)
  
  # Remove large dataset objects before exporting the next dataset. gc() alone
  # cannot reclaim objects that are still referenced in worker globals.
  clusterEvalQ(cl, {
    old_objects <- c(
      "benchmark_datasets", "data_combined", "X_train", "X_test",
      "Y_train", "Y_test", "training_data", "train_set", "test_set",
      "X", "Y", "current_data", "cv_folds", "cv_fold_data",
      "cv_lasso_sigma", "precomputed_cv_data", "final_splits"
    )
    rm(list = intersect(old_objects, ls(envir = .GlobalEnv)), envir = .GlobalEnv)
    gc()
  })
  
  cat("\nAnalysing dataset:", data_name, "\n")
  current_data <- benchmark_datasets[[data_name]]
  
  Y <- current_data$Y
  X <- current_data$X
  
  is_classification <- length(unique(na.omit(Y))) == 2
  metric_name       <- ifelse(is_classification, "Accuracy", "RMSE")
  cat("Detected task:  ", ifelse(is_classification, "Binary Classification", "Regression"), "\n")
  cat("Target Metric:  ", metric_name, "\n\n")
  
  if (is_classification) {
    unique_vals <- sort(unique(na.omit(Y)))
    Y <- as.integer(ifelse(Y == unique_vals[2], 1L, 0L))
  }
  
  data_combined <- data.frame(Y = Y, X)
  n             <- nrow(data_combined)
  ADAPTIVE_BURNIN <- if (n < 500) {
    ADAPTIVE_BURNIN_FINAL_small
  } else if (n < 2000) {
    ADAPTIVE_BURNIN_FINAL_medium
  } else {
    ADAPTIVE_BURNIN_FINAL_large
  }

  cat(sprintf("Adaptive MCMC schedule: burn-in = %d, draws = %d (n = %d)\n",
              ADAPTIVE_BURNIN, ADAPTIVE_DRAWS, n))
  train_indices <- sample(1:n, size = round(0.8 * n))
  
  train_set <- data_combined[train_indices, ]
  test_set  <- data_combined[-train_indices, ]
  
  X_train <- model.matrix(Y ~ . - 1, data = train_set)
  X_test  <- model.matrix(Y ~ . - 1, data = test_set)
  Y_train <- train_set$Y
  Y_test  <- test_set$Y
  
  non_constant_cols <- apply(X_train, 2, function(col) length(unique(col)) > 1)
  X_train <- X_train[, non_constant_cols, drop = FALSE]
  X_test  <- X_test[,  non_constant_cols, drop = FALSE]
  
  cv_folds            <- createFolds(Y_train, k = 5, returnTrain = TRUE)
  cv_lasso_sigma      <- if (!is_classification) {
    vapply(cv_folds, function(train_idx) {
      cv <- get_cv_data(X_train, Y_train, train_idx)
      fit_lasso <- cv.glmnet(as.matrix(cv$X_train), cv$Y_train)
      best_idx <- which(fit_lasso$lambda == fit_lasso$lambda.min)
      sqrt(fit_lasso$cvm[best_idx])
    }, numeric(1))
  } else {
    NULL
  }
  
  packages_list <- c("glmnet", "flexBART", "dbarts", "ranger", "xgboost")
  
  # Export the training matrix and fold indices; workers materialize one fold at a time.
  clusterExport(cl,
                varlist = c("X_train", "Y_train", "cv_folds",
                            "cv_lasso_sigma", "is_classification", "n",
                            "ADAPTIVE_BURNIN", "ADAPTIVE_DRAWS",
                            "data_combined"),
                envir   = environment())
  
  # CV blacklist — everything already on workers or too large to ship
  heavy_objects_cv <- c(
    "benchmark_datasets", "data_combined", "X_train", "X_test",
    "Y_train", "Y_test", "training_data", "train_set", "test_set",
    "X", "Y", "current_data",
    "cv_folds", "is_classification", "n", "ADAPTIVE_BURNIN", "ADAPTIVE_DRAWS",
    "MCMC_ITER_CV", "MCMC_BURNIN_CV", "DIRICHLET_WARMUP_CV"
  )
  
  # --------------------------------------------------------------------
  # TUNING: Thiessen Uniform  (param x fold flattened)
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Thiessen Uniform (%d combinations x 5 folds)...\n", nrow(grid_mode0)))
  pf_grid_0 <- expand.grid(param_idx = seq_len(nrow(grid_mode0)), fold = 1:5)
  
  raw_0 <- foreach(row = seq_len(nrow(pf_grid_0)), .combine = rbind,
                   .packages = packages_list, .noexport = heavy_objects_cv,
                   .options.snow = list(preschedule = FALSE)) %dopar% {
                     i      <- pf_grid_0$param_idx[row]; f <- pf_grid_0$fold[row]
                     params <- grid_mode0[i, , drop = FALSE]
                     cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                     score <- fit_thiessen_cv(
                       cv, params, "uniform", is_classification,
                       MCMC_BURNIN_CV, MCMC_ITER_CV - MCMC_BURNIN_CV,
                       chains = ADAPTIVE_CV_CHAINS
                       
                     )
                     
                     data.frame(param_idx = i, fold = f, score = score)
                   }
  cv_agg_0     <- aggregate(score ~ param_idx, data = raw_0, FUN = mean)
  cv_results_0 <- merge(cv_agg_0, cbind(param_idx = seq_len(nrow(grid_mode0)), grid_mode0), by = "param_idx")
  names(cv_results_0)[names(cv_results_0) == "score"] <- "avg_score"
  best_params_0 <- if (is_classification) cv_results_0[which.max(cv_results_0$avg_score), ] else
    cv_results_0[which.min(cv_results_0$avg_score), ]
  print(best_params_0)
  cat(sprintf("Thiessen Uniform tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))
  
  # --------------------------------------------------------------------
  # TUNING: Thiessen Dirichlet  (param x fold flattened)
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Thiessen Dirichlet (%d combinations x 5 folds)...\n", nrow(grid_mode1)))
  pf_grid_1 <- expand.grid(param_idx = seq_len(nrow(grid_mode1)), fold = 1:5)
  
  raw_1 <- foreach(row = seq_len(nrow(pf_grid_1)), .combine = rbind,
                   .packages = packages_list, .noexport = heavy_objects_cv,
                   .options.snow = list(preschedule = FALSE)) %dopar% {
                     i      <- pf_grid_1$param_idx[row]; f <- pf_grid_1$fold[row]
                     params <- grid_mode1[i, , drop = FALSE]
                     cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                     score <- fit_thiessen_cv(
                       cv, params, "dirichlet", is_classification,
                       MCMC_BURNIN_CV, MCMC_ITER_CV - MCMC_BURNIN_CV,
                       DIRICHLET_WARMUP_CV, chains = ADAPTIVE_CV_CHAINS
                       
                     )
                     
                     data.frame(param_idx = i, fold = f, score = score)
                   }
  cv_agg_1     <- aggregate(score ~ param_idx, data = raw_1, FUN = mean)
  cv_results_1 <- merge(cv_agg_1, cbind(param_idx = seq_len(nrow(grid_mode1)), grid_mode1), by = "param_idx")
  names(cv_results_1)[names(cv_results_1) == "score"] <- "avg_score"
  best_params_1 <- if (is_classification) cv_results_1[which.max(cv_results_1$avg_score), ] else
    cv_results_1[which.min(cv_results_1$avg_score), ]
  
  print(best_params_1)
  cat(sprintf("Thiessen Dirichlet tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))

  # --------------------------------------------------------------------
  # TUNING: Thiessen Adaptive using the selected Dirichlet parameters
  # --------------------------------------------------------------------
  grid_mode2 <- cbind(
    best_params_1[rep(1L, nrow(adaptive_grid)), names(grid_mode1), drop = FALSE],
    adaptive_grid
  )
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Thiessen Adaptive (%d combinations x 5 folds)...\n", nrow(grid_mode2)))
  pf_grid_2 <- expand.grid(param_idx = seq_len(nrow(grid_mode2)), fold = 1:5)

  raw_2 <- foreach(row = seq_len(nrow(pf_grid_2)), .combine = rbind,
                   .packages = packages_list, .noexport = heavy_objects_cv,
                   .options.snow = list(preschedule = FALSE)) %dopar% {
                     i      <- pf_grid_2$param_idx[row]; f <- pf_grid_2$fold[row]
                     params <- grid_mode2[i, , drop = FALSE]
                     cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                     score <- fit_thiessen_cv(
                       cv, params, "adaptive", is_classification,
                       MCMC_BURNIN_CV, MCMC_ITER_CV - MCMC_BURNIN_CV,
                       DIRICHLET_WARMUP_CV, MOMENTUM_SETTLE_CV, MOMENTUM_WINDOW_CV,
                       chains = ADAPTIVE_CV_CHAINS
                     )
                     data.frame(param_idx = i, fold = f, score = score)
                   }
  cv_agg_2     <- aggregate(score ~ param_idx, data = raw_2, FUN = mean)
  cv_results_2 <- merge(cv_agg_2, cbind(param_idx = seq_len(nrow(grid_mode2)), grid_mode2), by = "param_idx")
  names(cv_results_2)[names(cv_results_2) == "score"] <- "avg_score"
  best_params_2 <- if (is_classification) cv_results_2[which.max(cv_results_2$avg_score), ] else
    cv_results_2[which.min(cv_results_2$avg_score), ]
  print(best_params_2)
  cat(sprintf("Thiessen Adaptive tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))

  # --------------------------------------------------------------------
  # TUNING: Uniform BART  (param × fold flattened)
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Uniform BART (%d combinations × 5 folds)...\n", nrow(grid_uBART)))
  pf_grid_ubart <- expand.grid(param_idx = seq_len(nrow(grid_uBART)), fold = 1:5)
  
  raw_ubart <- foreach(row = seq_len(nrow(pf_grid_ubart)), .combine = rbind,
                       .packages = packages_list, .noexport = heavy_objects_cv,
                       .options.snow = list(preschedule = FALSE)) %dopar% {
                         i      <- pf_grid_ubart$param_idx[row]; f <- pf_grid_ubart$fold[row]
                         params <- grid_uBART[i, ]
                         cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                         predictions <- fit_uniform_dbarts(
                           cv$X_train, cv$Y_train, cv$X_val, is_classification,
                           ntree = params$ntree, base = params$base,
                           power = params$power, k = params$k,
                           sigdf = params$sigdf, sigquant = params$sigquant,
                           ndpost = MCMC_ITER_CV - MCMC_BURNIN_CV,
                           nskip = MCMC_BURNIN_CV,
                           chains = ADAPTIVE_CV_CHAINS,
                           sigma_estimate = if (is_classification) NA_real_ else cv_lasso_sigma[f]
                         )
                         score <- if (is_classification) {
                           mean(ifelse(predictions > 0.5, 1, 0) == cv$Y_val)
                         } else {
                           sqrt(mean((predictions - cv$Y_val)^2))
                         }
                         
                         # Release references; avoid forcing a collection per job.
                         rm(predictions, cv)
                         
                         data.frame(param_idx = i, fold = f, score = score)
                       }
  cv_agg_ubart     <- aggregate(score ~ param_idx, data = raw_ubart, FUN = mean)
  cv_results_ubart <- merge(cv_agg_ubart, cbind(param_idx = seq_len(nrow(grid_uBART)), grid_uBART), by = "param_idx")
  names(cv_results_ubart)[names(cv_results_ubart) == "score"] <- "avg_score"
  best_params_ubart <- if (is_classification) cv_results_ubart[which.max(cv_results_ubart$avg_score), ] else
    cv_results_ubart[which.min(cv_results_ubart$avg_score), ]
  cat(sprintf("Uniform BART tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))

  # --------------------------------------------------------------------
  # TUNING: Dirichlet flexBART  (param × fold flattened)
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Dirichlet flexBART (%d combinations × 5 folds)...\n", nrow(grid_dBART)))
  pf_grid_dbart <- expand.grid(param_idx = seq_len(nrow(grid_dBART)), fold = 1:5)
  
  raw_dbart <- foreach(row = seq_len(nrow(pf_grid_dbart)), .combine = rbind,
                       .packages = packages_list, .noexport = heavy_objects_cv,
                       .options.snow = list(preschedule = FALSE)) %dopar% {
                         i      <- pf_grid_dbart$param_idx[row]; f <- pf_grid_dbart$fold[row]
                         params <- grid_dBART[i, ]
                         cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                         x_train <- as.data.frame(cv$X_train, check.names = FALSE)
                         x_val <- as.data.frame(cv$X_val, check.names = FALSE)
                         y_train <- cv$Y_train
                         y_val <- cv$Y_val
                         rm(cv)
                         gc()
                         predictions <- fit_flexbart_chains(
                           x_train, y_train, x_val, is_classification,
                           ntree = params$ntree, base = params$base,
                           power = params$power,
                           ndpost = MCMC_ITER_CV - MCMC_BURNIN_CV,
                           nskip = MCMC_BURNIN_CV,
                           a_u = params$a_u, b_u = params$b_u,
                           k = params$k, nu = params$nu,
                           sigquant = params$sigquant,
                           chains = ADAPTIVE_CV_CHAINS
                         )
                         score <- if (is_classification) {
                           mean(ifelse(predictions > 0.5, 1, 0) == y_val)
                         } else {
                           sqrt(mean((predictions - y_val)^2))
                         }
                         
                         rm(predictions, x_train, x_val, y_train, y_val)
                         
                         data.frame(param_idx = i, fold = f, score = score)
                       }
  cv_agg_dbart     <- aggregate(score ~ param_idx, data = raw_dbart, FUN = mean)
  cv_results_dbart <- merge(cv_agg_dbart, cbind(param_idx = seq_len(nrow(grid_dBART)), grid_dBART), by = "param_idx")
  names(cv_results_dbart)[names(cv_results_dbart) == "score"] <- "avg_score"
  best_params_dbart <- if (is_classification) cv_results_dbart[which.max(cv_results_dbart$avg_score), ] else
    cv_results_dbart[which.min(cv_results_dbart$avg_score), ]
  cat(sprintf("Dirichlet flexBART tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))
  
  # --------------------------------------------------------------------
  # TUNING: Random Forest  (param × fold flattened)
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning Random Forest (%d combinations × 5 folds)...\n", nrow(grid_rf)))
  pf_grid_rf <- expand.grid(param_idx = seq_len(nrow(grid_rf)), fold = 1:5)
  
  raw_rf <- foreach(row = seq_len(nrow(pf_grid_rf)), .combine = rbind,
                    .packages = packages_list, .noexport = heavy_objects_cv,
                    .options.snow = list(preschedule = FALSE)) %dopar% {
                      i      <- pf_grid_rf$param_idx[row]; f <- pf_grid_rf$fold[row]
                      params <- grid_rf[i, ]
                      cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                      current_splitrule <- if (is_classification) ifelse(params$use_extratrees, "extratrees", "gini") else
                        ifelse(params$use_extratrees, "extratrees", "variance")
                      mtry_val <- max(1, floor(params$mtry_prop * ncol(cv$X_train)))
                      
                      if (is_classification) {
                        model <- ranger(x = cv$X_train, y = as.factor(cv$Y_train),
                                        num.trees = params$num.trees, mtry = mtry_val,
                                        min.node.size = params$min.node.size, sample.fraction = params$sample.fraction,
                                        replace = params$replace, splitrule = current_splitrule,
                                        importance = "none", num.threads = 1)
                        preds <- predict(model, cv$X_val)$predictions
                        score <- mean(as.numeric(as.character(preds)) == cv$Y_val)
                      } else {
                        model <- ranger(x = cv$X_train, y = cv$Y_train,
                                        num.trees = params$num.trees, mtry = mtry_val,
                                        min.node.size = params$min.node.size, sample.fraction = params$sample.fraction,
                                        replace = params$replace, splitrule = current_splitrule,
                                        importance = "none", num.threads = 1)
                        preds <- predict(model, cv$X_val)$predictions
                        score <- sqrt(mean((preds - cv$Y_val)^2))
                      }
                      
                      # Release references; avoid forcing a collection per job.
                      rm(model, cv, preds)
                      
                      data.frame(param_idx = i, fold = f, score = score)
                    }
  cv_agg_rf     <- aggregate(score ~ param_idx, data = raw_rf, FUN = mean)
  cv_results_rf <- merge(cv_agg_rf, cbind(param_idx = seq_len(nrow(grid_rf)), grid_rf), by = "param_idx")
  names(cv_results_rf)[names(cv_results_rf) == "score"] <- "avg_score"
  best_params_rf <- if (is_classification) cv_results_rf[which.max(cv_results_rf$avg_score), ] else
    cv_results_rf[which.min(cv_results_rf$avg_score), ]
  cat(sprintf("Random Forest tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))
  
  # --------------------------------------------------------------------
  # SVM tuning is disabled.
  # --------------------------------------------------------------------
  # tuning_started <- proc.time()[["elapsed"]]
  # cat(sprintf("Tuning SVM (%d combinations × 5 folds)...\n", nrow(grid_svm)))
  # pf_grid_svm <- expand.grid(param_idx = seq_len(nrow(grid_svm)), fold = 1:5)
  # raw_svm <- foreach(row = seq_len(nrow(pf_grid_svm)), .combine = rbind,
  #                    .packages = packages_list, .noexport = heavy_objects_cv,
  #                    .options.snow = list(preschedule = FALSE)) %dopar% {
  #   i <- pf_grid_svm$param_idx[row]; f <- pf_grid_svm$fold[row]
  #   params <- grid_svm[i, ]
  #   cv <- get_cv_data(X_train, Y_train, cv_folds[[f]])
  #   if (is_classification) {
  #     model <- svm(x = cv$X_train, y = as.factor(cv$Y_train),
  #                  kernel = as.character(params$kernel), cost = params$cost,
  #                  gamma = params$gamma, degree = params$degree, coef0 = params$coef0,
  #                  probability = FALSE)
  #     preds <- predict(model, cv$X_val)
  #     score <- mean(as.numeric(as.character(preds)) == cv$Y_val)
  #   } else {
  #     model <- svm(x = cv$X_train, y = cv$Y_train,
  #                  kernel = as.character(params$kernel), cost = params$cost,
  #                  gamma = params$gamma, degree = params$degree,
  #                  coef0 = params$coef0, epsilon = params$epsilon)
  #     preds <- predict(model, cv$X_val)
  #     score <- sqrt(mean((preds - cv$Y_val)^2))
  #   }
  #   rm(model, cv, preds)
  #   data.frame(param_idx = i, fold = f, score = score)
  # }
  # cv_agg_svm <- aggregate(score ~ param_idx, data = raw_svm, FUN = mean)
  # cv_results_svm <- merge(cv_agg_svm,
  #   cbind(param_idx = seq_len(nrow(grid_svm)), grid_svm), by = "param_idx")
  # names(cv_results_svm)[names(cv_results_svm) == "score"] <- "avg_score"
  # best_params_svm <- if (is_classification) {
  #   cv_results_svm[which.max(cv_results_svm$avg_score), ]
  # } else {
  #   cv_results_svm[which.min(cv_results_svm$avg_score), ]
  # }
  
  # --------------------------------------------------------------------
  # TUNING: XGBoost  (param × fold flattened, device = "cpu")
  # --------------------------------------------------------------------
  tuning_started <- proc.time()[["elapsed"]]
  cat(sprintf("Tuning XGBoost (%d combinations × 5 folds)...\n", nrow(grid_xgb)))
  pf_grid_xgb <- expand.grid(param_idx = seq_len(nrow(grid_xgb)), fold = 1:5)
  
  raw_xgb <- foreach(row = seq_len(nrow(pf_grid_xgb)), .combine = rbind,
                     .packages = packages_list, .noexport = heavy_objects_cv,
                     .options.snow = list(preschedule = FALSE)) %dopar% {
                       i        <- pf_grid_xgb$param_idx[row]; f <- pf_grid_xgb$fold[row]
                       params   <- grid_xgb[i, ]
                       cv       <- get_cv_data(X_train, Y_train, cv_folds[[f]])
                       obj_type <- ifelse(is_classification, "binary:logistic", "reg:squarederror")
                       model    <- xgboost(x = cv$X_train,
                                           y = if (is_classification) factor(cv$Y_train, levels = c(0, 1)) else cv$Y_train,
                                           objective = obj_type, nrounds = params$nrounds,
                                           max_depth = params$max_depth, learning_rate = params$learning_rate,
                                           subsample = params$subsample, colsample_bytree = params$colsample_bytree,
                                           tree_method = "hist", device = "cpu", nthread = 1, verbosity = 0)
                       preds <- predict(model, cv$X_val)
                       score <- if (is_classification) mean(ifelse(preds > 0.5, 1, 0) == cv$Y_val) else
                         sqrt(mean((preds - cv$Y_val)^2))
                       
                       # Release references; avoid forcing a collection per job.
                       rm(model, cv, preds)
                       
                       data.frame(param_idx = i, fold = f, score = score)
                     }
  cv_agg_xgb     <- aggregate(score ~ param_idx, data = raw_xgb, FUN = mean)
  cv_results_xgb <- merge(cv_agg_xgb, cbind(param_idx = seq_len(nrow(grid_xgb)), grid_xgb), by = "param_idx")
  names(cv_results_xgb)[names(cv_results_xgb) == "score"] <- "avg_score"
  best_params_xgb <- if (is_classification) cv_results_xgb[which.max(cv_results_xgb$avg_score), ] else
    cv_results_xgb[which.min(cv_results_xgb$avg_score), ]
  cat(sprintf("XGBoost tuning elapsed: %s (hh:mm:ss)\n",
              format_elapsed(proc.time()[["elapsed"]] - tuning_started)))
  
  # ====================================================================
  # FINAL EVALUATION
  # ====================================================================
  final_evaluation_started <- proc.time()[["elapsed"]]
  cat("Running final evaluation (20 runs × 9 models = 180 fits)...\n")
  
  # Step 1: Pre-generate all train/test split indices
  final_splits <- lapply(seq_len(n_runs), function(r) sample(1:n, size = round(0.8 * n)))
  final_sigma_estimate <- if (is_classification) NA_real_ else mean(cv_lasso_sigma)

  # CV objects are no longer needed by workers; retain only the full data used
  # to reconstruct final splits and the small final-fit state.
  clusterEvalQ(cl, {
    rm(list = intersect(
      c("X_train", "X_test", "Y_train", "Y_test", "cv_folds",
        "cv_lasso_sigma", "cv_fold_data", "precomputed_cv_data"),
      ls(envir = .GlobalEnv)
    ), envir = .GlobalEnv)
    gc()
  })
  
  # Step 2: Export best params + splits to workers once
  clusterExport(cl,
                varlist = c("final_splits",
                            "final_sigma_estimate",
                            "ADAPTIVE_BURNIN", "ADAPTIVE_DRAWS",
                            "best_params_0", "best_params_1", "best_params_2",
                            "best_params_ubart", "best_params_dbart", "best_params_rf",
                            "best_params_xgb"),
                envir = environment())

  rm(X_train, X_test, Y_train, Y_test, cv_folds, cv_lasso_sigma,
     current_data, X, Y, train_set, test_set)
  gc()
  
  # Step 3: Build (run × model) job grid
  model_names    <- c("rf", "xgb", "baseline")
  job_grid_final <- expand.grid(run = seq_len(n_runs), model = model_names,
                                stringsAsFactors = FALSE)
  
  # Comprehensive blacklist — everything already on workers via clusterExport
  heavy_objects_final <- c(
    "benchmark_datasets", "cv_folds",
    "X_train", "X_test", "Y_train", "Y_test", "training_data",
    "train_set", "test_set", "X", "Y", "current_data",
    "data_combined", "n", "is_classification",
    "MCMC_ITER_FINAL", "MCMC_BURNIN_FINAL", "DIRICHLET_WARMUP_FINAL",
    "ADAPTIVE_BURNIN", "ADAPTIVE_DRAWS", "final_sigma_estimate",
    "cv_lasso_sigma",
    "final_splits",
    "best_params_0", "best_params_1", "best_params_2",
    "best_params_ubart", "best_params_dbart", "best_params_rf", "best_params_xgb"
  )
  
  # Step 4: Run — each job = one model fit, one score returned
  raw_final <- foreach(
    job       = seq_len(nrow(job_grid_final)),
    .combine  = rbind,
    .packages = packages_list,
    .noexport = heavy_objects_final,
    .options.snow = list(preschedule = FALSE)
  ) %dopar% {
    
    run_id     <- job_grid_final$run[job]
    model_name <- job_grid_final$model[job]
    
    # Reconstruct this run's train/test split
    train_idx <- final_splits[[run_id]]
    train_set <- data_combined[train_idx, ]
    test_set  <- data_combined[-train_idx, ]
    X_tr      <- model.matrix(Y ~ . - 1, data = train_set)
    X_te      <- model.matrix(Y ~ . - 1, data = test_set)
    Y_tr      <- train_set$Y
    Y_te      <- test_set$Y
    
    non_const <- apply(X_tr, 2, function(col) length(unique(col)) > 1)
    X_tr <- X_tr[, non_const, drop = FALSE]
    X_te <- X_te[, non_const, drop = FALSE]
    
    # Fit the single assigned model, compute score, and IMMEDIATELY REMOVE object
    fit_started <- proc.time()[["elapsed"]]
    score_val <- if (model_name == "rf") {
      
      rf_mtry <- max(1, floor(best_params_rf$mtry_prop * ncol(X_tr)))
      if (is_classification) {
        current_splitrule <- ifelse(best_params_rf$use_extratrees, "extratrees", "gini")
        m <- ranger(x = X_tr, y = as.factor(Y_tr),
                    num.trees = best_params_rf$num.trees, mtry = rf_mtry,
                    min.node.size = best_params_rf$min.node.size,
                    sample.fraction = best_params_rf$sample.fraction,
                    replace = best_params_rf$replace, splitrule = current_splitrule,
                    importance = "none", num.threads = 1)
        p <- predict(m, X_te)$predictions
        
        # Release the model before the next job.
        rm(m)
        
        mean(as.numeric(as.character(p)) == Y_te)
      } else {
        current_splitrule <- ifelse(best_params_rf$use_extratrees, "extratrees", "variance")
        m <- ranger(x = X_tr, y = Y_tr,
                    num.trees = best_params_rf$num.trees, mtry = rf_mtry,
                    min.node.size = best_params_rf$min.node.size,
                    sample.fraction = best_params_rf$sample.fraction,
                    replace = best_params_rf$replace, splitrule = current_splitrule,
                    importance = "none", num.threads = 1)
        p <- predict(m, X_te)$predictions
        
        # Release the model before the next job.
        rm(m)
        
        sqrt(mean((p - Y_te)^2))
      }
      
    } else if (model_name == "xgb") {
      
      obj_type <- ifelse(is_classification, "binary:logistic", "reg:squarederror")
      m <- xgboost(x = X_tr,
           y = if (is_classification) factor(Y_tr, levels = c(0, 1)) else Y_tr,
                   objective = obj_type,
                   nrounds = best_params_xgb$nrounds, max_depth = best_params_xgb$max_depth,
                   learning_rate = best_params_xgb$learning_rate,
                   subsample = best_params_xgb$subsample,
                   colsample_bytree = best_params_xgb$colsample_bytree,
                   tree_method = "hist", device = "cpu", nthread = 1, verbosity = 0)
      p <- predict(m, X_te)
      
      # Release the model before the next job.
      rm(m)
      
      if (is_classification) mean(ifelse(p > 0.5, 1, 0) == Y_te) else sqrt(mean((p - Y_te)^2))
      
    } else { # baseline
      
      if (is_classification) max(sum(Y_te == 0), sum(Y_te == 1)) / length(Y_te) else
        sqrt(mean((mean(Y_te) - Y_te)^2))
      
    }
    
    # Return the clean scalar score
    data.frame(run = run_id, model = model_name, score = score_val,
           elapsed_seconds = proc.time()[["elapsed"]] - fit_started)
  }

  # Run each split on one worker; average chain predictions before returning.
  raw_flexbart_final <- foreach(
    run_id = seq_len(n_runs),
    .combine = rbind,
    .packages = packages_list,
    .noexport = heavy_objects_final,
    .options.snow = list(preschedule = FALSE)
  ) %dopar% {
    train_idx <- final_splits[[run_id]]
    train_set <- data_combined[train_idx, ]
    X_tr <- model.matrix(Y ~ . - 1, data = train_set)
    Y_tr <- train_set$Y
    rm(train_set)

    test_set <- data_combined[-train_idx, ]
    X_te <- model.matrix(Y ~ . - 1, data = test_set)
    Y_te <- test_set$Y
    rm(test_set)

    non_const <- apply(X_tr, 2, function(col) length(unique(col)) > 1)
    X_tr <- X_tr[, non_const, drop = FALSE]
    X_te <- X_te[, non_const, drop = FALSE]

    prediction_sum <- numeric(length(Y_te))
    elapsed_seconds <- 0
    for (chain_id in seq_len(ADAPTIVE_FINAL_CHAINS)) {
      fit_started <- proc.time()[["elapsed"]]
      chain_predictions <- fit_flexbart_chains(
        X_tr, Y_tr, X_te, is_classification,
        ntree = best_params_dbart$ntree, base = best_params_dbart$base,
        power = best_params_dbart$power,
        a_u = best_params_dbart$a_u, b_u = best_params_dbart$b_u,
        k = best_params_dbart$k,
        nu = best_params_dbart$nu,
        sigquant = best_params_dbart$sigquant,
        ndpost = MCMC_ITER_FINAL - MCMC_BURNIN_FINAL,
        nskip = MCMC_BURNIN_FINAL, chains = 1L
      )
      elapsed_seconds <- elapsed_seconds +
        proc.time()[["elapsed"]] - fit_started
      prediction_sum <- prediction_sum + chain_predictions
      rm(chain_predictions)
    }

    predictions <- prediction_sum / ADAPTIVE_FINAL_CHAINS
    score <- if (is_classification) {
      mean(ifelse(predictions > 0.5, 1, 0) == Y_te)
    } else {
      sqrt(mean((predictions - Y_te)^2))
    }
    data.frame(
      run = run_id, model = "dbart", score = score,
      elapsed_seconds = elapsed_seconds
    )
  }
  raw_final <- rbind(raw_final, raw_flexbart_final)
  rm(raw_flexbart_final)

  # Run final uniform dbarts fits in thread-budgeted batches, like Thiessen.
  bart_workers <- max(1L, floor(num_cores / ADAPTIVE_FINAL_THREADS))
  bart_run_groups <- split(
    seq_len(n_runs),
    rep(seq_len(min(bart_workers, n_runs)), length.out = n_runs)
  )
  raw_ubart_final <- foreach(
    group_id = seq_along(bart_run_groups),
    .combine = rbind,
    .packages = packages_list,
    .noexport = heavy_objects_final,
    .options.snow = list(preschedule = FALSE)
  ) %dopar% {
    do.call(rbind, lapply(bart_run_groups[[group_id]], function(run_id) {
      train_idx <- final_splits[[run_id]]
      train_set <- data_combined[train_idx, ]
      test_set <- data_combined[-train_idx, ]
      X_tr <- model.matrix(Y ~ . - 1, data = train_set)
      X_te <- model.matrix(Y ~ . - 1, data = test_set)
      Y_tr <- train_set$Y
      Y_te <- test_set$Y

      non_const <- apply(X_tr, 2, function(col) length(unique(col)) > 1)
      X_tr <- X_tr[, non_const, drop = FALSE]
      X_te <- X_te[, non_const, drop = FALSE]

      fit_started <- proc.time()[["elapsed"]]
      predictions <- fit_uniform_dbarts(
        X_tr, Y_tr, X_te, is_classification,
        ntree = best_params_ubart$ntree, base = best_params_ubart$base,
        power = best_params_ubart$power, k = best_params_ubart$k,
        sigdf = best_params_ubart$sigdf,
        sigquant = best_params_ubart$sigquant,
        ndpost = MCMC_ITER_FINAL - MCMC_BURNIN_FINAL,
        nskip = MCMC_BURNIN_FINAL,
        chains = ADAPTIVE_FINAL_CHAINS,
        threads = ADAPTIVE_FINAL_THREADS,
        sigma_estimate = final_sigma_estimate
      )
      score <- if (is_classification) {
        mean(ifelse(predictions > 0.5, 1, 0) == Y_te)
      } else {
        sqrt(mean((predictions - Y_te)^2))
      }
      elapsed_seconds <- proc.time()[["elapsed"]] - fit_started
      rm(predictions)
      data.frame(
        run = run_id, model = "ubart", score = score,
        elapsed_seconds = elapsed_seconds
      )
    }))
  }
  raw_final <- rbind(raw_final, raw_ubart_final)

  # Batch all final Thiessen fits to respect the shared thread budget.
  thiessen_workers <- max(1L, floor(num_cores / ADAPTIVE_FINAL_THREADS))
  thiessen_run_groups <- split(
    seq_len(n_runs),
    rep(seq_len(min(thiessen_workers, n_runs)), length.out = n_runs)
  )
  raw_thiessen <- foreach(
    group_id = seq_along(thiessen_run_groups),
    .combine = rbind,
    .packages = packages_list,
    .noexport = heavy_objects_final,
    .options.snow = list(preschedule = FALSE)
  ) %dopar% {
    do.call(rbind, lapply(thiessen_run_groups[[group_id]], function(run_id) {
      train_idx <- final_splits[[run_id]]
      train_set <- data_combined[train_idx, ]
      test_set  <- data_combined[-train_idx, ]
      X_tr <- model.matrix(Y ~ . - 1, data = train_set)
      X_te <- model.matrix(Y ~ . - 1, data = test_set)
      Y_tr <- train_set$Y
      Y_te <- test_set$Y

      non_const <- apply(X_tr, 2, function(col) length(unique(col)) > 1)
      X_tr <- X_tr[, non_const, drop = FALSE]
      X_te <- X_te[, non_const, drop = FALSE]

      cv_data <- list(X_train = X_tr, Y_train = Y_tr, X_val = X_te, Y_val = Y_te)
      schedule <- list(
        burn_in = MCMC_BURNIN_FINAL,
        draws = MCMC_ITER_FINAL - MCMC_BURNIN_FINAL,
        dirichlet_warmup = DIRICHLET_WARMUP_FINAL,
        momentum_settle = MOMENTUM_SETTLE_FINAL,
        momentum_window = MOMENTUM_WINDOW_FINAL,
        chains = ADAPTIVE_FINAL_CHAINS,
        threads = ADAPTIVE_FINAL_THREADS
      )
      adaptive_schedule <- schedule
      adaptive_schedule$burn_in <- ADAPTIVE_BURNIN
      adaptive_schedule$draws <- ADAPTIVE_DRAWS
      uniform_started <- proc.time()[["elapsed"]]
      uniform_score <- do.call(fit_thiessen_cv, c(
          list(cv = cv_data, params = best_params_0, prior = "uniform",
               is_classification = is_classification), schedule
        ))
      uniform_elapsed <- proc.time()[["elapsed"]] - uniform_started
      dirichlet_started <- proc.time()[["elapsed"]]
      dirichlet_score <- do.call(fit_thiessen_cv, c(
          list(cv = cv_data, params = best_params_1, prior = "dirichlet",
               is_classification = is_classification), schedule
        ))
      dirichlet_elapsed <- proc.time()[["elapsed"]] - dirichlet_started
      adaptive_started <- proc.time()[["elapsed"]]
      adaptive_score <- do.call(fit_thiessen_cv, c(
          list(cv = cv_data, params = best_params_2, prior = "adaptive",
           is_classification = is_classification), adaptive_schedule
        ))
      adaptive_elapsed <- proc.time()[["elapsed"]] - adaptive_started
      data.frame(
        run = run_id,
        model = c("uniform", "dirichlet", "adaptive"),
        score = c(uniform_score, dirichlet_score, adaptive_score),
        elapsed_seconds = c(uniform_elapsed, dirichlet_elapsed, adaptive_elapsed)
      )
    }))
  }
  raw_final <- rbind(raw_final, raw_thiessen)
  final_elapsed <- proc.time()[["elapsed"]] - final_evaluation_started
  final_runtime_by_model <- aggregate(
    elapsed_seconds ~ model, data = raw_final, FUN = sum
  )
  final_runtime_by_model$mean_seconds_per_run <-
    final_runtime_by_model$elapsed_seconds / n_runs
  cat("Final fit time by algorithm (summed across runs; hh:mm:ss):\n")
  for (i in seq_len(nrow(final_runtime_by_model))) {
    cat(sprintf(
      "  %-20s total %s, mean/run %s\n",
      final_runtime_by_model$model[i],
      format_elapsed(final_runtime_by_model$elapsed_seconds[i]),
      format_elapsed(final_runtime_by_model$mean_seconds_per_run[i])
    ))
  }
  cat(sprintf("Final evaluation wall time: %s (hh:mm:ss)\n",
              format_elapsed(final_elapsed)))
  
  # Step 5: Reshape long -> wide (one row per run)
  evaluation_results <- reshape(
    raw_final[c("run", "model", "score")],
    idvar = "run", timevar = "model", direction = "wide"
  )
  # Rename score.* columns to match original conventions
  col_map <- c(
    "score.uniform"  = "Thiessen_Uniform_Score",
    "score.dirichlet" = "Thiessen_Dirichlet_Score",
    "score.adaptive" = "Thiessen_Adaptive_Score",
    "score.ubart"    = "Uniform_BART_Score",
    "score.dbart"    = "Dirichlet_BART_Score",
    "score.rf"       = "RF_Score",
    "score.xgb"      = "XGB_Score",
    "score.baseline" = "baseline"
  )
  names(evaluation_results) <- c("Run", col_map[names(evaluation_results)[-1]])

  display_name_map <- c(
    Thiessen_Adaptive_Score = "AV-adapt",
    Thiessen_Dirichlet_Score = "AV-Dir",
    Thiessen_Uniform_Score = "AV-Uni",
    Dirichlet_BART_Score = "DART",
    Uniform_BART_Score = "BART"
  )
  display_score_names <- function(scores) {
    old_names <- names(scores)
    aliases <- unname(display_name_map[old_names])
    names(scores) <- ifelse(is.na(aliases), old_names, aliases)
    scores
  }
  
  cat(paste0("\n--- Average ", metric_name, " across 20 runs ---\n"))
  print(display_score_names(colMeans(evaluation_results[, -1])))
  
  cat(paste0("\n--- Standard Deviation across 20 runs ---\n"))
  print(display_score_names(apply(evaluation_results[, -1], 2, sd)))
  
  results_summary_full <- rbind(results_summary_full, data.frame(
    Dataset                = data_name,
    Thiessen_Uniform_Score = mean(evaluation_results$Thiessen_Uniform_Score),
    Thiessen_Dirichlet_Score = mean(evaluation_results$Thiessen_Dirichlet_Score),
    Thiessen_Adaptive_Score = mean(evaluation_results$Thiessen_Adaptive_Score),
    Uniform_BART_Score     = mean(evaluation_results$Uniform_BART_Score),
    Dirichlet_BART_Score   = mean(evaluation_results$Dirichlet_BART_Score),
    RF_Score               = mean(evaluation_results$RF_Score),
    XGB_Score              = mean(evaluation_results$XGB_Score),
    baseline_score         = mean(evaluation_results$baseline)
  ))
}

stopCluster(cl)

# ======================================================================
# RANKING LOGIC
# ======================================================================
algorithm_names <- c("Thiessen_Uniform", "Thiessen_Dirichlet", "Thiessen_Adaptive",
                     "Uniform_BART", "Dirichlet_BART", "RF", "XGBoost", "Baseline")
average_spot    <- setNames(rep(0, 8), algorithm_names)

for (i in 1:nrow(results_summary_full)) {
  scores <- as.numeric(results_summary_full[i, 2:9])
  current_ranks <- rank(scores, ties.method = "average")
  average_spot  <- average_spot + current_ranks
  print(current_ranks)
}

final_avg_rank <- average_spot / nrow(results_summary_full)

cat("\n--- Final Average Algorithm Ranks (1 = best) ---\n")
print(final_avg_rank)


# Extract the relevant columns (excluding Baseline)
algo_cols <- c("Thiessen_Uniform_Score", "Thiessen_Dirichlet_Score", 
               "Thiessen_Adaptive_Score", "Uniform_BART_Score", 
               "Dirichlet_BART_Score", "RF_Score", "XGB_Score")

# Calculate RMSE (square root of the MSE scores)
rmse_data <- sqrt(results_summary_full[, algo_cols])

# Find the best (minimum) RMSE for each dataset
best_rmse <- apply(rmse_data, 1, min)

# Calculate the Relative RMSE by dividing each row by its minimum
rel_rmse_data <- sweep(rmse_data, 1, best_rmse, "/")

# Clean the column names for the plot labels
colnames(rel_rmse_data) <- c("AV-Uni", "AV-Dir", 
                             "AV-Adapt", "BART", 
                             "DART", "RF", "XGB")

# Calculate the median Relative RMSE for each algorithm
algorithm_medians <- apply(rel_rmse_data, 2, median)

# Order the dataframe columns by median ascending.
# Since boxplot() plots the first column at the bottom, 
# this puts the best at the bottom and worst at the top.
ordered_rel_rmse <- rel_rmse_data[, order(algorithm_medians)]

# Widen the left margin so the algorithm names fit
par(mar = c(5, 5.5, 1, 1) + 0.1)

# Generate the horizontal boxplot
boxplot(
  ordered_rel_rmse,
  horizontal = TRUE,
  las = 1,
  xlab = "Relative RMSE",
  col = "lightblue",
  log = "x"
)

uniform_beats_both <- rmse_data[, "Thiessen_Uniform_Score"] <
  rmse_data[, "Thiessen_Dirichlet_Score"] &
  rmse_data[, "Thiessen_Uniform_Score"] <
  rmse_data[, "Thiessen_Adaptive_Score"]
plot_datasets <- !is.na(uniform_beats_both) & !uniform_beats_both

boxplot(
  ordered_rel_rmse[plot_datasets, 1:2, drop = FALSE],
  horizontal = TRUE,
  las = 1,
  xlab = "Relative RMSE (1.0 = Best)",
  col = "lightblue",
  log = "x"
)

colMeans(ordered_rel_rmse)
colMedians <- apply(ordered_rel_rmse, 2, median)
colMedians


results_summary_full <- data.frame(
  Dataset = c(
    "Seatbelts",
    "cpus",
    "Tecator",
    "QSAR_AquaticTox",
    "Ozone",
    "ResidentialBuilding",
    "Credit",
    "ChickWeight",
    "Soybean",
    "FRED_MD",
    "College",
    "ChemManuf",
    "BloodBrain",
    "mpg",
    "Cox2",
    "Hitters",
    "Auto",
    "Carseats",
    "Cars2004",
    "Sacramento",
    "Solubility",
    "quakes",
    "Superconductivity",
    "nlschools",
    "AmesHousing",
    "Wage",
    "DNA"
  ),

  Thiessen_Uniform_Score = c(
    12.6096843,
    48.5130675,
    1.8484638,
    0.5627693,
    4.2356548,
    28.3737987,
    41.3073436,
    7.8940589,
    2.1187460,
    0.2590424,
    1269.9324151,
    9.765297e-01,
    4.870239e-01,
    1.108254e+00,
    2.707595e+01,
    2.602153e+02,
    2.698590e+00,
    1.147314e+00,
    2.017282e+03,
    7.166518e+04,
    5.978651e-01,
    1.811400e-01,
    1.290492e+01,
    6.558573e+00,
    2.262407e+04,
    3.239497e-01,
    2.693690e-01
  ),

  Thiessen_Dirichlet_Score = c(
    1.232073e+01,
    4.256901e+01,
    1.737029e+00,
    5.978878e-01,
    4.212777e+00,
    2.327419e+01,
    2.914228e+01,
    7.969470e+00,
    1.930427e+00,
    9.877302e-02,
    1.264719e+03,
    8.424103e-01,
    5.442551e-01,
    1.181179e+00,
    2.772076e+01,
    2.796307e+02,
    2.545968e+00,
    1.118306e+00,
    2.047298e+03,
    7.004454e+04,
    6.003718e-01,
    1.811272e-01,
    1.283604e+01,
    6.638032e+00,
    2.228074e+04,
    2.092448e-01,
    2.460455e-01
  ),

  Thiessen_Adaptive_Score = c(
    12.3225962,
    41.0936907,
    1.6939152,
    0.5912673,
    4.1722355,
    23.6478019,
    28.6634527,
    8.0077026,
    1.9355839,
    0.1058665,
    1261.4024995,
    7.918739e-01,
    5.275722e-01,
    1.165579e+00,
    2.777170e+01,
    2.813428e+02,
    2.560412e+00,
    1.116542e+00,
    2.048310e+03,
    7.005132e+04,
    5.944358e-01,
    1.809966e-01,
    1.293631e+01,
    6.639478e+00,
    2.217036e+04,
    2.120996e-01,
    2.462974e-01
  ),

  Uniform_BART_Score = c(
    12.4865948,
    50.0890021,
    6.9549787,
    0.5808102,
    4.5624100,
    28.2475290,
    48.0581407,
    9.7305410,
    2.0767765,
    0.2881872,
    1309.5819282,
    1.006087e+00,
    5.031760e-01,
    1.239054e+00,
    2.703862e+01,
    2.656919e+02,
    2.752354e+00,
    1.135420e+00,
    2.071749e+03,
    7.297060e+04,
    6.138561e-01,
    1.821797e-01,
    1.365271e+01,
    6.569684e+00,
    2.253510e+04,
    8.006566e-01,
    2.668631e-01
  ),

  Dirichlet_BART_Score = c(
    12.4400631,
    45.4905850,
    5.5955167,
    0.5910566,
    4.1695541,
    22.6647471,
    35.0955906,
    8.5138663,
    2.0904534,
    0.1182740,
    1349.3916399,
    9.397629e-01,
    5.127500e-01,
    1.203656e+00,
    2.716355e+01,
    2.708053e+02,
    2.974935e+00,
    1.106650e+00,
    2.070573e+03,
    7.363798e+04,
    6.045061e-01,
    1.819754e-01,
    1.389704e+01,
    6.568482e+00,
    2.325151e+04,
    2.348033e-01,
    2.813389e-01
  ),

  RF_Score = c(
    12.4580317,
    47.4925442,
    6.4446291,
    0.6044852,
    4.2216397,
    33.9989037,
    97.5776865,
    18.8609494,
    2.1704391,
    0.3185137,
    1311.9418333,
    9.515376e-01,
    5.161889e-01,
    1.025221e+00,
    2.677442e+01,
    2.798251e+02,
    2.792215e+00,
    1.565657e+00,
    2.141263e+03,
    7.390631e+04,
    6.630347e-01,
    1.840820e-01,
    1.281753e+01,
    6.728633e+00,
    2.638121e+04,
    5.945976e-01,
    2.636534e-01
  ),

  XGB_Score = c(
    12.2578707,
    50.6408564,
    6.5804986,
    0.5942254,
    4.2431614,
    28.5197693,
    84.7630511,
    11.0447336,
    2.1129546,
    0.4266368,
    1429.4531620,
    8.774020e-01,
    5.382319e-01,
    1.169569e+00,
    2.636133e+01,
    2.907445e+02,
    2.813652e+00,
    1.280953e+00,
    2.136013e+03,
    7.321344e+04,
    6.166113e-01,
    1.834186e-01,
    1.318210e+01,
    6.603805e+00,
    2.377907e+04,
    1.252519e+00,
    2.584183e-01
  ),

  baseline_score = c(
    25.369183,
    142.918559,
    12.636732,
    1.271683,
    8.086109,
    155.093535,
    464.127213,
    69.981611,
    5.359660,
    11.448452,
    3888.052762,
    1.799456e+00,
    7.690661e-01,
    5.949543e+00,
    3.070161e+01,
    4.572734e+02,
    7.609161e+00,
    2.818774e+00,
    9.684240e+03,
    1.300588e+05,
    2.059940e+00,
    3.951982e-01,
    3.410028e+01,
    9.006389e+00,
    8.151541e+04,
    4.180524e+01,
    8.269984e-01
  )
)
