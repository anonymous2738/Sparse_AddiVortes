# ======================================================================
# MICE DATASET EVALUATION (80/20 SPLIT)
# Focus: High-dimensional regression (N = 1814, P = 10345)
# Models: Thiessen (Uniform, Dirichlet, Adaptive) + DART (BART package)
# ======================================================================

library(BART)
library(caret)

# Enable thiessen experimental features (for categorical/student-T etc if needed, safe to keep on)
Sys.setenv(THIESSEN_EXPERIMENTAL = 1)

# ----------------------------------------------------------------------
# 1. LOAD DATA & PREPROCESS
# ----------------------------------------------------------------------
cat("Loading Mice dataset (This may take a moment due to P > 10,000)...\n")

# Extract the Mice data loaded by the benchmark script
data <- benchmark_datasets$Wheat
X <- data$X
Y <- data$Y

# Clean up constants (BART hates zero-variance predictors)
non_const <- apply(X, 2, function(col) length(unique(col)) > 1)
X <- X[, non_const, drop = FALSE]

# ----------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (80/20)
# ----------------------------------------------------------------------
set.seed(4)  # For reproducible splits
n <- nrow(X)
train_idx <- sample(1:n, size = round(0.8 * n))

X_train <- X[train_idx, , drop = FALSE]
Y_train <- Y[train_idx]
X_test  <- X[-train_idx, , drop = FALSE]
Y_test  <- Y[-train_idx]

non_const <- apply(X_train, 2, function(col) length(unique(col)) > 1)
X_train <- X_train[, non_const, drop = FALSE]
X_test <- X_test[, non_const, drop = FALSE]

cat(sprintf("\nTraining Set: %d rows, %d features\n", nrow(X_train), ncol(X_train)))
cat(sprintf("Test Set:     %d rows, %d features\n\n", nrow(X_test), ncol(X_test)))

# Global MCMC Settings (Adjust these if you want longer chains)
MCMC_BURNIN <- 20000
MCMC_ITER   <- 10000
DIR_WARMUP  <- 2000

results <- data.frame(Model = character(), RMSE = numeric(), stringsAsFactors = FALSE)

fit_lasso <- cv.glmnet(as.matrix(X_train), Y_train)
best_idx <- which(fit_lasso$lambda == fit_lasso$lambda.min)
lasso_sigma <- sqrt(fit_lasso$cvm[best_idx])

# ----------------------------------------------------------------------
# MODEL 1: Thiessen (Uniform)
# ----------------------------------------------------------------------
cat("\nFitting Model 1/4: Thiessen (Uniform)...\n")
model_unif <- thiessen(
  x = X_train, y = Y_train, chains = 1, threads = 1,
  control = thiessen_control(
    outcome = gaussian_outcome(nu = 6, q = 0.85),
    mean_params = term_params(
      tessellations = 200, k = 3,
      structure = structure_params(inclusion = "uniform")
    ),
    general_params = general_params(burn_in = MCMC_BURNIN, draws = MCMC_ITER)
  )
)
preds_unif <- predict(model_unif, X_test)
rmse_unif  <- sqrt(mean((preds_unif - Y_test)^2))
results    <- rbind(results, data.frame(Model = "Thiessen_Uniform", RMSE = rmse_unif))

# Free memory before next heavy model
rm(model_unif, preds_unif); gc()

# ----------------------------------------------------------------------
# MODEL 2: Thiessen (Dirichlet)
# ----------------------------------------------------------------------
cat("\nFitting Model 2/4: Thiessen (Dirichlet)...\n")
model_dir <- thiessen(
  x = X_train, y = Y_train, chains = 1, threads = 1,
  control = thiessen_control(
    outcome = gaussian_outcome(nu = 6, q = 0.85,lasso_sigma),
    mean_params = term_params(
      tessellations = 200, k = 3,
      structure = structure_params(
        inclusion = dart_inclusion(
          alpha = 1.0, a_alpha = 0.5, b_alpha = 1.0
      )
    )),
    general_params = general_params(burn_in = MCMC_BURNIN, draws = MCMC_ITER)
)
)
preds_dir <- predict(model_dir, X_test)
rmse_dir  <- sqrt(mean((preds_dir - Y_test)^2))
results   <- rbind(results, data.frame(Model = "Thiessen_Dirichlet", RMSE = rmse_dir))

rm(model_dir, preds_dir); gc()

# ----------------------------------------------------------------------
# MODEL 3: Thiessen (Adaptive)
# ----------------------------------------------------------------------

cat("\nFitting Model 3/4: Thiessen (Adaptive)...\n")
control_adaptive <- thiessen_control(
  outcome = gaussian_outcome(nu = 6, q = 0.85,lasso_sigma),
  mean_params = term_params(
    tessellations = 200,  
    lambda_c = 4,
    structure = structure_params(                          # Maps to Omega = 1.5
      inclusion = dart_inclusion(
        alpha = 1, a_alpha = 0.5, b_alpha = 1.0,
        adapt_boost = 0.2,
        adapt_penalty = 0.2,
        momentum_decay = 0.5,
        dirichlet_warmup = 2000,
        momentum_settle = 1000, 
        momentum_window = 5000,
        momentum_trace_every = 50
    ),
      omega = 1.5
    )
  ),
  general_params = general_params(
    burn_in = 20000,                         # Maps to mcmcBurnIn = 2500
    draws = 10000,
    thinning = 1                        # Maps to thinning = 1
  )
)

model_adapt <- thiessen(
  x = X_train, y = Y_train, xtest = X_test, chains = 1, threads = 1,
  control = control_adaptive
)

preds_adapt <- predict(model_adapt, X_test)
sqrt(mean((preds_adapt - Y_test)^2))
sqrt(mean((model_adapt- Y_test)^2))
rmse_adapt  <- sqrt(mean((preds_adapt - Y_test)^2))
results     <- rbind(results, data.frame(Model = "Thiessen_Adaptive", RMSE = rmse_adapt))

rm(model_adapt, preds_adapt); gc()

# ----------------------------------------------------------------------
# MODEL 4: DART (via classic BART Package)
# ----------------------------------------------------------------------
cat("\nFitting Model 4/4: Standard DART (BART::wbart sparse=TRUE)...\n")
model_dart <- wbart(
  x.train = X_train, y.train = Y_train, x.test = X_test,
  ntree = 200, sparse = TRUE,
  ndpost = MCMC_ITER, nskip = MCMC_BURNIN,
  printevery = 500
)
preds_dart <- model_dart$yhat.test.mean
rmse_dart  <- sqrt(mean((preds_dart - Y_test)^2))
results    <- rbind(results, data.frame(Model = "Standard_DART_BART_Pkg", RMSE = rmse_dart))

rm(model_dart, preds_dart); gc()

# ----------------------------------------------------------------------
# BASELINE (Predicting the mean every time)
# ----------------------------------------------------------------------
rmse_base <- sqrt(mean((mean(Y_train) - Y_test)^2))
results   <- rbind(results, data.frame(Model = "Baseline_Mean", RMSE = rmse_base))

# ======================================================================
# RESULTS
# ======================================================================
cat("\n\n======================================================\n")
cat("      MICE DATASET (P > 10,000) EVALUATION RESULTS\n")
cat("======================================================\n")
results_sorted <- results[order(results$RMSE), ]
print(results_sorted)

