# ======================================================================
# LATIN HYPERCUBE SAMPLING (LHS) FOR ADAPTIVE MCMC DEFAULTS
# Focus: Regression Datasets Only
# Goal: Find the most robust global default hyperparameters
# ======================================================================

library(lhs)
library(doParallel)
library(doRNG)
library(caret)

# 1. Configuration
n_profiles <- 100              # Number of hyperparameter combinations to test
num_cores  <- 20

# MCMC settings (Keep these moderate for the search phase to save time)
MCMC_ITER   <- 5000
MCMC_BURNIN <- 2500
DIRICHLET_WARMUP <- 1000

# 2. Setup Parallel Backend
cl <- makeCluster(num_cores, type = "PSOCK")
registerDoParallel(cl)
registerDoRNG(seed = 42)

clusterEvalQ(cl, {
  Sys.setenv(THIESSEN_EXPERIMENTAL = 1)
  devtools::load_all("/home/jd191252/thiessen/r")
  
})


# Pick a diverse subset of regression datasets to evaluate "Global" performance
# You want a mix of high/low dimensional datasets here.
datasets_to_run <- c("ChemManuf")

# 4. Generate Latin Hypercube Sample
cat(sprintf("Generating %d Latin Hypercube profiles...\n", n_profiles))
set.seed(123)
k_params <- 5
raw_lhs <- randomLHS(n_profiles, k_params)

# Map the [0,1] uniform LHS probabilities to your actual parameter ranges
lhs_grid <- data.frame(
  profile_id     = 1:n_profiles,
  alpha          = qunif(raw_lhs[, 1], min = 1, max = 20.0),
  a_alpha        = qunif(raw_lhs[, 2], min = 1, max = 20.0),
  b_alpha        = qunif(raw_lhs[, 3], min = 1, max = 2.0),
  adapt_rate     = qunif(raw_lhs[, 4], min = 0.1, max = 20.0),
  momentum_decay = qunif(raw_lhs[, 5], min = 0.5, max = 0.99)
)

# Export required objects
clusterExport(cl, varlist = c("benchmark_datasets", "datasets_to_run", "lhs_grid", 
                              "MCMC_ITER", "MCMC_BURNIN", "DIRICHLET_WARMUP"), 
              envir = environment())

# 5. Run the LHS Evaluation Loop
cat("Running Adaptive MCMC LHS Search (This may take a while)...\n")

lhs_results <- foreach(i = 1:nrow(lhs_grid), .combine = rbind, .packages = c("thiessen", "caret")) %dopar% {
  
  params <- lhs_grid[i, ]
  dataset_scores <- numeric(length(datasets_to_run))
  names(dataset_scores) <- datasets_to_run
  
  for (d in seq_along(datasets_to_run)) {
    data_name <- datasets_to_run[d]
    current_data <- benchmark_datasets[[data_name]]
    
    X <- current_data$X
    Y <- current_data$Y
    n <- nrow(X)
    
    # Fast 80/20 train/test split for evaluation
    train_idx <- sample(1:n, size = round(0.8 * n))
    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- Y[train_idx]
    X_test  <- X[-train_idx, , drop = FALSE]
    Y_test  <- Y[-train_idx]
    
    # Remove constants
    non_const <- apply(X_train, 2, function(col) length(unique(col)) > 1)
    X_train <- X_train[, non_const, drop = FALSE]
    X_test  <- X_test[, non_const, drop = FALSE]
    
    # Configure the Adaptive Dirichlet Prior using the LHS parameters
    adaptive_prior <- thiessen::dart_inclusion(
      alpha            = params$alpha,
      a_alpha          = params$a_alpha,
      b_alpha          = params$b_alpha,
      adapt_boost      = params$adapt_rate,
      adapt_penalty    = params$adapt_rate,
      momentum_decay   = params$momentum_decay,
      dirichlet_warmup = DIRICHLET_WARMUP
    )
    
    # Fit the model
    # Note: Using standard Gaussian outcome for regression (no sigma_hat needed on this branch)
    model <- thiessen::thiessen(
      x = X_train, y = Y_train,
      chains = 1, threads = 1,
      control = thiessen::thiessen_control(
        outcome = thiessen::gaussian_outcome(nu = 6, q = 0.85),
        mean_params = thiessen::term_params(
          tessellations = 200, 
          k = 3, 
          structure = thiessen::structure_params(inclusion = adaptive_prior)
        ),
        general_params = thiessen::general_params(burn_in = MCMC_BURNIN, draws = MCMC_ITER)
      )
    )
    
    # Predict and evaluate
    preds <- predict(model, X_test)
    rmse  <- sqrt(mean((preds - Y_test)^2))
    
    # Calculate Baseline RMSE (predicting the mean)
    baseline_rmse <- sqrt(mean((mean(Y_train) - Y_test)^2))
    
    # CRITICAL: We calculate RELATIVE RMSE so that datasets with massive variance 
    # don't completely dominate the global score. 
    # Relative RMSE < 1.0 means it beat the baseline.
    relative_rmse <- rmse / baseline_rmse
    dataset_scores[d] <- relative_rmse
    
    # Free memory inside the worker
    rm(model, preds, X_train, X_test); gc()
  }
  
  # Return the parameters alongside the average relative score across all datasets
  result_row <- params
  result_row$Avg_Relative_RMSE <- mean(dataset_scores)
  
  # Append individual dataset scores to the row for detailed tracking
  for (name in names(dataset_scores)) {
    result_row[[paste0("Score_", name)]] <- dataset_scores[[name]]
  }
  
  result_row
}

stopCluster(cl)

# 6. Find the Best Global Defaults
cat("\n====================================================\n")
cat(" TOP 10 HYPERPARAMETER PROFILES FOR ADAPTIVE MCMC\n")
cat("====================================================\n")

# Sort by lowest Average Relative RMSE
lhs_results <- lhs_results[order(lhs_results$Avg_Relative_RMSE), ]

# Print the top 10 configurations
print(head(lhs_results, 10))

# Save the full results to investigate visually later
write.csv(lhs_results, "Adaptive_MCMC_LHS_Results.csv", row.names = FALSE)
cat("\nFull search results saved to Adaptive_MCMC_LHS_Results.csv\n")

