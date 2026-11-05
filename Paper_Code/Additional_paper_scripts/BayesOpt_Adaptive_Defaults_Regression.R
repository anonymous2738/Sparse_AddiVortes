# ======================================================================
# BAYESIAN OPTIMIZATION FOR ADAPTIVE MCMC DEFAULTS
# Focus: Regression Datasets Only
# Goal: Use Gaussian Processes to actively search for the best defaults
# ======================================================================

# Install required package if missing
if (!require("rBayesianOptimization")) install.packages("rBayesianOptimization")
library(rBayesianOptimization)
library(doParallel)
library(doRNG)

# 1. Configuration
n_init_points <- 20    # Number of random starting profiles to seed the Gaussian Process
n_bo_iters    <- 35    # Number of intelligent search iterations (active learning)

MCMC_ITER   <- 6000
MCMC_BURNIN <- 4000
DIRICHLET_WARMUP <- 200
MOMENTUM_SETTLE <- 200
MOMENTUM_WINDOW <- 800
TIMEOUT_SECONDS <- 190 # Maximum time (in seconds) allowed per dataset fit

# 2. Setup Datasets
datasets_to_run <- names(benchmark_datasets)

# 3. Setup Parallel Backend
# Since Bayesian Optimization is strictly sequential (it needs the result of 
# step N to guess step N+1), we parallelize the INNER loop: we evaluate the 
# 4 datasets simultaneously across your cores for lightning-fast scoring.
num_cores <- min(length(datasets_to_run), 10)
cl <- makeCluster(num_cores, type = "PSOCK")
registerDoParallel(cl)
registerDoRNG(seed = 11042000)

clusterExport(cl, varlist = c("benchmark_datasets", "datasets_to_run", 
                              "MCMC_ITER", "MCMC_BURNIN", "DIRICHLET_WARMUP",
                              "MOMENTUM_SETTLE", "MOMENTUM_WINDOW", "TIMEOUT_SECONDS"), 
              envir = environment())

# 4. Define the Global Objective Function
# rBayesianOptimization maximizes the Score, so we will return NEGATIVE Relative RMSE
thiessen_bo_score <- function(alpha, a_alpha, b_alpha, adapt_rate, momentum_decay) {
  
  # Evaluate this specific parameter profile across all datasets in parallel
  dataset_scores <- foreach(data_name = datasets_to_run, .combine = c, .packages = c("thiessen")) %dopar% {
    Sys.setenv(THIESSEN_EXPERIMENTAL = 1)
    
    current_data <- benchmark_datasets[[data_name]]
    X <- current_data$X
    Y <- current_data$Y
    n <- nrow(X)
    
    # 80/20 train/test split (fixed seed inside worker for stable BO surface)
    set.seed(42)
    train_idx <- sample(1:n, size = round(0.8 * n))
    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- Y[train_idx]
    X_test  <- X[-train_idx, , drop = FALSE]
    Y_test  <- Y[-train_idx]
    
    non_const <- apply(X_train, 2, function(col) length(unique(col)) > 1)
    X_train <- X_train[, non_const, drop = FALSE]
    X_test  <- X_test[, non_const, drop = FALSE]
    
    adaptive_prior <- thiessen::dart_inclusion(
      alpha            = alpha,
      a_alpha          = a_alpha,
      b_alpha          = b_alpha,
      adapt_boost      = adapt_rate,
      adapt_penalty    = adapt_rate,
      momentum_decay   = momentum_decay,
      dirichlet_warmup = DIRICHLET_WARMUP,
      momentum_settle  = MOMENTUM_SETTLE,
      momentum_window  = MOMENTUM_WINDOW
    )
    
    preds <- tryCatch({
      R.utils::withTimeout({
        thiessen::thiessen(
          x = X_train, y = Y_train,xtest = X_test,
          chains = 1, threads = 1,
          control = thiessen::thiessen_control(
            outcome = thiessen::gaussian_outcome(nu = 6, q = 0.85),
            mean_params = thiessen::term_params(
              tessellations = 200, # Kept moderate for tuning speed
              k = 3, 
              structure = thiessen::structure_params(inclusion = adaptive_prior)
            ),
            general_params = thiessen::general_params(burn_in = MCMC_BURNIN, draws = MCMC_ITER)
          )
        )
      }, timeout = TIMEOUT_SECONDS, onTimeout = "error")
    }, error = function(e) {
      NULL
    })
    
    if (is.null(preds)) {
      # If the model timed out or threw another error, heavily penalize this parameter set
      # by returning a relative RMSE of 5.0 (5x worse than just predicting the mean)
      return(5.0)
    }
    
    rmse  <- sqrt(mean((preds - Y_test)^2))
    
    # Baseline RMSE (predicting the mean)
    baseline_rmse <- sqrt(mean((mean(Y_train) - Y_test)^2))
    
    # Relative RMSE (Lower is better)
    relative_rmse <- rmse / baseline_rmse
    
    rm(preds); gc()
    return(relative_rmse)
  }
  
  # Calculate the global average across all datasets
  avg_relative_rmse <- mean(dataset_scores)
  
  # Return the negative RMSE because BO is trying to maximize the function
  list(Score = -avg_relative_rmse, Pred = 0)
}

# 5. Define Parameter Bounds
bounds <- list(
  alpha          = c(1, 45.0),
  a_alpha        = c(1, 45.0),
  b_alpha        = c(1, 2),
  adapt_rate     = c(0.1, 30.0),
  momentum_decay = c(0.1, 0.99)
)

# 6. Run Bayesian Optimization
cat("\n====================================================\n")
cat(" STARTING GAUSSIAN PROCESS BAYESIAN OPTIMIZATION\n")
cat("====================================================\n")

opt_res <- BayesianOptimization(
  FUN = thiessen_bo_score,
  bounds = bounds,
  init_points = n_init_points,
  n_iter = n_bo_iters,
  acq = "ucb",      # Upper Confidence Bound acquisition function
  kappa = 2.576,    # Exploration-Exploitation trade-off
  eps = 0.0,
  verbose = TRUE
)

stopCluster(cl)

# 7. Print Results
cat("\n====================================================\n")
cat(" BAYESIAN OPTIMIZATION COMPLETE - BEST DEFAULTS FOUND\n")
cat("====================================================\n")

print(opt_res$Best_Par)
cat("Best Average Relative RMSE Score:", -opt_res$Best_Value, "\n")


# ---------------------------------------------------------------------
# 8. Extract Good Ranges for CV Grid Search
# ---------------------------------------------------------------------
# The 'Value' is negative RMSE, so higher is better in this dataframe.
# Sort the history from best to worst score
history<-opt_res$History
sorted_history <- history[order(history$Value, decreasing = TRUE), ]

# Re-convert Value back to positive Relative RMSE for readability
sorted_history$Relative_RMSE <- -sorted_history$Value
sorted_history$Value <- NULL

top_10_combinations <- head(sorted_history, 10)
print(top_10_combinations, row.names = FALSE)

# Optionally, you can save the entire sorted history to a CSV to review later:
# write.csv(sorted_history, "BayesOpt_All_Combinations_Ranked.csv", row.names = FALSE)

# ---------------------------------------------------------------------
# 9. Analyze Hyperparameter Importance
# ---------------------------------------------------------------------
# We can figure out which hyperparameters matter most by fitting a quick 
# Random Forest to predict the RMSE from the hyperparameters, and checking 
# the feature importance.
if (requireNamespace("ranger", quietly = TRUE)) {
  cat("\n====================================================\n")
  cat(" HYPERPARAMETER IMPORTANCE (Higher is more impactful)\n")
  cat("====================================================\n")
  
  # Predict the RMSE using the five tuned hyperparameters
  importance_model <- ranger::ranger(
    Relative_RMSE ~ alpha + a_alpha + b_alpha + adapt_rate + momentum_decay,
    data = sorted_history,
    importance = "permutation",
    num.trees = 500
  )
  
  # Extract and sort importance
  imp <- importance_model$variable.importance
  # Handle negative importances which can happen with permutation importance
  imp[imp < 0] <- 0
  imp_sorted <- sort(imp, decreasing = TRUE)
  
  # Normalize to percentages for easier reading
  imp_percent <- round((imp_sorted / sum(imp_sorted)) * 100, 1)
  
  importance_df <- data.frame(
    Hyperparameter = names(imp_percent),
    Relative_Importance_Pct = paste0(imp_percent, "%")
  )
  print(importance_df, row.names = FALSE)
  
} else {
  cat("\n(Install the 'ranger' package if you want to see hyperparameter importance scores!)\n")
}

# ---------------------------------------------------------------------
# 10. Conditional Hyperparameter Importance
# ---------------------------------------------------------------------
# What matters most once the dominant parameter is in a good spot?
if (requireNamespace("ranger", quietly = TRUE)) {
  # Get the best values for the linked Dirichlet parameters
  best_b_alpha <- sorted_history$b_alpha[1]
  best_alpha <- sorted_history$alpha[1]
  best_a_alpha <- sorted_history$a_alpha[1]
  
  # Define 'near' as within +/- 3.0 of the best value (adjust as needed)
  tolerance <- 4.0
  tolerance_b <- 2.0

  # Subset history to only look at runs where all three are near their optimums
  dirichlet_subset <- subset(sorted_history, 
                             abs(b_alpha - best_b_alpha) <= tolerance_b &
                             abs(alpha - best_alpha) <= tolerance &
                             abs(a_alpha - best_a_alpha) <= tolerance)
  
  cat(sprintf("\n====================================================\n"))
  cat(sprintf(" CONDITIONAL IMPORTANCE (Near optimal Dirichlet Priors)\n"))
  cat(sprintf("   b_alpha ~ %.2f, alpha ~ %.2f, a_alpha ~ %.2f\n", 
              best_b_alpha, best_alpha, best_a_alpha))
  cat(sprintf(" Evaluating %d profiles in this neighborhood\n", nrow(dirichlet_subset)))
  cat(sprintf("====================================================\n"))
  
  # Only run if we have enough points in the subset
  if (nrow(dirichlet_subset) >= 10) {
    importance_model_cond <- ranger::ranger(
      Relative_RMSE ~ alpha + a_alpha + b_alpha + adapt_rate + momentum_decay,
      data = dirichlet_subset,
      importance = "permutation",
      num.trees = 500
    )
    
    imp_cond <- importance_model_cond$variable.importance
    imp_cond[imp_cond < 0] <- 0
    imp_sorted_cond <- sort(imp_cond, decreasing = TRUE)
    
    imp_percent_cond <- round((imp_sorted_cond / sum(imp_sorted_cond)) * 100, 1)
    
    importance_df_cond <- data.frame(
      Hyperparameter = names(imp_percent_cond),
      Relative_Importance_Pct = paste0(imp_percent_cond, "%")
    )
    print(importance_df_cond, row.names = FALSE)
  } else {
    cat("Not enough profiles in this neighborhood to train a Random Forest.\n")
    cat("Try increasing the 'tolerance' variable to include more points.\n")
  }
}

# ---------------------------------------------------------------------
# 11. Extract Hybrid CV Grid (Cross-Product of Linked Groups)
# ---------------------------------------------------------------------
if (requireNamespace("ranger", quietly = TRUE)) {
  cat("\n====================================================\n")
  cat(" BUILDING HYBRID CV GRID (48 Combinations)\n")
  cat("====================================================\n")
  
  # We extract the top distinct synergistic tuples for our linked groups,
  # and the top distinct single values for the rest.
  # 4 * 3 * 2 * 2 = 48 combinations
  
  get_top_unique <- function(df, cols, n) {
    df_base <- as.data.frame(df)
    res <- unique(df_base[, cols, drop = FALSE])
    head(res, n)
  }
  
  # Group A: The Dirichlet Prior (4 top linked triplets)
  group_A <- get_top_unique(sorted_history, c("alpha", "a_alpha", "b_alpha"), 4)
  group_A$A_id <- 1:nrow(group_A)
  
  # Single B: Adapt Rate (2 top values)
  group_C <- get_top_unique(sorted_history, "adapt_rate", 2)
  group_C$C_id <- 1:nrow(group_C)
  
  # Single D: Momentum Decay (2 top values)
  group_D <- get_top_unique(sorted_history, "momentum_decay", 2)
  group_D$D_id <- 1:nrow(group_D)
  
  # Create a Cartesian product of the Group IDs
  grid_ids <- expand.grid(A_id = group_A$A_id, C_id = group_C$C_id,
                          D_id = group_D$D_id)
  
  # Merge the actual parameter values back in
  final_grid <- merge(grid_ids, group_A, by = "A_id")
  final_grid <- merge(final_grid, group_C, by = "C_id")
  final_grid <- merge(final_grid, group_D, by = "D_id")
  
  # Clean up columns and ensure integer types
  cv_grid_exact <- final_grid[, c("alpha", "a_alpha", "b_alpha", 
                                  "adapt_rate", "momentum_decay")]
  
  cat(sprintf("Successfully extracted synergistic group combinations.\n"))
  
  # ------------------------------------------------------------------
  # Build grid_mode2 directly in R and save to CSV
  # ------------------------------------------------------------------
  
  # Format Group A
  grid_mode_alpha <- data.frame(
    alpha   = round(group_A$alpha,0),
    a_alpha = round(group_A$a_alpha, 2),
    b_alpha = round(group_A$b_alpha, 2)
  )
  
  # Format Single C (Mapped to adaptBoost and adaptPenalty)
  hyperparam_boost_penalty <- data.frame(
    adaptBoost   = round(group_C$adapt_rate,0),
    adaptPenalty = round(group_C$adapt_rate, 0)
  )
  
  # Format Single D (added directly to usual)
  grid_mode_usual <- expand.grid(
    m = c(200), 
    k = c(2), 
    Omega = c(1, 3), 
    LambdaRate = c(4),
    momentumDecay = round(group_D$momentum_decay, 3)
  )
  
  # Merge all linked groups together to form the combinations
  grid_mode_comb <- merge(grid_mode_usual, grid_mode_alpha, by = NULL)
  grid_mode2     <- merge(grid_mode_comb, hyperparam_boost_penalty, by = NULL)
    
  cat(sprintf("Successfully generated grid_mode2 with %d total combinations!\n", nrow(grid_mode2)))
  cat(sprintf("Saved to: %s\n\n", csv_filename))
  cat("In your CV script, you can now simply run:\n")
  cat("grid_mode2 <- read.csv(\"grid_mode2_optimized.csv\")\n")
}


  # --- Thiessen: Adaptive Dirichlet inclusion (Optimized Hybrid Grid) ---
  
  # 1. Standard parameters + top momentum decay values
  grid_mode_usual <- expand.grid(
    m = c(200), k = c(3), Omega = c(1.5, 3), LambdaRate = c(4),
    momentumDecay = c(0.5,0.7)
  )
  
  # 2. Linked Dirichlet Prior parameters
  grid_mode_alpha <- data.frame(
    alpha   = c(1,40),
    a_alpha = c(0.5, 40),
    b_alpha = c(1)
  )
  
  # 3. Linked Adaptive Boost/Penalty parameters
  hyperparam_boost_penalty <- data.frame(
    adaptBoost   = c(5, 15), 
    adaptPenalty = c(5, 15),
    stringsAsFactors = FALSE
  )
  
  # 4. Merge them all together (yields exactly 64 combinations)
  grid_mode_comb <- merge(grid_mode_usual, grid_mode_alpha, by = NULL)
  grid_mode2     <- merge(grid_mode_comb, hyperparam_boost_penalty, by = NULL)



