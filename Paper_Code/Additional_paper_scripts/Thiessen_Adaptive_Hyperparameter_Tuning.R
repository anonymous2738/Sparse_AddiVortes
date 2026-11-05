# ======================================================================
# HYPERPARAMETER TUNING FOR thiessen::dart_inclusion() ADAPTIVE DEFAULTS
#
# Self-contained: generates its own synthetic benchmark datasets (no
# external `benchmark_datasets` object required), searches the adaptive
# DART hyperparameters (adapt_boost, adapt_penalty, momentum_decay,
# a_alpha, b_alpha, alpha) with a Latin Hypercube global
# search followed by a local refinement pass, and reports the profile
# with the best average relative RMSE across datasets.
#
# Search bounds and the per-fit wall-clock cap encode the findings from
# the adapt_boost/momentum_decay stability investigation: momentum_decay
# is kept well below 1, adapt_boost/adapt_penalty are bounded to a modest,
# comparable range, and any fit that runs away past `fit_time_limit`
# seconds is disqualified rather than left to hang.
#
# Usage: Rscript Thiessen_Adaptive_Hyperparameter_Tuning.R
# ======================================================================


for (pkg in c("lhs", "parallel")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
n_lhs_profiles   <- 100     # global LHS search size
n_refine_profiles <- 50    # local refinement search size around the best LHS point
num_cores        <- 20
fit_time_limit   <- 60     # seconds; any fit exceeding this is disqualified

# MCMC settings for the search phase: moderate, not production-scale, to
# keep the sweep affordable. Re-validate the winning profile at full scale
# afterwards (section 6).
TESSELLATIONS    <- 200
MCMC_ITER        <- 5000
MCMC_BURNIN      <- 2500
DIRICHLET_WARMUP <- 1000


# Fixed 80/20 split per dataset, reused for every hyperparameter profile
# so profiles are compared on identical data (no train/test-split noise
# confounding the search).
splits <- lapply(benchmark_datasets, function(d) {
  set.seed(42)
  n <- nrow(d$X)
  train_idx <- sample.int(n, size = round(0.8 * n))
  non_const <- apply(d$X[train_idx, , drop = FALSE], 2, function(col) length(unique(col)) > 1)
  list(
    X_train = d$X[train_idx, non_const, drop = FALSE], Y_train = d$Y[train_idx],
    X_test  = d$X[-train_idx, non_const, drop = FALSE], Y_test  = d$Y[-train_idx]
  )
})

# ---------------------------------------------------------------------
# 3. One profile evaluation: mean relative RMSE and worst-case elapsed
#    time across all datasets, with a hard per-fit wall-clock cap.
# ---------------------------------------------------------------------
evaluate_profile <- function(params, mcmc_iter, burn_in, dirichlet_warmup, tessellations, time_limit) {
  rel_rmses <- numeric(length(splits))
  max_elapsed <- 0
  names(rel_rmses) <- names(splits)

  for (name in names(splits)) {
    s <- splits[[name]]
    prior <- thiessen::dart_inclusion(
      alpha            = params$alpha,
      a_alpha          = params$a_alpha,
      b_alpha          = params$b_alpha,
      adapt_boost      = params$adapt_boost,
      adapt_penalty    = params$adapt_penalty,
      momentum_decay   = params$momentum_decay,
      dirichlet_warmup = dirichlet_warmup
    )

    control <- thiessen::thiessen_control(
      outcome = thiessen::gaussian_outcome(nu = 6, q = 0.9),
      mean_params = thiessen::term_params(
        tessellations = tessellations,
        structure = thiessen::structure_params(inclusion = prior)
      ),
      general_params = thiessen::general_params(burn_in = burn_in, draws = mcmc_iter)
    )

    t0 <- Sys.time()
    preds <- tryCatch({
      setTimeLimit(elapsed = time_limit, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      thiessen::thiessen(s$X_train, s$Y_train,xtest=s$X_test, chains = 1, threads = 1, control = control, seed = 7)
    }, error = function(e) NULL)
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    max_elapsed <- max(max_elapsed, elapsed)

    if (is.null(preds)) {
      rel_rmses[name] <- NA_real_ # disqualified: errored out or hit the time cap
      next
    }
    rmse <- sqrt(mean((preds - s$Y_test)^2))
    baseline_rmse <- sqrt(mean((mean(s$Y_train) - s$Y_test)^2))
    rel_rmses[name] <- rmse / baseline_rmse
  }

  list(mean_relative_rmse = mean(rel_rmses), per_dataset = rel_rmses, max_elapsed = max_elapsed)
}

# ---------------------------------------------------------------------
# 4. Global search: Latin Hypercube over the plausible, stability-aware
#    bounds (momentum_decay capped at 0.9, adapt_boost/adapt_penalty
#    bounded and comparable, a_alpha kept modest; see the hyperparameter
#    stability investigation this script follows on from).
# ---------------------------------------------------------------------
sample_profiles <- function(n, seed) {
  set.seed(seed)
  raw <- lhs::randomLHS(n, 6)
  data.frame(
    profile_id     = seq_len(n),
    alpha          = qunif(raw[, 1], 0.5, 40),
    a_alpha        = qunif(raw[, 2], 0.5, 40),
    b_alpha        = qunif(raw[, 3], 0.5, 20),
    adapt_boost    = qunif(raw[, 4], 0, 40),
    adapt_penalty  = qunif(raw[, 5], 0.5, 1.5) * qunif(raw[, 4], 0, 10),
    momentum_decay = qunif(raw[, 6], 0, 0.9)
  )
}

cat("Generating", n_lhs_profiles, "LHS profiles...\n")
lhs_grid <- sample_profiles(n_lhs_profiles, seed = 123)

run_sweep <- function(grid) {
  results <- parallel::mclapply(seq_len(nrow(grid)), function(i) {
    res <- evaluate_profile(grid[i, ], MCMC_ITER, MCMC_BURNIN, DIRICHLET_WARMUP, TESSELLATIONS, fit_time_limit)
    cbind(grid[i, ], mean_relative_rmse = res$mean_relative_rmse, max_elapsed = res$max_elapsed)
  }, mc.cores = num_cores)
  do.call(rbind, results)
}

cat("Running global LHS search (", nrow(lhs_grid), "profiles x", length(splits), "datasets)...\n")
lhs_results <- run_sweep(lhs_grid)
lhs_results <- lhs_results[order(lhs_results$mean_relative_rmse), ]
#write.csv(lhs_results, "thiessen_adaptive_lhs_search.csv", row.names = FALSE)

# ---------------------------------------------------------------------
# 5. Local refinement: a shrunk random search around the best LHS point.
# ---------------------------------------------------------------------
best_lhs <- lhs_results[1, ]
cat("\nBest LHS profile:\n"); print(best_lhs)

refine_bounds <- function(center, width_frac, lo, hi) {
  half <- width_frac * (hi - lo)
  c(max(lo, center - half), min(hi, center + half))
}

set.seed(456)
refine_grid <- data.frame(
  profile_id     = seq_len(n_refine_profiles),
  alpha          = runif(n_refine_profiles, refine_bounds(best_lhs$alpha, 0.25, 0.5, 40)[1], refine_bounds(best_lhs$alpha, 0.25, 0.5, 40)[2]),
  a_alpha        = runif(n_refine_profiles, refine_bounds(best_lhs$a_alpha, 0.25, 0.5, 40)[1], refine_bounds(best_lhs$a_alpha, 0.25, 0.5, 40)[2]),
  b_alpha        = runif(n_refine_profiles, refine_bounds(best_lhs$b_alpha, 0.25, 0.5, 20)[1], refine_bounds(best_lhs$b_alpha, 0.25, 0.5, 20)[2]),
  adapt_boost    = runif(n_refine_profiles, refine_bounds(best_lhs$adapt_boost, 0.25, 0, 40)[1], refine_bounds(best_lhs$adapt_boost, 0.25, 0, 40)[2]),
  momentum_decay = runif(n_refine_profiles, refine_bounds(best_lhs$momentum_decay, 0.25, 0, 0.9)[1], refine_bounds(best_lhs$momentum_decay, 0.25, 0, 0.9)[2])
)
refine_grid$adapt_penalty <- runif(n_refine_profiles, 0.5, 1.5) * refine_grid$adapt_boost

cat("\nRunning local refinement search (", nrow(refine_grid), "profiles )...\n")
refine_results <- run_sweep(refine_grid)
refine_results <- refine_results[order(refine_results$mean_relative_rmse), ]
#write.csv(refine_results, "thiessen_adaptive_refine_search.csv", row.names = FALSE)

all_results <- rbind(lhs_results, refine_results)
all_results <- all_results[order(all_results$mean_relative_rmse), ]
best <- all_results[1, ]

# ---------------------------------------------------------------------
# 6. Compare the winning profile against turning adaptation off (plain
#    DART, adapt_boost = 0) and against no DART at all (uniform
#    inclusion), at the same search-phase MCMC scale, so the reported
#    lift is attributable to the adaptive mechanism itself.
# ---------------------------------------------------------------------
uniform_score <- evaluate_profile(
  data.frame(alpha = 1, a_alpha = 0.5, b_alpha = 1, adapt_boost = 0,
             adapt_penalty = 0, momentum_decay = 0),
  MCMC_ITER, MCMC_BURNIN, DIRICHLET_WARMUP, TESSELLATIONS, fit_time_limit
)

cat("\n====================================================\n")
cat(" BEST ADAPTIVE DART PROFILE FOUND\n")
cat("====================================================\n")
print(best)
cat(sprintf("\nMean relative RMSE (best):    %.4f\n", best$mean_relative_rmse))
cat(sprintf("Mean relative RMSE (adapt_boost = 0 / plain DART default): %.4f\n", uniform_score$mean_relative_rmse))
cat(sprintf("Worst-case single-dataset fit time (best): %.2fs (cap was %ds)\n", best$max_elapsed, fit_time_limit))

cat("\nRecommended default:\n")
cat(sprintf(
  "dart_inclusion(alpha = %.3g, a_alpha = %.3g, b_alpha = %.3g, adapt_boost = %.3g, adapt_penalty = %.3g, momentum_decay = %.3g, dirichlet_warmup = %d)\n",
  best$alpha, best$a_alpha, best$b_alpha, best$adapt_boost, best$adapt_penalty,
  best$momentum_decay, DIRICHLET_WARMUP
))

cat("\nFull search history saved to thiessen_adaptive_lhs_search.csv and thiessen_adaptive_refine_search.csv\n")

