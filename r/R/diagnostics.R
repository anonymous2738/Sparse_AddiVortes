# Accessors for the per-draw sampler diagnostics.

#' Per-draw sampler diagnostics of a fitted model
#'
#' The trace of the quantities the sampler records once per kept draw. For a
#' draws object the same quantities, with the mean function, come from
#' [posterior::as_draws_df()], which `bayesplot::mcmc_trace()` takes.
#'
#' @param object An object of class `"thiessen"`.
#'
#' @return A data frame with one row per kept draw and the columns `chain`,
#'   the chain the draw comes from; `draw`, its index within that chain;
#'   `sigma`, the residual standard deviation, under the Gaussian model
#'   only; `cell_count`, the mean cells per mean tessellation; and
#'   `dimension_count`, the mean active covariates per mean tessellation.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' head(thiessen_diagnostics(fit))
#' @export
thiessen_diagnostics <- function(object) {
  check_fit(object)
  state <- fit_state(object)
  counts <- core_call(core_diagnostics(state))
  sigma <- core_call(core_sigma(state))
  iterations <- object$n_draws / object$n_chains
  out <- data.frame(
    chain = rep(seq_len(object$n_chains), each = iterations),
    draw = rep(seq_len(iterations), times = object$n_chains)
  )
  if (length(sigma) > 0L) {
    out$sigma <- sigma
  }
  out$cell_count <- counts$cell_count
  out$dimension_count <- counts$dimension_count
  out
}

#' Recorded adaptive momentum during warmup
#'
#' Returns the optional momentum snapshots requested with
#' `dart_inclusion(momentum_trace_every = ...)`. Snapshots begin after
#' `dirichlet_warmup` and are not posterior draws. A value of zero disables
#' recording and returns a zero-row data frame.
#'
#' @param object An object of class `"thiessen"`.
#' @return A data frame with `chain`, `iteration`, and one `momentum[name]`
#'   column per predictor, using input column names or indices when unnamed.
#' @export
thiessen_momentum_trace <- function(object) {
  check_fit(object)
  trace <- object$momentum_trace
  predictor_names <- colnames(object$x)
  if (is.null(predictor_names)) {
    predictor_names <- as.character(seq_len(ncol(trace) - 2L))
  }
  colnames(trace) <- c(
    "chain", "iteration", paste0("momentum[", predictor_names, "]")
  )
  trace
}

#' Per-draw covariate inclusion counts
#'
#' Counts how many mean tessellations use each covariate in every kept
#' draw. Counts are derived on demand from the retained posterior, so
#' requesting them does not add work to the sampler.
#'
#' @param object An object of class "thiessen".
#' @return An integer matrix, one row per kept draw and one column per
#'   covariate. Rows are in chain order, matching
#'   [posterior::as_draws_matrix()].
#' @export
thiessen_inclusion_counts <- function(object) {
  check_fit(object)
  counts <- core_call(core_inclusion_counts(fit_state(object)))
  predictor_names <- colnames(object$x)
  if (is.null(predictor_names)) {
    predictor_names <- paste0("x", seq_len(ncol(counts)))
  }
  colnames(counts) <- predictor_names
  counts
}

#' Per-draw histogram of tessellation dimensions
#'
#' Counts the mean tessellations with each number of active covariates in
#' every kept draw.
#'
#' @param object An object of class "thiessen".
#' @return An integer matrix, one row per kept draw and one column per
#'   dimension from one through the number of predictors.
#' @export
thiessen_dimension_histograms <- function(object) {
  check_fit(object)
  histograms <- core_call(core_dimension_histograms(fit_state(object)))
  colnames(histograms) <- paste0("dimension_", seq_len(ncol(histograms)))
  histograms
}

#' Per-draw covariate counts used by the DART update
#'
#' Includes the latent augmentation counts after the DART warmup, as well
#' as the dimensions present in the fitted tessellations.
#'
#' @param object An object of class "thiessen".
#' @return A numeric matrix, one row per kept draw and one column per
#'   predictor; zero rows under inclusion priors other than DART.
#' @export
thiessen_augmented_counts <- function(object) {
  check_fit(object)
  counts <- core_call(core_augmented_counts(fit_state(object)))
  predictor_names <- colnames(object$x)
  if (is.null(predictor_names)) {
    predictor_names <- paste0("x", seq_len(ncol(counts)))
  }
  colnames(counts) <- predictor_names
  counts
}

#' Stored Dirichlet-adaptive posterior draws
#'
#' Returns the posterior quantities recorded for the DART inclusion prior
#' without calculating the fitted mean at every training row.
#'
#' @param object An object of class "thiessen".
#' @return A list with `concentration`, `inclusion_weight`, and `momentum`.
#'   The latter two are matrices with one row per kept draw and one column
#'   per covariate.
#' @export
thiessen_dart_draws <- function(object) {
  check_fit(object)
  draws <- core_call(core_posterior_draws(fit_state(object)))
  predictor_names <- colnames(object$x)
  if (is.null(predictor_names)) {
    predictor_names <- paste0("x", seq_len(ncol(draws$inclusion_weight)))
  }
  colnames(draws$inclusion_weight) <- predictor_names
  colnames(draws$momentum) <- predictor_names
  draws[c("concentration", "inclusion_weight", "momentum")]
}

#' Variable inclusion proportions of a fitted model
#'
#' The share of the active dimensions of the mean tessellations falling on
#' each covariate, averaged over the kept draws, as `dbarts::varcount`
#' summarises tree splits. The values sum to one.
#'
#' They report where the ensemble spent its dimensions, not which
#' covariates carry signal, and they inherit the covariate-inclusion prior:
#' at the default `omega` of `min(3, p)` every dimension is always active
#' when p is 3 or fewer, so the proportions are then uniform by
#' construction, exactly 1/p. Separation is weak at p = 4, where two
#' informative covariates measured 0.26 and 0.29 against pure noise at
#' 0.25 and 0.19. Do not read them as variable selection.
#'
#' @param object An object of class `"thiessen"`.
#'
#' @return A numeric vector, one value per column of the design, named as
#'   the design columns are and unnamed where the design has no column
#'   names.
#'
#' @references
#' Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART: Bayesian
#' additive regression trees. *The Annals of Applied Statistics* 4(1),
#' 266-298. \doi{10.1214/09-AOAS285}, s. 5.2.
#'
#' @examples
#' n <- 60
#' x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
#' y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
#' control <- thiessen_control(
#'   tessellations = 10,
#'   general_params = general_params(burn_in = 20, draws = 40)
#' )
#' fit <- thiessen(x, y, control, seed = 1)
#' variable_inclusion(fit)
#' @export
variable_inclusion <- function(object) {
  check_fit(object)
  proportions <- core_call(core_diagnostics(fit_state(object)))$inclusion
  names(proportions) <- colnames(object$x)
  proportions
}

#' @export
ensemble_inclusion <- function(object) {
  check_fit(object)
  proportions <- core_call(core_diagnostics(fit_state(object)))$ensemble_inclusion
  names(proportions) <- colnames(object$x)
  proportions
}

#' Refuse an object that is not a fit
#'
#' @param object The object to check.
#' @param call The calling environment to report.
#' @noRd
check_fit <- function(object, call = rlang::caller_env()) {
  if (!inherits(object, "thiessen")) {
    thiessen_abort("`object` must come from `thiessen()`.", call = call)
  }
  invisible(object)
}
