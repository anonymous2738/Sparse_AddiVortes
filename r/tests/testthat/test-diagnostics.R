test_that("the diagnostics carry one row per draw", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  diagnostics <- thiessen_diagnostics(fit)

  expect_identical(
    names(diagnostics),
    c("chain", "draw", "sigma", "cell_count", "dimension_count")
  )
  expect_identical(nrow(diagnostics), 20L)
  expect_identical(diagnostics$chain, rep(1L, 20L))
  expect_identical(diagnostics$draw, 1:20)
  expect_true(all(diagnostics$sigma > 0))
  expect_true(all(diagnostics$cell_count >= 1))
})

test_that("the sigma column is absent where the model has no sigma", {
  fixture <- small_fixture()
  y <- as.numeric(fixture$y > stats::median(fixture$y))

  probit <- thiessen(
    fixture$x, y, small_control(outcome = probit_outcome()), seed = 1,
    chains = 1
  )
  heteroscedastic <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)), seed = 1,
    chains = 1
  )

  expect_false("sigma" %in% names(thiessen_diagnostics(probit)))
  expect_false("sigma" %in% names(thiessen_diagnostics(heteroscedastic)))
})

test_that("the sigma column is the sigma draws of the draws object", {
  skip_if_not_installed("posterior")
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  draws <- posterior::as_draws_df(fit)

  expect_identical(thiessen_diagnostics(fit)$sigma, draws$sigma)
})

test_that("the momentum trace records the requested post-warmup sweeps", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  inclusion <- dart_inclusion(
    adapt_boost = 0.5, adapt_penalty = 0.5,
    dirichlet_warmup = 2, momentum_settle = 2,
    momentum_trace_every = 3
  )
  control <- small_control(mean_params = term_params(
    structure = structure_params(omega = 1, inclusion = inclusion)
  ))

  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, control, seed = 1, chains = 2)
  )
  trace <- thiessen_momentum_trace(fit)

  expect_identical(names(trace),
                   c("chain", "iteration", "momentum[1]", "momentum[2]"))
  expect_identical(trace$chain, c(1L, 1L, 2L, 2L))
  expect_identical(trace$iteration, c(5L, 8L, 5L, 8L))
  expect_true(all(is.finite(as.matrix(trace[3:4]))))
})

test_that("the momentum trace labels values with predictor names", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  colnames(fixture$x) <- c("signal", "noise")
  inclusion <- dart_inclusion(
    adapt_boost = 0.5, adapt_penalty = 0.5,
    dirichlet_warmup = 2, momentum_settle = 2,
    momentum_trace_every = 3
  )
  control <- small_control(mean_params = term_params(
    structure = structure_params(omega = 1, inclusion = inclusion)
  ))

  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, control, seed = 1, chains = 1)
  )

  expect_identical(names(thiessen_momentum_trace(fit)), c(
    "chain", "iteration", "momentum[signal]", "momentum[noise]"
  ))
})

test_that("momentum warmup adapts under uniform weights before Dirichlet", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  inclusion <- dart_inclusion(
    adapt_boost = 0.5, adapt_penalty = 0.5,
    dirichlet_warmup = 2, momentum_warmup = 3,
    momentum_settle = 2, momentum_trace_every = 2
  )
  control <- small_control(mean_params = term_params(
    structure = structure_params(omega = 1, inclusion = inclusion)
  ))

  fit <- thiessen(fixture$x, fixture$y, control, seed = 1, chains = 1)
  trace <- thiessen_momentum_trace(fit)

  expect_identical(trace$iteration, c(4L, 6L, 8L))
})

test_that("momentum tracing is disabled by default", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  control <- small_control(mean_params = term_params(
    structure = structure_params(inclusion = dart_inclusion(
      adapt_boost = 0.5, adapt_penalty = 0.5, dirichlet_warmup = 2
    ))
  ))

  fit <- thiessen(fixture$x, fixture$y, control, seed = 1, chains = 1)

  expect_identical(nrow(thiessen_momentum_trace(fit)), 0L)
})

test_that("the inclusion proportions sum to one and name the columns", {
  fixture <- small_fixture()
  frame <- data.frame(y = fixture$y, a = fixture$x[, 1], b = fixture$x[, 2])

  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1, chains = 1)
  inclusion <- variable_inclusion(fit)

  expect_identical(names(inclusion), c("a", "b"))
  expect_equal(sum(inclusion), 1, tolerance = 1e-12)
})

test_that("inclusion counts are derived per draw on request", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  counts <- thiessen_inclusion_counts(fit)

  expect_type(counts, "integer")
  expect_identical(colnames(counts), c("x1", "x2"))
  expect_identical(dim(counts), c(fit$n_draws, 2L))
  expect_equal(
    rowSums(counts),
    thiessen_diagnostics(fit)$dimension_count * fit$control$mean_params$tessellations
  )
})

test_that("dimension histograms summarize each retained draw", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  histograms <- thiessen_dimension_histograms(fit)
  dimensions <- seq_len(ncol(histograms))

  expect_type(histograms, "integer")
  expect_identical(dim(histograms), c(fit$n_draws, ncol(fixture$x)))
  expect_identical(colnames(histograms), paste0("dimension_", dimensions))
  expect_equal(
    rowSums(histograms),
    rep(fit$control$mean_params$tessellations, fit$n_draws)
  )
  expect_equal(
    drop(histograms %*% dimensions),
    thiessen_diagnostics(fit)$dimension_count *
      fit$control$mean_params$tessellations
  )
  expect_identical(
    dim(thiessen_augmented_counts(fit)),
    c(0L, ncol(fixture$x))
  )
})

test_that("DART augmented counts are available per kept draw", {
  skip_if_not(core_experimental())
  fixture <- small_fixture()
  control <- small_control(mean_params = term_params(
    structure = structure_params(inclusion = dart_inclusion(
      dirichlet_warmup = 2
    ))
  ))
  fit <- suppressWarnings(
    thiessen(fixture$x, fixture$y, control, seed = 1, chains = 2)
  )
  counts <- thiessen_augmented_counts(fit)

  expect_identical(dim(counts), c(fit$n_draws, ncol(fixture$x)))
  expect_identical(colnames(counts), c("x1", "x2"))
  expect_true(all(is.finite(counts)))
  expect_true(all(counts >= 0))
  restored <- unserialize(serialize(fit, NULL))
  expect_identical(thiessen_augmented_counts(restored), counts)
})

test_that("DART draws are available without fitted-value draws", {
  skip_if_not(core_experimental())
  skip_if_not_installed("posterior")
  fixture <- small_fixture()
  control <- small_control(mean_params = term_params(
    structure = structure_params(inclusion = dart_inclusion(
      adapt_boost = 0.5, adapt_penalty = 0.5, dirichlet_warmup = 2
    ))
  ))
  fit <- thiessen(fixture$x, fixture$y, control, seed = 1, chains = 1)
  dart <- thiessen_dart_draws(fit)
  draws <- posterior::as_draws_df(fit)
  weight_columns <- grep("^inclusion_weight\\[", names(draws), value = TRUE)
  momentum_columns <- grep("^momentum\\[", names(draws), value = TRUE)

  expect_identical(names(dart), c("concentration", "inclusion_weight", "momentum"))
  expect_equal(dart$concentration, draws$concentration)
  expect_equal(
    unname(dart$inclusion_weight),
    unname(suppressWarnings(as.matrix(draws[weight_columns])))
  )
  expect_equal(
    unname(dart$momentum),
    unname(suppressWarnings(as.matrix(draws[momentum_columns])))
  )
})

test_that("the accessors refuse an object that is not a fit", {
  expect_error(thiessen_diagnostics(1), class = "thiessen_error")
  expect_error(thiessen_dart_draws(1), class = "thiessen_error")
  expect_error(thiessen_inclusion_counts(1), class = "thiessen_error")
  expect_error(thiessen_dimension_histograms(1), class = "thiessen_error")
  expect_error(thiessen_augmented_counts(1), class = "thiessen_error")
  expect_error(variable_inclusion(1), class = "thiessen_error")
})
