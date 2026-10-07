test_that("a fit reports its dimensions and schedule", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)

  expect_s3_class(fit, "thiessen")
  expect_identical(fit$model, "gaussian")
  expect_identical(fit$n_draws, 20L)
  expect_identical(nobs(fit), 40L)
  expect_length(fitted(fit), 40L)
  expect_length(residuals(fit), 40L)
  expect_equal(residuals(fit), fixture$y - fitted(fit))
})

test_that("xtest returns predictions without creating a fitted model", {
  fixture <- small_fixture()
  expected <- thiessen(
    fixture$x, fixture$y, small_control(), seed = 1, chains = 1
  )
  actual <- thiessen(
    fixture$x, fixture$y, small_control(), seed = 1, chains = 1,
    threads = 2, xtest = fixture$x
  )

  expect_type(actual, "double")
  expect_length(actual, nrow(fixture$x))
  expect_equal(actual, predict(expected, fixture$x))
})

test_that("the resolved configuration is on the fit", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)

  expect_s3_class(fit$control, "thiessen_control")
  expect_identical(fit$control$mean_params$tessellations, 8L)
  # omega is unset in the control and resolves to min(3, p) at fit.
  expect_identical(fit$control$mean_params$structure$omega, 2)
})

test_that("a vector design is one column", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x[, 1], fixture$y, small_control(), seed = 1,
                  chains = 1)

  expect_identical(fit$n_features, 1L)
})

test_that("the same seed gives the same fit", {
  fixture <- small_fixture()

  first <- thiessen(fixture$x, fixture$y, small_control(), seed = 3, chains = 1)
  again <- thiessen(fixture$x, fixture$y, small_control(), seed = 3, chains = 1)

  expect_identical(fitted(again), fitted(first))
})

test_that("a different seed gives a different fit", {
  fixture <- small_fixture()

  first <- thiessen(fixture$x, fixture$y, small_control(), seed = 3, chains = 1)
  other <- thiessen(fixture$x, fixture$y, small_control(), seed = 4, chains = 1)

  expect_false(identical(fitted(other), fitted(first)))
})

test_that("set.seed governs when the seed is left unset", {
  fixture <- small_fixture()

  set.seed(11)
  first <- thiessen(fixture$x, fixture$y, small_control(), chains = 1)
  set.seed(11)
  again <- thiessen(fixture$x, fixture$y, small_control(), chains = 1)

  expect_identical(again$seed, first$seed)
  expect_identical(fitted(again), fitted(first))
})

test_that("the seed used is stored", {
  fixture <- small_fixture()

  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 12345,
                  chains = 1)

  expect_identical(fit$seed, 12345)
})

test_that("an invalid seed is rejected", {
  fixture <- small_fixture()

  expect_error(
    thiessen(fixture$x, fixture$y, small_control(), seed = -1, chains = 1),
    class = "thiessen_error"
  )
  expect_error(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1.5, chains = 1),
    class = "thiessen_error"
  )
  expect_error(
    thiessen(fixture$x, fixture$y, small_control(), seed = c(1, 2), chains = 1),
    class = "thiessen_error"
  )
})

test_that("the design and the response must agree", {
  fixture <- small_fixture()

  expect_error(
    thiessen(fixture$x, fixture$y[-1], small_control(), seed = 1, chains = 1),
    class = "thiessen_error"
  )
})

test_that("a missing value is rejected", {
  fixture <- small_fixture()
  x <- fixture$x
  x[1, 1] <- NA
  y <- fixture$y
  y[1] <- NA

  expect_error(
    thiessen(x, fixture$y, small_control(), seed = 1, chains = 1),
    class = "thiessen_error"
  )
  expect_error(
    thiessen(fixture$x, y, small_control(), seed = 1, chains = 1),
    class = "thiessen_error"
  )
})

test_that("a matrix fit refuses a data frame at predict", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)

  expect_error(
    predict(fit, as.data.frame(fixture$x)),
    class = "thiessen_error"
  )
})

test_that("a response the model does not admit is refused by the core", {
  fixture <- small_fixture()

  expect_error(
    thiessen(
      fixture$x, fixture$y, small_control(outcome = probit_outcome()),
      seed = 1, chains = 1
    ),
    class = "thiessen_error"
  )
})

test_that("more features than observations warns", {
  x <- matrix(seq_len(24) / 24, nrow = 4)
  y <- c(0.1, 0.2, 0.3, 0.4)

  expect_warning(
    thiessen(x, y, small_control(), seed = 1, chains = 1),
    class = "thiessen_warning"
  )
})

test_that("the probit model fits a binary response", {
  fixture <- small_fixture()
  labels <- as.double(fixture$y >= stats::median(fixture$y))

  fit <- thiessen(
    fixture$x, labels, small_control(outcome = probit_outcome()), seed = 1,
    chains = 1
  )

  expect_identical(fit$model, "probit")
  expect_true(all(fitted(fit) >= 0 & fitted(fit) <= 1))
})

test_that("the heteroscedastic model fits", {
  fixture <- small_fixture()

  fit <- thiessen(
    fixture$x, fixture$y,
    small_control(variance_params = term_params(tessellations = 4)),
    seed = 1, chains = 1
  )

  expect_identical(fit$model, "heteroscedastic")
})

test_that("a non-finite value is rejected, naming its position", {
  fixture <- small_fixture()
  x <- fixture$x
  x[3, 1] <- Inf

  expect_error(
    thiessen(x, fixture$y, small_control(), seed = 1, chains = 1),
    "not finite",
    class = "thiessen_error"
  )
  y <- fixture$y
  y[3] <- -Inf
  expect_error(
    thiessen(fixture$x, y, small_control(), seed = 1, chains = 1),
    "not finite",
    class = "thiessen_error"
  )
})

test_that("zero-length data is rejected", {
  expect_error(
    thiessen(matrix(numeric(0), ncol = 1L), numeric(0), small_control(),
             seed = 1, chains = 1),
    "observations",
    class = "thiessen_error"
  )
})

test_that("a constant column is rejected, naming the column", {
  fixture <- small_fixture()
  x <- cbind(fixture$x[, 1], 1)

  expect_error(
    thiessen(x, fixture$y, small_control(), seed = 1, chains = 1),
    "constant",
    class = "thiessen_error"
  )
})

test_that("a list column is refused", {
  fixture <- small_fixture()
  frame <- data.frame(y = fixture$y, a = fixture$x[, 1])
  frame$b <- I(as.list(seq_len(nrow(frame))))

  expect_error(thiessen(y ~ ., frame, small_control(), seed = 1,
                        chains = 1), "list")
})

test_that("a numeric column with foreign attributes fits as its values", {
  fixture <- small_fixture()
  decorated <- fixture$x[, 1]
  attr(decorated, "unit") <- "metres"
  frame <- data.frame(y = fixture$y, a = decorated, b = fixture$x[, 2])
  plain <- data.frame(y = fixture$y, a = fixture$x[, 1], b = fixture$x[, 2])

  with_attributes <- thiessen(y ~ ., frame, small_control(), seed = 1,
                              chains = 1)
  without <- thiessen(y ~ ., plain, small_control(), seed = 1, chains = 1)

  expect_identical(fitted(with_attributes), fitted(without))
})
