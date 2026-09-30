frame_fixture <- function(n = 40) {
  fixture <- small_fixture(n)
  data.frame(
    y = fixture$y,
    a = fixture$x[, 1],
    b = fixture$x[, 2],
    g = factor(rep(c("low", "mid", "high"), length.out = n),
               levels = c("low", "mid", "high"))
  )
}

test_that("a formula fit takes the response and the named covariates", {
  frame <- frame_fixture()

  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1, chains = 1)

  expect_s3_class(fit, "thiessen")
  expect_identical(fit$n_features, 2L)
  expect_identical(nobs(fit), 40L)
})

test_that("a dot on the right side takes every remaining column", {
  frame <- frame_fixture()

  fit <- thiessen(y ~ ., frame, small_control(), seed = 1, chains = 1)

  # a, b and the two indicators of the three-level factor.
  expect_identical(fit$n_features, 4L)
})

test_that("a factor becomes d - 1 treatment-contrast indicators", {
  frame <- frame_fixture()

  fit <- thiessen(y ~ g, frame, small_control(), seed = 1, chains = 1)

  expect_identical(fit$n_features, 2L)
  expect_identical(colnames(fit$x), c("gmid", "ghigh"))
  # The first level is the reference, so its rows carry no indicator.
  expect_true(all(rowSums(fit$x[frame$g == "low", ]) == 0))
})

test_that("a data frame fit matches the formula fit over the same columns", {
  frame <- frame_fixture()

  formula_fit <- thiessen(y ~ a + b, frame, small_control(), seed = 5,
                          chains = 1)
  frame_fit <- thiessen(frame[c("a", "b")], frame$y, small_control(), seed = 5,
                        chains = 1)

  expect_identical(fitted(frame_fit), fitted(formula_fit))
})

test_that("a matrix fit matches the data frame fit over the same columns", {
  frame <- frame_fixture()

  matrix_fit <- thiessen(
    as.matrix(frame[c("a", "b")]), frame$y, small_control(), seed = 5,
    chains = 1
  )
  frame_fit <- thiessen(frame[c("a", "b")], frame$y, small_control(), seed = 5,
                        chains = 1)

  expect_identical(fitted(matrix_fit), fitted(frame_fit))
})

test_that("reordered columns predict identically", {
  frame <- frame_fixture()
  fit <- thiessen(y ~ a + b + g, frame, small_control(), seed = 1, chains = 1)

  straight <- predict(fit, frame)
  shuffled <- predict(fit, frame[c("g", "y", "b", "a")])

  expect_identical(shuffled, straight)
})

test_that("a missing column is reported by name", {
  frame <- frame_fixture()
  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1, chains = 1)

  expect_error(predict(fit, frame["a"]), class = "thiessen_error")
  expect_error(predict(fit, frame["a"]), "b")
})

test_that("a level not seen at fit is refused", {
  frame <- frame_fixture()
  fit <- thiessen(y ~ g, frame, small_control(), seed = 1, chains = 1)
  novel <- frame
  levels(novel$g) <- c(levels(frame$g), "other")
  novel$g[1] <- "other"

  # hardhat removes the level and leaves NA, which the design then refuses.
  expect_warning(
    expect_error(predict(fit, novel), class = "thiessen_error"),
    "Novel level"
  )
})

test_that("predict at the training frame is the fitted values", {
  frame <- frame_fixture()

  fit <- thiessen(y ~ a + b + g, frame, small_control(), seed = 1, chains = 1)

  expect_identical(predict(fit, frame), fitted(fit))
})

test_that("a two-level factor response becomes 0 and 1", {
  frame <- frame_fixture()
  frame$label <- factor(ifelse(frame$y >= stats::median(frame$y), "yes", "no"))

  fit <- thiessen(label ~ a + b, frame,
                  small_control(outcome = probit_outcome()),
                  seed = 1, chains = 1)

  expect_identical(fit$response$levels, c("no", "yes"))
  expect_true(all(fitted(fit) >= 0 & fitted(fit) <= 1))
})

test_that("an unordered factor response of more than two levels is rejected", {
  frame <- frame_fixture()

  expect_error(
    thiessen(g ~ a + b, frame, small_control(), seed = 1, chains = 1),
    "Unordered factor responses with more than two levels are not supported",
    class = "thiessen_error"
  )
})

test_that("a declared metric passes a factor as level codes", {
  frame <- frame_fixture()
  control <- small_control(mean_params = term_params(
    geometry = geometry_params(
      metric = c("euclidean", "euclidean", "categorical")
    )
  ))

  fit <- thiessen(y ~ a + b + g, frame, control, seed = 1, chains = 1)

  expect_identical(fit$n_features, 3L)
  expect_identical(sort(unique(fit$x[, 3])), c(0, 1, 2))
})

test_that("a factor whose metric is not categorical is refused", {
  frame <- frame_fixture()
  control <- small_control(mean_params = term_params(
    geometry = geometry_params(
      metric = c("euclidean", "euclidean", "euclidean")
    )
  ))

  expect_error(
    thiessen(y ~ a + b + g, frame, control, seed = 1, chains = 1),
    class = "thiessen_error"
  )
})

test_that("a categorical fit predicts on new rows", {
  frame <- frame_fixture()
  control <- small_control(mean_params = term_params(
    geometry = geometry_params(
      metric = c("euclidean", "euclidean", "categorical")
    )
  ))
  fit <- thiessen(y ~ a + b + g, frame, control, seed = 1, chains = 1)

  expect_identical(predict(fit, frame), fitted(fit))
})

test_that("update refits with the argument replaced", {
  frame <- frame_fixture()
  fit <- thiessen(y ~ a + b, frame, small_control(), seed = 1, chains = 1)

  again <- update(fit, seed = 2)

  expect_s3_class(again, "thiessen")
  expect_identical(again$seed, 2)
  expect_false(identical(fitted(again), fitted(fit)))
})

test_that("update works on a matrix fit", {
  fixture <- small_fixture()
  fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)

  again <- update(fit, seed = 2)

  expect_identical(again$seed, 2)
})

test_that("the blueprint is stored on a formula fit and absent on a matrix fit", {
  frame <- frame_fixture()
  fixture <- small_fixture()

  expect_s3_class(
    thiessen(y ~ a, frame, small_control(), seed = 1, chains = 1)$blueprint,
    "hardhat_blueprint"
  )
  expect_null(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1,
             chains = 1)$blueprint
  )
})
