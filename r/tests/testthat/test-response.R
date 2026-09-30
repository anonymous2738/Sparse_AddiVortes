# The response selects the outcome family where the control names none,
# and a named family that disagrees is an error naming both. These rules
# are the package's own, so they hold in either build.

test_that("the response selects the family where the control names none", {
  fixture <- small_fixture()
  labels <- factor(fixture$x[, 2] > 0.3)

  gaussian <- thiessen(fixture$x, fixture$y, small_control(), seed = 1,
                       chains = 1)
  probit <- thiessen(fixture$x, labels, small_control(), seed = 1, chains = 1)

  expect_null(small_control()$outcome)
  expect_identical(attr(gaussian$control$outcome, "kind"), "gaussian")
  expect_identical(attr(probit$control$outcome, "kind"), "probit")
  expect_identical(probit$response$levels, levels(labels))
})

test_that("a named family that disagrees with the response is an error", {
  fixture <- small_fixture()
  n <- nrow(fixture$x)
  ordered <- factor(c("a", "b")[(seq_len(n) %% 2) + 1], ordered = TRUE)

  condition <- rlang::catch_cnd(
    thiessen(
      fixture$x, ordered, small_control(outcome = gaussian_outcome()),
      seed = 1, chains = 1
    )
  )

  expect_s3_class(condition, "thiessen_error")
  expect_match(conditionMessage(condition), "ordinal")
  expect_match(conditionMessage(condition), "gaussian")
  expect_error(
    thiessen(
      fixture$x, fixture$y, small_control(outcome = probit_outcome()),
      seed = 1, chains = 1
    ),
    class = "thiessen_error"
  )
})

test_that("a Surv of another type is rejected by name", {
  skip_if_not_installed("survival")
  fixture <- small_fixture()
  n <- nrow(fixture$x)
  y <- survival::Surv(rep(0, n), exp(fixture$y), rep(1, n), type = "counting")

  expect_error(
    thiessen(fixture$x, y, small_control(), seed = 1, chains = 1),
    "counting",
    class = "thiessen_error"
  )
})

test_that("a factor of more than two unordered levels is unsupported", {
  fixture <- small_fixture()
  n <- nrow(fixture$x)
  y <- factor(c("a", "b", "c")[(seq_len(n) %% 3) + 1])

  expect_error(
    thiessen(fixture$x, y, small_control(), seed = 1, chains = 1),
    "Unordered factor responses with more than two levels are not supported",
    class = "thiessen_error"
  )
})

test_that("the control prints an unnamed family as taken from the response", {
  expect_output(print(thiessen_control()), "from the response")
})

test_that("the sampler resolves the family as thiessen() does", {
  fixture <- small_fixture()
  labels <- factor(fixture$x[, 2] > 0.3)

  sampler <- thiessen_sampler(fixture$x, labels, small_control(), seed = 1)
  sampler$step(2)
  sampler$keep()

  expect_identical(attr(sampler$finish()$control$outcome, "kind"), "probit")
})
