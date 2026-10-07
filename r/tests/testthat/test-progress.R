# progressr signals nothing in a non-interactive session unless it is
# enabled, and `with_progress()` muffles every condition before a handler
# established outside it runs, so collect the conditions with the option set
# and no progressr handler in the way.
progressions <- function(expr) {
  old <- options(progressr.enable = TRUE)
  on.exit(options(old), add = TRUE)
  seen <- data.frame(
    type = character(0), amount = numeric(0), message = character(0),
    sticky = logical(0)
  )
  withCallingHandlers(
    expr,
    progression = function(condition) {
      seen[nrow(seen) + 1L, ] <<- list(
        condition$type,
        if (is.null(condition$amount)) NA_real_ else condition$amount,
        paste(conditionMessage(condition), collapse = ""),
        inherits(condition, "sticky")
      )
    }
  )
  seen
}

updates_of <- function(seen) {
  seen[seen$type == "update", ]
}

# The steps the whole fit takes, sweeps and the phases that follow them.
total_advance <- function(seen) {
  as.integer(sum(updates_of(seen)$amount))
}

# The steps the sweeps take: those the first phase after them precedes.
sweep_updates <- function(seen) {
  updates <- updates_of(seen)
  pooling <- which(updates$message == "pooling the draws")[1L]
  as.integer(sum(updates$amount[seq_len(pooling - 1L)]))
}

phase_messages <- function(seen) {
  updates <- updates_of(seen)
  updates$message[nzchar(updates$message)]
}

test_that("a fit signals every progression it requests", {
  fixture <- small_fixture()
  control <- thiessen_control(
    tessellations = 8,
    general_params = general_params(burn_in = 40, draws = 80)
  )

  expect_identical(progress_updates(control), 100L)
  seen <- progressions(thiessen(fixture$x, fixture$y, control, seed = 1,
                                chains = 1))
  expect_identical(sweep_updates(seen), 100L)
  expect_identical(
    total_advance(seen), progress_steps(control, nrow(fixture$x))
  )
})

test_that("a fit shorter than a hundred sweeps signals one per sweep", {
  fixture <- small_fixture()

  expect_identical(progress_updates(small_control()), 30L)
  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  )
  expect_identical(sweep_updates(seen), 30L)
})

test_that("xtest reports progress across its prediction schedule", {
  fixture <- small_fixture()

  seen <- progressions(thiessen(
    fixture$x, fixture$y, small_control(), seed = 1, chains = 1,
    xtest = fixture$x
  ))

  expect_identical(total_advance(seen), progress_updates(small_control()))
})

test_that("the sweeps of every chain are reported", {
  fixture <- small_fixture()

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1)
  ))

  expect_identical(sweep_updates(seen), progress_updates(small_control(), 2L))
  expect_identical(
    total_advance(seen),
    progress_steps(small_control(), nrow(fixture$x), 2L)
  )
})

# The sweeps once took every step of the progressor, which finished the
# report at the last sweep and left pooling and the convergence summary
# running under a closed handler.
test_that("the sweeps leave the budget the phases after them need", {
  fixture <- small_fixture()
  sweeps <- progress_updates(small_control())

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  )

  expect_identical(sweep_updates(seen), sweeps)
  expect_identical(
    total_advance(seen) - sweeps,
    sum(progress_phase_steps(small_control(), nrow(fixture$x)))
  )
  expect_identical(
    phase_messages(seen),
    c(
      "sampling", "pooling the draws", "saving the state",
      "summarising the draws"
    )
  )
})

# The phases after the sweeps are each one call into the core or into
# posterior, so the bar rests where the sweeps leave it for as long as each
# runs. Their share of the bar follows their cost: they scale with the kept
# draws on one thread, the sweeps with the training rows on the threads.
test_that("the sweeps take most of the bar on a one-chain default fit", {
  control <- thiessen_control()

  fraction <- progress_updates(control) / progress_steps(control, n = 200L)

  expect_gt(fraction, 0.6)
})

test_that("the phases after the sweeps take more of the bar as they cost more", {
  tail_steps <- function(n, chains = 1L, threads = 1L, ...) {
    sum(progress_phase_steps(thiessen_control(...), n, chains, threads))
  }
  longer <- general_params(burn_in = 200, draws = 2000)

  expect_lt(tail_steps(1000L), tail_steps(200L))
  expect_gt(tail_steps(200L, general_params = longer), tail_steps(200L))
  expect_gt(tail_steps(200L, 4L, 4L), tail_steps(200L, 4L, 1L))
  expect_identical(tail_steps(200L, 4L, 4L), tail_steps(200L, 4L, 8L))
})

test_that("every phase after the sweeps takes at least one step", {
  steps <- progress_phase_steps(small_control(), 40L)
  expect_named(steps, c("pooling", "saving", "summarising"))
  expect_true(all(steps >= 1L))

  steps <- progress_phase_steps(thiessen_control(), 100000L, 4L, 4L)
  expect_true(all(steps >= 1L))
})

test_that("the bar completes at the last phase and not before", {
  fixture <- small_fixture()

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  )
  updates <- updates_of(seen)
  total <- progress_steps(small_control(), nrow(fixture$x))

  expect_lt(sum(updates$amount[-nrow(updates)]), total)
  expect_identical(as.integer(sum(updates$amount)), total)
})

# Under a terminal handler a plain message is overwritten by the next bar
# redraw, so only a sticky one leaves the phase on screen.
test_that("every phase names itself in a sticky progression", {
  fixture <- small_fixture()

  seen <- progressions(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  )
  named <- seen[nzchar(seen$message), ]

  expect_identical(
    named$message,
    c(
      "sampling", "pooling the draws", "saving the state",
      "summarising the draws"
    )
  )
  expect_true(all(named$sticky))
})

test_that("the chains of a fit are named as they are sampled", {
  fixture <- small_fixture()

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1)
  ))

  expect_identical(
    phase_messages(seen),
    c(
      "sampling 2 chains", "pooling the draws", "saving the state",
      "summarising the draws"
    )
  )

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 2, seed = 1,
             threads = 2)
  ))

  expect_identical(
    phase_messages(seen),
    c(
      "sampling 2 chains on 2 threads", "pooling the draws",
      "saving the state", "summarising the draws"
    )
  )
})

test_that("the sweeps of threaded chains are reported", {
  fixture <- small_fixture()

  seen <- progressions(suppressWarnings(
    thiessen(fixture$x, fixture$y, small_control(), chains = 3, seed = 1,
             threads = 2)
  ))

  expect_identical(sweep_updates(seen), progress_updates(small_control(), 3L))
  expect_identical(
    total_advance(seen),
    progress_steps(small_control(), nrow(fixture$x), 3L, 2L)
  )
})

test_that("the number of updates does not exceed the sweeps", {
  expect_identical(
    progress_updates(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2))
    ),
    3L
  )
  expect_identical(
    progress_updates(
      thiessen_control(general_params = general_params(burn_in = 1, draws = 2)),
      chains = 2
    ),
    6L
  )
  expect_identical(
    progress_updates(
      thiessen_control(
        general_params = general_params(burn_in = 100, draws = 100)
      )
    ),
    100L
  )
})

# progressr's terminal handlers write nothing in a non-interactive session,
# so an assertion on captured output reports on the printed fit rather than
# on the progress bar. The counts above are the check on the mechanism; this
# one is that the handler pipeline runs at all.
test_that("a fit runs under a handler", {
  fixture <- small_fixture()

  fit <- progressr::with_progress(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1),
    handlers = progressr::handler_void()
  )

  expect_s3_class(fit, "thiessen")
})

test_that("nothing is printed without a handler", {
  fixture <- small_fixture()

  expect_silent(thiessen(fixture$x, fixture$y, small_control(), seed = 1,
                         chains = 1))
})

test_that("a sampler assembles a fit without a progressor", {
  fixture <- small_fixture()

  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(5)
  sampler$keep()

  expect_s3_class(sampler$finish(), "thiessen")
})

test_that("reporting progress does not change the draws", {
  fixture <- small_fixture()

  plain <- thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1)
  reported <- progressr::with_progress(
    thiessen(fixture$x, fixture$y, small_control(), seed = 1, chains = 1),
    handlers = progressr::handler_void()
  )

  expect_identical(fitted(reported), fitted(plain))
  expect_identical(thiessen_diagnostics(reported), thiessen_diagnostics(plain))
})
