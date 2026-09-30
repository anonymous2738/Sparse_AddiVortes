# Parity with the core's configuration spec, in both directions: every
# option of the core's serialised defaults is reachable from the surface,
# and the fully populated surface is accepted by the core, so there is no
# silent extra.

full_term <- function(tessellations) {
  term_params(
    tessellations = tessellations,
    k = 2,
    lambda_c = 4,
    geometry = geometry_params(
      metric = c("euclidean", "euclidean"), sigma_c = 0.7,
      membership = "hard"
    ),
    structure = structure_params(omega = 1.5, inclusion = "uniform"),
    cell = cell_params(basis = "constant")
  )
}

full_control <- function() {
  thiessen_control(
    outcome = gaussian_outcome(nu = 5, q = 0.9),
    mean_params = full_term(8),
    variance_params = full_term(4),
    general_params = general_params(
      burn_in = 5, draws = 6, thinning = 2, prior_only = TRUE
    )
  )
}

config_paths <- function(tree, prefix = "") {
  paths <- character(0)
  for (name in names(tree)) {
    path <- paste0(prefix, name)
    value <- tree[[name]]
    if (is.list(value) && length(value) > 0L && !is.null(names(value))) {
      paths <- c(paths, config_paths(value, paste0(path, ".")))
    } else {
      paths <- c(paths, path)
    }
  }
  paths
}

# Groups the core serialises empty at their defaults, so the path is the
# group itself; the surface names their one field.
UNEXPOSED <- c(
  "mean_params.cell", "variance_params.cell", "outcome.gaussian.sigma_hat"
)

test_that("every surface argument is a core option", {
  expect_no_error(full_control())
  expect_no_error(thiessen_control(outcome = probit_outcome(offset = 0.5)))
})

test_that("every core option is reachable from the surface", {
  core <- config_paths(
    jsonlite::fromJSON(core_defaults(), simplifyVector = FALSE)
  )
  surface <- config_paths(
    jsonlite::fromJSON(config_json(full_control()), simplifyVector = FALSE)
  )

  missing <- setdiff(setdiff(core, surface), UNEXPOSED)
  expect_identical(missing, character(0))
})

test_that("every outcome family option is a constructor argument", {
  for (family in core_catalogue()) {
    kind <- names(family)[[1L]]
    constructor <- get(
      paste0(kind, "_outcome"),
      envir = asNamespace("thiessen")
    )

    expect_setequal(names(formals(constructor)), names(family[[kind]]))
    expect_setequal(
      names(unclass(constructor())), names(formals(constructor))
    )
  }
})

test_that("the factor encoding is the shared fixture", {
  # Nine rows, three levels: the encoding the Python suite asserts as well.
  n <- 9
  continuous <- (seq_len(n) - 1) / (n - 1)
  level <- factor(c("l0", "l1", "l2")[(seq_len(n) - 1) %% 3 + 1])
  frame <- data.frame(y = continuous + 0.5 * (as.integer(level) - 1),
                      a = continuous, g = level)

  fit <- thiessen(
    y ~ a + g, frame,
    thiessen_control(
      tessellations = 4,
      general_params = general_params(burn_in = 2, draws = 2)
    ),
    seed = 1, chains = 1
  )

  expected <- cbind(
    continuous,
    as.double(level == "l1"),
    as.double(level == "l2")
  )
  expect_identical(unname(fit$x), unname(expected))
})

test_that("the package masks nothing attached by default", {
  # `gaussian()` masked `stats::gaussian()`, so a `glm(family = gaussian)`
  # after `library(thiessen)` passed a `thiessen_outcome` to `glm` and
  # failed obscurely. `R CMD check` does not flag it.
  exports <- getNamespaceExports("thiessen")
  attached <- c("stats", "graphics", "grDevices", "utils", "datasets",
                "methods", "base")

  masked <- Reduce(
    union,
    lapply(attached, function(package) {
      intersect(exports, getNamespaceExports(package))
    }),
    character(0)
  )

  expect_identical(masked, character(0))
})
