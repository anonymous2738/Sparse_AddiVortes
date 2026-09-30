# The response shapes the package takes, in the forms the R ecosystem
# already uses, and the outcome family each selects: `glm`'s rule, the
# declared family authoritative and the response checked against it, with
# "not declared" representable.

#' Parse a response into the shape the core takes
#'
#' A `survival::Surv` object of type `"right"` selects the AFT family and
#' one of type `"interval"` (the `"interval2"` encoding) the
#' interval-censored family; an ordered factor selects the ordinal family,
#' a two-level factor the probit family, and a numeric vector the Gaussian
#' family. An unordered factor with more than two levels is unsupported.
#'
#' @param y The response as the caller gave it.
#' @param call The calling environment to report.
#' @return An object of class `"thiessen_response"`: a list of `kind`, the
#'   core entry point the response reaches (`"plain"`, `"aft"` or
#'   `"interval_censored"`); `shape`, the shape of the response;
#'   `family`, the outcome family the shape selects; `n`, the number of
#'   observations; the numeric vector or the censoring columns the core
#'   takes; and the levels of a factor response.
#' @noRd
as_response <- function(y, call = rlang::caller_env()) {
  if (inherits(y, "Surv")) {
    return(surv_response(y, call = call))
  }
  if (is.factor(y)) {
    return(factor_response(y, call = call))
  }
  if (!is.numeric(y) || !is.null(dim(y))) {
    thiessen_abort(
      paste0(
        "`y` must be a numeric vector, a factor or a `survival::Surv` ",
        "object."
      ),
      call = call
    )
  }
  if (anyNA(y)) {
    thiessen_abort("`y` must not contain missing values.", call = call)
  }
  new_response("plain", "numeric", y = as.double(y))
}

#' A factor response: ordered as ordinal codes and two levels as probit
#' labels; unordered factors with more levels are unsupported.
#'
#' The codes are 0 to K - 1 with the first level as 0, as `glm` treats a
#' factor response and `MASS::polr` an ordered one.
#'
#' @param y A factor.
#' @param call The calling environment to report.
#' @return An object of class `"thiessen_response"`.
#' @noRd
factor_response <- function(y, call) {
  if (anyNA(y)) {
    thiessen_abort("`y` must not contain missing values.", call = call)
  }
  codes <- as.double(as.integer(y) - 1L)
  if (is.ordered(y)) {
    return(new_response("plain", "ordered", y = codes, levels = levels(y)))
  }
  if (nlevels(y) != 2L) {
    thiessen_abort(
      "Unordered factor responses with more than two levels are not supported; use an ordered factor or a numeric response.",
      call = call
    )
  }
  new_response("plain", "binary", y = codes, levels = levels(y))
}

#' A `Surv` response as the censoring columns the core takes
#'
#' A `Surv` is a numeric matrix with a `type` attribute. Type `"right"`
#' holds `time` and `status` (1 an event, 0 right-censored). Type
#' `"interval"`, which `Surv(lower, upper, type = "interval2")` also
#' produces, holds `time1`, `time2` and a `status` of 0 (right-censored
#' at `time1`), 1 (exact at `time1`), 2 (left-censored at `time1`) or 3
#' (between `time1` and `time2`); the bounds reach the core with an
#' infinite endpoint for one-sided censoring and an equal pair for an
#' exact value.
#'
#' @param y A `Surv` object.
#' @param call The calling environment to report.
#' @return An object of class `"thiessen_response"`.
#' @noRd
surv_response <- function(y, call) {
  type <- attr(y, "type")
  columns <- unclass(y)
  if (identical(type, "right")) {
    times <- columns[, "time"]
    status <- columns[, "status"]
    if (anyNA(times) || anyNA(status)) {
      thiessen_abort("`y` must not contain missing values.", call = call)
    }
    return(new_response(
      "aft", "right",
      times = as.double(times), events = status == 1
    ))
  }
  if (identical(type, "interval")) {
    time1 <- columns[, "time1"]
    time2 <- columns[, "time2"]
    status <- columns[, "status"]
    if (anyNA(time1) || anyNA(status) || anyNA(time2[status == 3])) {
      thiessen_abort("`y` must not contain missing values.", call = call)
    }
    lower <- ifelse(status == 2, -Inf, time1)
    upper <- ifelse(status == 3, time2, ifelse(status == 0, Inf, time1))
    return(new_response(
      "interval_censored", "interval",
      lower = as.double(lower), upper = as.double(upper)
    ))
  }
  thiessen_abort(
    sprintf(
      "A `Surv` response must be of type \"right\" or \"interval\", not \"%s\".",
      type
    ),
    call = call
  )
}

#' Construct a parsed response
#'
#' @param kind The core entry point: `"plain"`, `"aft"` or
#'   `"interval_censored"`.
#' @param shape The response shape: `"numeric"`, `"binary"`, `"ordered"`,
#'   `"right"` or `"interval"`.
#' @param y The numeric response of a plain kind.
#' @param levels The levels of a factor response.
#' @param times,events The columns of a right-censored response.
#' @param lower,upper The columns of an interval-censored response.
#' @return An object of class `"thiessen_response"`.
#' @noRd
new_response <- function(kind, shape, y = NULL, levels = NULL, times = NULL,
                         events = NULL, lower = NULL, upper = NULL) {
  structure(
    list(
      kind = kind,
      shape = shape,
      family = SHAPE_FAMILY[[shape]],
      n = NROW(if (kind == "plain") y else if (kind == "aft") times else lower),
      y = y,
      levels = levels,
      times = times,
      events = events,
      lower = lower,
      upper = upper
    ),
    class = "thiessen_response"
  )
}

# The outcome family each response shape selects when none is named.
SHAPE_FAMILY <- c(
  numeric = "gaussian",
  binary = "probit",
  ordered = "ordinal",
  right = "aft",
  interval = "interval_censored"
)

# The response shapes each outcome family accepts when named: the
# families over a real line take a numeric vector, the probit family the
# labels of a two-level factor or the numbers 0 and 1, and each other
# family the one shape that selects it.
FAMILY_SHAPES <- list(
  gaussian = "numeric",
  tobit = "numeric",
  student_t = "numeric",
  laplace = "numeric",
  probit = c("binary", "numeric"),
  ordinal = "ordered",
  aft = "right",
  interval_censored = "interval"
)

#' Describe a response shape in an error message
#'
#' @param shape A response shape.
#' @return A noun phrase.
#' @noRd
shape_label <- function(shape) {
  switch(
    shape,
    numeric = "a numeric vector",
    binary = "a two-level factor",
    ordered = "an ordered factor",
    right = "a `Surv` object of type \"right\"",
    interval = "a `Surv` object of type \"interval\""
  )
}

#' Refuse a named outcome family the response does not fit
#'
#' @param kind The core's name for the named family.
#' @param response An object of class `"thiessen_response"`.
#' @param call The calling environment to report.
#' @noRd
check_outcome_response <- function(kind, response,
                                   call = rlang::caller_env()) {
  if (response$shape %in% FAMILY_SHAPES[[kind]]) {
    return(invisible(NULL))
  }
  thiessen_abort(
    sprintf(
      "The response is %s, which selects the %s family, but `outcome` names the %s family.",
      shape_label(response$shape), response$family, kind
    ),
    call = call
  )
}

#' The control with its outcome family resolved against the response
#'
#' `NULL` takes the family the response selects; a named family is checked
#' against the response and never coerced. The ordinal family takes its
#' category count from the levels of the response where the constructor
#' left it unset.
#'
#' @param control An object of class `"thiessen_control"`.
#' @param response An object of class `"thiessen_response"`.
#' @param call The calling environment to report.
#' @return The control with a named outcome family.
#' @noRd
resolve_outcome <- function(control, response, call = rlang::caller_env()) {
  outcome <- control$outcome
  if (is.null(outcome)) {
    control$outcome <- family_of(response)
    return(control)
  }
  check_outcome_response(attr(outcome, "kind"), response, call = call)
  control$outcome <- resolve_family(outcome, response, call = call)
  control
}

#' A named outcome family with the fields the response supplies
#'
#' @param outcome An object of class `"thiessen_outcome"`.
#' @param response An object of class `"thiessen_response"`.
#' @param call The calling environment to report.
#' @return The family, its unset fields taken from the response.
#' @noRd
resolve_family <- function(outcome, response, call = rlang::caller_env()) {
  UseMethod("resolve_family")
}

#' @export
resolve_family.default <- function(outcome, response,
                                   call = rlang::caller_env()) {
  outcome
}

#' @export
resolve_family.thiessen_ordinal <- function(outcome, response,
                                            call = rlang::caller_env()) {
  categories <- length(response$levels)
  if (is.null(outcome$categories)) {
    outcome$categories <- categories
  } else if (outcome$categories != categories) {
    thiessen_abort(
      sprintf(
        "`outcome` names %d categories but the response has %d levels.",
        outcome$categories, categories
      ),
      call = call
    )
  }
  outcome
}

#' The outcome family a response selects, at its defaults
#'
#' @param response An object of class `"thiessen_response"`.
#' @return An object of class `"thiessen_outcome"`.
#' @noRd
family_of <- function(response) {
  switch(
    response$family,
    gaussian = gaussian_outcome(),
    probit = probit_outcome(),
    ordinal = ordinal_outcome(categories = length(response$levels)),
    aft = aft_outcome(),
    interval_censored = interval_censored_outcome()
  )
}

#' Construct the core's sampler over a parsed response
#'
#' @param config The configuration JSON.
#' @param design The numeric design.
#' @param response An object of class `"thiessen_response"`.
#' @param seed The resolved seed.
#' @param chain The chain index, from 0.
#' @return An external pointer to the sampler.
#' @noRd
new_sampler_handle <- function(config, design, response, seed, chain) {
  switch(
    response$kind,
    plain = core_sampler_new(config, design, response$y, seed, chain),
    aft = core_sampler_new_aft(
      config, design, response$times, response$events, seed, chain
    ),
    interval_censored = core_sampler_new_interval_censored(
      config, design, response$lower, response$upper, seed, chain
    )
  )
}

#' Replace a sampler's response with a parsed one
#'
#' @param handle An external pointer to the sampler.
#' @param response An object of class `"thiessen_response"`.
#' @noRd
set_sampler_response <- function(handle, response) {
  switch(
    response$kind,
    plain = core_sampler_set_response(handle, response$y),
    aft = core_sampler_set_aft_response(
      handle, response$times, response$events
    ),
    interval_censored = core_sampler_set_interval_censored_response(
      handle, response$lower, response$upper
    )
  )
}

#' The pointwise log-likelihood of a parsed response
#'
#' @param state An external pointer to the fitted state.
#' @param design The numeric design.
#' @param response An object of class `"thiessen_response"`.
#' @return A double matrix, one row per kept draw.
#' @noRd
log_lik_of <- function(state, design, response) {
  switch(
    response$kind,
    plain = core_log_lik(state, design, response$y),
    aft = core_log_lik_survival(
      state, design, response$times, response$events
    ),
    interval_censored = core_log_lik_interval_censored(
      state, design, response$lower, response$upper
    )
  )
}
