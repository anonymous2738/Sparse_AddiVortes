library(posterior)
devtools::clean_dll("/home/jd191252/thiessen/r")
devtools::load_all("/home/jd191252/thiessen/r")

# Generate some toy data resembling Fridman setup (sparse)
set.seed(42)
n <- 200
p <- 1000
X <- matrix(rnorm(n * p), n, p)
Y <- 10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5] + rnorm(n)

cat("Running thiessen...\n")
control <- thiessen_control(
  outcome = gaussian_outcome(nu = 6, q = 0.9),
  mean_params = term_params(
    tessellations = 200,
    structure = structure_params(
      inclusion = dart_inclusion(
        a_alpha = 5.0,
        b_alpha = 0.5,
        adapt_boost = 5.0,
        adapt_penalty = 0.1,
        momentum_decay = 0.90,
        dirichlet_warmup = 500
      ),
      omega = 1.5
    )
  ),
  general_params = general_params(
    burn_in = 2500,
    draws = 2500,
    thinning = 1
  )
)

fit <- thiessen(X, Y, control = control)
draws <- as_draws_df(fit)
cat("Mean alpha after warmup: ", mean(draws$concentration), "\n")
