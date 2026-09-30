library(posterior)
devtools::load_all("/home/jd191252/thiessen/r")
set.seed(123)
X_train <- matrix(rnorm(100 * 5), 100, 5)
Y_train <- rnorm(100)
control_adaptive <- thiessen_control(
  outcome = gaussian_outcome(nu = 6, q = 0.9, sigma_hat = 1.0),
  mean_params = term_params(
    tessellations = 10,  
    structure = structure_params(
      inclusion = dart_inclusion(adapt_boost = 5, adapt_penalty = 5, dirichlet_warmup = 10), omega = 1.5
    )
  ),
  general_params = general_params(burn_in = 20, draws = 20, thinning = 1)
)
fit <- thiessen(X_train, Y_train, chains = 1, threads = 1, control = control_adaptive)
draws <- as_draws_df(fit)
print(head(colnames(draws), 20))
