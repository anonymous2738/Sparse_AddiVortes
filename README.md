The purpose of this repository is to reproduce the graphs in paper "Adaptive MCMC Sampling and Dirichlet Prior for Variable Inclusion to Induce Sparsity using AddiVortes".

## Run the Friedman simulation

You need R 4.2 or newer and Rust/Cargo 1.74 or newer. Install Rust from [rustup.rs](https://rustup.rs), then restart RStudio so it can find Cargo. On Windows, also install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version. For standard 64-bit Rtools, run `rustup target add x86_64-pc-windows-gnu`.

In RStudio, run:

```r
install.packages(c("remotes", "glmnet", "BART", "pbapply", "progressr"))
Sys.setenv(THIESSEN_EXPERIMENTAL = "1")
remotes::install_github(
  "anonymous2738/Sparse_AddiVortes",
  subdir = "r",
  ref = "backup/thiessen-2026-10-13"
)

```

## Licence

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT licence ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
licence, shall be dual licensed as above, without any additional terms or
conditions.
