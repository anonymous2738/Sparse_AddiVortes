The purpose of this repository is to reproduce the graphs in paper "Adaptive MCMC Sampling and Dirichlet Prior for Variable Inclusion to Induce Sparsity using AddiVortes".

## Pre-installation requirements

You need R 4.2 or newer and Rust/Cargo 1.74 or newer. Install Rust from [rustup.rs](https://rustup.rs), then restart RStudio so it can find Cargo. On Windows, also install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version. For standard 64-bit Rtools, run `rustup target add x86_64-pc-windows-gnu`.

## Simulation Studies

The Friedman Simulation study in the paper can be run in Rstudio using:
```R

source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Friedman_Graphs")
```

or find file located at Paper_Code > Additional_paper_scripts > Simulation Studies > Friedman_Graphs in this repository and download and run the file.

If exact graphs are not produced, then check the warnings in the console as different versions of packages/software may affect the reproducibility of the code.

The whole script will take about 20-30 minutes to run depending on how many cores you have available, with the majority of the time spent on the 20 simulated runs at the end of the script. Additional graphs are included in the code.

Similarly, to run the scripts from supplementary material, in Rstudio run:
```R
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Binary_Graphs_1")
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Binary_Graphs_2")
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Correlated_Graphs")
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Correlated_Graphs_2")
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Friedman_Graphs_s")
source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Simulation%20Studies/Non-sparse_Graphs")
```
or download and run them individually.

## Benchmark Experiments

Download Paper_Code > Additional_paper_scripts > Benchmark_datasets.R. Find and run the line:
```R
benchmark_datasets<-list()
```
Then find the dataset or datasets that you would like to use, and run the block of code that downloads and assigns them to benchmark_datasets. For example for ChemManuf from the paper, run the code:
```R
    # 1. Chemical Manufacturing Process (AppliedPredictiveModeling) - 176 obs, 57 covariates     x
  {data(ChemicalManufacturingProcess, package = "AppliedPredictiveModeling")
  ChemicalManufacturingProcess<-na.omit(ChemicalManufacturingProcess)
  benchmark_datasets$ChemManuf <- list(
    X = model.matrix(Yield ~ . - 1, data = ChemicalManufacturingProcess),
    Y = as.numeric(ChemicalManufacturingProcess$Yield)
  )
  rm(ChemicalManufacturingProcess)}
```

You can run several datasets at the same time and the order of the datasets doesn't matter since the seed is linked to the name of the dataset. 

Warning: Do not run all the datasets, this will take +24hours for the current experimental settings. You can reduce the number of iterations/#chains/Hyperparameter grid size, to reduce the time but this will not produce the results from the paper.

It is not recommended to run datasets with +500 observations unless you have 20+ cores and sufficient RAM, otherwise it will take a long time or crash.

To run the experiments either

Licensed under either
```R

source("https://raw.githubusercontent.com/anonymous2738/Sparse_AddiVortes/backup/thiessen-2026-10-13/Paper_Code/Additional_paper_scripts/Cross_Validation_Benchmark_VS")
```
or download and run. This script will give you a table of the RMSE values stored as results_ and the boxplot from the paper. The RMSE results from the paper are commented out at the bottom of this script.  

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT licence ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
licence, shall be dual licensed as above, without any additional terms or
conditions.
