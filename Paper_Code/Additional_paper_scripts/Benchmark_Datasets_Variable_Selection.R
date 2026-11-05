
  # 2. Parkinson's Telemonitoring (UCI) - 5,875 obs, 16 covariates
  # Substitutes ADNI. Exact same profile: predicting continuous neurological decline (UPDRS) from dense, highly correlated biomedical sensor metrics.
  {# Target: Continuous motor UPDRS score
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
  Parkinsons <- read.csv(url, header = TRUE)
  Parkinsons <- na.omit(Parkinsons)
  benchmark_datasets$Parkinsons <- list(
    X = as.matrix(Parkinsons[, !names(Parkinsons) %in% c("subject.", "motor_UPDRS", "total_UPDRS")]),
    Y = as.numeric(Parkinsons$motor_UPDRS)
  )
  rm(Parkinsons, url)}

  # 3. FRED-MD Macroeconomic Database (CRAN: BVAR) - 728 obs, 128 covariates
  {# Target: Industrial Production Index (INDPRO)
  if (!requireNamespace("BVAR", quietly = TRUE)) install.packages("BVAR")
  data(fred_md, package = "BVAR")
  fred_md <- na.omit(fred_md)
  benchmark_datasets$FRED_MD <- list(
    X = as.matrix(fred_md[, !names(fred_md) %in% "INDPRO"]),
    Y = as.numeric(fred_md$INDPRO)
  )
  rm(fred_md)}


# 6. Aquatic Toxicity (CRAN: QSARdata) - 322 obs, 220 covariates
  # Substitutes QSAR BCF. Exact same profile: predicting continuous chemical activity from sparse molecular descriptors.
  {# Target: Continuous LC50 toxicity 
  if (!requireNamespace("QSARdata", quietly = TRUE)) install.packages("QSARdata")
  data(AquaticTox, package = "QSARdata")
  AqTox_Data <- cbind(AquaticTox_moe2D[, -1], Outcome = AquaticTox_Outcome$Activity)
  AqTox_Data <- na.omit(AqTox_Data)
  benchmark_datasets$QSAR_AquaticTox <- list(
    X = as.matrix(AqTox_Data[, !names(AqTox_Data) %in% "Outcome"]),
    Y = as.numeric(AqTox_Data$Outcome)
  )
  rm(list = ls(pattern = "^AquaticTox"), AqTox_Data)}
  

  # 8. Residential Building Data (UCI) - 372 obs, 105 covariates
  {# Target: Continuous construction cost (Column 109)
  if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")
  temp <- tempfile()
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx", temp, mode = "wb")
  ResBuild <- readxl::read_excel(temp, skip = 1)
  unlink(temp)
  ResBuild <- na.omit(ResBuild)
  benchmark_datasets$ResidentialBuilding <- list(
    X = as.matrix(ResBuild[, 5:107]),
    Y = as.numeric(ResBuild[[109]])
  )
  rm(ResBuild, temp)}

  # 9. Superconductivity Dataset (UCI) - Subsampled to 2000 obs, 81 covariates
  {# Target: Continuous critical temperature
  temp <- tempfile()
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip", temp)
  Superconduct <- read.csv(unz(temp, "train.csv"))
  unlink(temp)
  Superconduct <- na.omit(Superconduct)
  set.seed(42)
  Superconduct <- Superconduct[sample(nrow(Superconduct), 2000), ]
  benchmark_datasets$Superconductivity <- list(
    X = as.matrix(Superconduct[, !names(Superconduct) %in% "critical_temp"]),
    Y = as.numeric(Superconduct$critical_temp)
  )
  rm(Superconduct, temp)}

  # 10. Communities and Crime Unnormalized (UCI) - 2215 obs, 125 covariates (Regression)
  {# Target: Continuous violent crimes per capita (Column 146)
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt"
  CommCrime <- read.csv(url, header = FALSE, na.strings = "?")
  CommCrime <- na.omit(CommCrime)
  benchmark_datasets$CommunitiesCrimeUnnorm <- list(
    X = as.matrix(CommCrime[, 6:130]),
    Y = as.numeric(CommCrime[, 146])
  )
  rm(CommCrime, url)}




















# 1. Aquatic Toxicity (QSARdata) - 322 obs, 220 covariates
  # Using the MOE 2D descriptors. We exclude the first column ("Molecule" string IDs).
  {data(AquaticTox, package = "QSARdata")
  AqTox_Data <- cbind(AquaticTox_moe2D[, -1], Outcome = AquaticTox_Outcome$Activity)
  AqTox_Data <- na.omit(AqTox_Data)
  benchmark_datasets$AquaticTox <- list(
    X = as.matrix(AqTox_Data[, !names(AqTox_Data) %in% "Outcome"]),
    Y = as.numeric(AqTox_Data$Outcome)
  )
  rm(list = ls(pattern = "^AquaticTox"), AqTox_Data)}

  # 2. Melting Point (QSARdata) - 4,173 obs, 202 covariates
  # The descriptors object is correctly named MP_Descriptors.
  {data(MeltingPoint, package = "QSARdata")
  MP_Data <- cbind(MP_Descriptors, Outcome = MP_Outcome)
  MP_Data <- na.omit(MP_Data)
  benchmark_datasets$MeltingPoint <- list(
    X = as.matrix(MP_Data[, !names(MP_Data) %in% "Outcome"]),
    Y = as.numeric(MP_Data$Outcome)
  )
  rm(list = ls(pattern = "^MP_"), MP_Data)}

  # 3. CT Slices Subset (UCI) - Subsampled to 3,000 obs, 384 covariates
  # Signal Sparsity: The features are continuous, dense spatial histograms extracted from bone structures. While there are no explicit zeroes, only a few highly specific histogram bins anchor the true axial slice location, making the underlying signal sparse.
  {temp <- tempfile()
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip", temp)
  CTSlices <- read.csv(unz(temp, "slice_localization_data.csv"))
  unlink(temp)
  CTSlices <- na.omit(CTSlices)
  set.seed(42) # Ensures a reproducible subset 
  CTSlices_subset <- CTSlices[sample(nrow(CTSlices), 3000), ]
  benchmark_datasets$CTSlices_Subset <- list(
    X = as.matrix(CTSlices_subset[, !names(CTSlices_subset) %in% c("patientId", "reference")]),
    Y = as.numeric(CTSlices_subset$reference)
  )
  rm(CTSlices, CTSlices_subset, temp)}

  # 2. Geographical Origin of Music (UCI) - 1059 obs, 68 covariates
  {music_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00315/default_features_1059tracks.txt"
  GeoMusic <- read.csv(music_url, header = FALSE)
  GeoMusic <- na.omit(GeoMusic)
  # V1 and V2 represent Latitude and Longitude respectively. We target Latitude (V1) and remove Longitude (V2).
  benchmark_datasets$GeoMusic <- list(
    X = model.matrix(V1 ~ . - V2 - 1, data = GeoMusic),
    Y = as.numeric(GeoMusic$V1)
  )
  rm(GeoMusic, music_url)}

# 4. UJI Indoor Localisation (UCI) - Subsampled to 2,500 obs, 520 covariates
  {temp <- tempfile()
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip", temp)
  UJIIndoorLoc <- read.csv(unz(temp, "UJIndoorLoc/trainingData.csv"))
  unlink(temp)
  UJIIndoorLoc <- na.omit(UJIIndoorLoc)
  set.seed(42)
  UJI_subset <- UJIIndoorLoc[sample(nrow(UJIIndoorLoc), 2500), ]
  # Predict continuous LONGITUDE
  benchmark_datasets$UJIIndoorLoc <- list(
    X = as.matrix(UJI_subset[, 1:520]),
    Y = as.numeric(UJI_subset$LONGITUDE)
  )
  rm(UJIIndoorLoc, UJI_subset, temp)}

# 3. Permeability (AppliedPredictiveModeling) - 165 obs, 1107 covariates (Regression)
  {# Target: Continuous permeability of molecules
  # Covariates: 1107 binary molecular fingerprints
  data(permeability, package = "AppliedPredictiveModeling")
  valid <- !is.na(permeability)
  benchmark_datasets$Permeability <- list(
    X = as.matrix(fingerprints[valid, ]),
    Y = as.numeric(permeability[valid])
  )
  rm(fingerprints, permeability, valid)}
  
  # 4. NIR Soil Spectroscopy (prospectr) - 825 obs, 1050 covariates (Regression)
  {# Target: Soil Carbon concentration (C)
  # Covariates: 1050 near-infrared reflectance spectral bands
  data(NIRsoil, package = "prospectr")
  nir_df <- data.frame(C = NIRsoil$CEC, NIRsoil$spc)
  nir_df <- na.omit(nir_df)
  benchmark_datasets$NIRsoil <- list(
    X = model.matrix(C ~ . - 1, data = nir_df),
    Y = as.numeric(nir_df$C)
  )
  rm(NIRsoil, nir_df)}
  
  # 5. Wheat Yield Markers (BGLR) - 599 obs, 1279 covariates (Regression)
  {# Target: Continuous wheat trait/yield (first column of wheat.Y)
  # Covariates: 1279 distinct DArT molecular markers
  data(wheat, package = "BGLR")
  valid <- !is.na(wheat.Y[, 1])
  benchmark_datasets$Wheat <- list(
    X = as.matrix(wheat.X[valid, ]),
    Y = as.numeric(wheat.Y[valid, 1])
  )
  rm(wheat.X, wheat.Y, wheat.A, valid)}






### Triazines ###
{# Import the dataset into a data frame
triazines_data <- read.csv("https://raw.githubusercontent.com/s3628730/datasets/master/triazines.csv")

# View the first few rows of the data
head(triazines_data)
X_triazines <- triazines_data[,1:60]
X_triazines <- X_triazines[,-c(43,44)]
Y_triazines <- triazines_data[,61]
}

### Boston Housing ###
{
  boston_data <- read.csv("https://raw.githubusercontent.com/s3628730/datasets/master/boston_housing.csv")
  
  # View the first few rows of the data
  head(boston_data)
  X_boston <- boston_data[,1:13]
  Y_boston <- boston_data[,14]
}


### Gisette ###
{
  gisette_data <- read.csv("C:/Users/adam6/Downloads/gisette.csv/gisette.csv")
  
  # View the first few rows of the data
  head(gisette_data)
  X_gisette <- gisette_data[,1:5000]
  Y_gisette <- gisette_data[,5001]
}


install.packages("hdi")

set.seed(123)

data <- data.frame(Y = Y_gisette, X_gisette)
# Determine the number of rows
n <- nrow(data)
train_indices <- sample(1:n, size = round(0.8 * n))
train_set <- data[train_indices, ]
test_set  <- data[-train_indices, ]
X_train <- model.matrix(Y ~ . -1, data = train_set)
X_test  <- model.matrix(Y ~ . -1, data = test_set)
Y_train <- train_set$Y
Y_test  <- test_set$Y
training_data<-data.frame(Y=Y_train,X_train)

{# Identify columns in X_train that have more than one unique value
non_constant_cols <- apply(X_train, 2, function(col) length(unique(col)) > 1)

# Subset both X_train and X_test using the same logical vector
X_train <- X_train[, non_constant_cols]
X_test  <- X_test[, non_constant_cols]

# Update the training_data data frame with the filtered predictors
training_data <- data.frame(Y = Y_train, X_train)}


cat("Fitting local update algorithm...\n")
time_local <- system.time({
  Model_AddiVortes_local <- AddiVortes(Y_train, X_train,m=20,totalMCMCIter = 2000, mcmcBurnIn = 1000,dirichletWarmup = 500,a_alpha = 0.5, b_alpha = 1,adaptBoost = 1, adaptPenalty = 1, momentumDecay = 0.6, kappa = 0.6,numChains = 1,IntialSigma = "Naive")
})

cat("Time taken for local:\n")
print(time_local)
##Graphs
{
  # Assuming you have run the algorithm and saved the output to 'results':
  # results <- Homo_AddiVortes_Algorithm(...)
  
  # 1. Extract the two metrics from the results list
  inclusion_probs <- Model_AddiVortes_local$ensembleInclusionProbabilities
  
  s_means <- Model_AddiVortes_local$posteriorDirichletWeightsMean
  s_lower <- Model_AddiVortes_local$posteriorDirichletWeightsLower
  s_upper <- Model_AddiVortes_local$posteriorDirichletWeightsUpper
  covariate_index <- 1:length(s_means)
  
  plot(covariate_index, s_means, type = "p", pch = 19, col = "darkorange",
       ylim = c(0, max(s_upper)+0.05),
       main = "Posterior Dirichlet Weights with 95% Credible Intervals",
       xlab = "Covariate Index", ylab = "Weight")
  
  arrows(x0 = covariate_index, y0 = s_lower, x1 = covariate_index, y1 = s_upper,
         angle = 90, code = 3, length = 0.05, col = "darkgray")
  # 4. Plot the Ensemble Inclusion Probabilities
  # We set ylim = c(0,1) because this is a strict probability bound
  inclusion_probs <- Model_AddiVortes_local$ensembleInclusionProbabilities
  
  barplot(inclusion_probs, 
          main = "Ensemble Inclusion Probabilities", 
          xlab = "Covariate Index", 
          ylab = "Probability (0 to 1)",
          col = "steelblue",
          border = NA,
          ylim = c(0, 1),
          names.arg = 1:length(inclusion_probs))
  
  #plot(Model_AddiVortes_local$posteriorAlphaTrace)
}
prediction_AddiVortes<-predict(Model_AddiVortes_local,X_test)
rmse_AddiVortes<-sqrt(mean((prediction_AddiVortes-Y_test)^2))
print(paste("RMSE for AddiVortes:", rmse_AddiVortes))
print(paste("Benchmark:", sqrt(mean((mean(Y_test)-Y_test)^2))))

cat("Fitting Orginal update algorithm...\n")
time_local <- system.time({
  Model_AddiVortes_Original <- AddiVortes(Y_train, X_train,varSelMode = 0,m=200,totalMCMCIter = 5000, mcmcBurnIn = 2000,dirichletWarmup = 1000,a_alpha = 0.5, b_alpha = 1,adaptBoost = 1, adaptPenalty = 1, momentumDecay = 0.9, kappa = 0.6,numChains = 1,IntialSigma = "LASSO")
})

cat("Time taken for local:\n")
print(time_local)
##Graphs
{
  # Assuming you have run the algorithm and saved the output to 'results':
  # results <- Homo_AddiVortes_Algorithm(...)
  
  # 1. Extract the two metrics from the results list
  inclusion_probs <- Model_AddiVortes_Original$ensembleInclusionProbabilities
  
  s_means <- Model_AddiVortes_Original$posteriorDirichletWeightsMean
  s_lower <- Model_AddiVortes_Original$posteriorDirichletWeightsLower
  s_upper <- Model_AddiVortes_Original$posteriorDirichletWeightsUpper
  covariate_index <- 1:length(s_means)
  
  plot(covariate_index, s_means, type = "p", pch = 19, col = "darkorange",
       ylim = c(0, max(s_upper)+0.05),
       main = "Posterior Dirichlet Weights with 95% Credible Intervals",
       xlab = "Covariate Index", ylab = "Weight")
  
  arrows(x0 = covariate_index, y0 = s_lower, x1 = covariate_index, y1 = s_upper,
         angle = 90, code = 3, length = 0.05, col = "darkgray")
  # 4. Plot the Ensemble Inclusion Probabilities
  # We set ylim = c(0,1) because this is a strict probability bound
  inclusion_probs <- Model_AddiVortes_Original$ensembleInclusionProbabilities
  
  barplot(inclusion_probs, 
          main = "Ensemble Inclusion Probabilities", 
          xlab = "Covariate Index", 
          ylab = "Probability (0 to 1)",
          col = "steelblue",
          border = NA,
          ylim = c(0, 1),
          names.arg = 1:length(inclusion_probs))
  
  #plot(Model_AddiVortes_Original$posteriorAlphaTrace)
}
prediction_AddiVortes_Original<-predict(Model_AddiVortes_Original,X_test)
rmse_AddiVortes_Original<-sqrt(mean((prediction_AddiVortes_Original-Y_test)^2))
print(paste("RMSE for AddiVortes:", rmse_AddiVortes_Original))

cat("Fitting Orginal update algorithm...\n")
time_local <- system.time({
  Model_AddiVortes_Dir <- AddiVortes(Y_train, X_train,varSelMode = 1,m=200,totalMCMCIter = 5000, mcmcBurnIn = 2000,dirichletWarmup = 1000,a_alpha = 0.5, b_alpha = 1,adaptBoost = 1, adaptPenalty = 1, momentumDecay = 0.9, kappa = 0.6,numChains = 1,IntialSigma = "LASSO")
})

cat("Time taken for local:\n")
print(time_local)
##Graphs
{
  # Assuming you have run the algorithm and saved the output to 'results':
  # results <- Homo_AddiVortes_Algorithm(...)
  
  # 1. Extract the two metrics from the results list
  inclusion_probs <- Model_AddiVortes_Dir$ensembleInclusionProbabilities
  
  s_means <- Model_AddiVortes_Dir$posteriorDirichletWeightsMean
  s_lower <- Model_AddiVortes_Dir$posteriorDirichletWeightsLower
  s_upper <- Model_AddiVortes_Dir$posteriorDirichletWeightsUpper
  covariate_index <- 1:length(s_means)
  
  plot(covariate_index, s_means, type = "p", pch = 19, col = "darkorange",
       ylim = c(0, max(s_upper)+0.05),
       main = "Posterior Dirichlet Weights with 95% Credible Intervals",
       xlab = "Covariate Index", ylab = "Weight")
  
  arrows(x0 = covariate_index, y0 = s_lower, x1 = covariate_index, y1 = s_upper,
         angle = 90, code = 3, length = 0.05, col = "darkgray")
  # 4. Plot the Ensemble Inclusion Probabilities
  # We set ylim = c(0,1) because this is a strict probability bound
  inclusion_probs <- Model_AddiVortes_Dir$ensembleInclusionProbabilities
  
  barplot(inclusion_probs, 
          main = "Ensemble Inclusion Probabilities", 
          xlab = "Covariate Index", 
          ylab = "Probability (0 to 1)",
          col = "steelblue",
          border = NA,
          ylim = c(0, 1),
          names.arg = 1:length(inclusion_probs))
  
  #plot(Model_AddiVortes_Dir$posteriorAlphaTrace)
}
prediction_AddiVortes_Dir<-predict(Model_AddiVortes_Dir,X_test)
rmse_AddiVortes_Dir<-sqrt(mean((prediction_AddiVortes_Dir-Y_test)^2))
print(paste("RMSE for AddiVortes:", rmse_AddiVortes_Dir))

task<- system.time({
  set.seed(123)
  Model_flexBART <- flexBART(Y~bart(.), train_data = training_data,M_vec=20,save_trees = TRUE,n.chains=1)
  prediction_flexBART<-predict.flexBART(Model_flexBART,test_set)
})
print(task)
rmse_flexBART<-sqrt(mean((colMeans(prediction_flexBART)-Y_test)^2))
#rmse_flexBART<-sqrt(mean((Model_flexBART$yhat.test.mean-test_data$mu)^2))
print(paste("RMSE for flexBART:", rmse_flexBART))

