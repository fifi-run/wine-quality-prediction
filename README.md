# Wine Quality Prediction: LASSO and Random Forest

## Overview
This project uses physicochemical measurements from the UCI Red Wine Quality dataset to study wine quality scores. The dataset contains 1,599 red-wine observations, 11 physicochemical predictors and a quality score. The CSV is semicolon-delimited.

The coursework explored both numeric score prediction and categorical quality classification. These are distinct prediction tasks and their metrics should be reported separately.

## Methods
- Exploratory data analysis and correlation inspection.
- IQR-based outlier review; the original coursework also applied domain-informed filters.
- LASSO regression with cross-validation for penalty selection.
- Random Forest regression and Random Forest classification.
- Holdout evaluation using RMSE, MAE and R-squared for regression, and confusion-matrix/class-level metrics for classification.

## Results in the original submitted report
| Model / task | Reported result |
|---|---:|
| LASSO regression | RMSE 0.675; MAE 0.537; R-squared 0.370 |
| Random Forest classification | Accuracy 0.942; Kappa 0.910 |
| Random Forest numeric-score comparison (as reported) | RMSE 0.261; MAE 0.0614; R-squared 0.896 |

These are historical coursework figures, not outputs from the revised script. The original report's adjusted R-squared value of 8.92777 is not reproduced because it is outside the valid range for the usual adjusted R-squared statistic. The numeric Random Forest comparison is also not directly comparable with the LASSO regression unless both are evaluated as the same task on the same holdout observations.

## Curated code and data
- `R/wine_quality_analysis.R`: project-relative workflow with a fixed 80/20 stratified split. It evaluates LASSO and Random Forest regression on the same holdout set, then evaluates Random Forest classification separately.
- `data/DATASET_ATTRIBUTION.md`: dataset citation and CC BY 4.0 attribution details.

The raw CSV is not committed. Download the red-wine subset from the UCI record and place it at `data/winequality-red.csv`.

## Run
Install R and the packages `readr`, `dplyr`, `glmnet`, `randomForest` and `caret`. From the repository root:

```r
source("R/wine_quality_analysis.R")
```

The revised script has not been executed end-to-end in this update. Its results should not be quoted until the authorised input data is supplied and the script is run in a recorded R environment.

## Dataset citation
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

The UCI record specifies CC BY 4.0: https://archive.ics.uci.edu/dataset/186/wine+quality

## Limitations and next steps
Use repeated or nested resampling for more stable estimates, report class-level recall/F1 and balanced accuracy, justify outlier filters, and consider ordinal modelling because quality scores are ordered categories.
