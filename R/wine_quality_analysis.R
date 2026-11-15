# Red Wine Quality: regression and classification baselines
# Place the UCI red-wine file at data/winequality-red.csv (semicolon-delimited).
required <- c("readr","dplyr","glmnet","randomForest","caret")
missing <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop("Install packages: ",paste(missing,collapse=", "))
suppressPackageStartupMessages({library(readr);library(dplyr);library(glmnet);library(randomForest);library(caret)})
input <- file.path("data","winequality-red.csv")
if(!file.exists(input)) stop("Missing ",input,"; see data/DATASET_ATTRIBUTION.md.")
wine <- read_delim(input,delim=";",show_col_types=FALSE) |> filter(complete.cases(.))
if(!"quality" %in% names(wine)) stop("Expected target column quality.")
set.seed(123); idx <- caret::createDataPartition(wine$quality,p=.8,list=FALSE)
train <- wine[idx,,drop=FALSE]; test <- wine[-idx,,drop=FALSE]
xtr <- as.matrix(select(train,-quality)); xte <- as.matrix(select(test,-quality))
ytr <- train$quality; yte <- test$quality
# Regression: tune lambda on training data only; test set remains held out.
set.seed(123); lasso <- cv.glmnet(xtr,ytr,alpha=1,nfolds=10)
lp <- as.numeric(predict(lasso,newx=xte,s="lambda.min"))
metrics <- data.frame(model="LASSO regression",RMSE=sqrt(mean((yte-lp)^2)),MAE=mean(abs(yte-lp)),R_squared=1-sum((yte-lp)^2)/sum((yte-mean(yte))^2))
# Random Forest regression on exactly the same holdout observations.
set.seed(123); rfreg <- randomForest(x=xtr,y=ytr,ntree=500,importance=TRUE)
rp <- as.numeric(predict(rfreg,newdata=xte))
metrics <- bind_rows(metrics,data.frame(model="Random Forest regression",RMSE=sqrt(mean((yte-rp)^2)),MAE=mean(abs(yte-rp)),R_squared=1-sum((yte-rp)^2)/sum((yte-mean(yte))^2)))
# Classification is a separate task; do not compare accuracy with regression R-squared.
levels_q <- sort(unique(wine$quality))
trc <- train |> mutate(quality=factor(quality,levels=levels_q))
tec <- test |> mutate(quality=factor(quality,levels=levels_q))
set.seed(123); rfcls <- randomForest(quality~.,data=trc,ntree=500,importance=TRUE)
pred <- predict(rfcls,newdata=tec); cm <- caret::confusionMatrix(pred,tec$quality)
write.csv(metrics,"regression_metrics.csv",row.names=FALSE)
write.csv(as.data.frame(cm$table),"classification_confusion_matrix.csv",row.names=FALSE)
write.csv(as.data.frame(cm$byClass),"classification_per_class_metrics.csv")
write.csv(data.frame(metric=names(cm$overall),value=as.numeric(cm$overall)),"classification_overall_metrics.csv",row.names=FALSE)
write.csv(randomForest::importance(rfreg),"rf_regression_feature_importance.csv")
write.csv(randomForest::importance(rfcls),"rf_classification_feature_importance.csv")
message("Saved regression and classification results separately. This is one holdout split; use repeated/nested resampling for stronger estimates.")
