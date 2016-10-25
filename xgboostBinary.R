# load libraries
library(caret)
library(mlbench)
library(xgboost)

data(PimaIndiansDiabetes)

set.seed(1)

#split the dataset
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p=0.80, list=FALSE)
dataTrain <- PimaIndiansDiabetes[train_index,]
dataTest <- PimaIndiansDiabetes[-train_index,]

#prepare the dataset for xgboost
xgbTrain <- xgb.DMatrix(data.matrix(dataTrain[,1:8]), label=(dataTrain$diabetes=="pos"))
xgbTest <- xgb.DMatrix(data.matrix(dataTest[,1:8]), label=(dataTest$diabetes=="pos"))

#train the model
param <- list(booster = "gblinear", max.depth = 5, eta = 1, eval_metric = "error", objective = "binary:logistic")
bst <- xgb.train(param, data = xgbTrain, nthread = 2, nround = 2, watchlist)

#predic
pred <- predict(bst, xgbTest)

#make binary prediction 
prediction <- as.numeric(pred > 0.5)

#find the accuracy
target <- as.numeric((dataTest$diabetes=="pos"))
test.confMatrx <- confusionMatrix(data=prediction, reference=target, positive="1", mode = "prec_recall")
print(test.confMatrx)
