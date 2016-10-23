# load libraries
library(caret)
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- PimaIndiansDiabetes[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataTrain <- PimaIndiansDiabetes[validation_index,]

# prepare resampling method
control <- trainControl(method="repeatedcv", number=10, repeats=5)
set.seed(7)
fit <- train(diabetes~., data=dataTrain, method="glm", metric="Accuracy", trControl=control)
# display results
#print(fit)
trainingAccuracy <- fit$results$Accuracy
cat("Training Accuracy: ", trainingAccuracy)


# estimate skill of the validation dataset
predictions <- predict(fit, validation)
testResult <- confusionMatrix(predictions, validation$diabetes, positive="pos")
#print(testResult)

################Built-in Method###########################
testAccuracy <- testResult$overall["Accuracy"]
precision <- testResult$byClass['Pos Pred Value']
recall <- testResult$byClass['Sensitivity']
f1score <- 2 * ((precision * recall) / (precision + recall))

cat("Test Accuracy: ", testAccuracy)
cat("Precision: ", precision)
cat("Recall: ", recall)
cat("F1score: ", f1score)
#########################################################
