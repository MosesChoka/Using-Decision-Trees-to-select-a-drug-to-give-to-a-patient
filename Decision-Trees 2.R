# Install packages
library(tidyverse)
library(rsample)

# modelling packages
library(rpart) # direct engine for decision tree application
library(caret)

# model interpretability packages

library(rpart.plot)
library(vip)
library(pdp)

# model accuracy
library(Metrics)

# LOADING DATA
library(readr)
drug200 <- read_csv("drug200.csv")
View(drug200)

# check structure of the dataset
str(drug200)

# convert the data set to a data frame
as.data.frame(drug200)

class(drug200)
str(drug200)

# Convert the string data types to factor
drug200$Sex <- as.factor(drug200$Sex)
drug200$BP <- as.factor(drug200$BP)
drug200$Cholesterol <- as.factor(drug200$Cholesterol)
drug200$Drug <- as.factor(drug200$Drug)


# get the descriptive statistics of the data frame
summary(drug200)
head(drug200)

# DATA CLEANING
# Check for any missing values
colSums(is.na(drug200))

# TRAIN AND TEST SET SPLIT
set.seed(123)

split <- initial_split(drug200, prop = 0.8, strata = NULL)

drug200_train <- training(split)
str(drug200_train)
head(drug200_train)

drug200_test <- testing(split)

str(drug200_test)

# CLASSIFICATION TREE
# part a
dt1 <- rpart(
  formula = Drug ~ . , 
  data = drug200_train,
  method = "class")

# plot the first classification tree
rpart.plot(dt1)

# part b
# Make predictions
pred <- predict(dt1, drug200_test, type = "class")
confusionMatrix(pred,reference = drug200_test$Drug )

# Accuracy test
accuracy(actual = drug200_test$Drug,
         predicted = pred)


# Check Variable importance by plotting
vip(dt1, num_features = 6, bar = FALSE)

# PRUNING - Using cross validation

# Identify the best cp value to use - minimum cp values
printcp(dt1)

#plotting cross parameters (cp)
plotcp(dt1)
dt1$cptable

# retrieve of optimal cp value based on cross validation error
index <- which.min(dt1$cptable[,"xerror"])
index

cp_optimal <- dt1$cptable[index, "CP"]
cp_optimal

# pruning tree based on optimal CP value
pruned_tree <- prune(tree = dt1, cp = cp_optimal)


# plot the optimized model
rpart.plot(x = pruned_tree,yesno = 2, type = 0, extra = 0,
           main="Pruned Classification Tree")

# make predictions and evaluate
pred2 <- predict(pruned_tree, drug200_test, type = "class")

accuracy(actual = drug200_test$Drug, predicted = pred2)

