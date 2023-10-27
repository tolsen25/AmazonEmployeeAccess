library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(discrim)
library(themis)

trainData = vroom("train.csv") %>% mutate(ACTION = as.factor(ACTION))
testData = vroom("test.csv")


# bayesRegModel = naive_Bayes(Laplace = tune(), smoothness= tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")


my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .54321) %>%
  step_smote(all_outcomes(), neighbors= 27)

source("logReg.R")
source("randForest.R")
source("naiveBayes.R")
source("penLogreg.R")
source("knn.R")
source("pcaKNN.R")
source("pca.R")
source("svm.R")



