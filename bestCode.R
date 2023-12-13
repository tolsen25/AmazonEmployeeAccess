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
  #step_pca(all_predictors(), threshold = .9) %>%
  step_smote(all_outcomes(), neighbors= 5)

treeModel = rand_forest(mtry = tune(), min_n =  tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


# my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
#   #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


forestReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(treeModel)

tuning_grid = grid_regular(mtry(range = c(1,7)), min_n(), levels = 5)# idk what this does

folds = vfold_cv(trainData, v = 5, repeats = 1)

CV_results = forestReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                              metrics = metric_set(roc_auc, precision))

bestTune = CV_results %>% select_best("roc_auc")

final_wf = forestReg_workflow %>% finalize_workflow(bestTune) %>% fit(trainData)


amazon_preds = predict(final_wf, new_data = testData, type = "prob")


sub = testData %>% mutate(
  Action = amazon_preds$.pred_1,
  Id = id
  
) %>% select(Id, Action)


vroom_write(sub, "local_smote_forestLogit_3.csv", delim = ",")

