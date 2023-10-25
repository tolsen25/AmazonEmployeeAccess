library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)


trainData = vroom("train.csv") %>% mutate(ACTION = as.factor(ACTION))
testData = vroom("test.csv")


knnModel = nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .87654321)

knn_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(knnModel)

tuning_grid = grid_regular(neighbors(), levels = 5)

folds = vfold_cv(trainData, v = 5, repeats = 1)

CV_results = knn_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                        metrics = metric_set(roc_auc, precision))

bestTune = CV_results %>% select_best("roc_auc")

final_wf = knn_workflow %>% finalize_workflow(bestTune) %>% fit(trainData)


amazon_preds = predict(final_wf, new_data = testData, type = "prob")


sub = testData %>% mutate(
  Action = amazon_preds$.pred_1,
  Id = id
  
) %>% select(Id, Action)


vroom_write(sub, "pca_kknn2.csv", delim = ",")
