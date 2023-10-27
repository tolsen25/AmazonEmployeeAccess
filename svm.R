library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(discrim)


trainData = vroom("train.csv") %>% mutate(ACTION = as.factor(ACTION))
testData = vroom("test.csv")


bayesRegModel = naive_Bayes(Laplace = tune(), smoothness= tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")


my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .54321)

svmRadial = svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")


svmReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(svmRadial)

tuning_grid = grid_regular(rbf_sigma(), cost(), levels = 2)

folds = vfold_cv(trainData, v = 5, repeats = 1)

CV_results = svmReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                             metrics = metric_set(roc_auc, precision))

bestTune = CV_results %>% select_best("roc_auc")

final_wf = svmReg_workflow %>% finalize_workflow(bestTune) %>% fit(trainData)


amazon_preds = predict(final_wf, new_data = testData, type = "prob")

sub = testData %>% mutate(
  Action = amazon_preds$.pred_1,
  Id = id
  
) %>% select(Id, Action)


vroom_write(sub, "smote_pca_SVM_2.csv", delim = ",")
