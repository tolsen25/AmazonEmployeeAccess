# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(vroom)
# 
# 
# trainData = vroom("train.csv") %>% mutate(ACTION = as.factor(ACTION))
# testData = vroom("test.csv")


bayesRegModel = naive_Bayes(Laplace = tune(), smoothness= tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


# my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
#   #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  #   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

bayesReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(bayesRegModel)

tuning_grid = grid_regular(Laplace(), smoothness(), levels = 5)

folds = vfold_cv(trainData, v = 5, repeats = 1)

CV_results = bayesReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
                                           metrics = metric_set(roc_auc, precision))

bestTune = CV_results %>% select_best("roc_auc")

final_wf = bayesReg_workflow %>% finalize_workflow(bestTune) %>% fit(trainData)


amazon_preds = predict(final_wf, new_data = testData, type = "prob")

sub = testData %>% mutate(
  Action = amazon_preds$.pred_1,
  Id = id
  
) %>% select(Id, Action)


vroom_write(sub, "smote_bayesLogit_2.csv", delim = ",")
