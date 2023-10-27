# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(vroom)
# library(ggmosaic)
# 
# trainData = vroom("train.csv") %>% mutate(ACTION = as.factor(ACTION))
# testData = vroom("test.csv")
# 

logRegModel = logistic_reg() %>%
  set_engine("glm")


# my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
#   step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value6
#   step_dummy(all_nominal_predictors())


logReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(logRegModel) %>%
  fit(data = trainData)

amazon_preds = predict(logReg_workflow, new_data = testData, type = "prob")


sub = testData %>% mutate(
  Action = amazon_preds$.pred_1,
  Id = id
  
) %>% select(Id, Action)


vroom_write(sub, "smote_logit.csv", delim = ",")



