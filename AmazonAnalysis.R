library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(ggmosaic)

trainData = vroom("train.csv")
testData = vroom("test.csv")


trainData %>% ggplot(mapping = aes(x = 'ACTION')) +
  geom_mosaic()


trainData %>% ggplot(mapping = aes(x = ACTION, fill = as.factor(ACTION))) +
  geom_bar()


titles =  trainData %>%  mutate (
  roleTitle = as.factor(ROLE_TITLE)
    
  ) %>% group_by(ROLE_TITLE) %>% summarise(
   
    num = n()
  
  ) %>% arrange(desc(num))

titles %>% ggplot(mapping = aes(x = num)) +
  geom_histogram()+
  labs(title = "Histogram of the number of titles")

# collinearity a problem with regression
# two things tellings them apart can't decide which one to use 

#rFormula = ACTION ~ RESOURCE + MGR_ID + ROLE_ROLLUP_1 + ROLE_ROLLUP_2 + ROLE_DEPTNAME + ROLE_TITLE + ROLE_FAMILY_DESC + ROLE_FAMILY + ROLE_CODE 

my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value6
  step_dummy(all_nominal_predictors()) # dummy variable encoding7
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding8
# also step_lencode_glm() a
  
preppedRecipe = prep(my_recipe)
bakedRecipe = bake(preppedRecipe, new_data = trainData)  
  