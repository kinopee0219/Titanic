package_list = c("readr","tidyverse","randomForest")
new_package_list = package_list[!(package_list %in% installed.packages()[, "Package"])]
if(length(new_package_list)){
  install.packages(new_package_list)
}
library(readr)
library(tidyverse)
library(randomForest)

# setup your workspace
workspace_dir = "~/Desktop/pjt/learn/enjoy_myself/titanic"
data_dir = paste0(workspace_dir, "/data/")
setwd(workspace_dir)

# input the two data sets
train_data = paste0(data_dir, "train.csv") %>% read_csv()
test_data = paste0(data_dir, "test.csv") %>% read_csv()
tmp_data = paste0(data_dir, "gender_submission.csv") %>% read_csv()

# combine the two datasets
train_data = train_data %>% mutate(IsTrain = TRUE)
test_data = test_data %>% mutate(IsTrain = FALSE ,
                                 Survived = NA)
full_data = rbind(train_data, test_data)
full_data$Cabin = NULL 

# check which column have missing value, except for Survived. 
# The answer is Age, Embarked, and Fare. 
apply(full_data, 2, function(x) sum(is.na(x)))

# clean my data
# Age：線形回帰モデルを作成して補完
boxplot(full_data$Age)
age_filter = full_data$Age < boxplot.stats(full_data$Age)$stats[5]
age_formula = "Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked"
age_model = lm(age_formula, data = full_data[age_filter,])

age_row = full_data[is.na(full_data$Age), c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked")]
age_pred = predict(age_model, newdata = age_row)
full_data[is.na(full_data$Age), "Age"] = age_pred

# Embarked：最も多い乗船場所で補完
table(full_data$Embarked)
full_data[which(is.na(full_data$Embarked)),]$Embarked = "S"

# Fare:線形回帰モデルを作成して補完
boxplot(full_data$Fare)
outliers_filter = full_data$Fare < boxplot.stats(full_data$Fare)$stats[5]
fare_formula = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare_model = lm(fare_formula, data = full_data[outliers_filter,])

fare_row = full_data[is.na(full_data$Fare), c("Pclass","Sex", "Age", "SibSp", "Parch", "Embarked")]
fare_pred = predict(fare_model, newdata = fare_row)
full_data[is.na(full_data$Fare), "Fare"] = fare_pred

full_data$Name = NULL

# categorical casting
str(full_data)
full_data = full_data %>% 
  mutate_at(c("Pclass", "Sex", "Embarked"), as.factor) # factor型にしないと分類問題ではなく、回帰問題として扱われてしまう。

# split the dataset back out into train and test datasets. 
train_data2 = full_data[full_data$IsTrain == TRUE, ]
test_data2 = full_data[!full_data$IsTrain == TRUE, ]
train_data2$Survived = as.factor(train_data2$Survived)

# train my model
train_data2$PassengerId = NULL
train_data2$Ticket = NULL
train_data2$IsTrain = NULL
View(train_data2)
tmp_rf = tuneRF(train_data2[,-1], unlist(train_data2[,1]), doBest = TRUE)

my_formula = "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked" %>% as.formula()
rf_model = randomForest(my_formula, data = train_data2, ntree = 500, mtry = tmp_rf$mtry, nodesize = 0.01*nrow(test_data2))

# predict Survived
Survived = predict(rf_model , newdata = test_data2)
PassengerId = test_data2$PassengerId
output_df = data_frame(PassengerId, Survived)

# output my data
write_csv(output_df, paste0(data_dir, "my_output.csv")) # Top 35%, 0.78947,  09152018

# 分類に寄与した変数の選択
varImpPlot(rf_model)

