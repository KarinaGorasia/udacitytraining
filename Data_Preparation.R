########################
# Author: Karina Gorasia
# Created: May 2022
# FT Case study
########################

#● Ability to communicate your insights to a non-technical audience.
#● Use of visualisations to express findings - both through verbal and visual
#communication.
#● Ability to communicate and justify the technical choices and assumptions you
#make.
#● Ability to identify data discrepancies and take action where appropriate.
#
#The dataset includes the following columns:
#  ● CustomerId
#● CreditScore : Refers to the credit score of the customer.
#● Geography : Refers to the country or origin of the Customer.
#● Gender : Refers to gender (Female or Male)
#● Age : Refers to the age of the customer.
#● Tenure : It refers to the number of years that the customer is a customer of
#the bank.
#● Balance : Customer’s account balance.
#● NumOfProducts : Refers to the number of products a customer has
#purchased through the bank.
#● HasCrCard : Indicates whether the customer has a credit card.
#● IsActiveMember : It expresses whether the customer is active in using the
#bank’s products.
#● EstimatedSalary : Estimated salary of the customer.
#● Churned : Whether the customer has churned from the bank.
#
#1. How’s the Balance distributed across customers?
#  2. Does the level of activity of a customer (IsActiveMember column) affect how they
#churn? How significant is this relationship?
#  3. Can you create a model that predicts the probability for a customer to churn?


# Key drivers of mobile performance and where the opportunities in mobile may lie
# Over the past few years WR has seen huge growth in the number of bookings made on Mobile devices (Smartphones and Tablets).
# In order to fully exploit the opportunities in Mobile, we need to understand how customers are using Mobile devices, to help us to:
#  - Tailor the app and mobile website to the needs of our customers
#  - Target our marketing efforts appropriately
# what the data suggests in terms of the key drivers of Mobile performance, and where the opportunities in Mobile may lie. 
# what other data or information you would request, in order to gain additional insight.

#Net Sent Amount GBP	The total £ amount that customers transferred. The value of cancelled bookings is removed
#Net Orders	The total number of transactions made - the number of transactions cancelled

### --------------------------------------------------------------- environment setup ------------------------------------------------------------ ###
options(scipen=999)
# Load packages
library(dplyr)
library(tidyverse)
library(ggplot2)
library(plotly)
library(ggridges)
library(fastDummies)
library(caret)
library(pROC)
library(ConfusionTableR)

### ----------------------------------------------------------------- load data ----------------------------------------------------------------- ###
raw_data <- read_csv("churn_(3).csv")
df <- raw_data %>% 
  mutate_at((c("Gender", "Geography", "HasCrCard", "IsActiveMember", "Churned")), ~as.factor(.)) %>%
  select(-X1)

### ----------------------------------------------------------------- tidy data ----------------------------------------------------------------- ###
# assess for nulls
# dedupe
# remove outliers?
# create features

# check data for dupes - 4 cases found
df %>% group_by(CustomerId) %>% filter(n() > 1)
#check for NAs - cases found. As these account for <2%, will exclude. In future would decide handle these edge cases with imputation or to omit.
any(is.na(df)) 
df %>% drop_na() %>% nrow()

#  quick data summary - check for: any negative/nonsensical values?  any negative values in numerical cols?
# some variables need to be checked and cases excluded - see age max 999, estimated salary -176137
summary(df)

# excluding outliers as edge cases - can be reviewed separately at later stage
# Anyone aged 62+ excluded - 2.4% customers
ggplotly(ggplot(df %>% select(Age), aes(y=Age)) + geom_boxplot())
Q1_Age <- quantile(df$Age, .25)
Q3_Age <- quantile(df$Age, .75)
IQR_Age <- IQR(df$Age)

# exclusions to apply, applied retrospectively after some tidying. 94% data remains.
df_tidy <- df %>% 
  distinct() %>% 
  drop_na() %>%
  filter(Age < (Q3_Age + 1.5*IQR_Age) &
         EstimatedSalary > 0)


### ----------------------------------------------------------------- Produce final tables for analysis ----------------------------------------------------------------- ###
df_analysis <- df_tidy

### ----------------------------------------------------------------- feature engineering ----------------------------------------------------------------- ###
# note: not all of these features are used - but useful to assess in EDA and create features for use during analysis and/or modelling

# How do features correlate with Churn or show trends? Can we create some features to use here should the original prove insignifcant in the model?
# focus on continuous features 

# num of products - churned customers have median 1 product - does having > 1 product correlate with not churning? more interest in product?
ggplotly(df_analysis %>% 
           select(NumOfProducts, Churned) %>% 
           ggplot(., aes(y=NumOfProducts, x=Churned, fill=Churned)) + 
           theme_bw() +
           geom_boxplot())

NumOfProducts_bucket <- df_analysis %>% 
  mutate(NumOfProducts_Only1 = as.factor(ifelse(NumOfProducts==1, 1, 0))) %>% # note this assumes no customers have 0 prods.
  select(CustomerId, NumOfProducts_Only1) %>% 
  distinct()
# note there isn't much different in the spread of this variable - however we can assess how closely this correlates with churn in model

# Tenure
ggplotly(df_analysis %>% 
           select(Tenure, Churned) %>%
           ggplot(., (aes(y=Tenure, x=Churned, fill=Churned))) +
           geom_boxplot())

Tenure_buckets <- df_analysis %>% 
  mutate(Tenure_buckets = ifelse(Tenure<=1, 1,
                                 ifelse(Tenure<=2, 2,
                                        ifelse(Tenure <=3,3,
                                               ifelse(Tenure<=4,4,
                                                      ifelse(Tenure<=5,5,0)))))) %>%
  select(CustomerId, Tenure_buckets) 

Tenure_buckets2 <- df_analysis %>% 
  mutate(Tenure_buckets1ornot = ifelse(Tenure <=1,1,0)) %>%
  select(CustomerId, Tenure_buckets1ornot) 

# Creditscore - the plots are not significantly different. Let's pass on creating a feature here.
ggplotly(df_analysis %>% 
           #filter(CreditScore<600) %>%
           select(CreditScore, Churned) %>%
           ggplot(., (aes(y=CreditScore, x=Churned, fill=Churned))) +
           geom_boxplot())

CreditScore_bucket <- df_analysis %>% 
  mutate(CreditScore_buckets = ifelse(CreditScore < 600,1,0)) %>%
  select(CustomerId, CreditScore_buckets)

# Estimated Salary - the plots are not significantly different again. Let's pass on creating a feature here.
ggplotly(df_analysis %>% 
           select(EstimatedSalary, Churned) %>%
           ggplot(., (aes(y=EstimatedSalary, x=Churned, fill=Churned))) +
           geom_boxplot())

#library("gridExtra")
#grid.arrange(a, b, c, d , 
#          labels = c("A", "B", "C", "D"),
#          ncol = 2, nrow = 2)


### ----------------------------------------------------------------- Modelling ----------------------------------------------------------------- ###

# Consider is active as baseline model? what would recall be with this alone?
# build contender model - how many more would we capture? how much does recall improve?

# create binary vbls from categorical vbls
# scaling numerical values not required here as chosen model for task is logistic regression 

df_model_data <- df_analysis %>%
  left_join(., NumOfProducts_bucket, by=c("CustomerId")) %>%
  left_join(., Tenure_buckets, by= c("CustomerId")) %>%
  left_join(., CreditScore_bucket, by= c("CustomerId")) %>%
  dummy_cols(., 
             select_columns = c('Geography', 'Tenure_buckets'), 
             remove_selected_columns = TRUE,
             remove_first_dummy = TRUE) %>%
  mutate_at(c("Geography_Y", "Geography_Z",
              "Tenure_buckets_1", "Tenure_buckets_2","Tenure_buckets_3",
              "Tenure_buckets_4","Tenure_buckets_5",
              "CreditScore_buckets"), ~as.factor(.))

# Splitting the data into train and test
set.seed(1)
index <- createDataPartition(df_model_data$Churned, p = .75, list = FALSE)
train <- df_model_data[index, ]
test <- df_model_data[-index, ]

########################################################### BASELINE MODEL
# Training the baseline model
baseline_model <- glm(Churned ~ IsActiveMember, family = binomial, data=train)
summary(baseline_model)

# Predicting in the test dataset
pred_prob <- predict(baseline_model, test, type = "response")

# Converting from probability to actual output
train$pred_class <- ifelse(baseline_model$fitted.values >= 0.5, 1,0)
test$pred_class <- ifelse(pred_prob >= 0.5, 1,0)

# Generating the classification table
ctab_train <- table(train$Churned, train$pred_class)
ctab_train
ctab_test <- table(test$Churned, test$pred_class)
ctab_test

# IsActiveMember is so strongly correlated with Churn that building a model with this feature alone identifies 80% of non churners and no actual churners, 
# which is a good accuracy score however meaningless in this scenario as the data is imbalanced in this way. We want to identify churners and need more variables
# to be able to identify those cases.

###########################################################  INITIAL MODEL WITH ALL FEATURES
set.seed(2)
logistic_model <- glm(Churned ~  CreditScore + Gender + Age  + 
                        Tenure + Balance + NumOfProducts + HasCrCard +
                        IsActiveMember + EstimatedSalary +
                        Geography_Z, 
                      family = binomial, data=train)
# Checking the model
summary(logistic_model)

# Predicting in the test dataset
pred_prob <- predict(logistic_model, test, type = "response")
# Converting from probability to actual output
train$pred_class <- ifelse(logistic_model$fitted.values >= 0.5, 1, 0)
test$pred_class  <- ifelse(pred_prob >= 0.5, 1,0)

# Assess Performance
# Precision and recall most important for this business case as we want to insure most churners are captured before they churn.
# Accuracy is not the most helpful metric here as dataset imbalanced, but it helps us identify is model is overfitting in training.

# Converting from probability to actual output
# Generating the classification table for test dataset
ctab_train <- table(train$Churned, train$pred_class)
ctab_train
ctab_test  <- table(test$Churned, test$pred_class)
ctab_test
#Accuracy = (TP + TN)/(TN + FP + FN + TP)
accuracy_train <- sum(diag(ctab_train))/sum(ctab_train)*100
accuracy_test  <- sum(diag(ctab_test))/sum(ctab_test)*100
accuracy_train
accuracy_test
#Recall/ Sensitivity = (TP) / (TP + FN)
#indicates how often does our model predicts actual TRUE from the overall TRUE events.
Recall <- (ctab_test[2,2]/sum(ctab_test[2,]))*100
Recall
#Precision indicates how often does your predicted TRUE values are actually TRUE. TP/(FP + TP)
# Precision in Train dataset
Precision <- (ctab_test[2,2]/sum(ctab_test[,2]))*100
Precision

#The value 27.1 tells us that the logistic regression model is able to catch 27.1% of customers, who in reality left company. 
#The value of precision 65.4 means that 65.4% of customers predicted to leave in fact left the company.
#This ratio is not so big in comparison with the sensitivity, but is acceptable because of the higher importance
#of sensitivity for the company

# so of all the predictions we make, we are getting a fair number correct, but we are also missing out a lot of churned customers.

###########################################################  FINAL MODEL FOLLOWING BUCKETED VBLS AND FEATURE SELECTION
set.seed(23) # 12 tenure imp, 13 credit imp, 19 cred imp, ten v close , same 23
logistic_model2 <- glm(Churned ~ CreditScore + Gender + Age +
                        NumOfProducts_Only1 + 
                        IsActiveMember  +
                        Geography_Z, 
                      family = binomial, data=train)

# Checking the model
summary(logistic_model2)

# Predicting in the test dataset
pred_prob <- predict(logistic_model2, test, type = "response")
# Converting from probability to actual output
train$pred_class <- ifelse(logistic_model2$fitted.values >= 0.5, 1, 0)
test$pred_class  <- ifelse(pred_prob >= 0.5, 1,0)

# Assess Performance
# Precision and recall most important for this business case as we want to insure most churners are captured before they churn.
# Accuracy is not the most helpful metric here as dataset imbalanced, but it helps us identify is model is overfitting in training.

# Generating the classification table for test dataset
ctab_train <- table(train$Churned, train$pred_class)
ctab_train
ctab_test  <- table(test$Churned, test$pred_class)
ctab_test
#Accuracy = (TP + TN)/(TN + FP + FN + TP)
accuracy_train <- sum(diag(ctab_train))/sum(ctab_train)*100
accuracy_test  <- sum(diag(ctab_test))/sum(ctab_test)*100
accuracy_train
accuracy_test
#Recall/ Sensitivity = (TP) / (TP + FN)
#indicates how often does our model predicts actual TRUE from the overall TRUE events.
Recall <- (ctab_test[2,2]/sum(ctab_test[2,]))*100
Recall
#Precision indicates how often does your predicted TRUE values are actually TRUE. TP/(FP + TP)
# Precision in Train dataset
Precision <- (ctab_test[2,2]/sum(ctab_test[,2]))*100
Precision


# Recall has marginally improved with significant variables included only, but model still isn't performant.


# Interpretation of values:
odds_ratio2 <- exp(coef(logistic_model2))
percent_change_in_odds2 <- (odds_ratio2 - 1)*100
percent_change_in_odds2
#(Intercept)          CreditScore           GenderMale                  Age      NumOfProducts_Only11      IsActiveMember1         Geography_Z1 
#-99.71504056          -0.02875147         -41.41306215          12.06713082             148.07779289         -57.47016827         141.35865001 
# for each increment of +1 in your credit score, the odds of you churning are you are .02% less 
# the odds of churning are 41% less if you're male
# for each +1 year in age, the odds of churning increase by 12%  
# the odds of churning are 148% greater if you have just one product
# the odds of churning are 58% lower if you are an active member
# the odds of churning are 141% greater if you are in geographyZ






















#------------------- cross-validation--------------------- #
# how to implement?

## define training control
#train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)
#
## train the model on training set
#model2 <- train(Churned ~ CreditScore + Gender + Age + 
#                 NumOfProducts_Only1 + EstimatedSalary + 
#                 IsActiveMember  +
#                 Geography_Z,
#               data = train,
#               trControl = train_control,
#               method = "glm",
#               family=binomial())
#
## print cv scores
#summary(model2)

roc <- roc(train$Churned, logistic_model$fitted.values)
auc(roc)


# Model is not performing well from a recall perspective - how can we improve this?
# engineer variables from those which were insignificant to see if we can have more detail on customers and utilise the data we have better
# could SMOTE to balance classes better 
# test alternative methods which deal with imbalanced classes better
# more data
# more balanced data/ more examples of churn in dataset


#------------------- refactoring notes --------------------- #
# create function to dynamically group and prevent code repetition 
# as a case study in the short time and as this is not being productionised it was faster to repeat when thinknig through useful features

# 