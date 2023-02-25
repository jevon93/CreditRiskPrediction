########
##Credit Default Prediction Code
##Data used: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
##Written by Rhys Jevon on 25/02/2023
########
########
##Last Update: N/A
##Change Log
##N/A
########

##Load dependencies
library(dplyr)
library(caret)
library(scales)
library(ggplot2)

##Set working directory to where data is stored and load data into R
setwd("C:/Data/")
credit<-read.csv("credit_risk_dataset.csv",header=TRUE)

##A bit of exploratory analysis to look for outliers
head(credit)

plot(credit$person_income)

plot(credit$person_emp_length)

plot(credit$loan_amnt, xlim=c(0,32581), ylim=c(0,100000))

##Remove outliers
##Outliers can  effect model accuracy and predictive strength
##Depending on use case this can be edited
creditfixed<-credit %>%
  filter(person_income < 200000)

creditfixed<-creditfixed %>%
  filter(person_emp_length < 100)

##log transform the percentage of income variable
##Zero to One percentage variables can be log transformed to increase their variability
##This can increase their usefulness as independent variables
log_loan_percent_income<- log(creditfixed$loan_percent_income)

creditfixed<-cbind(creditfixed,log_loan_percent_income)

##Checking data types on factor variables
##typeof(creditfixed$cb_person_default_on_file)
##typeof(creditfixed$loan_grade)
##typeof(creditfixed$loan_intent)
##typeof(creditfixed$person_home_ownership)
##typeof(creditfixed$cb_person_default_on_file)

##Fixing data types on factor variables
creditfixed$cb_person_default_on_file <- as.factor(creditfixed$cb_person_default_on_file)
creditfixed$loan_grade <- as.factor(creditfixed$loan_grade)
creditfixed$loan_intent <- as.factor(creditfixed$loan_intent)
creditfixed$person_home_ownership <- as.factor(creditfixed$person_home_ownership)


hist(creditfixed$loan_status)

##Good amount of positive cases


##For Clearing Working Memory
##rm(Graph_Data)
##rm(Graph_Data_Final)



##Initialise tables needs in for loop
##Values default to being store as a character
##We will transform them after, outside of the for loop
Graph_Data<-data.frame(Iteration=character(),Fold=character(),PPV=character(),
                       NPV=character(),Threshold=character())
Graph_Data_Final<-data.frame(Iteration=character(),Fold=character(),PPV=character(),
                             NPV=character(),Threshold=character())

##Will be used to index our results from iterations on the k-folds
start_row<-seq(1, 400, by = 4)

###################################################################################
##In this example case we have decided to use a stratified 5-fold analysis and 
##analyse the PPV (Positive Predictive Value) and NPV (Negative Predictive Value)
##to try to determine model fit.
##
##We use the first fold of 20% of the data as the training set and test it on the
##remaining four folds of data. We then repeat this process twenty times to 
##ensure we account for any variability in the random distribution of data between
##the five folds.
##
##Due to this code being run on my laptop we have computing power limitations.
##Therefore, I chose a reductive model building technique which started with a 
##fully saturated logistic model (all possible independent variables and interactions 
##included) and then removed terms with insignificant p-values one by one.
##
##In a business environment where computing power is not an issue, model selection
##can be done by adding a third layer to the two layer nested loop below which
##then iterates through a list of all possible models (perhaps logistic, glmboost 
##and random forest) and then the best model out of the thousands of possible 
##permutations can be selected based on whatever business criteria are needed
##
##The below code is an example of how each model can be tested and validated
##based on PPV and NPV in this case and then graphed based on threshholds.
###################################################################################


Threshold<-c('20%','40%','60%','80%')

##What is a threshold?##

##Since probabilistic models (in the case below, a logistic one) return a probability between
##zero and one, a threshold is needed to classify whether or not we predicted a customer
##default. For example, an 80% threshold means for probabilities of 0.8 and above
##we assign a value of 1 (as in yes - we predict the customer will default) and for any
##probability below 0.8 we predict a value of 0 (as in no - we predict the customer will not
##default)

for (a in 1:20)
  {
  flds <- createFolds(creditfixed$loan_status, k = 5, list = TRUE, returnTrain = FALSE)
  for (x in 1:4) 
    {
      ##Progress Tracker
      ##percent(a*5/100)
    
      ##Build logistic model on training fold
      log_model <- glm(loan_status ~ person_age:person_income 
                  + person_age:loan_intent
                  + person_income:loan_intent
                  + loan_intent:cb_person_default_on_file
                  + person_income:person_home_ownership
                  + person_income:loan_amnt
                  + person_age
                  + person_income
                  + person_home_ownership + person_emp_length
                  + loan_intent + loan_grade + loan_amnt
                  + cb_person_default_on_file 
                  + log_loan_percent_income
                  ,family=binomial(link='logit'),data=creditfixed[ flds[[1]], ])
      
      ##Predict probabilities of default on test fold
      probabilities<-predict.glm(log_model, newdata = creditfixed[ flds[[x+1]], ], type="response")
  
      ##Create data frame of actual credit default results from test data
      check_predict_values <- as.data.frame(creditfixed[ flds[[x+1]], 9 ]
                                            , responseName = "Actuals")
      
      ##Add column of predicted default probability to the data frame
      check_predict_values<-cbind(check_predict_values,probabilities)
      
      
      ##Create default predictions for different thresholds
      check_predict_values<-check_predict_values %>% 
        mutate(predicted_eighty = case_when(
        probabilities >= 0.8  ~ 1 ,
        probabilities <  0.8  ~ 0)
        )
      
      check_predict_values<-check_predict_values %>% 
        mutate(predicted_sixty = case_when(
          probabilities >= 0.6  ~ 1 ,
          probabilities <  0.6  ~ 0)
        )
      
      check_predict_values<-check_predict_values %>% 
        mutate(predicted_forty = case_when(
          probabilities >= 0.4  ~ 1 ,
          probabilities <  0.4  ~ 0)
        )
      
      check_predict_values<-check_predict_values %>% 
        mutate(predicted_twenty = case_when(
          probabilities >= 0.2  ~ 1 ,
          probabilities <  0.2  ~ 0)
        )
      
      ##Create column names for default prediction table
      colnames(check_predict_values) <- c("Actuals", "Probabilities"
                                          , "Predicted_Eighty"
                                          , "Predicted_Sixty"
                                          , "Predicted_Forty"
                                          , "Predicted_Twenty")
      
      ##Tally up the True Positives, False Positives, True Negative and False Negatives
      ##for each of the four threshold values
      check_predict_values<-check_predict_values %>% 
        mutate(Threshold_Eighty_Matrix = case_when(
          (Predicted_Eighty == 1 & Actuals ==1)  ~ 'TP' , #True Positive
          (Predicted_Eighty == 1 & Actuals ==0)  ~ 'FP' , #False Positive
          (Predicted_Eighty == 0 & Actuals ==0)  ~ 'TN' , #True Negative
          (Predicted_Eighty == 0 & Actuals ==1)  ~ 'FN' ) #False Negative
        )
      check_predict_values<-check_predict_values %>% 
        mutate(Threshold_Sixty_Matrix = case_when(
          (Predicted_Sixty == 1 & Actuals ==1)  ~ 'TP' , #True Positive
          (Predicted_Sixty == 1 & Actuals ==0)  ~ 'FP' , #False Positive
          (Predicted_Sixty == 0 & Actuals ==0)  ~ 'TN' , #True Negative
          (Predicted_Sixty == 0 & Actuals ==1)  ~ 'FN' ) #False Negative
        )
      check_predict_values<-check_predict_values %>% 
        mutate(Threshold_Forty_Matrix = case_when(
          (Predicted_Forty == 1 & Actuals ==1)  ~ 'TP' , #True Positive
          (Predicted_Forty == 1 & Actuals ==0)  ~ 'FP' , #False Positive
          (Predicted_Forty == 0 & Actuals ==0)  ~ 'TN' , #True Negative
          (Predicted_Forty == 0 & Actuals ==1)  ~ 'FN' ) #False Negative
        )
      check_predict_values<-check_predict_values %>% 
        mutate(Threshold_Twenty_Matrix = case_when(
          (Predicted_Twenty == 1 & Actuals ==1)  ~ 'TP' , #True Positive
          (Predicted_Twenty == 1 & Actuals ==0)  ~ 'FP' , #False Positive
          (Predicted_Twenty == 0 & Actuals ==0)  ~ 'TN' , #True Negative
          (Predicted_Twenty == 0 & Actuals ==1)  ~ 'FN')  #False Negative
        )
      
      ##Calculate PPV (Positive Predictive Value) And NPV (Negative ...)
      ##The formula for PPV is TP/TP+FP and the formulas for NPV is TN/TN+FN
      
      #Threshold == 80%
      TP_eighty<-sum(check_predict_values$Threshold_Eighty_Matrix == "TP")
      FP_eighty<-sum(check_predict_values$Threshold_Eighty_Matrix == "FP")
      TN_eighty<-sum(check_predict_values$Threshold_Eighty_Matrix == "TN")
      FN_eighty<-sum(check_predict_values$Threshold_Eighty_Matrix == "FP")
      
      PPV_eighty<-TP_eighty/(TP_eighty+FP_eighty)
      NPV_eighty<-TN_eighty/(TN_eighty+FN_eighty)
      
      #Threshold == 60%
      TP_sixty<-sum(check_predict_values$Threshold_Sixty_Matrix == "TP")
      FP_sixty<-sum(check_predict_values$Threshold_Sixty_Matrix == "FP")
      TN_sixty<-sum(check_predict_values$Threshold_Sixty_Matrix == "TN")
      FN_sixty<-sum(check_predict_values$Threshold_Sixty_Matrix == "FP")
      
      PPV_sixty<-TP_sixty/(TP_sixty+FP_sixty)
      NPV_sixty<-TN_sixty/(TN_sixty+FN_sixty)
      
      #Threshold == 40%
      TP_forty<-sum(check_predict_values$Threshold_Forty_Matrix == "TP")
      FP_forty<-sum(check_predict_values$Threshold_Forty_Matrix == "FP")
      TN_forty<-sum(check_predict_values$Threshold_Forty_Matrix == "TN")
      FN_forty<-sum(check_predict_values$Threshold_Forty_Matrix == "FP")
      
      PPV_forty<-TP_forty/(TP_forty+FP_forty)
      NPV_forty<-TN_forty/(TN_forty+FN_forty)
      
      #Threshold == 20%
      TP_twenty<-sum(check_predict_values$Threshold_Twenty_Matrix == "TP")
      FP_twenty<-sum(check_predict_values$Threshold_Twenty_Matrix == "FP")
      TN_twenty<-sum(check_predict_values$Threshold_Twenty_Matrix == "TN")
      FN_twenty<-sum(check_predict_values$Threshold_Twenty_Matrix == "FP")
      
      PPV_twenty<-TP_twenty/(TP_twenty+FP_twenty)
      NPV_twenty<-TN_twenty/(TN_twenty+FN_twenty)
      
      ##Add the NPV and PPV Values to a vector for insertion into graphing table
      PPV_graph<-c(PPV_twenty, PPV_forty, PPV_sixty, PPV_eighty)
      NPV_graph<-c(NPV_twenty, NPV_forty, NPV_sixty, NPV_eighty)
      
      ##Add columns to table (from right to left, Iteration, Fold, PPV, NPV and Threshold)
      Graph_Data[start_row[x],]<-c(a,x,PPV_graph[1],NPV_graph[1],Threshold[1])
      Graph_Data[start_row[x]+1,]<-c(a,x,PPV_graph[2],NPV_graph[2],Threshold[2])
      Graph_Data[start_row[x]+2,]<-c(a,x,PPV_graph[3],NPV_graph[3],Threshold[3])
      Graph_Data[start_row[x]+3,]<-c(a,x,PPV_graph[4],NPV_graph[4],Threshold[4])
  }
  ##Add the data to the final graph table from the previous iteration
  Graph_Data_Final <- Graph_Data_Final %>%
    bind_rows(Graph_Data)
}

##Transform data in final table to correct format for graphing
Graph_Data_Final$PPV<-as.numeric(Graph_Data_Final$PPV)
Graph_Data_Final$NPV<-as.numeric(Graph_Data_Final$NPV)
Graph_Data_Final$Threshold<-as.factor(Graph_Data_Final$Threshold)

##Plot a basic scatter graph with colour based on threshold value
p <- ggplot(Graph_Data_Final, aes(PPV, NPV)) +
  geom_point(aes(color=Threshold))

##View Graph to see how different threshold values influence PPV and NPV
p

##Use median PPV and NPV in the 80% threshold to estimate model predictive power
Graph_Data_Final %>% filter(Threshold == '80%') %>% summarise(PPV = median(PPV, na.rm = TRUE),
                                                              NPV = median(NPV, na.rm = TRUE))

##We have a median PPV of 89% and a median NPV of 99% for this model
##meaning we classify 89% of positives (credit defaults) correctly
##and only have a false positive rate of <1%

##A lot of work could go into the model building process here, for example
##xgboost, glmboost or random forest models will likely have even higher PPV.
##We could also experiment with higher thresholds as well.

##Hopefully this code has given you a basic overview of how you can build and test
##different credit risk models and a couple of the model fitting metrics that can be used
##to select credit risk models.