## Project title
DS Write a blog post - Bike Sharing Demand

## Motivation
Data Science Udacity course requires writing a blog post. In this case I have explored Kaggle's popular 'Bike Sharing Demand' competition, and written a blog post on how to model this problem. The notebook uploaded to this repo shows EDA and modelling techniques used throughout to develop DS skills using this project.

The final blog post can be found here:
https://medium.com/@karina.gorasia/how-can-we-predict-bike-sharing-demand-45f2cb750c00


## Business Understanding
Ever since the roll out of bike hire schemes for on the fly bike usage, companies have tried to forecast how many bicycles they’ll need in set locations at a given time in order to meet demand, and it hasn’t been an easy feat. In an age where alternative transport methods are being ever more widely explored, this issue has become more prominent and of bigger interest. What helps us understand bike sharing demand and can we forecast it?

Understanding what drivers influence the count of hired bikes is key to being able to estimate how many will be in demand given these conditions.
1. What would help us predict bike demand? 
    a. Does weather impact bike hire?
    b. Is demand the same regardless of what day of the week it is? Are weekends different to weekdays? Is every month the same?
    c. Does demand fluctuate across daytime?
2. What does bike sharing demand look like? Is it consistent? A good grasp of what we are looking to predict helps ensure more accurate predictions can be made.
3. How can we go about predicting demand? What models work best?

## Data Understanding
Kaggle provides a dataset for this very problem, containing 2 years of data surrounding weather(temperatures, humidity, conditions) and counts of bike demand hourly and daily, which have been used for this analysis. The dataset consists of 10886 records with the following columns:
datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count     

## Prepare Data
The features were explored individually compared with the target variable, as well as combining features to understand how they impacted bike sharing demand. The "Prepare Data" section breaks this down in relation to the key business questions being assessed.

## Data Modeling
Three models are evaluated in this work, each with varying performance:
Model | Training RMSLE | Validation RMSLE | 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Linear Regression, log transformed target | 0.270600 | 1.089280 | 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Random Forest with bucketed variables, no dummies, no log transformed target | 0.696200 | 0.807400 | 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Random Forest with bucketed variables, dummies and log transformed target | 0.215500 | 0.735010 |

## Evaluate the Results

### Let's cycle back to our three key questions:

1. What would help us predict bike demand?
    a. Does weather impact bike hire? 
    Yes. More settled conditions and warmer temperatures drive higher demand, though there are other less expected conditions to consider when forecasting bike sharing demand too, such as cooler settled weather or windier conditions when its warm.
    b. Is demand the same regardless of what day of the week it is? 
    When analysing trends across days alone, there was no significant difference in average bike sharing demand.
    When looking across months, the winter months were unsurprisingly quieter. 
    c. Does demand fluctuate across daytime?
    However when we separate weekends from weekdays, we see that peak commuting hours signal higher bike demand on weekdays, whereas weekend hire follows a more normal distribution. This is a useful factor to consider.
 
3. How well can we predict bike share demand? 
It's clear that modelling bike sharing demand is no easy feat. Though we did find some factors repeatedly useful in helping understand and predict demand (temperatures, time of day), in spite of trying different modelling methods and approaches such as scaling, dummying difference variables and log transformation, it stills proves a challenge to accurately predict bike sharing demand. We also see that performance in validation drops greatly, signalling there is some clear overfitting happening during training. Linear Regression is arguably not the way forwads for this problem, though we have demonstrated that considering many features at different levels of granularity (datetime) and exploring different models, we can work towards understanding bike sharing demand more.

## Tech/libraries used
Developed in jupyter notebook
Libraries used: numpy, pandas, matplotlib, seaborn, sklearn, scipy, statsmodels, datetime

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation
Install Anaconda Navigator and open Jupyter notebooks

## How to use?
Create a pull request for the notebook or download to local environment. Ensure all the packages are installed locally so you can import. I have used the Anaconda environment which provides inbuilt packages so importing is easy and no need to pip install.

## Credits
https://www.kaggle.com/c/bike-sharing-demand

Udacity - Data Science 

## License
Anaconda Navigator

KarinaGorasia
