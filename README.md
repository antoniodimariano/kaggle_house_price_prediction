# Description 
AI Learning Program Challenge 03 : Houses' price prediction using kaggle dataset


* Download the initial data from https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Goal:
Help our wizards to predict the sales price for each house. For each Id in the test set, you must predict the value of the "SalePrice" variable using Linear Regression algorithms
It will be adding extra value and score to have trying different algorithms to achieve best metrics
The metrics to be consider will be the following:
Code and solution
Model accuracy
Bias & Variance
Price Accuracy Metric:
Submissions are evaluated on the difference between the predicted value and the observed final sale price
Help and tutorials:
There are several tutorials and sources to look at when developing this model:

* https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
* https://www.kaggle.com/dgawlik/house-prices-eda
* https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
* https://www.kaggle.com/apapiu/regularized-linear-models

# Dataset and Process Flow: 
This dataset is commonly used in data science education, it has so many variables in dataset that can describe almost every aspect of the real states. The objective of this project is to utilize supervised machine learning techniques to predict the housing prices for each home in the dataset
The steps towards creating a highly accurate model may be as follows:

# Exploratory Data Analysis (EDA)
Data cleaning
Categorization
Value Missing and Imputing
Dummification (Advanced and Expert levels only)
Feature engineering
Add new features
Scaling
Pre-Modeling
Cross-Validation (Tuning)
Modeling
Exploratory Data Analysis (EDA):

# Categorization:
You may start by exploring and understanding the dataset. For example: divide the variables into categories: General or Continuous, Nominal Categorical, Ordinal Categorical, Lable...
Lable:
Sale price is the value we are looking to predict in this project, so it makes sense to examine this variable first. The sale price has righ-skewed distribution, and may need to be considered

# Value Missing and Imputing:
Next, you may want to look at missing values by feature in the full dataset. There is significant amount of missing values by feature across both the dataset. Most of the missing data corresponded to the absence of a feature. For example, the Garage features, mentioned in the below table, showed up as "NA" if the house did not have a garage. These normally are imputed as 0 or "None" depending on the feature type
This particular task may impact heavily in the final model accuracy so try not skipping it
Correlation:
As an example: Sale Price is strongly correlated with these continuous variables
Find the highest correlations in this dataset
Higher Correlation with the Lable
Ground Living Area : ~0.70
Garage #Cars : ~0.64
Garage Area: ~0.62
Total Basement Surface : ~0.61
1st Floor Surface : ~0.60
Full Bath : ~0.56


# Requirements 
* Python 3.6 or higher

# How to run 

```
python main.py 
```
