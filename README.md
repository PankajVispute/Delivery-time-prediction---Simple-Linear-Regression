# Prediction-with-Simple-linear-Regression
Implementation of simple linear regression to predict delivery time.

# Description
* Dataset is fall under supervised.
* In this project, simple linear regression model is used for prediction.

# Data set format
* CSV (Comma Separated Values) format.
* Attributes can be integer or real values.
* Responses can be integer, real or categorical.

# Overview
The primary goal is predict delivery time based on sorting time.

# liabrary 
* pandas, numpy, matplotlib,seaborn,sklearn,joblib used in project

# Methodology
1. ## Machine learning life cycle:
   - followed indistry standard practice of machine learning life cycle steps.
2. ## Preprocessing and EDA:
   - implement necessary transformation, preprocessing of dataset.
   - conduct exploratory data analysis on dataset.
3. ## Visualization:
   - visualised data using visualisation library like matplotlib, seaborn.
4. ## Algorithm:
   - scikit library use for linear regression.
5. ## model validation:
   - model validate with r2_score and root mean square error.
6. ## save model:
   - joblib library used to dump model.
   - model is saved in .ipynb formate as Delivery_pr1.
# EDA:
- delivery time is float and sorting time is int data type.
- No null values in dataset.
- No outliers in dataset.
- delivery time is having 82.5% correlation with sorting time.
- maximum occurence of sorting time is 7.
- max delivery time is 29, min delivery time is 8, average delivery time is 17
- max sorting time is 10, min sorting time is 2, average sorting time is 6. max sorting time count is 7.

  
# Simple Linear Regression model:
- model intercept_ is 6.6 and coef_ is 1.60
- model r2_score is 57% and RMSE is 2.33
