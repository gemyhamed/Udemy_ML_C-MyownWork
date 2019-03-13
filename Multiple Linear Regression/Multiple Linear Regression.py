# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Change Directory to load file
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression')

# Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Dealing With Dummy Variables
from sklearn.preprocessing import  LabelEncoder
l_e = LabelEncoder()
X[:,3] = l_e.fit_transform(X[:,3])

from sklearn.preprocessing import  OneHotEncoder
o_h_e = OneHotEncoder(categorical_features=[3])
X = o_h_e.fit_transform(X).toarray()

# Avoiding Dummy Variables Trap although it's not necessary as python can do it
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

# Predicting On the Test Set
y_predict = reg.predict(X_test)


# Model Optimization , use SL of 0.05

 # Add Column of Ones to the X Matrix ( Because we are using stats model library which can't deal with X0
ones = np.ones((50,1)).astype('int')
X = np.append(arr= ones,values=X,axis=1)

# Find the feature with the biggest p-value
import statsmodels.formula.api as sm

X_Opt = X[:,[0,1,2,3,4,5]]

#reg_ols = sm.OLS(endog = y,exog = X_Opt).fit()
#print(reg_ols.summary())

# Delete x2 and rerun the Regressor to find the biggest P-value for the remaining features and delete it
X_Opt = X[:,[0,1,3,4,5]]

#reg_ols = sm.OLS(endog = y,exog = X_Opt).fit()
#print(reg_ols.summary())

# Delete X1 and Rerun
X_Opt = X[:,[0,3,4,5]]

#reg_ols = sm.OLS(endog = y,exog = X_Opt).fit()
#print(reg_ols.summary())

# Delete X4 and Rerun

X_Opt = X[:,[0,3,5]]

reg_ols = sm.OLS(endog = y,exog = X_Opt).fit()
print(reg_ols.summary())

# The final X matrix is the one with columns 0,3,5
