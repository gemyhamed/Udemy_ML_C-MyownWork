import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# change directory
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression')
# import data
df = pd.read_csv('Position_Salaries.csv')

x = df.iloc[:,1:2].values
y = df.iloc[:,2].values

# Building Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Getting Input to What the User Wants to predict and to plot
degre = int(input("What's the degree to plot ? "))
n = float(input(("What's the value to predict at ? ")))
n = np.array(n).reshape(-1,1)

# Polynomial Regression plots and predictions

for degre in range(degre):
    poly = PolynomialFeatures(degree=degre)
    x_poly = poly.fit_transform(x)
    reg = LinearRegression()
    reg.fit(x_poly, y)
    plt.subplot(3,2,degre+1)
    plt.scatter(x, y)
    plt.plot(x, reg.predict(x_poly))
    plt.title(' Polynomial Regression Of '+ str(degre) + ' polynomial Degree')
    print("Value of Prediction at " + str(degre) + " polynomial Degree = " + str(reg.predict(poly.fit_transform(n))))

plt.show()





