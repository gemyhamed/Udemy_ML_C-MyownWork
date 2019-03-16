# Regression Template

# Importing the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X,y)

# Predicting a new result
n = np.array(6.5).reshape(-1,1)
y_pred = reg.predict(n)

print(y_pred)

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()