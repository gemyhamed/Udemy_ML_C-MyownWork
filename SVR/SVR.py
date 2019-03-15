# Importing the libraries
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import  os

# Change Directory to load file
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)')


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
y = np.array(y).reshape(-1,1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X,y)



# Predicting a new result
n = float(input(("What's the value to predict at ? ")))
n = np.array(n).reshape(-1,1)

y_pred = sc_y.inverse_transform(reg.predict((sc_y.transform(n))))
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
