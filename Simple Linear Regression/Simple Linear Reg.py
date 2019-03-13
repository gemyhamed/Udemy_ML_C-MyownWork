import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.preprocessing import Imputer

# change directory
os.chdir('E:\\test\\dir-test')
# import data
df = pd.read_csv('Salary_Data.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:, 1].values

print(y.shape)
# Train test Split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=1/3,random_state=0)

# Training Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
l_r = LinearRegression()
l_r.fit(x_train,y_train)
# Prediction
y_predict = l_r.predict(x_test)

# Visualization Training Data
plt.scatter(x_train,y_train,color = 'blue')
plt.plot(x_train,l_r.predict(x_train),color = 'red')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs Experience For the Training Set')
plt.show()

# Visualization Test Data
plt.scatter(x_test,y_test,color = 'blue')
plt.plot(x_train,l_r.predict(x_train),color = 'red')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs Experience For the Test Set')
plt.show()

# Printing The Score
print(l_r.score(x_train,y_train))