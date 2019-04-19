# Importing the libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)')

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Data (Male/Female)
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

 # Changing The Column of Country
l_e_X_1 = LabelEncoder()
X[:,1] = l_e_X_1.fit_transform(X[:,1])

 # Changing The Column of Gender
l_e_X_2 = LabelEncoder()
X[:,2] = l_e_X_2.fit_transform(X[:,2])

 # One Hot Encoding for Country Column
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
 # Dropping a column to avoid dummy variables
X = X[:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN
import keras
# Intiliaze our NN
from keras.models import Sequential
from keras.layers import Dense
cls = Sequential()
# Building The first Hidden Layer & The Input Layer
cls.add(Dense(output_dim = 6 ,init = 'uniform',activation='relu',input_dim= 11))

# Building The Second Hidden Layer
cls.add(Dense(output_dim = 6 ,init = 'uniform',activation='relu'))

# Building The Output Layer
cls.add(Dense(output_dim = 1 ,init = 'uniform',activation='sigmoid'))

# Compiling the NN
cls.compile(optimizer='adam',loss = 'binary_crossentropy' ,metrics = ['accuracy'])


# Fitting the NN to the Data
cls.fit(X_train,y_train,batch_size=10,nb_epoch=100)


# Predicting

y_pred = cls.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

