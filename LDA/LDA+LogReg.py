
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import  seaborn as sns
from sklearn.preprocessing import Imputer

# change directory
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 9 - Dimensionality Reduction\\Section 44 - Linear Discriminant Analysis (LDA)')
df = pd.read_csv('Wine.csv')

x = df.iloc[:,0:13].values
y = df.iloc[:, 13].values


# Train test Split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Apply LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
lda = lda(n_components= 2)
x_train=lda.fit_transform(x_train,y_train)
x_test = lda.transform(x_test)


# Fitting Model
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(x_train,y_train)

y_predict = cls.predict(x_test)

# Confusion Matrix to evaluate our model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print(cm)

# Visualising the Test set results
from matplotlib.colors import ListedColormap

# create a copy of X_train and y_train
# for code resusability
X_set, y_set = x_train, y_train
# generate coordinate matrics using meshgrid
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
 np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
# now we need to plot the prediction for every coordinate from X1 and X2 therefore we need to first get the prediction from coordinate matrics
# convert X1 coordinate matrix to a flattened array - putting all the elements in one column
X1_ravel = X1.ravel()
# convert X2 coordinate matrix to a flattened array - putting all the elements in one column
X2_ravel = X2.ravel()
# create an array having 2 rows by placing X1_ravel over X2_ravel
X1X2_array = np.array([X1_ravel, X2_ravel])
# Since predict function takes an array which has 2 columns therefore we need to generate Transpose of X1X2_array - columns are converted into rows
X1X2_array_t = X1X2_array.T
# predict result using the classifier
X1X2_pred = cls.predict(X1X2_array_t)
#result of prediction will be used to plot againt the coordinate matrics therefore we need to reshape the result to match the shape of coordinate matrics
#generated array contains prediction for every coordinate value
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
#plot the predictions against coordinate matrics using contourf (filled)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape,
 alpha=0.75,
 cmap = ListedColormap(('red', 'green','blue'))
)
#not mandatory
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plot the actual points on the graph
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
 c = ListedColormap(('red', 'green','blue'))(i), label = j)
#for housekeeping
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

