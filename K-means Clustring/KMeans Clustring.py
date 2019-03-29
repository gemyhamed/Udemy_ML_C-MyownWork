# importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

# Importing the dataset
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 4 - Clustering\\Section 24 - K-Means Clustering')
df = pd.read_csv('Mall_Customers.csv')
print(df)
x = df.iloc[:,[3,4]].values

# Using Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    k = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    k.fit(x)
    wcss.append(k.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title('Elbow Method ')
plt.xlabel('No.of.Clusters')
plt.ylabel('WCSS Score')
plt.show()

# Fitting The Model To 5 Clusters
k = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_k = k.fit_predict(x)
print(y_k)

# Scatter Plot The Clusters
plt.scatter(x[y_k==0,0],x[y_k==0,1],c='red',label = 'Cluster 1')
plt.scatter(x[y_k==1,0],x[y_k==1,1],c='blue',label = 'Cluster 2')
plt.scatter(x[y_k==2,0],x[y_k==2,1],c='green',label = 'Cluster 3')
plt.scatter(x[y_k==3,0],x[y_k==3,1],c='yellow',label = 'Cluster 4')
plt.scatter(x[y_k==4,0],x[y_k==4,1],c='cyan',label = 'Cluster 5')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clustering Of Mall Clients')
plt.legend()
plt.show()


