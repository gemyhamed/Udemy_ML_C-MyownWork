# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  seaborn as sns
import  os

# Importing The Dataset
os.chdir('E:\\Programing\\UdemyML\\Machine Learning A-Z Template Folder\\Part 4 - Clustering\\Section 25 - Hierarchical Clustering')
df = pd.read_csv('Mall_Customers.csv')
x = df.iloc[:,[3,4]].values

# Using Dendogram To Find optimal No.of Clusters
from  scipy.cluster.hierarchy  import  linkage,dendrogram

hc1 = linkage(x,method='ward')
dendrogram(hc1,leaf_rotation=90)
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.show()

# Fitting Hierarchical Clustering Using Sklearn (use it because it's easier to fit and predict )

from sklearn.cluster import AgglomerativeClustering
hc2 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage = 'ward')
y_hc = hc2.fit_predict(x)
print(y_hc)

# Visualization of the Results
y_k = y_hc
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
