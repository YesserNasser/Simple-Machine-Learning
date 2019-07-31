# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:05:00 2019

@author: Yesser
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import data
data = pd.read_csv('DataR3.csv')
df=pd.DataFrame(data, columns=['sepal length','sepal width','petal length','petal width'])
print(df)

# isolate data values for preprocessing
    #defining features
features = ['sepal length','sepal width','petal length', 'petal width']
x=df.loc[:,features].values
    #scaling the data for any future operation
x1=StandardScaler().fit_transform(x)

# clustering the data
kmeans=KMeans(n_clusters=3)
y=kmeans.fit_predict(x1)
df['cluster']=y

# pricipal component analysis
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x1)
ReducedData=pd.DataFrame(principalComponents, columns=['pca1','pca2'])
ReducedData['cluster']=y

# ploting the data
fig=plt.figure(figsize=(8,8))
plt.scatter(ReducedData.pca1, ReducedData.pca2, c=kmeans.labels_.astype(float), s=50, Alpha=0.5)
plt.xlabel('Principal Component 1', fontsize=15)
plt.ylabel('Principal Component 2', fontsize=15)
plt.title('Principal Components Analysis (2)', fontsize=20)
plt.grid()    
# scalling the data is very import it produced different results with clustering and princiapl component analysis