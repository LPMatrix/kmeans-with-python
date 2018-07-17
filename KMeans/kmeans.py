import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data = make_blobs(n_samples=200,n_features=2,centers=4,random_state=101,cluster_std=1.8)

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])