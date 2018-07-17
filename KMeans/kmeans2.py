import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


data = pd.read_csv('College_Data',index_col=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop('Private',axis=1))

def converter(private):
	if private == 'Yes':
		return 1
	else:
		return 0

data['cluster'] = data['Private'].apply(converter)

#Evaluating our model
#In real world scenario, we won't have this luxury as we will not have the true values to confirm with
print(classification_report(data['cluster'],kmeans.labels_))