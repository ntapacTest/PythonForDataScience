# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:33:19 2019

@author: Дмитрий
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#мы создаем фиктивный набор данных вместо импорта реального.
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=300, n_features=2, 
                           centers=5, cluster_std=1.8,random_state=101)


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='brg')


kmeans = KMeans(n_clusters=5) #Выбираем количество кластеров. Пусть К=5
kmeans.fit(data[0])
#При этом параметры, беруться по умолчанию, в данном случае –
#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#    n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
#    random_state=None, tol=0.0001, verbose=0)


kmeans.cluster_centers_
kmeans.labels_


### Проверка
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='brg')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='brg')


