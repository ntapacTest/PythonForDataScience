# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:28:59 2019

@author: Дмитрий
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold


#Пример встроенных наборов
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=2) 
plt.scatter(X[:, 0], X[:, 1], c=y, s=75,alpha=0.75) 
plt.xlabel("first feature") 
plt.ylabel("second feature")

#Встроенный  набор c 20 признаками
from sklearn.datasets import make_classification
X, y = make_classification(100) 
plt.scatter(X[:, 0], X[:, 1], c=y, s=75,alpha=0.5) 
plt.xlabel("first feature") 
plt.ylabel("second feature")

#Пример на искуственных данных
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,
                            random_state = 42,stratify = y)
knn = KNeighborsClassifier(n_neighbors=1)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, 
#                     metric='minkowski', metric_params=None, 
#                     n_neighbors=1, p=2, weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (knn.score(X_test, y_test))

#Используем на совершенно новых данных не работает
# y_pred = knn.predict([[4,-3],[2,1]])
# plt.scatter([4,2],[-3,1],  c=y_pred, alpha=0.75,marker ='*')



#Реальный пример
df = pd.read_csv('Data/diabetes.csv')
df.shape
df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin",
              "bmi","Diabetes_Pedigree_Function","age","outcome"]

np.bincount(df.outcome)  

df.glucose.replace(0,np.nan,inplace = True)
df.insulin.replace(0,np.nan,inplace = True)
df.blood_pressure.replace(0,np.nan,inplace = True)
df.bmi.replace(0,np.nan,inplace = True)
df.skin_thickness.replace(0,np.nan,inplace = True)
df.age.replace(0,np.nan,inplace = True)
df.Diabetes_Pedigree_Function.replace(0,np.nan,inplace = True)

df.info()

df = df.fillna(df.mean())

df.describe()

df['insulin'].describe()

df['insulin'] = scale(df['insulin'])


y = df['outcome'].values
X = df.drop('outcome',axis =1).values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

#Три следующие операции эквивалентны 
knn.score(X_test,y_test)
metrics.accuracy_score(y_test, y_pred)
np.mean(y_pred == y_test)


#Изменение точности в зависимости от количества ближайших соседей

neighbors  = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)


plt.title('KNN : Accuracy Curve')
plt.plot(neighbors,test_accuracy,label = 'Testing Accuracy')
plt.plot(neighbors,train_accuracy,label = 'Training Accuracy')
plt.xlabel('No of Neighbors')
plt.ylabel('Accuracy')
plt.legend(loc = 'upper right')
plt.show()


knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
knn.score(X_test,y_test)
metrics.accuracy_score(y_test, y_pred)


kf = KFold(n_splits=5) 
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn = KNeighborsClassifier(n_neighbors = 8)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print(knn.score(X_test,y_test))

    





