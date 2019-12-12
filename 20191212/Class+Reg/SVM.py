# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:05:45 2019

@author: Дмитрий
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

bankdata = pd.read_csv("bill_authentication.csv")
bankdata.shape
bankdata.head()
#Делим данные на обучающую и тренировочную последовательности
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



#Kernel SVM 
#1. Polynomial Kernel

svclassifier = SVC(kernel='poly', degree=8) # degree -  степень полинома
svclassifier.fit(X_train, y_train)
#2. Gaussian Kernel
svclassifier = SVC(kernel='rbf')

#3. Sigmoid Kernel
svclassifier = SVC(kernel='sigmoid')
