# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:13:06 2019

@author: Дмитрий
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics
#from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import KFold

df = pd.read_csv('Data/diabetes.csv')
df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin",
              "bmi","Diabetes_Pedigree_Function","age","outcome"]
df.glucose.replace(0,np.nan,inplace = True)
df.insulin.replace(0,np.nan,inplace = True)
df.blood_pressure.replace(0,np.nan,inplace = True)
df.bmi.replace(0,np.nan,inplace = True)
df.skin_thickness.replace(0,np.nan,inplace = True)
df.age.replace(0,np.nan,inplace = True)
df.Diabetes_Pedigree_Function.replace(0,np.nan,inplace = True)
df.info()
df = df.fillna(df.mean())
df['insulin'] = scale(df['insulin'])
y = df['outcome'].values
X = df.drop('outcome',axis =1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


dtc = DecisionTreeClassifier(min_samples_split=4, random_state=0)

fig1=dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
 

y_pred = dtc.predict([[ 1, 90, 66, 32,-0.25, 34.9, 0.8, 56]])
print(y_pred[0])



#Баггинг
dtc = DecisionTreeClassifier()
# n_estimators - количество класификаторов которые будут использованы
bag = BaggingClassifier(dtc, n_estimators=100, max_samples=0.8,
random_state=1)
bg=bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# Boosting
# n_estimators - количество класификаторов которые будут использованы
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

y_pred = rfc.predict([[ 1, 90, 66, 32,-0.25, 34.9, 0.8, 56]])
print(y_pred[0])
