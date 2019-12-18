# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import statsmodels.api as sm

#Ручной расчет регрессии
np.random.seed(123456)
x=np.arange(1,301)
y=np.random.normal(x+2,50)

plt.scatter(x,y)
numerator=np.sum((x-np.mean(x))*(y-np.mean(y)))
denominator=np.sum((x-np.mean(x))**2)
b1=numerator/denominator
b0=np.mean(y)-b1*np.mean(x)
y_pred=b0+b1*x
plt.plot(x,y_pred,c='g')
#Считаем RMSE
rmse=np.sqrt((np.sum((y_pred-y)**2))/len(y))



#На реальных данных

dataset = pd.read_csv  ("Data/Weather.csv")
dataset.shape
dataset.describe()

dataset.plot(x='MinTemp', y='MaxTemp', style='o')  

plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.hist(dataset['MaxTemp'],bins=35,density=True)

sbn.distplot(dataset['MaxTemp'],bins=35)


X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Можно проверить "ручной" алгоритм
#x=X_train
#y=y_train
#plt.scatter(x,y)
#numerator=np.sum((x-np.mean(x))*(y-np.mean(y)))
#denominator=np.sum((x-np.mean(x))**2)
#b1=numerator/denominator
#b0=np.mean(y)-b1*np.mean(x)
#
#y_pred=b0+b1*x
#plt.plot(x,y_pred,c='g')
#rmse=np.sqrt((np.sum((y_pred-y)**2))/len(y))


regressor = LinearRegression()  
rModel=regressor.fit(X_train, y_train)
r_sq = regressor.score(X_train, y_train)

print(regressor.intercept_)   

print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(100)

df1.plot(kind='bar',figsize=(16,10))

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Множественная регрессия
dataset = pd.read_csv('Data/winequality.csv')
dataset.shape
dataset.describe()


dataset.isnull().any()
#Если бы Nan присутствовал, то мы бы выкинули соответствующие строки
dataset = dataset.fillna(method='ffill')

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values

sbn.distplot(dataset['quality'],kde=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

r_sq = regressor.score(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, dataset.columns[:-1], columns=['Coefficient'])  
coeff_df

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




#Полиноминальная регрессия
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

transformer = PolynomialFeatures(degree=2, include_bias=False)


transformer.fit(x)
x1 = transformer.transform(x)


x1 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

model = LinearRegression().fit(x1, y)
r_sq = model.score(x1, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

y_pred = model.predict(x1)


x2 = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)

model2 = LinearRegression(fit_intercept=False).fit(x2, y)
r_sq2 = model2.score(x2, y)
print('coefficient of determination:', r_sq2)
print('intercept:', model2.intercept_)
print('coefficients:', model2.coef_)

y_pred2 = model2.predict(x2)



# Многомерный случай
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
x3 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
model3 = LinearRegression().fit(x3, y)
r_sq3 = model3.score(x3, y)
intercept, coefficients = model3.intercept_, model3.coef_
y_pred = model3.predict(x3)


#линейная регрессия со statsmodels
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

x = sm.add_constant(x)
print(x)


model = sm.OLS(y, x)

results = model.fit()

print(results.summary())
print('coefficient of determination:', results.rsquared)
print('adjusted coefficient of determination:', results.rsquared_adj)

print('regression coefficients:', results.params)



print('predicted response:', results.fittedvalues, sep='\n')
print('predicted response:', results.predict(x), sep='\n')


x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print(x_new)
y_new = results.predict(x_new)
print(y_new)
