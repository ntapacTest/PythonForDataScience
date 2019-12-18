# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:29:09 2019

@author: Дмитрий
"""
#from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA

#Получаем ДФ
df = pd.read_csv('Data/a10.csv', parse_dates=['date']) 
df = pd.read_csv('Data/a10.csv',parse_dates=['date'],index_col='date')

#Визуализируем
fig, ax = plt.subplots(1, 1, figsize=(16,5))
plt.plot(df.index, df.value, color='tab:red')

plt.fill_between(df.index, y1=df.value, alpha=0.5, linewidth=2, color='seagreen')

#Построим гистограмму
df.value.hist(bins=30,density=True)
df.value.plot(kind='kde')
df.describe()

#Распарсим дату
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

#Coздадим свою палитру
mycolors= np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), 
                          len(years), replace=False)


plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', 
                 data=df.loc[df.year==y, :], 
                 color=mycolors[i], 
                 label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, 
                 df.loc[df.year==y, 
                'value'][-1:].values[0],
                    y, fontsize=12, color=mycolors[i])
        


#В виде box-plots
fig, axes = plt.subplots(1, 2, figsize=(20,7))
sns.boxplot(x='year', y='value', data=df, ax=axes[0])
sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])
axes[0].set_title('Изменения по годам\n(Тренд)', fontsize=18); 
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[1].set_title('зменения по месяцам\n(Сезонность)', fontsize=18)


#Сгенерируем данные из тренда и сезонности

fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
dft=pd.read_csv('Data/trend.csv', parse_dates=['date'])
axes[0].plot(dft.date,dft.value)

dfs=pd.read_csv('Data/seasone.csv', parse_dates=['date'])
axes[1].plot(dfs.date,dfs.value)

dfm=pd.merge(dft, dfs,left_on="date", right_on="date")
dfm['T+S']=dfm.value_x+dfm.value_y
axes[2].plot(dfm.date,dfm['T+S'])


#Как обнаружить наличие тренда и сезонности
fig, axes = plt.subplots(3,1)
axes[0].plot(df.index,df.value)
sm.graphics.tsa.plot_acf(df.value, ax=axes[1])
sm.graphics.tsa.plot_pacf(df.value, ax=axes[2])

pd.plotting.autocorrelation_plot(df.value.tolist())


#Декомпозирует тренд и сезонность 
df = pd.read_csv('Data/a10.csv',parse_dates=['date'],index_col='date')
result_mul = seasonal_decompose(df['value'], 
                model='multiplicative')
result_add = seasonal_decompose(df['value'],
                model='additive')               
result_mul.plot().suptitle('Мультипликативная декомпозиция', fontsize=22)
result_add.plot().suptitle('Аддитивная декомпозиция', fontsize=22)


#Посмотрим результаты в цифрах
df_reconstructed = pd.concat([result_mul.seasonal, 
                              result_mul.trend, result_mul.resid, 
                              result_mul.observed], axis=1)
df_reconstructed.columns = ['Сезонность', 
                            'Тренд', 'Ошибка', 
                            'Реальные значения']
df_reconstructed.head(15)


#Стационарный ряд
df.plot()
df_log = np.log(df.value)
df_log.plot()


fig, axes = plt.subplots(3,1)
axes[0].plot(df_log)
sm.graphics.tsa.plot_acf(df_log, ax=axes[1])
sm.graphics.tsa.plot_pacf(df_log, ax=axes[2])


#Взять разность первого прядка ручным методм
fig, axes = plt.subplots(3,1)
dataset=np.random.random(200)+np.linspace(-1,1,200)
axes[0].plot(dataset)
diff = list()
diff1 = list()
for i in range(1, len(dataset)):
    value = dataset[i] - dataset[i - 1]
    diff.append(value)
axes[1].plot(diff)
axes[2].hist(diff,bins=40,density=True)


dsf=pd.DataFrame(dataset)
diff2=dsf.diff(periods=1).dropna()
axes[2].plot(diff2)



#Удаление тренда с помощью построения регрессии
df = pd.read_csv('Data/a10.csv', parse_dates=['date']) 
fig, axes = plt.subplots(2,1)
regressor = LinearRegression()  
x=np.arange(len(df.index)).reshape(-1,1)
values=np.array(df['value'].values.reshape(-1,1))
regressor.fit(x, values)
Y=list()
for i in df.index:
    Y.append(regressor.intercept_[0]+ i*regressor.coef_[0,0])  
axes[0].plot(df.index, df.value)
axes[0].plot(x,Y[:])
diff=df['value'].values-Y[:]
axes[1].plot(x,diff)
axes[1].hlines(0, x.min(), x.max(), color = 'k')


#Удаление тренда с помощью скользящего среднего
df['MoveAvarage']=df['value'].rolling(window=12).mean()
axes[0].plot(df.index, df.MoveAvarage)
diff=df.value-df.MoveAvarage
axes[1].plot(x,diff)
axes[1].hlines(0, x.min(), x.max(), color = 'k')


# Экспонециальное сглаживание
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
df['ExpSmooth']=exponential_smoothing(df.value, 0.3)
axes[0].plot(df.index,df.ExpSmooth)
diff=df.value-df.ExpSmooth
axes[1].plot(x,diff)
axes[1].hlines(0, x.min(), x.max(), color = 'c')


df['ExpSmooth2'] =df.value.ewm(alpha=0.3).mean()
axes[0].plot(df['ExpSmooth2'], color='r')





df1 = pd.read_csv('Data/wwwusage.csv',names=['value'], header=0)
#df1 = pd.read_csv('austa.csv',names=['value'], header=0)
fig, ax = plt.subplots(3,2)
ax[0,0].plot(df1.value)
ax[1,0].plot(df1.value.diff())            # 1st Differencing
ax[2,0].plot(df1.value.diff().diff())      # 2st Differencing
sm.graphics.tsa.plot_pacf(df1.value.dropna(), ax=ax[0,1])
sm.graphics.tsa.plot_pacf(df1.value.diff().dropna(), ax=ax[1,1])
sm.graphics.tsa.plot_pacf(df1.value.diff().diff().dropna(), ax=ax[2,1])



#Строим модель ARIMA
model = ARIMA(df1.value, order=(1,1,2))
model_fit = model.fit(disp=0)   
print(model_fit.summary())
fig=model_fit.plot_predict()

#Сравним с скользящим среднем
axs=fig.gca()
df1['MoveAvarage']=df1['value'].rolling(window=3).mean()
axs.plot(df1.index,df1.MoveAvarage,axes=axs)


#График солнечной активности
dta = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')
dta.plot()
fig, ax = plt.subplots(3,2)
ax[0,0].plot(dta.SUNACTIVITY)
ax[1,0].plot(dta.SUNACTIVITY.diff())            # 1st Differencing
ax[2,0].plot(dta.SUNACTIVITY.diff().diff())      # 2st Differencing
sm.graphics.tsa.plot_pacf(dta.SUNACTIVITY.dropna(), ax=ax[0,1])
sm.graphics.tsa.plot_pacf(dta.SUNACTIVITY.diff().dropna(), ax=ax[1,1])
sm.graphics.tsa.plot_pacf(dta.SUNACTIVITY.diff().diff().dropna(), ax=ax[2,1])

res = sm.tsa.ARMA(dta, (3, 0)).fit()
fig, ax = plt.subplots()
ax = dta.loc['1950':].plot(ax=ax)
fig = res.plot_predict('1990', '2012', dynamic=True, ax=ax,
                        plot_insample=False)




df1 = pd.read_csv('Data/wwwusage.csv',names=['value'], header=0)
plt.plot(df1.value)

train = df1.value[:85]
test = df1.value[85:]
model = ARIMA(train, order=(1,1,1))  
fitted = model.fit(disp=-1)  

fc, se, conf = fitted.forecast(15, alpha=0.05)  

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)


# Модель с другими параметрами
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  

fc, se, conf = fitted.forecast(15, alpha=0.05)  

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)




