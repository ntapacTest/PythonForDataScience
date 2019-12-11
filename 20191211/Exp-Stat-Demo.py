# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import statistics as sc
import math

data=np.random.randn(1000)
#Строим таблицу средствами Numpy
hist_dataN, hist_binsN = np.histogram(data,bins=75)
#Строим таблицу и строим гистограмму средствами Matplotlib 
hist_dataM=plt.hist(data,bins=75)
#Представим гистограмму средствами Pandas
pd.Series(data).plot.hist(bins=75)
#Представим диаграмму средствами bar-графиков
b = [i for i in range(len(hist_dataN))]
plt.bar(b,hist_dataN)
plt.xticks(b, np.around(hist_binsN[:-1],decimals=3),rotation=90)
plt.ylabel('Повторяемость')

#Строим аппроксимацию функции плотности распределения
pd.Series(data).plot.hist(bins=75,density=True)
pd.Series(data).plot.kde()


######### Использование NumPy 
#
# np.mean(data)
# np.median(data)
# np.var(data) - дисперсия генеральной совокупности
# np.std(data) - стандартное отклонение генеральной совокупности (корень от дисперсии)
# np.var(data,ddof=1) - дисперсия выборки
# np.std(data,ddof=1) стандартное отклонение выборки
#
######### Использование Statistics
#
# sс.mean(data)
# sс.median(data)
#       st.median_low()
#       st.median_high()
# sс.mode(data)
# sс.pvariance(data) - дисперсия генеральной совокупности
# sс.pstdev(data) - стандартное отклонение генеральной совокупности (the square root of the population variance)
# sс.variance(data) - дисперсия выборки
# sс.stdev(data)-стандартное отклонение выборки (the square root of the sample variance) 
#
######### Использование Scipy.Stats
# 
# ss.describe(data)[2] - mean
# ss.mode(data)
# ss.describe(data)[3] -стандартное отклонение генеральной совокупности
# ss.skew (data) - скос = sc.stats.describe(data)[4]
# ss.kurtosis (data) - эксцесс = sc.stats.describe(data)[5]
# ss.variation (data)



#Box-Plot
plt.boxplot(data)


#Средняя и матожидание
def mean(nums):
    return sum(nums) / len(nums)
mu=mean(data)

mu2=np.mean(data) 
mu3=data.mean()  
mu4=sc.mean(data) 
mu5=ss.describe(data)[2]

#В Pandas
df["weight"].mean()
df.groupby('Year').mean()
df.groupby('Year').mean().plot(kind='bar')


#Сравнение среднего и медианы
d1=np.array([31,34,35,36,43,46,46,47,52,54,56,58,59])
d2=np.array([31,34,35,36,43,46,46,47,52,54,56,58,93])
mu1=np.mean(d1) 
mu2=np.mean(d2)
md1=np.median(d1) 
md2=np.median(d2) 


#Создаем объект-распределение
norm_data=ss.norm()
norm_data=ss.norm(loc=13,scale=0.05)
mean, var, skew, kurtosis = norm_data.stats(moments='mvsk')
#Куммулятивная функция в точке
norm_data.cdf(3.2)
#Точка отсукающая заданный процент (для квартилей, например,0.25,0.5,0.75)
norm_data.ppf(0.908789)
#Границы симметричного двустороннего интервала с заданной веростностью.
norm_data.interval(0.999312)
#Нарисуем график
fig=plt.subplot(2,1,1)
sample_space = np.arange(12.5, 13.5, 0.001)
pdf = norm_data.pdf(sample_space)
plt.plot(sample_space, pdf)
fig=plt.subplot(2,1,2)
cdf = norm_data.cdf(sample_space)
plt.plot(sample_space, cdf)

#Генерируем набор
data1=norm_data.rvs(size=1000)
plt.plot(data1)
plt.plot(data1,'kd',linestyle='')


#Равномерное распределение
a=5
b=10
N=10000
data_u= y1=ss.uniform.rvs(loc=a,scale=10,size=N)
hist_data=plt.hist(data_u,bins=75,density=True)


#Проверка параметров закона распределения
mu=50
data=np.array([49.88,50.50,49.10,50.94,49.31,50.01,50.51,50.42,50.34,
               50.70,50.55,50.18,50.81,50.45,51.27,50.23,49.70,50.22,
               50.22,50.98,50.07,50.30,49.72,50.42,50.68
])
data.mean()
#"Ручной" метод
tfact1=(data.mean()-mu)/data.std(ddof=1)*math.sqrt(len(data))
#Используем Scipy
tfact2,pv=ss.ttest_1samp(data, mu)
ss.t.ppf(q=0.025,  # Quantile to check
            df=len(data)-1) 

#Cтроим доверительный интервал
sigma = data.std()/math.sqrt(len(data))
intval=ss.t.interval(0.95,df=len(data)-1,loc = data.mean(),scale= sigma)

#Двувыборочный тест на совпадение двух выборок
data1=np.array([27.68,26.25,26.62,28.69,28.11,
               28.73,26.76,30.94,30.84,24.28,
               27.02,30.06,24.74,27.74,26.88,
               26.34,29.16,26.30,27.91,30.37,
               25.14,26.66,26.64,27.35,28.08])
data2=np.array([23.88,29.11,25.40,28.44,24.18,
                32.60,26.27,26.45,23.07,27.58,
                28.15,27.63,24.75,28.00,24.82,
                28.62,23.33,29.52])
ss.ttest_ind(data1,data2,
             equal_var=False) # У выборок равные диспесрии??



#XИ-квадрат - тест

national = pd.DataFrame(["Украинцы"]*100000 + ["Русские"]*60000 +\
                        ["Евреи"]*50000 + ["Поляки"]*15000 + ["Остальные"]*35000)
minnesota = pd.DataFrame(["Украинцы"]*600 + ["Русские"]*300 + \
                         ["Евреи"]*250 +["Поляки"]*75 + ["Остальные"]*150)
national_table = pd.crosstab(index=national[0], columns="count")
minnesota_table = pd.crosstab(index=minnesota[0], columns="count")

print( "Вся старна")
print(national_table)
print(" ")
print( "Область Х")
print(minnesota_table)
observed = minnesota_table
national_ratios = national_table/len(national)  
expected = national_ratios * len(minnesota)   

crit = ss.chi2.ppf(q = 0.95,df = 4) 
chi_squared_stat = (((observed-expected)**2)/expected).sum()
#Примечание: нас интересует только правый хвост распределения хи-квадрат
p_value = 1 - ss.chi2.cdf(x=chi_squared_stat,  df=4)

#Можно автоматически выполнить критерий соответствия по
#критерию хи-квадрат, используя функцию scipy
ss.chisquare(f_obs= observed,   
                f_exp= expected)  


#Корреляция
n=[i for i in range(0,9)]
x=[500,790,870,1500,2300,5600,100,20,5]
y=[5.4, 4.2, 4, 3.4, 2.5, 1.0, 6.1, 8.2, 14.6]
plt.plot (n,x)
plt.show ()
plt.plot (n,y)
ss.pearsonr(x,y)
ss.t.ppf(q=0.05, df=len(x)-1) 