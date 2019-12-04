import pandas as pd
import numpy as np

data1=pd.Series(data=[0,1,2,3,4,5,6,7,8,9])
data2=pd.Series(data=[9,8,7,6,5,4,3,2,1,0])

df1=pd.DataFrame({"col1":data1,"col2":data2})

df1.head()

d3=[{'a':i,'b':2*i}for i in range(10)]
print(d3)
df3=pd.DataFrame(d3)
df3.head()

df4=pd.DataFrame(np.random.rand(3,2),columns=['col1','col2'],index=['a','b','c'])
df4.head()

df5=pd.DataFrame(np.random.rand(3,2),columns=['col1','col2'])
df5.head()

# Операции с датафреймами

df1=pd.DataFrame(np.random.rand(3,2),columns=['col1','col2'])
df2=pd.DataFrame(np.random.rand(3,2),columns=['col1','col2'])

print(df1.add(df2))
print(df1.sub(df2))
print(df1.mul(df2))
print(df1.div(df2))
print(df1.floordiv(df2))    # //
print(df1.mod(df2))         # %
print(df1.pow(df2))         # возведение квадрат

# Среднее построчно
fill=df1.mean()

# Среднее по всему датафрейму
fill=df1.stack().mean()

# Заполнение NaN значением fill
print(df1.add(df2,fill_value=fill))

# Информация о всей матрице относительно одной строки, 
# по столбцам (относительно 0 элемента),
# каждый элемент делится на нулевой
df1/df1.iloc[0]

# То же самое но относительно нулевого столбца
df1.div(df1['col1'],axis=0)

# Информация о датафрейме
vixcls=pd.read_csv('Data/VIXCLS.csv',sep=',',index_col='DATE')

# Получение информации о значениях
df1.describe()
vixcls.describe()

# Получение общей информации о датафрейме
df1.info()
vixcls.info()

# Количество уникальных значений в каждом из столбцов
df1.nunique()
vixcls.info()

# Просмотр первые 5 строк, как аргумент можно передать количество
vixcls.head()

# Просмотреть случайную строку, как аргумент можно передать количество
vixcls.sample()

# Просмотр последних 5 строк, как аргумент можно передать количество
vixcls.tail()

# График, ось x - индексы
df1.plot()

zoo=pd.read_csv('Data/zoo.csv',sep=',',index_col='uniq_id')

zoo.plot()
