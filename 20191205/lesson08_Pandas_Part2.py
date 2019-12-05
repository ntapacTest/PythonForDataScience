import pandas as pd
import numpy as np

df1=pd.DataFrame(np.random.rand(5,5),columns=['col1','col2','col3','col4','col5'])
print(df1)
# Вырезать 2 столбца (col2, col4) из датафрейма
cc=df1[['col2','col4']]

# Вырезать 2 столбца (col2, col4) из датафрейма и первые 3 строчки
cc=df1[['col2','col4']][:3]

# Вырезать столбец col3 из датафрейма
c1=df1.col3

# Вырезать строки по значению индекса
ss=df1.loc[2]

# Выбор строк в которых значения элементов в столбце col2>0.5
ss=df1.loc[df1['col2']>0.5]

filter1=df1['col2']<0.7

filter2=df1['col2']>0.3

ss=df1.loc[filter1&filter2]
print(ss)

# Выбор строк по номеру строки (по индексу)
df1.iloc[1:3]
df1.iloc[-1] # последняя строка
df1.iloc[-2:] # последние 2 строки

# df1.ix по строке и столбце, но лучше не использовать

df1=pd.DataFrame(np.random.rand(5,5),columns=['col1','col2','col3','col4','col5'])

# Удаление столбцов 
# inplace True - удалять в исходном датафрейме
# inplace False - удалять в исходном датафрейме (default)
# axis=0 - строки (default) 
# axis=1 - столбцы
df2=df1.drop(['col2','col4'],inplace=False, axis=1)
print(df1)
print(df2)

df1.drop(['col2','col4'],inplace=True, axis=1)


# Удаление строк по значению индекса
# inplace True - удалять в исходном датафрейме
# inplace False - удалять в исходном датафрейме (default) 
# axis=0 - строки (default) 
# axis=1 - столбцы
df2=df1.drop([2,4],inplace=False, axis=0)

# Информация о данных для тех столбцов где есть количественные данные
df1.describe(include='all')

zoo=pd.read_csv('Data/zoo.csv',sep=',',index_col='uniq_id')
zoo.describe(include='all')

# min max

df1['col3'].max() # по одному столбцу

df1.max() # по всем столбцам

# df1.stack.max # по всему датафрейму не работает

df1.max(axis=1) # по строкам

# count
df1.count() # по столбцам

df1.count(axis=1) # по строкам

# first last
df1.first()
df1.last()

# mean median std var
zoo.mean()
zoo.median() # по столбцам
zoo.median(axis=1) # по строкам

# по столбцу water_need
zoo.water_need.std()
zoo['water_need'].std()

# количество появлений каждого уникального значения в столбце animal 
# сумма в группах
# сортировка по уменьшению
zoo.animal.value_counts()
c1=zoo['animal'].value_counts()

c1[:3].plot(kind='bar')

vixcls=pd.read_csv('Data/VIXCLS.csv',sep=',',index_col='DATE')
vixcls.info()

sunspots=pd.read_csv('Data/monthly-sunspots.csv',sep=',',index_col='Month')
sunspots.info()

cars=pd.read_csv('Data/car_data.csv',sep=',')
cars.info()
df1.isnull()

# выбрать данные где есть null
df1.isnull()

# выбрать данные не null
df1.notnull()

# удалить все строки где есть null
df1['col1'].dropna(inplace=True)
df1.dropna(inplace=True)

# заполнение null данными 
df1['col1'].fillna(df1['col1'].mean(),inplace=True)

# проверка на уникальность
df1.col1.is_unique

# групировка значений
zoo.groupby('animal').mean() # по всем столбцам
zoo.groupby('animal').mean()[['water_need']] # по одному столбцу
zoo.groupby('animal').water_need.median() # то же, но результат в виде Series

# по нескольким полям
zoo.groupby(['animal','water_need'])

# к результатам групировки можно применить функцию
def func1(x):
    print(x)
    return x['water_need']*100

zoo.groupby('animal').apply(func1)

# объединение датафреймов
df1=pd.DataFrame(np.random.rand(5,5),columns=['col1','col2','col3','col4','col5'])
df2=pd.DataFrame(np.random.rand(5,5),columns=['col1','col2','col3','col4','col5'])

combo=pd.concat([df1,df2], axis=0) # Добавить снизу
combo=pd.concat([df1,df2], axis=1) # Добавить справа
print(combo)

# Добавить снизу, с проверкой уникальности индексов, 
# при совпадении выброс исключения
combo2=pd.concat([df1,df2], axis=0, verify_integrity=True) 

# Добавить снизу, с генерацией новых индексов при повторении
combo2=pd.concat([df1,df2], axis=0, ignore_index=True) 
print(combo2)

# добавление данных из подчиненных таблиц
zoo=pd.read_csv('Data/zoo.csv',sep=',',index_col='uniq_id')
zoo_eat=pd.DataFrame([['elefant','vegetables'],['tiger','meat'],['zebra','grass'],['kangaroo','fruits'],['giraffe','leaves']],columns=['animal','food'])

# объединение фреймов значения (строки) которых не в обоих фреймах исключаются
# по общему столбцу, если в обеих есть одинаковый столбец
zoo.merge(zoo_eat)
zoo.merge(zoo_eat, on='animal')
zoo_eat.merge(zoo)
df_result=pd.merge(zoo_eat,zoo)

# объединение фреймов значения (строки) которых не в обоих фреймах исключаются
# по общему столбцам left_on='animal' и right_on='animal'
zoo.merge(zoo_eat, left_on='animal', right_on='animal')

# С указанием как 
# inner - default (пересечение)
# outer - объеденение
# left - дополнение значениями слева
# right- дополнение значениями справа
zoo.merge(zoo_eat, how='inner')
zoo.merge(zoo_eat, how='outer')
zoo.merge(zoo_eat, how='left')
zoo.merge(zoo_eat, how='right')

# Слияние по индексам нужны правильные индексы
# zoo.join(zoo_eat)


# Изменение типа столбца
zoo['water_need'].astype(int)

# Изменение значений на названые (кластерные)
def change(x):   
    if x>400:
        y= 'C'
    elif x<200:
        y= 'A'
    else:
        y= 'B'
    return y

zoo['water_need']=zoo['water_need'].apply(lambda x:change(x))

# Добавление вычисленного столбца
zoo['water_need_str']=zoo['water_need'].apply(lambda x:change(x))

# Сортировка по значению в колонке
df1=pd.DataFrame(np.random.rand(10,5),columns=['col1','col2','col3','col4','col5'])

df1.sort_values('col2')
df1.sort_values(by='col2')

df1.sort_values('col2',ascending=False)

# Переустановка индексов после сортировки, 
# чтобы значения индексов соотв значению номера строки
df1.sort_values('col2',ascending=False).reset_index() # добавление нового
df1.sort_values('col2',ascending=False).reset_index(drop=True) # добавление нового и удаление старого


# Работа со временем


