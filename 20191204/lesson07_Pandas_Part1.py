import pandas as pd

# Вызов информации о версии, 
# должно быть во всех правильных модулях
print(pd.__version__)

# Чтение из файла
# zoo=pd.read_csv('Data/zoo.csv',sep=',',index_col='uniq_id',na_values=['NO CLUE','tiger'])
zoo=pd.read_csv('Data/zoo.csv',sep=',',index_col='uniq_id')
# na_values список запрещенных значений которые будут заменены на NaN
vixcls=pd.read_csv('Data/VIXCLS.csv',sep=',',index_col='DATE')
sunspots=pd.read_csv('Data/monthly-sunspots.csv',sep=',',index_col='Month')
zoo.head()
vixcls.head()
sunspots.head()

# Запись в файл
zoo.to_csv('Data/zoo_new.csv')
zoo.to_json('Data/zoo_new.json')

# Основные объекты pandas
# series, dataframe, index, datatime index


# Series - одномерный индексируемый массив
data=pd.Series(data=[0,1,2,3,4,5,6,7,8,9])

print(data.index)
print(data.values)

print(data[1:3])

# Отдельно созданный индекс
data=pd.Series(data=[0,1,2,3,4,5,6,7,8,9], index=['a','b','c','d','e','g','h','j','k','l'])
print(data.index)
print(data.values)
print(data['a':'d'])

# Повторяющийся индекс
data=pd.Series(data=[0,1,2,3,4,5,6,7,8,9], index=['a','b','c','a','e','a','h','j','k','l'])
print(data.index)
print(data.values)
print(data['a'])

# Фильтр по условиям в индексе
print(data[data.index<'f'])

# Индексов бошльше чем значений 
# значения добавляются
data=pd.Series(data=5, index=['a','b','c','a','e','a','h','j','k','l'])
print(data.index)
print(data.values)

data=pd.Series(data='val', index=['a','b','c','a','e','a','h','j','k','l'])
print(data.index)
print(data.values)

# Можно создавать из словаря
data=pd.Series({1:'a',2:'b',3:'c','d':4})
print(data.index)
print(data.values)

# Сложение 
data1=pd.Series({1:'a',2:'b',3:'c','d':4})
data2=pd.Series({1:'a',2:'b',3:'c','d':4})
data3=pd.Series({1:'a',4:'b',3:'c','e':4})
data4=data1+data2
print(data4)
data5=data1+data3
print(data5)
data6=data1.add(data2)
print(data6)


