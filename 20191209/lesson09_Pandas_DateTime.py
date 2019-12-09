import pandas as pd
import numpy as np
import datetime as dt

date= pd.to_datetime("09/12/2019")
date2= pd.to_datetime("25/12/2019")

date3=date2+pd.to_timedelta(np.arange(12),'D')
date4=date2+dt.timedelta(days=-7)
date4=date2+dt.timedelta(weeks=2)

# Список дат от 1.1.2019 до 08.01.2019 с шагом час
date_range01=pd.date_range('1/1/2019',end='01/08/2019',freq='H')

# Список дат от 1.1.2019 с шагом час количеством шагов 20
date_range02=pd.date_range('1/1/2019',periods=20,freq='H')

# Создание датафрейма с одним столбцом на основании date_range01
df=pd.DataFrame(date_range01,columns=['date'])
print(df)

# Добавление столбца с данными
df['data2']=np.random.randint(0,100,size=len(date_range01))
print(df)

df.info()

# Установка индекса по дате и времени
df['datetime']=pd.to_datetime(df['date'])
df.set_index('datetime',inplace=True)
df.drop('date',axis=1,inplace=True)


# Конвертация списка строковых представлений дат в список дат
string_date=[str(x) for x in date_range01]

dates=pd.to_datetime(string_date)

dates2=pd.to_datetime(string_date,infer_datetime_format=True)

string_date_1=['June-01-2018','june-02-2018', 'jul-03-2018','jun-05-2018']
dates=pd.to_datetime(string_date,infer_datetime_format=True)

string_date=['June-01-2018','june-02-2018','july-05-2018']
time_stamp=[dt.datetime.strptime(x,'%B-%d-%Y') for x in string_date]

string_date_2='12/11/2019 09:15:03'
ts=pd.to_datetime(string_date_2,dayfirst=True)

string_date_2='2019/11/12 09:15:03'
ts=pd.to_datetime(string_date_2,dayfirst=True)

string_date_2='2019-11-12 09:15:03'
ts=pd.to_datetime(string_date_2,dayfirst=True)

string_date_2='12/11/2019 09:15:03'
ts=[dt.datetime.strptime(string_date_2,'%d/%m/%Y %H:%M:%S')]

df.head()

# Выборка данных за второе число каждого месяца
df[df.index.day==2]

df['2019-01-03'] # За один день

df['2019-01-03':'2019-01-05'] # За период


df.resample('D').mean() # Среднее по дням
df.resample('M').mean() # Среднее по месяцам
df.resample('6H').mean() # Среднее за каждые 6 часов

# Скользящее среднее, заменяет каждые 3 точки на стреднее
df['rolling_mean']=df.rolling(3).mean()

# Заполнение NaN значениями из следующего заполненного
df['rolling_mean_fill']=df['rolling_mean'].fillna(method='backfill')


# Unix (эпохальное) time, число секунд с 01.01.1970

ep=1234574124

ts=pd.to_datetime(ep,unit='s')