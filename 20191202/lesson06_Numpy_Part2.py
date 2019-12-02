import numpy as np

print('02.12.2019')

z=np.arange(start=0, stop=6,step=1)

# Переупорядочивает элемены списка в другой размерности,
# при этом не создавая нового списка, 
# работа ведется со ссылкой на исходный, 
# если значение задано -1, то другие 
# рассчитываются для создания кратной величины
z1=z.reshape(2,3)
print(z1)

z2=z.reshape(-1,2)
print(z2)

z3=z.reshape(2,-1)
print(z3)

# Создание НОВОГО списка с новой размерностью, 
# на основании данных из исходного списка
z4=np.resize(z,(3,2))
print(z4)

z5=np.resize(z,(3,-1))
print(z5)

# array to list
ls=z.tolist()
print(ls)

ls2=z4.tolist()
print(ls2)

# Транспонирование массива (оборот по диагонали)
arr=np.arange(start=0, stop=9,step=1)
arr2=np.resize(arr,(3,3))
arrTr=arr.transpose()
print(arrTr)

# Превращение многомерного массива в одномерный
arr=np.arange(start=0, stop=9,step=1)
arr2=np.resize(arr,(3,3))

# Новый объект
arr3=arr2.flatten()
# По ссылке
arr4=arr2.ravel()


# Конкатенация массивов с одинаковой размерности
arr1=np.array([1,2])
arr2=np.array([3,4,5,6])
arr3=np.array([7,8,9])

arr4=np.concatenate((arr1,arr2,arr3))

arr1=np.array([[1,2],[3,4]])
arr2=np.array([[5,6],[7,8]])

# Аргумент axis указывает по какой оси производится конкатенация
arr3=np.concatenate((arr1,arr2))
arr3=np.concatenate((arr1,arr2),axis=0)
arr4=np.concatenate((arr1,arr2),axis=1)

# Добавление размерности (измерения) к массиву 
z=np.arange(start=0, stop=6,step=1)
z2=z[:,np.newaxis]
print(z2)
print(z2.shape)
arr1=np.array([[1,2],[3,4]])
arr2=arr1[:,np.newaxis]
print(arr2)
print(arr2.shape)

# Математические операции над одноразмерными массивами
# поэлементно
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[10,11,12]])

arr3=arr1+arr2
arr3_2=np.add(arr1,arr2)
print(arr3)
print(arr3_2)

arr4=arr1-arr2
arr4_2=np.subtract(arr1,arr2)
print(arr4)
print(arr4_2)

arr5=arr1*arr2
arr5_2=np.multiply(arr1,arr2)
print(arr5)
print(arr5_2)

arr6=arr1/arr2
arr6_2=np.divide(arr1,arr2)
print(arr6)
print(arr6_2)

arr7=arr1%arr2
print(arr7)

arr8=arr1//arr2
print(arr8)

# Если нехватает размерности, но в размерности хватает элементов, 
# то размерность разширяется копированием существующей
a=np.zeros([2,2])
b=np.array([-1,3])

print(a)
print(b)

c=a+b
print(c)

# Добавление размерностей перед операцией
b1=b[np.newaxis,:]
print(b1)

b2=b[:,np.newaxis]
print(b2)

c1=a+b1
print(c1)

c2=a+b2
print(c2)

# Если нехватает элементов при совпадающей размерности то будет ошибка
d=np.array([1,2,3])
e=np.array([4,5])
f=d+e

# Перебор элементов массива
a=np.array([1,2,3,4,5,6,7,8])

for x in a:
    print(x)

a=np.array([[1,2,3],[4,5,6]])
b=np.array([[1,2],[3,4],[5,6]])

# Проход по первой оси, по вертикали
for x in a:
    print(x)

# Перебор всех элементов
for (x,y) in b:
    print(x,y)

for (x,y,z) in a:
    print(x, y,z)

for x in a.flat:
    print(x)

for x in b.flat:
    print(x)

# Действия над массивом (свойства массива)

# сумма
print(a.sum())
print(b.sum())

# Min
print(a.min())

# Max
print(a.max())

import time

arr=np.random.random(10000000)
start_time=time.perf_counter()
s=0
for itm in arr:s+=itm

t1=time.perf_counter()-start_time
print(s,t1)

start_time=time.perf_counter()
s=arr.sum()

t1=time.perf_counter()-start_time
print(s,t1)

# Разница между мин и мах (размах)
print(arr.ptp())

# Среднее
print(arr.mean())

# Дисперсия
print(arr.var())

# Стандартное отклонение
print(arr.std())

# Медиана
print(np.median(arr))

# Кореляция
a=np.array([[1,2,3],[1,5,6]])
print(np.corrcoef(a))

# Индекс мин и мак элемента одномерного массива
arr=np.random.random(10000)
print(arr.argmin())
print(arr.argmax())


# Среднее значение по столбцу
print(a.mean(axis=0))

# Среднее значение по строке
print(a.mean(axis=1))

# Дисперсии и отклонения так же

# Сортировка массива
arr=np.array([2,1,3,4,5,2,6,2])
print(arr)
# print(arr.sort())
print(np.sort(arr))

# Получение массива уникальных значений
print(np.unique(arr))

# Обрезка данных до диапазона, 
# числа которые выходят заменяются на мин и макс соотв.
print(arr.clip(0,4))


# Поэлементное сравнение массивов одной размерности
a=np.array([[1,2,3],[5,6,7]])
b=np.array([[3,4,6],[8,2,5]])
print(a<b)

# Поэлементное сравнение массива с числом
print(a>2)

# Получение индексов значений удовлетворяющих условие
print(np.where(a>2))

# Проверка удовлетворяет ли условие хоть один елемент массива
print(np.any(a>2))

# Проверка удовлетворяет ли условие все елементы массива
print(np.all(a>2))

# Компоновка условий (logical_and, logical_or, logical_not)
b=np.logical_and(a>2,a<5)

# Отсутсвующее значение NaN
# Проверка на присутствие Nan, false если NaN отсутствует
b=np.array([1,np.NaN,3,4,5])
b[0]=np.NaN
print(np.isnan(a))
print(np.isnan(b))

c=np.array([[1,np.NaN,3,4,5],[3,5,6,7,np.nan]])
c[0,0]=np.NaN
print(np.isnan(c))
print(np.isnan(c))

