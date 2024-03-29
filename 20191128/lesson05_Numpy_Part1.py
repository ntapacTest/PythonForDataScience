import numpy as np

# Двумерный массив
x=np.array([[1,2,3],[4,5,6],[7,8,9]])

print(x)

lst=[[1,2,3],[4,'5',6],[7,8,9]]

x1=np.array(lst,float)

x2=np.array(lst,str)

print(x2)

x3=np.array(lst)

print(x[1,2])

# Установить значение
# x[строки,столбцы]
x[0,2]=50

print(x)

# Вывод нулевого ряда
print(x[0,:])

# Вывод 1-го столбца
print(x[:,1])

# Вывод подмассива от нулевого до указанного не включая его
print(x[:2,:2])

# Получение одномерного массива из первой строки
arr=x[:1]

# Размерность
print(x.shape)
print(np.shape(x))

# Тип переменных
print(x.dtype)

# Количество измерений массива
print(x.ndim)

# Количество элементов
print(x.size)

# Длинна первого измерения
print(len(x))

# Наличие элемента в массиве
print(7 in x)
print(77 in x)

# Создание массива 5х2 типа int и заполнить нулями
a=np.zeros((5,2),int)
a=np.zeros(shape=(5,2),dtype=int)

# Создание массива 5х2 типа int и заполнить еденицами
b=np.ones((5,2),int)
b=np.ones(shape=(5,2),dtype=int)

# Создание массива 5х2 типа int 
# из памяти не очищая ее перед созданием 
# остается мусор
c=np.empty((5,2),float)
c=np.empty((5,2),int)

# Создание массива 5х2 типа int и заполнить семерками
d=np.full((5,2),7,float)

# Создание квадратной матрицы размерности 3 
# и заполнение диагонали еденицами, остальное нули
e=np.eye(3)

# # Создание квадратной матрицы и заполнение ее диагонали 
# значениями из списка, остальное нули
f=np.diag([4,3,2,1])

# Создание вектора начиная с start до stop (не включая stop) 
# и заполнение его последовательными значениями с шагом step
g=np.arange(start=10, stop=50,step=1)
g=np.arange(0.2, 3.2,0.2)

# Создание вектора начиная с start до stop (не включая stop) 
# количеством 21 элементов (точек), 
# соотв. 20 равных частей
np.linspace(1.2,5.7,21)
np.linspace(0,100,21, dtype=int)

# Заполнение массива значениями функции от индексов
def fff(x, y):
    return x**2+y**2

np.fromfunction(fff,(2,2))

x=np.linspace(1.2,5.7,21)
# Взять последний элемент
x[-1]

# Перевернуть одномерный массив
x[::-1]

