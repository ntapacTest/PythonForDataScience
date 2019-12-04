import numpy as np
from numpy import random as rd

# Перемножение матриц по правилам линейной алгебры

a=np.array([[0,1],[2,3]])
b=np.array([2,3])
c=np.array([[2,3],[1,1]])

print(np.dot(a,b))
print(np.dot(b,a))

print(np.dot(a,c))
print(np.dot(c,a))

# Генераторы псевдослучайных чисел
# Не повторяемый результат
rd.seed()
print(rd.randint(0,100))

# Повторяемый результат
rd.seed(10)
print(rd.randint(0,100))
print(rd.randint(0,100))
print(rd.randint(0,100))
rd.seed(10)
print(rd.randint(0,100))
print(rd.randint(0,100))
print(rd.randint(0,100))
rd.seed(10)
print(rd.randint(0,100))
print(rd.randint(0,100))
print(rd.randint(0,100))

# Получение данных состоянии о генератора
rd.seed(10)
print(rd.randint(0,100))
state=rd.get_state()

print(rd.randint(0,100))
print(rd.randint(0,100))
print(rd.randint(0,100))

print(state)

# Установка состояния генератора, 
# для продолжения генерации с сохраненного состояния
rd.set_state(state)
print(rd.randint(0,100))
print(rd.randint(0,100))
print(rd.randint(0,100))

# Генерация n (10) элементов 
# равномерно распределенных от 0 до 1
print(rd.random(10))

# Генерация n (1000) элементов 
# равномерно распределенных от 0 до 100
r=rd.randint(0,100,100000)
print(r)
print(np.mean(r))
print(np.median(r))

# Генерация двумерного массива 10*20 элементов 
# равномерно распределенных от 0 до 1
print(rd.randn(10,20))

# Нормальный закон распределения 
# mean=median=moda

# Генерация массива значений 
# по закону нормального распределения с заданым средним (m) и 
# среднеквадратичным отклонением (c) количеством s
# rd.normal(m,c,s)
rAdv=rd.normal(1.5,4,10000)
print(rAdv)
print(np.mean(rAdv))
print(np.std(rAdv))


# Перемешать элементы
# работает по ссылке
rd.shuffle(rAdv)
print(rAdv)

# Выбор size случайных элементов из массива
# replace=True - с повторением, допускается повторный выбор элементов 
# replace=False - без повторения, не допускается повторный выбор элементов 
# p - массив вероятностей выпадания конкретного элемента массива, 
# длинна должна быть равна длинне исходного массива, 
# сумма всех элементов не может превышать 1
r=np.array([0,1,2,3,4,5,6,7,8,9])
print(r)
r2=rd.choice(r,size=5, replace=True, p=None)
r3=rd.choice(r,size=5, replace=False, p=None)
r4=rd.choice(r,size=5, replace=True, p=[0.7,0,0,0,0,0,0,0,0.2,0.1])
print(r2)
print(r3)
print(r4)


