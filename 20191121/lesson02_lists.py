# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:45:16 2019

@author: user
"""

# Лист из 4-х єлементов a b c d
lst=list("abcd")
print(lst)

# Лист из 1 єлемента abcd
lst=list(["absd"])
print(lst)

# Лист из 1 єлемента abcd
lst=["absd"]
print(lst)

# Лист из 1 єлемента abcd
lst=["abcd"]
print(lst)


lst=[x**2 for x in range(10)]
print(lst)

lst=[x**2 for x in range(10) if x%2==1]
print(lst)

# Пустой список
lst=[]
print(lst)

lst=list("abcd")
print(lst[3])

# Срез от i до j с шагом d lst[i:j:d]
lst=[1,2,3,4,5,6,7,8,9,10]
print(lst[2:8:2])

# n с конца элемент lst[-n::] lst[-n:] 
print(lst[-1::])
print(lst[-1:])
print(lst[-1])

print(lst[0:])

# Каждый n элемент списка lst[::n]
print(lst[::3])

# Добавить элементы в начало списка
lst[0:0]=[-3,-2,-1]
print(lst)

# Начиная с n до предпоследнего lst[n:-1]
print(lst[2:-1])



lst[1:3]=[99,88]
print(lst)

# Те что можно заменяются лишние вставляются как новые, сдвигая остальные в право
lst[1:3]=[99,88,77]
print(lst)

lst[1:1]=[99,88,77]
print(lst)

# Количество елементов первого уровня
print(len(lst))

# Проверка вхождения значения в список x in lst
print(3 in lst)
print(13 in lst)

# Проверка не вхождения значения в список x not in lst
print(3 not in lst)
print(13 not in lst)

# Максимальное и минимальное значение
print(max(lst))
print(min(lst))

lstStr=['d','f','w','f']
print(max(lstStr))
print(min(lstStr))

# Удаление элементов списка по срезу
del(lst[2:6:2])
print(lst)

# Добавление элементов в список
lst.append(88)
print(lst)

# Расширение списка другим списком
lst.extend([55,66,77])
print(lst)

# Количество елементов в списке равных x lst.count(x)
lst.count(77)

# Наименьший индекс элемента равный x lst.index(x)
lst.index(55)

lst=[1,2,3,4,3,2,1,5,6,7,2,3]
lst.index(2,2)

# Удалить первый найденый элемент списка равный lst.remove(x)
lst.remove(5)
print(lst)

# Вставка элементов в список
lst=[1,2,3,4,5,6,7,8,9,10]
lst.insert(7,88)
print(lst)

# Получение и удаление элемента из списка по индексу
x=lst.pop(7)
print(x,lst)

# Поменять местами в обратном порядке 
lst.reverse()
print(lst)

# Сортировка
lst.sort()
print(lst)

zipObject=zip([1,2,3,4,5],[6,7,8,9])
print(zipObject)
print(list(zipObject))

# Присвоение массивов происходит по ссылке
lst1=[10,20,30,40,50,60]
lst2=lst1
lst2[2]=100

# Сравнение массивов 
# L==M - сравнение по значениям, 
# L is M сравнение ссылок на объекты

lst3=[1,2,3,4]
lst4=[1,2,3,4]
lst5=lst3

# Копия списка (только для одномерных списков)
lst6=lst3[:]
lst7=list(lst3)
lst8=lst3.copy()

# Копия списка, включая многомерные и списки объектов
import copy
lst9= copy.deepcopy(lst3)

print(lst3==lst4)
print(lst3==lst5)
print(lst3==lst6)
print(lst3==lst7)
print(lst3==lst8)
print(lst3==lst9)
print(lst3 is lst4)
print(lst3 is lst5)
print(lst3 is lst6)
print(lst3 is lst7)
print(lst3 is lst8)
print(lst3 is lst9)

import math as m
# К каждому элементу листа lst применяет функцию func
# map(func,lst)

lst=list(map(m.sin,[0,m.pi/6,m.pi/3,m.pi/2,m.pi]))

# Генерация списков (комбинация всех букв в первом слове со всеми буквами во втором слове)
comb=[a+b for a in "Hello" for b in "all"]

comb1=[letters[1] for letters in  [a+b for a in "Hello" for b in "all"]]

# Нахождение общих для двух списков элементов
a="Hello"
b="all"
result=[i for i in a if i in b]

