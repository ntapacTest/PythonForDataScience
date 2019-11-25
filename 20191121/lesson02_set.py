my_set={1,2,3,4,5,5}

# Добавление
my_set.add(6)
# Если значение существует, то оно не добавляется
my_set.add(6)

# Удаление
my_set.remove(6)

# Проверка существования в сете
print(3 in my_set)
print(30 in my_set)

set1={1,2,3,4,5}
set2={1,3,5,6,7,8}

# Объеденение сетов
set3=set1.union(set2)

# Пересечение сетов
set4=set1.intersection(set2)

# Вичитание элементов которые входят в сет 1 и 2, остаются только уникальные
set5=set1.difference(set2)

# Проверка входит ли сет 2 в сет 1 как сабсет
result=set.issubset(set2)




