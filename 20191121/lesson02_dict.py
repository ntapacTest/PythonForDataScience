
D1={'name':'ivan','age':35,'sex':True}
print(D1)

print(D1['name'])

print('name' in D1)

# При добавлении, если ключа нет то создается
D1['address']='africa'

# Удаление
del(D1['address'])

# Объект списка ключей
print(D1.keys())

print(list(D1.keys()))

# Все значения
print(D1.values())
print(list(D1.values()))

# Все пары 
print(D1.items())
print(list(D1.items()))

for key,val in D1.items():
    print(key,val)
    
for key in D1.keys():
    print(D1[key])

for key in sorted(D1.keys()):
    print(key)

for key in sorted(D1):
    print(key)

# Получение и удаление элемента из списка по ключу key, 
# если ключ не найден то возвращается значение value
# pop(key, value)

age=D1.pop('age',-1)
age=D1.pop('AGE',-1)
age=D1.pop('AGE')

address=D1.pop('address','no address')

# Генерация словаря
words=['word', 'names', 'list']
length=[len(word) for word in words]
d1={w:l for(w,l) in zip(words,length)}

# или так
d2={word:len(word) for word in words}