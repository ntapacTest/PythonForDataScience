def cost(hrn,kop=0):
    return ('%i,%i UAH'%(hrn, kop))

print(cost(15,45))

def emptyFunc():
    pass

emptyFunc()

def f(x):
    # Работа с глобальной, по отношению к функции, переменной
    global z
    z=33
    y=5
    return y

z=43
y=22
f(4)
print(y)

def f2(x):
    return 5*x


print(f2(4))

# Умножение списка
print(f2([1,2,3]))

def prod(x,y):
    return x*y
L=[6,3]

print(prod(*L))

print(type(prod))

fProd=prod
print(fProd(*L))

# Сохранения состояния объекта 'c' между вызовами функций
# func(c=[])
def funcStat(a,b,c=[]):
    c.append(a*b)
    return c

print(funcStat(1,1))
print(funcStat(2,2))
print(funcStat(3,3))
print(funcStat(4,4))

# Функция с неихвестным количеством параметров
def summa(*x):
    s=0
    print(x)
    print(type(x))
    for i in x: s+=i
    return s


print(summa(10,20,30))
