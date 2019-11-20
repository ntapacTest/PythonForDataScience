import keyword
import math

print(keyword.kwlist)

x=int(input('Enter the number:'))

if x%2==0:
    print('Четное')
else:
    print('Нечетное')

n=0

while n<50:
    if n%10==0:
        print(n)
    n+=1
else:
    print('n = '+str(n))



while True:
    i=int(input('Enter the number: '))
    
    if i<0:
        break
    if i>20:
        continue
    if 0<i and i<=10:
        print(i**2)
    else:
        print(i)

i=0

summ=0
for i in [1,2,3,4]:
    summ+=i

print(summ)

for str in 'hello':
    print(str)
   
# range(a,b,c) 
# a - начиная с, по умолчанию 0
# b - последнее значение, обязательный параметр
# c - шаг, по умолчанию 1
i=0

summ=0
for i in range(1,10,1):
    summ+=i
    print(i)

print(summ)

i=1
summ=0
for i in range(10):
    summ+=i
    print(i)

print(summ)

i=1
summ=0
for i in range(3,10):
    summ+=i
    print(i)

print(summ)

i=0
for i in range(1,11):print(i)

i=0
lst=[3,6,8,2,0,2,6,4,7,1]

for i in range(len(lst)):print(lst[i])

i=0
lst=[3,6,8,2,0,2,6,4,7,1]

for i in range(len(lst)):
    lst[i]=lst[i]**2
print(lst)

# enumerate
for i,x in enumerate(lst):
    print(i+1,x)
    
# enumerate
for i,x in enumerate('hello world'):
    print(i,x)


# int
i=500000000000000000
i=i**2

print(i)


# float
print(math.pi)

p40=math.pi**40
print(p40)


print(round(math.pi,2))

print(round(2.5))
print(round(3.5))
print(round(4.5))
print(round(5.5))
print(round(6.5))
print(round(7.5))

print(math.ceil(2.5))
print(math.floor(2.5))


# bool
for i in (False,True):
    for j in (False,True):
        print(i,j,i and j,i or j, not i)
        
for i in (False,True):
    for j in (False,True):
        print('i = '+str(i),',j = '+str(j),', and '+str(i and j), ', or '+str(i or j),', not '+ str(not i))
        
# string

# 'str'
# "str"
# '''str'''
# """str"""
        
str='''
str
'''
print(str)

print("""
      
hello

""")

str='aa'+'bb'
print(str)

str='ab'*10
print(str)

str='val %s %i'%('str',12)
print(str)

str="Text+\u03c0\u03a3\u2660\u263a\u222b"
print(str)
        
print(ord('A'))

print(chr(65))

print(ord('\u222b'))

print(chr(8747))

print('ABCD EFJ'.split())

print('ABCD,EFJ'.split(','))

print('_'.join(['a','b','c']))

print(' ABCD EFJ '.lstrip())
print(' ABCD EFJ '.rstrip())
print(' ABCD EFJ '.strip())

print('345'.isdigit())
print('345d'.isdigit())

print('ddddd'.isalpha())
print('345d'.isalpha())

name=input('Привет медвед, ты кто? ')
print('Привет медвед по имени %s'%(name))


# tuple
p=(1,4,'r',4.2,('rrr',43,2.43,['fff',43]))
print(p)
print(p[4])
