
ex1=0
try:
    # x=int(input("x: "))
    # y=int(input("y: "))
    x="sdddssad"
    y=0
    res=x/y
    print(res)
except IOError:
    print("IOError")    
except ZeroDivisionError:
    print("ZeroDivisionError")
except KeyboardInterrupt:
    print("KeyboardInterrupt")
except Exception as ex:
    ex1
    print(ex)
    raise ex
else:
    print("Ok")
finally:
    print("Close")


print(ex1)
print(type(ex1))


# Создание своего исключения
class MyExcept(Exception):
    pass

try:
    raise MyExcept("Message")
except MyExcept as ex:
    print("Error", ex) 
    print(type(ex))






