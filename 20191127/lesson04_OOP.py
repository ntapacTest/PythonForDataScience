class classA():
    pass

a=classA()
b=classA()

a.arg=1
b.field="AAA"

print(a.arg)
print(b.field)

class classB:
    # Конструктор
    def __init__(self):
        pass
    def __del__(self):
        pass
 
    field1="fff"
    def sayHello(self, name):
        print("Hello %s"%(name))  
    def sayAll(self):
        print(self.getHelloAllText())
    def getHelloText(self,name):
        return "Hello "+name
    def getHelloAllText(self):
        return "Hello all"


b=classB()
#b.__del__()
classB.__del__(b)

b.sayHello("name")
b.sayAll()

print(b.getHelloText("name"))
print(b.getHelloAllText())
print(b.field1)

# Второй способ вызова метода
classB.sayHello(b,"ggg")
classB.sayAll(b)

print(b.field1)
b.field1=list('fffffff')
print(b.field1)

class classC():
    def __init__(self,name):
        self.name=name


c=classC("NAME")
print(c.name)


# Геттеры и сеттеры v1
class classCar():
    def __init__(self,name,color,year):
        self.name=name
        self.color=color
        self.year=year

    def get_name(self):
        return self.name
    def set_name(self,name):
        self.name=name

    def get_color(self):
        return self.color
    def set_color(self,color):
        self.color=color

    def get_year(self):
        return self.year
    def set_year(self,year):
        self.year=year

    Name=property(get_name,set_name,)
    Color=property(get_color,set_color)
    Year=property(get_year,set_year)



car=classCar("Tesla","red",2018)

car.Color="green"

print(car.Color)

# Если нужно закрыть какуюто переменную внутри класа ее имя нужно начать с __
# Геттеры и сеттеры v2, с помощью декораторов
class classCarV2():
    def __init__(self,name,color,year):
        self.__name=name
        self.color=color
        self.year=year
    @property 
    def Name(self):
        return self.__name
    @Name.setter
    def Name(self,name):
        self.__name=name

    @property 
    def Color(self):
        return self.color
    @Color.setter
    def Color(self,color):
        self.color=color

    @property 
    def Year(self):
        return self.year
    @Year.setter
    def Year(self,year):
        self.year=year


car2=classCarV2("BMW","black",2000)

print(car2.Color)
car2.Color="white"
print(car2.Color)

print(car2.Name)
car2._classCarV2__name="RRR"
print(car2.Name)


