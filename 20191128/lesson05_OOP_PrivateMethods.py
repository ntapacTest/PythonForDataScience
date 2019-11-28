class Person():
    
    def __init__(self,name, age):
        self.name=name
        self.age=age

    # Типа "приватный" метод, только по соглашению имен
    def _privateMethod(self):
        print("I am private method", self.name,self.age)


person=Person("Name",100)

person._privateMethod()