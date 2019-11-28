class A():
    count=0

    def __init__(self):
        A.count+=1        
    
    def exclime(self):
        print("I am an A")

    # cls = self
    # self = cls
    @classmethod
    def kids(cls):
        print("A bla bla bla",cls.count,"little object")


easy_a=A()
greasy_a=A()
wheezy_a=A()
wheezy_a.count=10
A.kids()
easy_a.kids()
print(easy_a.count)
print(wheezy_a.count)
print(wheezy_a.kids())

