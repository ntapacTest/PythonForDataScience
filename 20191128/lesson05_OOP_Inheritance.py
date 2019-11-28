class AAA():
    def exclime(self):
        print("I am an AAA")
    def exclime2(self):
        print("I am an AAA2")

class BBB(AAA):
    def exclime(self): 
        super().exclime()       
        print("I am a BBB")
    def exclime3(self):        
        print("I am a BBB3")


a=AAA()
b=BBB()
a.exclime()
a.exclime2()
b.exclime()
b.exclime2()
b.exclime3()


