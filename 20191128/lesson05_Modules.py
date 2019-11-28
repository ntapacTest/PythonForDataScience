# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! При импортировании всего модуля (не отдельного     !!!
# !!! объекта или метода) он выполняется полностью       !!!
# !!! чтобы избежать этого в коде модуля нужно добавить  !!!
# !!! перед исполняемым кодом (не методами):             !!!
# !!! if __name__ == "__main__":                         !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Импорт модуля
# import moduleName
# import moduleName as mn

# Импорт объeкта или метода из модуля
# from moduleName import method
# import moduleName.subClass as sc

# Импорт всего из модуля
# from moduleName import *

#import my_module as mym

import my_module
# from my_module import MyMod

mm=my_module.MyMod()

mm.exclaim()

# Пути к модулям
#import sys 
#print(sys.path)
