# Чтение данных
# fileObject=open("path","mode")
# Modes:
# r - read only
# w - write
# a - append

# Узнать текущую рабочую директорию
import os
os.getcwd() 

# Задать текущую рабочую директорию
os.chdir("path")

# Получить список содержимого
os.listdir()


print(os.getcwd())

print(os.listdir())