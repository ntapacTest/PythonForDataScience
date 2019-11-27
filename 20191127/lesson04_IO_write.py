# part 1
# Чтение данных
# fileObject=open("path","mode")
# Modes:
# r - read only
# w - write
# a - append
# x - с проверкой на существование файла, 
# если файл существует то не пишем, выброс исключения

# Узнать текущую рабочую директорию
import os
os.getcwd() 

# Задать текущую рабочую директорию
os.chdir("path")

# Получить список содержимого
os.listdir()


print(os.getcwd())

print(os.listdir())

# part 2

a="asdfghj"
b="ffffff"

# Write to file
# w - write; t - text file
fout=open("test_text_file.txt","wt")
fout.write(a)
fout.close()

# Write to file x mode
# x - check exist; t - text file
try:
    fout_x=open("test_text_file_x.txt","xt")
    fout_x.write(a)
    fout_x.close()
except FileExistsError as ex:
    print(ex)

# Append o file
# a - append; t - text file
fout=open("test_text_file.txt","at")
fout.write(a)
fout.close()

# Запись с помощью print
fout=open("test_text_file.txt","wt")
print(a,b,file=fout,sep="\n")
fout.close()


# Алгоритм записи в файл больших строк, пошагово
a1="qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
size=len(a1)
fout=open("test_text_file_2.txt","at")
offset=0
ch=10
while True:
    if(offset>size):
        break
    fout.write(a1[offset:offset+ch])
    fout.write("\n")
    offset+=ch

fout.close()