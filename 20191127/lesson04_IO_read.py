import os

# Write whole to file
# r - read; t - text file
fout=open("test_text_file.txt","rt")
read_str=fout.read()
fout.close()

print(read_str)

# Чтение из файла пошагово
read_str=""
fin=open("test_text_file_2.txt","rt")
ch=10
while True:
    fragment=fin.read(ch)
    if not fragment:
        break
    read_str+=fragment
fin.close()

print(read_str)

# Чтение построчно V1
read_str=""
fin=open("test_text_file_2.txt","rt")
for line in fin:
    read_str+=line
fin.close()

print(read_str)

# Чтение построчно V2
read_str=""
fin=open("test_text_file_2.txt","rt")
lines=fin.readlines()
fin.close()

print(lines)


# Менеджер контекста
a1="1111wwwwwweeeeeeeeeejjjjjjjjjffffffffffffffuuuuuuuuuusaaaaaaaaaannnnnnnnnnnnfddddddddd[[[[[[[[[[]]]]]]]]]]"
with open("test_text_file_3.txt","wt") as fout:    
    fout.write(a1)

