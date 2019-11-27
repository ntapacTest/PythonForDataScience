import csv

# Wrtite csv
data=[["Ivanov","PHP",1],["Petrov","Java",10],["Sidorov","C#",5]]

with open("file.csv","w",newline="") as csv_file:
    writer=csv.writer(csv_file, delimiter=';',)
    # for line in data:
    #     writer.writerow(line)
    # или так
    writer.writerows(data)

# Read csv

with open("file.csv","r",newline="") as fin:
    cin=csv.reader(fin,delimiter=";")
    info=[row for row in cin]

print(info)



# Dictionary in csv
data_d=[{"name":"name1", "age":34, "gender":"m"},{"name":"name2", "age":20, "gender":"f"},{"name":"name3", "age":3524, "gender":"m"}]
with open("file_d.csv","w",newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames = ["name", "age", "gender"],delimiter=";")
    writer.writeheader()
    writer=csv.writer(csv_file,delimiter=";")
    writer.writerows(data)

with open("file_d.csv","r",newline="") as fin:
    cin=csv.DictReader(fin,delimiter=";")    
    info=[row for row in cin]

print(info)

