import pickle

data=[["Ivanov","PHP",1],["Petrov","Java",10],["Sidorov","C#",5]]

# Write dump
with open("pickle_dump_file.txt","wb") as fin:
    pickle.dump(data,fin)

info=""
# Read dump
with open("pickle_dump_file.txt","rb") as fin:
    info=pickle.load(fin)

print(info)