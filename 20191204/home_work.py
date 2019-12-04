import numpy as np

# Simple feature scaling method
data=np.array([[100,2,12,251,1,78,123,7,31,323,5,87,741,9,33,682,2,20]])
print(data)

print(data.argmin())
print(data.argmax())
print(data.min(),data.max())


data=data/data.max()
print(data)

print(data.argmin())
print(data.argmax())
print(data.min(),data.max())

# min-max method
data=np.array([[100,2,12,251,1,78,123,7,31,323,5,87,741,9,33,682,2,20]])
print(data)

print(data.argmin())
print(data.argmax())
print(data.min(),data.max())


data=(data-data.min())/(data.max()-data.min())
print(data)

print(data.argmin())
print(data.argmax())
print(data.min(),data.max())

