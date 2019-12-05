import numpy as np

date=np.array('2019-12-25', dtype=np.datetime64)
print(date)

date+np.arange(12) # создать массив дат начиная с исходной

# установка точности (ns - наносекунды)
date=np.datetime64('2019-12-25', 'ns')
print(date)