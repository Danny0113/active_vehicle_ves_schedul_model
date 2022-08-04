import numpy as np

D2_list = [[1, 2, 3], [2, 3, 4]]

np_array = np.array(D2_list)

print(np)
max = np.max(np_array, axis=0)
min = np.min(np_array, axis=0)
print(max, min)