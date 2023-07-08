import numpy as np

# arr = np.random.normal(size=9)
# print(arr)
# min_val = min(arr)
# max_val = max(arr)
# print((arr - min_val) / (max_val - min_val))

count = 0
for i in range(100):
    arr = np.random.randint(low=1, high=30, size=11)
    arr = np.sort(arr)
    if len(set(arr)) == 11:
        count += 1
        print(arr)
