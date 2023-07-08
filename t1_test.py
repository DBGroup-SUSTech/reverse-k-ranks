import numpy as np

user = np.array([[1.5, 0.9]])
item = np.array([[0.6, 0.9], [0.2, 2.7], [2.1, 1.8], [1.8, 2.7], [0.3, 0.6], [2.4, 0.9], [0.9, 0.0], [2.7, 0.6]])
print(np.dot(user, item.T))
