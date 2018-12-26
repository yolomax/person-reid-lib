import numpy as np


a = [53, 13, 84, 32, 1]
b = np.random.choice(len(a), 3, True)
print(a[b])