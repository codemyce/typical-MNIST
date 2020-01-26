#!../venv/bin/python

import numpy as np
import matplotlib.pyplot as plt

filename = '../data/t10k-images-idx3-ubyte'

file = np.fromfile(filename, dtype='>f')

arr = np.array(file)
print(arr[4:748])
images = np.reshape(arr[4:], (2500, 28, 28))

plt.imshow(images[0], cmap="gray"), plt.title('test')
plt.xticks([]), plt.yticks([])
plt.show()

input()
