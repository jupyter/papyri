import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()

x = np.arange(0, 10, 0.1) - 5
y = np.sin(x)

ax.plot(x, y)
ax.plot(y, x)
