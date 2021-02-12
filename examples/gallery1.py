import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()
x = np.arange(0, 10, 0.1)
ax.plot(x, np.sin(x))
ax.plot(x, np.cos(x))
