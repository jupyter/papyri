import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(0, 10, 50)
y = np.sin(x)

ax.plot(x, y)
