import numpy as np
import matplotlib.pyplot as plt
with open("dataset1.csv") as file_name:
    array = np.loadtxt(file_name, delimiter=" ")


array_t = np.linspace(0, 75, num=len(array))
quake_array = []
plt.plot(array_t, array)
plt.show()