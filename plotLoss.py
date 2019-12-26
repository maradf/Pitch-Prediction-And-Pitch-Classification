# Written by Mara Fennema
#
# Plots the loss of the LSTM from train.py

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


path = "PATH-TO-LOSS-FILE"

f = open(path, "r")

y = []
if f.mode == 'r':
    f1 = f.readlines()

    for line in f1:
        y.append(float(line[:2]))

# Calculate x-axis
ndatapoints = len(y)
maxX = 50*ndatapoints
x = list(range(0, maxX, 50))

# Calculate the trendline of all the datapoints
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x, y, label="Loss")
plt.plot(x, p(x), color="r", linestyle="dashed", label="Average")
plt.hlines(22.02122688293457, xmin=x[0], xmax=x[-1:], colors="m",linestyles="solid", label="Loss of average pitch")
plt.title("Loss over time")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.show()
