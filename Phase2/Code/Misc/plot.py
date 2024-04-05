import json
import numpy as np
import matplotlib.pyplot as plt

path = "../train_plot.json"
with open(path) as f:
    data = json.load(f)

length_data = len(data)
print(length_data)
steps = range(length_data)
values = [entry[2] for entry in data]
print(steps)
print(len(values))
plt.plot(steps, values, linestyle='-', color='b')
plt.xlabel("Number of Epochs")
plt.ylabel("MSE Loss")
plt.title("Validation loss vs. Number of Epochs")
plt.grid(True)
plt.show()