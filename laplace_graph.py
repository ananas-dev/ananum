import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import linregress

n = []
iter = []

with open("laplace.txt") as f:
    for line in f:
        entry = json.loads(line)
        n.append(entry["n"])
        iter.append(entry["iter"])

slope, intercept, r_value, p_value, std_err = linregress(n, iter)

# Generate regression line
x_fit = np.linspace(n[0], n[-1], 100)
y_fit = slope * x_fit + intercept

plt.plot(n, iter)
plt.plot(x_fit, y_fit)

print(slope)

plt.xlabel("n")
plt.ylabel("Nombre d'itérations")


plt.grid(True)

plt.savefig("1d.svg")

plt.show()

log2_n = np.log2(n)
log2_iter = np.log2(iter)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(log2_n, log2_iter)

# Generate regression line
log2_x_fit = np.linspace(np.log2(n[0]), np.log2(n[-1]), 100)
log2_y_fit = slope * log2_x_fit + intercept

# Convert back to original scale for plotting
x_fit = 2 ** log2_x_fit
y_fit = 2 ** log2_y_fit

plt.plot(n, iter)
plt.plot(x_fit, y_fit)

plt.xscale("log", base=2)
plt.yscale("log", base=2)


print(slope)

plt.xlabel("n")
plt.ylabel("Nombre d'itérations")

plt.savefig("1d-log.svg")

plt.grid(True)


plt.show()