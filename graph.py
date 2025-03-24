import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import linregress

n = []
time = []

with open("bench.txt") as f:
    for line in f:
        entry = json.loads(line)
        n.append(entry["n"] * entry["n"])
        time.append(entry["time"])

log2_n = np.log2(n)
log2_time = np.log2(time)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(log2_n, log2_time)

# Generate regression line
log2_x_fit = np.linspace(int(np.log2(4)), int(np.log2(30 * 30)) + 1, 100)
log2_y_fit = slope * log2_x_fit + intercept

# Convert back to original scale for plotting
x_fit = 2 ** log2_x_fit
y_fit = 2 ** log2_y_fit


plt.scatter(n, time)
plt.plot(x_fit, y_fit, color="red", label=f"Fit: y = {2**intercept:.2f} * x^{slope:.2f}")

print(slope)

plt.xscale("log", base=2)
plt.yscale("log", base=2)

plt.xlabel("nx=ny")
plt.ylabel("Temps [s]")

plt.grid(True)

plt.savefig("2d.svg")

plt.show()