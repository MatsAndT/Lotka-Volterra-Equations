# Description: Solving the differential equation using Taylor series

import numpy as np
import matplotlib.pyplot as plt

# Prey parameters
alpha = 1.0  # Prey birth rate
K = 100  # Carrying capacity

# Predation parameters
beta = 0.1  # Attack rate
h = 0.1  # Handling time per prey item

# Predator parameters
delta = 0.1  # Efficiency of converting consumed prey into predator reproduction
gamma = 0.1  # Predator death rate

# Initial conditions
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time settings
t_start = 0
t_end = 1000
dt = 0.01  # Time steps
t_points = np.arange(t_start, t_end, dt)

# Initialize arrays
x = np.zeros(len(t_points))
y = np.zeros(len(t_points))
x[0] = x0
y[0] = y0


# Define derivative functions
def rosenzweig_macarthur(x, y):
    # Functional response
    predation = (beta * x * y) / (1 + beta * h * x)
    # Prey growth
    dx1 = alpha * x * (1 - (x / K)) - predation
    # Predator growth
    dy1 = delta * predation - gamma * y
    return dx1, dy1


# Taylor series method of order 1
for i in range(len(t_points) - 1):
    # Current values
    xi = x[i]
    yi = y[i]

    # First derivatives
    dx1, dy1 = rosenzweig_macarthur(xi, yi)

    # Update x and y using Taylor series
    x[i + 1] = xi + dx1 * dt
    y[i + 1] = yi + dy1 * dt

    # Ensure populations remain non-negative
    x[i + 1] = max(x[i + 1], 0)
    y[i + 1] = max(y[i + 1], 0)

# Plotting
plt.figure(figsize=(12, 5))

# Time series plot
plt.subplot(1, 2, 1)
plt.plot(t_points, x, label="Prey Population")
plt.plot(t_points, y, label="Predator Population")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population over Time")
plt.legend()

# Phase space plot
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Phase Space")
plt.grid()

plt.tight_layout()
plt.show()
