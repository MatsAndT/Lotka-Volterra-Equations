# Description: Using first-order Taylor series

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0  # Prey birth rate
beta = 0.1  # Predation rate
gamma = 1.5  # Predator death rate
delta = 0.075  # Predator reproduction rate

# Initial conditions
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time setting
t0 = 0
t_end = 150
dt = 0.01  # Time steps
t_points = np.arange(t0, t_end, dt)

# Initialize arrays
x = np.zeros(len(t_points))
y = np.zeros(len(t_points))
x[0] = x0
y[0] = y0


# Define derivative functions
def first_derivatives(x, y):
    dx1 = alpha * x - beta * x * y
    dy1 = delta * x * y - gamma * y
    return dx1, dy1


# First-order Taylor-series method
for i in range(len(t_points) - 1):
    # Current values
    xi = x[i]
    yi = y[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Update x and y using first-order Taylor-series
    x[i + 1] = xi + dx1 * dt
    y[i + 1] = yi + dy1 * dt

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
plt.text(0.85, 0.7, f'$x_0 = {x0}$\n $y_0 = {y0}$\n $\\alpha = {alpha}$\n$\\beta = {beta}$\n$\delta = {delta}$\n$\gamma = {gamma}$', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.show()
