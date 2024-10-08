# Description: Introduced carrying capacity through a saturation point K

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0  # Prey birth rate
beta = 0.1  # Predation rate
gamma = 1.5  # Predator death rate
delta = 0.075  # Predator reproduction rate
K = 200  # Saturation point for prey (carrying capacity)

# Initial conditions
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time setting
t0 = 0
t_end = 150
h = 0.01  # Time steps
t_points = np.arange(t0, t_end, h)

# Initialize arrays
x = np.zeros(len(t_points))
y = np.zeros(len(t_points))
x[0] = x0
y[0] = y0


# Define derivative functions
def first_derivatives(x, y):
    dx1 = alpha * x * ((K - x) / K) - beta * x * y
    dy1 = delta * x * y - gamma * y
    return dx1, dy1


def second_derivatives(x, y, dx1, dy1):
    dx2 = alpha * dx1 - beta * (dx1 * y + x * dy1)
    dy2 = delta * (dx1 * y + x * dy1) - gamma * dy1
    return dx2, dy2


def third_derivatives(x, y, dx1, dy1, dx2, dy2):
    dx3 = alpha * dx2 - beta * (dx2 * y + 2 * dx1 * dy1 + x * dx2)
    dy3 = delta * (dx2 * y + 2 * dx1 * dy1 + x * dy2) - gamma * dy2
    return dx3, dy3


def fourth_derivatives(x, y, dx1, dy1, dx2, dy2, dx3, dy3):
    dx4 = alpha * dx3 - beta * (dx3 * y + 3 * dx2 * dy1 + 3 * dx1 * dy2 + x * dy3)
    dy4 = delta * (dx3 * y + 3 * dx2 * dy1 + 3 * dx1 * dy2 + x * dx3) - gamma * dy3
    return dx4, dy4


# Taylor series method of order 4
for i in range(len(t_points) - 1):
    # Current values
    xi = x[i]
    yi = y[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Second derivatives
    dx2, dy2 = second_derivatives(xi, yi, dx1, dy1)

    # Third derivatives
    dx3, dy3 = third_derivatives(xi, yi, dx1, dy1, dx2, dy2)

    # Fourth derivatives
    dx4, dy4 = fourth_derivatives(xi, yi, dx1, dy1, dx2, dy2, dx3, dy3)

    # Update x and y using Taylor series up to fourth order
    x[i + 1] = xi + dx1 * h + (dx2 * h**2) / 2 + (dx3 * h**3) / 6 + (dx4 * h**4) / 24
    y[i + 1] = yi + dy1 * h + (dy2 * h**2) / 2 + (dy3 * h**3) / 6 + (dx4 * h**4) / 24

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
