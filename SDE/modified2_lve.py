# Description: Implementing the Milstein Method for SDEs

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0  # Prey birth rate
beta = 0.1  # Predation rate
gamma = 1.5  # Predator death rate
delta = 0.075  # Predator reproduction rate

# Noise intensities
sigma_x = 0.2  # Noise intensity for prey
sigma_y = 0.2  # Noise intensity for predator

# Initial conditions
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time setting
t0 = 0
t_end = 100
dt = 0.01  # Time step size
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

    # Generate independent Wiener increments
    dW_x = np.random.normal(0.0, np.sqrt(dt))
    dW_y = np.random.normal(0.0, np.sqrt(dt))

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Second derivatives
    dx2, dy2 = second_derivatives(xi, yi, dx1, dy1)

    # Third derivatives
    dx3, dy3 = third_derivatives(xi, yi, dx1, dy1, dx2, dy2)

    # Fourth derivatives
    dx4, dy4 = fourth_derivatives(xi, yi, dx1, dy1, dx2, dy2, dx3, dy3)

    # Calculate the deterministic parts using Taylor series up to fourth order
    # dx_det = xi + dx1 * dt + (dx2 * dt**2) / 2 + (dx3 * dt**3) / 6 + (dx4 * dt**4) / 24
    # dy_det = yi + dy1 * dt + (dy2 * dt**2) / 2 + (dy3 * dt**3) / 6 + (dx4 * dt**4) / 24
    dx_det = (alpha * xi - beta * xi * yi) * dt
    dy_det = (delta * xi * yi - gamma * yi) * dt

    # Diffusion function
    g_x = sigma_x * xi
    g_y = sigma_y * yi

    # Derivatives of diffusion functions
    dg_x_dt = sigma_x
    dg_y_dt = sigma_y

    # Milstein correction terms
    dx_milstein = g_x * dW_x + 0.5 * g_x * dg_x_dt * (dW_x**2 - dt)
    dy_milstein = g_y * dW_y + 0.5 * g_y * dg_y_dt * (dW_y**2 - dt)

    # Calculate the stochastic parts
    # dx_stoch = sigma_x * xi * np.random.normal(0.0, np.sqrt(dt))
    # dy_stoch = sigma_y * xi * np.random.normal(0.0, np.sqrt(dt))

    # Update populations
    x[i + 1] = xi * dx_det + dx_milstein
    y[i + 1] = yi * dy_det + dy_milstein

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
