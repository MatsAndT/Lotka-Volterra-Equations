# Description: Using the Euler-Maruyama Method based on First-Order Taylor Expansions for SDEs

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha = 1.0  # Prey birth rate
beta = 0.1  # Predation rate
gamma = 1.5  # Predator death rate
delta = 0.075  # Predator reproduction rate

# Noise intensities
sigma_x = 0.1  # Noise intensity for prey
sigma_y = 0.1  # Noise intensity for predator

# Initial populations
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time settings
t0 = 0
t_end = 15
dt = 0.0001  # Time step size
t_points = np.arange(t0, t_end, dt)  # Time vector

# Initialize arrays
x = np.zeros(len(t_points))
y = np.zeros(len(t_points))
x[0] = x0
y[0] = y0


# Define derivative function
def first_derivatives(x, y):
    dx1 = alpha * x - beta * x * y
    dy1 = delta * x * y - gamma * y
    return dx1, dy1


# Euler-Maruyama simulation
for i in range(len(t_points) - 1):
    # Current values
    xi = x[i]
    yi = y[i]

    # Deterministic components
    dx1, dy1 = first_derivatives(xi, yi)
    dx_det = dx1 * dt
    dy_det = dy1 * dt

    # Stochastic components
    dx_stoch = sigma_x * xi * np.random.normal(0.0, np.sqrt(dt))
    dy_stoch = sigma_y * xi * np.random.normal(0.0, np.sqrt(dt))

    # Update populations
    x[i + 1] = xi + dx_det + dx_stoch
    y[i + 1] = yi + dy_det + dy_stoch

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
