import numpy as np
import matplotlib.pyplot as plt

# Prey parameters
alpha = 1.0  # Prey birth rate
K = 100  # Carrying capacity

# Predation parameters
beta = 0.1  # Predation rate
h = 0.1  # Handling time per prey item

# Predator parameters
delta = 0.1  # Efficiency of converting conumed prey into predator reproduction
gamma = 0.1  # Predator death rate

# Initial conditions
x0 = 40  # Initial prey population
y0 = 9  # Initial predator population

# Time setting
t0 = 0
t_end = 200
dt = 0.01  # Time steps
t_points = np.arange(t0, t_end, dt)

# Initialize arrays
x = np.zeros(len(t_points))
y = np.zeros(len(t_points))
x[0] = x0
y[0] = y0


# Define derivative functions
def first_derivatives(x, y):

    fx = (beta * x) / (1 + beta * h * x)

    dx1 = alpha * x * (1 - (x / K)) - fx * y
    dy1 = delta * fx * y - gamma * y
    return dx1, dy1


# Taylor series method of order 1
for i in range(len(t_points) - 1):
    # Current values
    xi = x[i]
    yi = y[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Update x and y using Taylor series up to first order
    x[i + 1] = xi + dx1 * dt
    y[i + 1] = yi + dy1 * dt

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
