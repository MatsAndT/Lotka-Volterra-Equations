import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
alpha = 2.7  # Prey birth rate
beta = 1.1  # Predation rate
gamma = 1.7  # Predator death rate
delta = 0.9  # Predator reproduction rate

# Equilibrium
xe = gamma / delta
ye = alpha / beta

# Initial conditions
x0 = xe+0.3  # Initial prey population
y0 = ye+0.2  # Initial predator population

# Time setting
t0 = 0
t_end = 100  # End time
dt = 0.0001  # Time steps
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

# Define Taylor series function
def V(x, y):
    return delta * x - gamma * np.log(x) + beta * y - alpha * np.log(y)

V_delta = V(x0, y0) - V(xe, ye)
A = np.sqrt(2 * V_delta * gamma) / delta
B = np.sqrt(2 * V_delta * alpha) / beta

omega = math.sqrt(alpha * gamma)
theta = math.atan((A*(y0 - ye)) / (B*(x0 - xe)))
t = t0

# Taylor series arrays
x_tay = []
y_tay = []

# Define Taylor series functions
def x_t(t):
    return gamma / delta + (math.sqrt(2 * V_delta * gamma) / delta) * math.cos(omega * t + theta)

def y_t(t):
    return alpha / beta + (math.sqrt(2 * V_delta * alpha) / beta) * math.sin(omega * t + theta)



# Print initial conditions to check alignment
print(f"x0 = {x0}, x_t(0) = {x_t(0)}")
print(f"y0 = {y0}, y_t(0) = {y_t(0)}")

# Taylor series method of order 4
for i in range(1, len(t_points)):  # Start loop from i = 1
    x_tay.append(x_t(t))
    y_tay.append(y_t(t))
    t += dt
    
    # Current values for the Phase Space model
    xi = x[i - 1]
    yi = y[i - 1]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Update x and y using Taylor series up to the first order
    x[i] = xi + dx1 * dt
    y[i] = yi + dy1 * dt

# Plotting
plt.figure(figsize=(12, 5))

# Plotting both lines on the same graph with different colors
plt.plot(x0, y0, 'o', label='Initial Conditions', color='red')
plt.plot(xe, ye, 'o', label='Equilibrium', color='green')
plt.plot(x_t(0), y_t(0), 'o', label='Taylor Initial Conditions', color='blue')
plt.plot(x_tay, y_tay, label='Taylor Model', color='blue')
plt.plot(x, y, label='Phase Space Model', color='red')

# Adding labels, title, and legend
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Comparison of Taylor and Phase Space Models")
plt.grid()
plt.legend()  # Add a legend to describe the colors

plt.tight_layout()
plt.show()
