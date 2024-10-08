# Description Solving the differential equation using built-in functions to solve ODEs

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Prey parameters
r = 1.0  # Intrinsic growth rate
K = 100.0  # Carrying capacity

# Predation parameters
a = 0.1  # Attack rate
h = 0.1  # Handling time

# Predator parameters
e = 0.1  # Conversion efficiency
d = 0.1  # Predator death rate

# Initial conditions
x0 = 40.0  # Initial prey population
y0 = 9.0  # Initial predator population

# Time settings
t_start = 0.0
t_end = 1000.0
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)


# Define the Rosenzweig-MacArthur model
def rosenzweig_macarthur(t, Z):
    x, y = Z
    # Ensure populations remain non-negative
    x = max(x, 0)
    y = max(y, 0)
    # Functional response
    predation = (a * x * y) / (1 + a * h * x)
    # Prey growth
    dxdt = r * x * (1 - x / K) - predation
    # Predator growth
    dydt = e * predation - d * y
    return [dxdt, dydt]


# Initial conditions vector
Z0 = [x0, y0]

# Solve the system
solution = solve_ivp(
    rosenzweig_macarthur, [t_start, t_end], Z0, t_eval=t_eval, method="RK45"
)

# Extract solutions
x = solution.y[0]
y = solution.y[1]
t = solution.t

# Plotting
plt.figure(figsize=(12, 5))

# Time series plot
plt.subplot(1, 2, 1)
plt.plot(t, x, label="Prey Population", color="blue")
plt.plot(t, y, label="Predator Population", color="orange")
plt.xlabel("Time")
plt.ylabel("Population Size")
plt.title("Rosenzweig-MacArthur Model: Population Dynamics Over Time")
plt.legend()
plt.grid(True)

# Phase space plot
plt.subplot(1, 2, 2)
plt.plot(x, y, color="green")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Rosenzweig-MacArthur Model: Phase Space Trajectory")
plt.grid(True)

plt.tight_layout()
plt.show()
