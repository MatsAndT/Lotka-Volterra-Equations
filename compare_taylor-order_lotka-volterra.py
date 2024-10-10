# Description: Using Taylor-series up to fourth order

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1  # Prey birth rate
beta = 1.7  # Predation rate
gamma = 2.2  # Predator death rate
delta = 0.4  # Predator reproduction rate

# Initial conditions
x0 = 10  # Initial prey population
y0 = 2  # Initial predator population

# Time setting
t0 = 0
t_end = 6
dt = 0.0001  # Time steps
t_points = np.arange(t0, t_end, dt)

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



# Initialize arrays
x_actual = [x0]
y_actual = [y0]

# The actual curve (with small enough time steps)
for i in range(len(t_points) - 1):
    # Current values
    xi = x_actual[i]
    yi = y_actual[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)


    x_actual.append(
        xi + dx1 * dt
    )
    y_actual.append(
        yi + dy1 * dt
    )

# New time steps to show the difference between the actual curve and the Taylor series approximations
dt = 0.3  # Time steps (much larger to show how the Taylor series approximations diverge from the actual curve)
dtt = 0.001
t_points = np.arange(t0, t_end, dt)

# Fourth order
x_fourth = [x0]
y_fourth = [y0]
for i in range(len(t_points) - 1):
    # Current values
    xi = x_fourth[i]
    yi = y_fourth[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Second derivatives
    dx2, dy2 = second_derivatives(xi, yi, dx1, dy1)

    # Third derivatives
    dx3, dy3 = third_derivatives(xi, yi, dx1, dy1, dx2, dy2)

    # Fourth derivatives
    dx4, dy4 = fourth_derivatives(xi, yi, dx1, dy1, dx2, dy2, dx3, dy3)

    x_fourth.append(
        xi + dx1 * dt + (dx2 * dt**2) / 2 + (dx3 * dt**3) / 6 + (dx4 * dt**4) / 24
    )
    y_fourth.append(
        yi + dy1 * dt + (dy2 * dt**2) / 2 + (dy3 * dt**3) / 6 + (dx4 * dt**4) / 24
    )
    
    x_local = [x_fourth[i]]
    y_local = [y_fourth[i]]
    for i_t in np.arange(0, dt, dtt):
        x_local.append(
            xi + dx1 * i_t + (dx2 * i_t**2) / 2 + (dx3 * i_t**3) / 6 + (dx4 * i_t**4) / 24
        )
        y_local.append(
            yi + dy1 * i_t + (dy2 * i_t**2) / 2 + (dy3 * i_t**3) / 6 + (dx4 * i_t**4) / 24
        )

    plt.plot(x_local, y_local, color='red')


# Third order
x_third = [x0]
y_third = [y0]
for i in range(len(t_points) - 1):
    # Current values
    xi = x_third[i]
    yi = y_third[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Second derivatives
    dx2, dy2 = second_derivatives(xi, yi, dx1, dy1)

    # Third derivatives
    dx3, dy3 = third_derivatives(xi, yi, dx1, dy1, dx2, dy2)

    x_third.append(
        xi + dx1 * dt + (dx2 * dt**2) / 2 + (dx3 * dt**3) / 6
    )
    y_third.append(
        yi + dy1 * dt + (dy2 * dt**2) / 2 + (dy3 * dt**3) / 6
    )
    x_local = [x_third[i]]
    y_local = [y_third[i]]
    for i_t in np.arange(0, dt, dtt):
        x_local.append(
            xi + dx1 * i_t + (dx2 * i_t**2) / 2 + (dx3 * i_t**3) / 6
        )
        y_local.append(
            yi + dy1 * i_t + (dy2 * i_t**2) / 2 + (dy3 * i_t**3) / 6
        )

    plt.plot(x_local, y_local, color='turquoise')


# Second order
x_second = [x0]
y_second = [y0]

for i in range(len(t_points) - 1):
    # Current values
    xi = x_second[i]
    yi = y_second[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Second derivatives
    dx2, dy2 = second_derivatives(xi, yi, dx1, dy1)

    x_second.append(
        xi + dx1 * dt + (dx2 * dt**2) / 2
    )
    y_second.append(
        yi + dy1 * dt + (dy2 * dt**2) / 2
    )

    x_local = [x_second[i]]
    y_local = [y_second[i]]
    for i_t in np.arange(0, dt, dtt):
        x_local.append(
            xi + dx1 * i_t + (dx2 * i_t**2) / 2
        )
        y_local.append(
            yi + dy1 * i_t + (dy2 * i_t**2) / 2
        )

    plt.plot(x_local, y_local, color='brown')


# First order
x_first= [x0]
y_first = [y0]
for i in range(len(t_points) - 1):
    # Current values
    xi = x_first[i]
    yi = y_first[i]

    # First derivatives
    dx1, dy1 = first_derivatives(xi, yi)

    # Update x and y using Taylor series up to fourth order
    x_first.append(
        xi + dx1 * dt 
    )
    y_first.append(
        yi + dy1 * dt 
    )

# Plotting

# Phase space plot
plt.rcParams.update({'font.size': 14})
plt.plot(x_actual, y_actual, label="Actual")
plt.plot(x_fourth, y_fourth, 'o',label="Fourth Order", color='red')
plt.plot(x_third, y_third, 'o', label="Third Order", color='turquoise')
plt.plot(x_second, y_second, 'o', label="Second Order", color='brown')
plt.plot(x_first, y_first, label="First Order", marker='o')

plt.text(0.75, 0.7, f'$x_0 = {x0}$\n $y_0 = {y0}$\n $\\alpha = {alpha}$\n$\\beta = {beta}$\n$\delta = {delta}$\n$\gamma = {gamma}$', transform=plt.gca().transAxes, fontsize=18)
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.title("Phase Space")
plt.grid()


plt.legend()
plt.tight_layout()
plt.show()