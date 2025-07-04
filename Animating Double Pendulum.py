import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81 # Constants
L1 = L2 = 1.0
m1 = m2 = 1.0
t_max = 30 # Time parameters
dt = 0.05
t_eval = np.arange(0, t_max, dt)

def deriv(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    M11 = (m1 + m2) * L1  # Mass matrix
    M12 = m2 * L2 * np.cos(delta)
    M21 = L1 * np.cos(delta)
    M22 = L2

    # Right-hand side (forcing terms)
    RHS1 = -m2 * L2 * omega2**2 * np.sin(delta) - (m1 + m2) * g * np.sin(theta1)
    RHS2 = L1 * omega1**2 * np.sin(delta) - g * np.sin(theta2)
    M = np.array([[M11, M12], [M21, M22]])
    RHS = np.array([RHS1, RHS2])
    accel = np.linalg.solve(M, RHS)  # Solve the 2x2 linear system

    domega1, domega2 = accel
    return omega1, domega1, omega2, domega2

y0_1 = [np.pi / 2, 0, np.pi / 2, 0] # Initial conditions
y0_2 = [np.pi / 2 + 0.01, 0, np.pi / 2, 0]

sol1 = solve_ivp(deriv, [0, t_max], y0_1, t_eval=t_eval, rtol=1e-10, atol=1e-10) # Solve ODEs
sol2 = solve_ivp(deriv, [0, t_max], y0_2, t_eval=t_eval, rtol=1e-10, atol=1e-10)

def get_xy(sol): # Convert angles to (x, y)
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

x1_1, y1_1, x2_1, y2_1 = get_xy(sol1)
x1_2, y1_2, x2_2, y2_2 = get_xy(sol2)

fig, ax = plt.subplots() # Set up animation
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.set_title("Double Pendulum")

line1, = ax.plot([], [], 'o-', lw=2, label='Pendulum 1')
line2, = ax.plot([], [], 'o-', lw=2, label='Pendulum 2', color='red')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text

def update(i):
    
    line1.set_data([0, x1_1[i], x2_1[i]], [0, y1_1[i], y2_1[i]])
    line2.set_data([0, x1_2[i], x2_2[i]], [0, y1_2[i], y2_2[i]])
    time_text.set_text(f'Time = {i * dt:.1f}s')
    return line1, line2, time_text

from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                    blit=True, interval=dt * 1000)

plt.legend()
plt.show()

