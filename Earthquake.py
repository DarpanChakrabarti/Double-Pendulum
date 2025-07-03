import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81 # Constants
L1 = L2 = 1.0
m1 = m2 = 1.0
t_max = 30
dt = 0.05
t_eval = np.arange(0, t_max, dt)

earthquake_amplitude = 5.0  # (A) m/s² Earth quack parameters
earthquake_frequency = 1.0  # f (Hz)

def earthquake_acceleration(t):
    return earthquake_amplitude * np.sin(2 * np.pi * earthquake_frequency * t)

def deriv(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    a_t = earthquake_acceleration(t)  # Get current ground acceleration
    M11 = (m1 + m2) * L1
    M12 = m2 * L2 * np.cos(delta)
    M21 = L1 * np.cos(delta)
    M22 = L2

    RHS1 = (-m2 * L2 * omega2**2 * np.sin(delta) - (m1 + m2) * g * np.sin(theta1) 
            + (m1 + m2) * a_t * np.cos(theta1))  # Horizontal forcing
    RHS2 = (L1 * omega1**2 * np.sin(delta) - g * np.sin(theta2) + a_t * np.cos(theta2))  # Horizontal forcing
    M = np.array([[M11, M12], [M21, M22]])
    RHS = np.array([RHS1, RHS2])
    accel = np.linalg.solve(M, RHS)

    domega1, domega2 = accel
    return omega1, domega1, omega2, domega2

y0_1 = [0, 0, 0, 0]  # Pendulum conditions
sol1 = solve_ivp(deriv, [0, t_max], y0_1, t_eval=t_eval, rtol=1e-10, atol=1e-10)

def get_xy(sol):
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

x1_1, y1_1, x2_1, y2_1 = get_xy(sol1)
fig, ax = plt.subplots()
ax.set_xlim(-3, 3) 
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_title("Double Pendulum with Earthquake")

line1, = ax.plot([], [], 'o-', lw=2, label='Pendulum')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line1.set_data([], [])
    time_text.set_text('')
    return line1, time_text

def update(i):

    if i == 500:
        plt.savefig("D:\Bacchuga\Projects\Images\Figure8f.eps", format='eps', dpi=300)

    line1.set_data([0, x1_1[i], x2_1[i]], [0, y1_1[i], y2_1[i]])
    time_text.set_text(f'Time = {i * dt:.1f}s')
    return line1, time_text

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                    blit=True, interval=dt * 1000)

plt.legend()
plt.show()