import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Constants
g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0

t_max = 30
dt = 0.05
t_eval = np.arange(0, t_max, dt)

# Default earthquake parameters
A_default = 5.0
f_default = 1.0

# Earthquake acceleration function
def make_acceleration(A, f):
    return lambda t: A * np.sin(2 * np.pi * f * t)

# Derivative function (depends on A and f)
def make_deriv(A, f):
    a_func = make_acceleration(A, f)
    def deriv(t, y):
        theta1, omega1, theta2, omega2 = y
        delta = theta1 - theta2
        a_t = a_func(t)

        M11 = (m1 + m2) * L1
        M12 = m2 * L2 * np.cos(delta)
        M21 = L1 * np.cos(delta)
        M22 = L2

        RHS1 = -m2 * L2 * omega2**2 * np.sin(delta) - (m1 + m2)*g*np.sin(theta1) + (m1 + m2)*a_t*np.cos(theta1)
        RHS2 = L1 * omega1**2 * np.sin(delta) - g*np.sin(theta2) + a_t*np.cos(theta2)

        M = np.array([[M11, M12], [M21, M22]])
        RHS = np.array([RHS1, RHS2])
        accel = np.linalg.solve(M, RHS)
        domega1, domega2 = accel

        return omega1, domega1, omega2, domega2
    return deriv

# Convert angles to (x,y)
def get_xy(sol):
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

# Initial conditions
y0 = [0, 0, 0, 0]

# Solve initial system
deriv = make_deriv(A_default, f_default)
sol = solve_ivp(deriv, [0, t_max], y0, t_eval=t_eval)
x1, y1, x2, y2 = get_xy(sol)

# Figure setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_title("Double Pendulum With Earthquake")

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update
def update(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    time_text.set_text(f"Time = {i*dt:.1f}s")
    return line, time_text

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=dt*1000, blit=True)

# Slider axes
ax_A = plt.axes([0.2, 0.12, 0.6, 0.03])
ax_f = plt.axes([0.2, 0.06, 0.6, 0.03])

A_slider = Slider(ax_A, 'A (m/s²)', 0, 20, valinit=A_default)
f_slider = Slider(ax_f, 'f (Hz)', 0, 5, valinit=f_default)

# Slider update callback
def update_sliders(val):
    global sol, x1, y1, x2, y2
    A_new = A_slider.val
    f_new = f_slider.val

    deriv = make_deriv(A_new, f_new)
    sol = solve_ivp(deriv, [0, t_max], y0, t_eval=t_eval)
    x1, y1, x2, y2 = get_xy(sol)

A_slider.on_changed(update_sliders)
f_slider.on_changed(update_sliders)

plt.show()