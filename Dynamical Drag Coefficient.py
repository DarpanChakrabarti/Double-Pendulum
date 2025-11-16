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

# Derivative function (depends on damping b)
def make_deriv(b):
    def deriv(t, y):
        theta1, omega1, theta2, omega2 = y
        delta = theta1 - theta2

        M11 = (m1 + m2) * L1
        M12 = m2 * L2 * np.cos(delta)
        M21 = L1 * np.cos(delta)
        M22 = L2

        RHS1 = -m2 * L2 * omega2**2 * np.sin(delta) - (m1 + m2)*g*np.sin(theta1) - b*omega1
        RHS2 = L1 * omega1**2 * np.sin(delta) - g*np.sin(theta2) - b*omega2

        M = np.array([[M11, M12], [M21, M22]])
        RHS = np.array([RHS1, RHS2])
        accel = np.linalg.solve(M, RHS)
        domega1, domega2 = accel

        return omega1, domega1, omega2, domega2
    return deriv

# Convert angles to Cartesian
def get_xy(sol):
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

# Initial conditions
y0_1 = [np.pi/2, 0, np.pi/2, 0]
y0_2 = [np.pi/2+0.01, 0, np.pi/2, 0]

# Initial damping
b_initial = 0.05

# Compute initial solutions
deriv = make_deriv(b_initial)
sol1 = solve_ivp(deriv, [0, t_max], y0_1, t_eval=t_eval)
sol2 = solve_ivp(deriv, [0, t_max], y0_2, t_eval=t_eval)

x1_1, y1_1, x2_1, y2_1 = get_xy(sol1)
x1_2, y1_2, x2_2, y2_2 = get_xy(sol2)

# Figure setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.set_title("Double Pendulum with Damping Slider")

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2, color='red')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update
def update(i):
    line1.set_data([0, x1_1[i], x2_1[i]], [0, y1_1[i], y2_1[i]])
    line2.set_data([0, x1_2[i], x2_2[i]], [0, y1_2[i], y2_2[i]])
    time_text.set_text(f"Time = {i*dt:.1f}s")
    return line1, line2, time_text

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=dt*1000, blit=True)

# Slider
ax_b = plt.axes([0.2, 0.1, 0.6, 0.03])
b_slider = Slider(ax_b, 'b (damping)', 0, 5, valinit=b_initial)

# Slider update behavior
def update_b(val):
    global sol1, sol2, x1_1, y1_1, x2_1, y2_1
    b_new = b_slider.val

    deriv = make_deriv(b_new)
    sol1 = solve_ivp(deriv, [0, t_max], y0_1, t_eval=t_eval)
    sol2 = solve_ivp(deriv, [0, t_max], y0_2, t_eval=t_eval)

    x1_1, y1_1, x2_1, y2_1 = get_xy(sol1)
    x1_2_, y1_2_, x2_2_, y2_2_ = get_xy(sol2)

    global x1_2, y1_2, x2_2, y2_2
    x1_2, y1_2, x2_2, y2_2 = x1_2_, y1_2_, x2_2_, y2_2_

b_slider.on_changed(update_b)

plt.show()