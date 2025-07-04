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

t = sol1.t # Extract solutions
t1_1, t2_1 = sol1.y[0], sol1.y[2]
t1_2, t2_2 = sol2.y[0], sol2.y[2]

plt.figure(figsize=(12, 6)) # Plot theta1 and theta2 for both cases
plt.subplot(2, 1, 1)
plt.plot(t, t1_1, label='$\\theta_1$ (original)')
plt.plot(t, t1_2, label='$\\theta_1$ (perturbed)', linestyle='--')
plt.plot(t, t2_1, label='$\\theta_2$ (original)')
plt.plot(t, t2_2, label='$\\theta_2$ (perturbed)', linestyle='--')
plt.ylabel("Angular Displacement (rad)")
plt.title("Angle vs Time")
plt.legend()

plt.subplot(2, 1, 2) # Plot difference t2 - t1 for both
d1 = t2_1 - t1_1
d2 = t2_2 - t1_2
plt.plot(t, d1, label='$\Delta$ = t2 - t1 (original)')
plt.plot(t, d2, label='$\Delta$ = t2 - t1 (perturbed)', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("$\Delta$ (rad)")
plt.legend()
plt.tight_layout()
plt.show()
