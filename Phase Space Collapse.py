import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81 # Constants
L1 = L2 = 1.0
m1 = m2 = 1.0
t_max = 30 # Time parameters
dt = 0.05
b1 = b2 = 0.5
t_eval = np.arange(0, t_max, dt)

def deriv(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    M11 = (m1 + m2) * L1  # Mass matrix
    M12 = m2 * L2 * np.cos(delta)
    M21 = L1 * np.cos(delta)
    M22 = L2

    # Right-hand side (forcing terms)
    RHS1 = -m2 * L2 * omega2**2 * np.sin(delta) - (m1 + m2) * g * np.sin(theta1) - b1*omega1
    RHS2 = L1 * omega1**2 * np.sin(delta) - g * np.sin(theta2) - b2*omega2
    M = np.array([[M11, M12], [M21, M22]])
    RHS = np.array([RHS1, RHS2])
    accel = np.linalg.solve(M, RHS)  # Solve the 2x2 linear system

    domega1, domega2 = accel
    return omega1, domega1, omega2, domega2

y0_1 = [np.pi / 2, 0, np.pi / 2, 0] # Initial conditions
y0_2 = [np.pi / 2 + 0.01, 0, np.pi / 2, 0]

sol1 = solve_ivp(deriv, [0, t_max], y0_1, t_eval=t_eval, rtol=1e-10, atol=1e-10) # Solve ODEs
sol2 = solve_ivp(deriv, [0, t_max], y0_2, t_eval=t_eval, rtol=1e-10, atol=1e-10)

# SAME

# Extract θ and ω
theta1_1, omega1_1 = sol1.y[0], sol1.y[1]
theta2_1, omega2_1 = sol1.y[2], sol1.y[3]

theta1_2, omega1_2 = sol2.y[0], sol2.y[1]
theta2_2, omega2_2 = sol2.y[2], sol2.y[3]

# Plot phase space
plt.figure(figsize=(12, 5))

# θ1 vs ω1
plt.subplot(1, 2, 1)
plt.plot(theta1_1, omega1_1, label='Original')
plt.plot(theta1_2, omega1_2, label='Perturbed')
plt.xlabel(r'$\theta_1$ (rad)')
plt.ylabel(r'$\omega_1$ (rad/s)')
plt.title('Phase Space: $\\theta_1$ vs $\\omega_1$')
plt.grid(True)
plt.legend()

# θ2 vs ω2
plt.subplot(1, 2, 2)
plt.plot(theta2_1, omega2_1, label='Original')
plt.plot(theta2_2, omega2_2, label='Perturbed')
plt.xlabel(r'$\theta_2$ (rad)')
plt.ylabel(r'$\omega_2$ (rad/s)')
plt.title('Phase Space: $\\theta_2$ vs $\\omega_2$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
