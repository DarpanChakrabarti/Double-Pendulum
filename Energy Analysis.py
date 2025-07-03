import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81 # Constants
L1 = L2 = 1.0
m1 = m2 = 1.0
b1 = b2 = 0.05
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

def compute_energy(sol):
    theta1, omega1 = sol.y[0], sol.y[1]
    theta2, omega2 = sol.y[2], sol.y[3]

    T1 = 0.5 * m1 * (L1 * omega1)**2    # Kinetic Energy
    T2 = 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 +
    2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = T1 + T2

    y1 = -L1 * np.cos(theta1)     # Potential Energy
    y2 = y1 - L2 * np.cos(theta2)
    V = m1 * g * y1 + m2 * g * y2

    return T + V

energy = compute_energy(sol1)
plt.plot(sol1.t, energy)
plt.xlabel('Time(s)')
plt.ylabel('Total Energy(J)')
plt.title('Total Mechanical Energy vs Time')
plt.grid(True)
plt.show()
