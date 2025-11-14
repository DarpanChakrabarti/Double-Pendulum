import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0
b1 = b2 = 0

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

def lyapunov_exponent_with_graph(y0, delta0, dt, T):

    t_span = [0, dt]
    y0_perturbed = y0 + np.array([delta0, 0, 0, 0])

    times = []
    log_divergence = []

    t = 0
    N = int(T / dt)
    sum_log = 0

    for _ in range(N):

        sol1 = solve_ivp(deriv, t_span, y0, t_eval=[t + dt],
                         rtol=1e-10, atol=1e-10)
        sol2 = solve_ivp(deriv, t_span, y0_perturbed, t_eval=[t + dt],
                         rtol=1e-10, atol=1e-10)

        y1 = sol1.y[:, -1]
        y2 = sol2.y[:, -1]

        diff = y2 - y1
        dist = np.linalg.norm(diff)

        # store data for graph
        times.append(t)
        log_divergence.append(np.log(dist / delta0))

        # accumulate for Lyapunov exponent
        sum_log += np.log(dist / delta0)

        # renormalize
        diff = diff * (delta0 / dist)
        y0 = y1
        y0_perturbed = y1 + diff

        # update time
        t += dt
        t_span = [t, t + dt]

    lambda_est = sum_log / (N * dt)

    plt.figure(figsize=(8, 5))
    plt.plot(times, log_divergence, lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\ln(\delta(t)/\delta_0)$")
    plt.title("Lyapunov Exponent Growth Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return lambda_est

y0 = np.array([np.pi/2, 0, np.pi/2, 0])
lambda_est = lyapunov_exponent_with_graph(y0, delta0=0.01, dt=0.01, T=30)

print("Largest Lyapunov exponent λ ≈", lambda_est)
