import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def MassSpringDamper(t, state):
    k = 124e3  # spring constant, kN/m
    m = 64.2  # mass, Kg
    c = 3  # damping coefficient
    g = 9.8  # metres per second*2
    omega = 1.0  # frequency
    phi = 0.0  # phase shift
    A = 5.0  # amplitude

    x, xd = state  # displacement, x and velocity x'
    xdd = -k x / m - c * xd - g + A * np.cos(2 * np.pi * omega * t - phi)

    return [xd, xdd]

#Initial conditions
state0 = [0.0, 1.2]  # [x0, v0] [m, m/sec]
ti = 0.0  # initial time
tf = 4.0  # final time
t = np.arange(ti, tf, 0.001)

#Solve the differential equation
solution = solve_ivp(MassSpringDamper, [ti, tf], state0, t_eval=t)

x = solution.y[0] * 1e3  # Displacement in mm
xd = solution.y[1]  # Velocity in m/s

#Plotting displacement and velocity
plt.figure(figsize=(15, 12))
plt.plot(t, x, 'b', label=r'$x$ (mm)', linewidth=2.0)
plt.plot(t, xd, 'g--', label=r'$\dot{x}$ (m/sec)', linewidth=2.0)
plt.xlabel('time (sec)')
plt.ylabel('disp (mm)', color='b')
plt.twinx()
plt.ylabel('velocity (m/s)', color='g')
plt.title('Mass-Spring System with $V0=1.2 \frac{m}{s}$ and $\delta{max}=22.9$mm')
plt.legend()
plt.grid()

plt.show()