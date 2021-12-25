# Python implementation of simulating Bekker and Reece-Wong derived Terramechanics models
# As described by Ishigami

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

# Vehicle Constants
beta = np.deg2rad(5)  # Slip Angle [deg]
s = 0.5  # Wheel Slip
W = 64.68  # Normal Load [N]
r = 0.09  # Wheel radius
b = 0.11  # Wheel width

# Soil Constants
c = 0.8 * 1000  # Cohesion stress
phi = np.deg2rad(37.2)  # Friction angle
X_c = np.deg2rad(26.4)  # Soil destructive angle
k_c = 1.37 * (10 ** 3)  # Pressure-sinkage module
k_phi = 8.14 * (10 ** 5)  # Pressure-sinkage module
n = 1.00  # Sinkage exponent
a_0 = 0.40  #
a_1 = 0.15  #
rho_d = 1.6 * (10 ** 3)  # Soil density
l_s = 0.9  # Wheel sinkage ratio
k_x = 0.043 * beta + 0.036  # Soil deformation module
k_y = 0.020 * beta + 0.013  # Soil deformation module


def get_static_behavior():
    k = (r ** (n + 1)) * (k_c + (k_phi * b))

    def integrand(_theta, _theta_s): return ((np.cos(_theta) - np.cos(_theta_s)) ** n) * np.cos(_theta)

    def integral(_theta_s): return W - (k * quad(integrand, -_theta_s, _theta_s, args=_theta_s)[0])

    vint = np.vectorize(integral)

    _theta_s = fsolve(vint, np.array([0]))
    _h_s = r * (1 - np.cos(_theta_s))

    return _theta_s[0], _h_s[0]


def get_contact_angles():
    # Solve for h
    def integrand(_theta, _h): return (tau_x_h(_theta, _h) * np.sin(_theta)) + (sigma_h(_theta, _h)[0] * np.cos(_theta))

    # _h_s = quad(integrand1, -theta_s, theta_s, args=theta_s)[0]
    def eq(_h): return W - (r*b*quad(integrand, theta_r_h(_h), theta_f_h(_h), args=_h)[0])

    veq = np.vectorize(eq)
    _h = fsolve(veq, np.array([0.05]))

    # H = np.linspace(0, 1, 100)
    # T = np.zeros(100)
    #
    # for _i in range(0, 100):
    #     T[_i] = eq(H[_i])
    #
    # plt.plot(H, T)
    # plt.legend(["Slip " + str(s)])
    # plt.axvline(x=_h, color='black')
    # plt.axhline(y=0, color='black')
    # plt.show()

    return theta_f_h(_h[0]), theta_r_h(_h[0])


def sigma(theta):
    _theta_m = (a_0 + (a_1 * s)) * theta_f
    k = (r ** n) * ((k_c / b) + k_phi)
    _sigma = -1
    if (_theta_m < theta) & (theta < theta_f):
        _sigma = \
            k * (np.cos(theta) - np.cos(theta_f) ** n)
    elif (theta_r < theta) & (theta < _theta_m):
        _sigma = \
            k * ((np.cos(theta_f - (((theta - theta_r) / (_theta_m - theta_r)) * (theta_f - _theta_m))) - np.cos(theta_f)) ** n)
    return _sigma, _theta_m


def theta_f_h(h):
    return np.arccos(1 - (h/r))


def theta_r_h(h):
    return -np.arccos(1 - (l_s * h/r))


def tau_x_h(theta, h):
    return (c + (sigma_h(theta, h)[0] * np.tan(phi))) * (1 - np.exp(-j_x_h(theta, h) / k_x))


def sigma_h(theta, h):
    _theta_m = (a_0 + (a_1 * s)) * theta_f_h(h)
    k = (r ** n) * ((k_c / b) + k_phi)
    _sigma = -1
    if (_theta_m < theta) & (theta < theta_f_h(h)):
        _sigma = \
            k * (np.cos(theta) - np.cos(theta_f_h(h)) ** n)
    elif (theta_r_h(h) < theta) & (theta < _theta_m):
        _sigma = \
            k * ((np.cos(theta_f_h(h) - (((theta - theta_r_h(h)) / (_theta_m - theta_r_h(h))) * (theta_f_h(h) - _theta_m))) - np.cos(theta_f_h(h))) ** n)
    return _sigma, _theta_m


def j_x(theta):
    return r * (theta_f - theta - ((1 - s) * (np.sin(theta_f) - np.sin(theta))))


def j_x_h(theta, h):
    return r * (theta_f_h(h) - theta - ((1 - s) * (np.sin(theta_f_h(h)) - np.sin(theta))))


def tau_x(theta):
    return (c + (sigma(theta)[0] * np.tan(phi))) * (1 - np.exp(-j_x(theta) / k_x))


def get_fx():
    def integrand(theta): return (tau_x(theta) * np.cos(theta)) - (sigma(theta)[0] * np.sin(theta))
    return r * b * quad(integrand, theta_r, theta_f)[0]


def get_fz():
    def integrand(theta): return (tau_x(theta) * np.sin(theta)) + (sigma(theta)[0] * np.cos(theta))
    return r * b * quad(integrand, theta_r, theta_f)[0]

# Calculate stresses


Sigma = np.zeros(200)
Tau = np.zeros(200)

theta_s, h_s = get_static_behavior()
theta_f, theta_r = get_contact_angles()
a, theta_m = sigma(99)

Theta = np.linspace(theta_r, theta_f, 200)
for i in range(0, 200):
    t = Theta[i]
    Sigma[i] = sigma(t)[0] * np.cos(t) * r * b
    Tau[i] = tau_x(t) * np.sin(t) * r * b

Sum = Tau + Sigma

plt.plot(Theta, Sigma)
plt.plot(Theta, Tau)
plt.plot(Theta, Sum)
plt.axvline(x=theta_m)
plt.title("Stresses on Single Wheel along Z axis (Slip Ratio = "+ str(s) + ")")
plt.legend(['Normal Stress(z)', 'Shear Stress (z)', 'Sum'])
plt.xlabel("Theta [Rad]")
plt.ylabel("Stress [Pa]")
plt.show()

# Calculate Forces
start = 0
end = 0.8
iter = 20

S = np.linspace(start, end, iter)
F_x = np.zeros(iter)
F_z = np.zeros(iter)

theta_s, h_s = get_static_behavior()

for i in range(0, iter):
    s = S[i]
    theta_f, theta_r = get_contact_angles()
    print("theta_s", theta_s, "h_s", h_s, "theta_f", theta_f, "theta_r", theta_r)

    F_x[i] = get_fx()
    F_z[i] = get_fz()
    if not math.isclose(F_z[i], W, abs_tol=0.001):
         print('Warning! F_z != W [F_z = ', F_z[i], ' W = ', W, "]")

plt.plot(S, F_x, color="red")
plt.title("Single Wheel Setup (Drawbar Pull)")
plt.xlabel("Slip Ratio")
plt.ylabel("Drawbar Pull [N]")
plt.grid()
plt.show()
plt.figure()

plt.plot(S, F_z, color="red")
plt.title("Single Wheel Setup (Vertical Force)")
plt.xlabel("Slip Ratio")
plt.ylabel("Vertical Force [N]")
plt.grid()
plt.show()
