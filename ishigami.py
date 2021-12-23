# Python implementation of simulating Bekker and Reece-Wong derived Terramechanics models
# As described by Ishigami

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
X_c = np.deg2rad(26.4)  # Soil distractive angle
k_c = 1.37 * (10 ** 3)  # Pressure-sinkage module
k_phi = 8.14 * (10 ** 5)  # Pressure-sinkage module
n = 1.00  # Sinkage exponent
a_0 = 0.40  #
a_1 = 0.15  #
rho_d = 1.6 * (10 ** 3)  # Soil density
l_s = 0.9 # Wheel sinkage ratio
k_x = 0.043 * beta + 0.036  # Soil deformation module
k_y = 0.020 * beta + 0.013  # Soil deformation module


def get_static_behavior():
    k = (r ** (n + 1)) * (k_c + (k_phi * b))

    def integrand(_theta, _theta_s): return ((np.cos(_theta) - np.cos(_theta_s)) ** n) * np.cos(_theta)

    def integral(_theta_s): return W - (k * quad(integrand, -_theta_s, _theta_s, args=_theta_s)[0])

    vint = np.vectorize(integral)

    _theta_s = fsolve(vint, 0)
    _h_s = r * (1 - np.cos(_theta_s))

    return _theta_s[0], _h_s[0]


def get_contact_angles():
    _theta_f = np.arccos(1 - (h_s / r))
    _theta_r = -np.arccos(1 - (l_s * h_s / r))
    return _theta_f, _theta_r


def sigma(theta):
    _theta_m = (a_0 + (a_1 * s)) * theta_f
    k = (r ** n) * ((k_c / b) + k_phi)
    _sigma = -1
    if (_theta_m <= theta) & (theta < theta_f):
        _sigma = \
            k * (np.cos(theta) - np.cos(theta_f) ** n)
    elif (theta_r < theta) & (theta <= _theta_m):
        _sigma = \
            k * ((np.cos(theta_f - (((theta - theta_r) / (_theta_m - theta_r)) * (theta_f - _theta_m))) - np.cos(theta_f)) ** n)
    return _sigma, _theta_m


def j_x(theta):
    return r * (theta_f - theta - ((1 - s) * (np.sin(theta_f) - np.sin(theta))))


def tau_x(theta):
    return (c + (sigma(theta)[0] * np.tan(phi))) * (1 - np.exp(-j_x(theta) / k_x))


def get_fx():
    def integrand(theta): return (tau_x(theta) * np.cos(theta)) - (sigma(theta)[0] * np.sin(theta))
    return r * b * quad(integrand, theta_r, theta_f)[0]


def get_fz():
    def integrand(theta): return (tau_x(theta) * np.sin(theta)) + (sigma(theta)[0] * np.cos(theta))
    return r * b * quad(integrand, theta_r, theta_f)[0]

# Static values


Sigma = np.zeros(200)
Tau = np.zeros(200)

theta_s, h_s = get_static_behavior()
theta_f, theta_r = get_contact_angles()

Theta = np.linspace(theta_r, theta_f, 200)
for i in range(0, 200):
    s = 0
    t = Theta[i]
    Sigma[i] = sigma(t)[0] * np.cos(t) * r * b
    Tau[i] = tau_x(t) * np.sin(t) * r * b

Sum = Tau + Sigma

plt.plot(Theta, Sigma)
plt.plot(Theta, Tau)
plt.plot(Theta, Sum)
plt.title("Shear and Normal Stresses on Single Wheel")
plt.legend(['Normal Stress(x)', 'Shear Stress (x)', 'Sum'])
plt.xlabel("Theta [Rad]")
plt.ylabel("Stress [Pa]")
plt.show()
f_z = get_fz()

# Variable values

start = 0
end = 0.8
iter = 25

S = np.linspace(start, end, iter)
F_x = np.zeros(iter)
F_z = np.zeros(iter)

theta_s, h_s = get_static_behavior()
theta_f, theta_r = get_contact_angles()

print("theta_s", theta_s, "h_s", h_s, "theta_f", theta_f, "theta_r", theta_r)
for i in range(0, iter):
    s = S[i]
    F_x[i] = get_fx()
    F_z[i] = get_fz()
    if F_z[i] != W:
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

# start = -0.3
# end = 0.3
# iter = 200
#
# T = np.linspace(start, end, iter)
# S_t = np.zeros(iter)
# for i in range(0, iter):
#     theta_s, h_s = get_static_behavior()
#     theta_f, theta_r = get_contact_angles()
#     theta = T[i]
#     S_t[i], theta_m = sigma(theta)
#
#
# print(theta_f, theta_r)
# print(h_s, theta_s)
#
# plt.plot(T, S_t)
# plt.axvline(x=theta_m)
# plt.show()
