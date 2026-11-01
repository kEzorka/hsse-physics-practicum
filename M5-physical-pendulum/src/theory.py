import numpy as np
from scipy.special import ellipk

# ---------------------------------------------------------------------------
# Свободные колебания без трения
# ---------------------------------------------------------------------------


def small_angle_frequency(inertia, mass, com_distance, g):
    return np.sqrt(mass * g * com_distance / inertia)


def small_angle_period(inertia, mass, com_distance, g):
    return 2 * np.pi / small_angle_frequency(inertia, mass, com_distance, g)


def exact_period(amplitude, inertia, mass, com_distance, g):
    k = np.sin(np.asarray(amplitude) / 2)
    return 4 * np.sqrt(inertia / (mass * g * com_distance)) * ellipk(k**2)


def period_series(amplitude, inertia, mass, com_distance, g):
    a = np.asarray(amplitude)
    t0 = small_angle_period(inertia, mass, com_distance, g)
    return t0 * (1 + a**2 / 16 + 11 * a**4 / 3072)


# ---------------------------------------------------------------------------
# Затухающие колебания
# ---------------------------------------------------------------------------


def viscous_decay_rate(viscous, inertia):
    return viscous / (2 * inertia)


def quality_factor(omega0, gamma):
    return omega0 / (2 * gamma)


def damped_linear_solution(t, amplitude, omega0, gamma):
    omega_d = np.sqrt(omega0**2 - gamma**2)
    return (
        amplitude
        * np.exp(-gamma * t)
        * (np.cos(omega_d * t) + gamma / omega_d * np.sin(omega_d * t))
    )
