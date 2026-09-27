import numpy as np
from scipy.optimize import brentq

import model


def depth(y, radius1, radius2):
    separation = np.linalg.norm(y[4:6] - y[0:2], axis=0)
    return radius1 + radius2 - separation


def compression_rate(y):
    separation_vector = y[4:6] - y[0:2]
    relative_velocity = y[6:8] - y[2:4]
    separation = np.linalg.norm(separation_vector, axis=0)
    return -np.sum(separation_vector * relative_velocity, axis=0) / separation


def kinetic_energy(y, m1, m2):
    return 0.5 * m1 * np.sum(y[2:4]**2, axis=0) + 0.5 * m2 * np.sum(y[6:8]**2, axis=0)


def potential_energy(y, radius1, radius2, stiffness, exponent):
    delta = np.maximum(depth(y, radius1, radius2), 0.0)
    return stiffness * delta**(exponent + 1) / (exponent + 1)


def momentum(y, m1, m2):
    return m1 * y[2:4] + m2 * y[6:8]


def pair_stiffness(ball1, ball2) -> float:
    return model.hertz_stiffness(
        ball1.young_modulus, ball1.poisson_ratio, ball1.radius,
        ball2.young_modulus, ball2.poisson_ratio, ball2.radius)


def time_to_touch(ball1, ball2) -> float:
    d0 = np.subtract(ball2.position, ball1.position)
    u = np.subtract(ball2.velocity, ball1.velocity)
    contact = ball1.radius + ball2.radius

    a = np.dot(u, u)
    b = 2 * np.dot(d0, u)
    c = np.dot(d0, d0) - contact**2

    if c <= 0:
        raise ValueError("Шары в начальный момент уже касаются или вмяты друг в друга.")
    discriminant = b**2 - 4 * a * c
    if a == 0 or discriminant < 0:
        raise ValueError("Шары не встретятся.")
    t_touch = (-b - np.sqrt(discriminant)) / (2 * a)
    if t_touch < 0:
        raise ValueError("Шары удаляются друг от друга.")
    return t_touch


def run(params, ball1, ball2):
    t_end = 2 * time_to_touch(ball1, ball2)
    return model.simulate(
        ball1, ball2, (0.0, t_end),
        params.HERTZ_EXPONENT,
        params.MIN_SEPARATION, params.MIN_RELATIVE_SPEED,
        params.SOLVER_METHOD, params.SOLVER_RTOL, params.SOLVER_ATOL)


def contact_interval(sol, radius1, radius2):
    delta = depth(sol.y, radius1, radius2)
    inside = np.flatnonzero(delta > 0)
    if inside.size == 0:
        return None

    first = inside[0]
    last = inside[-1]
    if first == 0 or last == len(sol.t) - 1:
        raise ValueError(
            "Шары уже внутри столкновения или не успели разойтись.")

    def depth_at(t):
        return depth(sol.sol(t), radius1, radius2)

    t_start = brentq(depth_at, sol.t[first - 1], sol.t[first])
    t_end = brentq(depth_at, sol.t[last], sol.t[last + 1])
    return t_start, t_end


def peak_compression(sol, t_start, t_end, radius1, radius2):
    t_peak = brentq(lambda t: compression_rate(sol.sol(t)), t_start, t_end)
    return t_peak, depth(sol.sol(t_peak), radius1, radius2)


def collision_summary(sol, ball1, ball2):
    m1, m2 = ball1.mass, ball2.mass
    y_before = sol.y[:, 0]
    y_after = sol.y[:, -1]

    if depth(y_after, ball1.radius, ball2.radius) >= 0:
        raise ValueError("К концу окна шары ещё в контакте.")

    interval = contact_interval(sol, ball1.radius, ball2.radius)
    if interval is None:
        raise ValueError("Столкновения не было.")
    t_start, t_end = interval
    t_peak, delta_max = peak_compression(sol, t_start, t_end, ball1.radius, ball2.radius)

    momentum_scale = (m1 * np.linalg.norm(y_before[2:4])
                      + m2 * np.linalg.norm(y_before[6:8]))
    momentum_change = momentum(y_after, m1, m2) - momentum(y_before, m1, m2)
    ke_before = kinetic_energy(y_before, m1, m2)
    ke_after = kinetic_energy(y_after, m1, m2)

    return {
        "steps": len(sol.t),
        "v1_before": y_before[2:4],
        "v2_before": y_before[6:8],
        "v1_after": y_after[2:4],
        "v2_after": y_after[6:8],
        "momentum_change": np.linalg.norm(momentum_change) / momentum_scale,
        "energy_change": (ke_after - ke_before) / ke_before,
        "contact_start": t_start,
        "contact_end": t_end,
        "contact_duration": t_end - t_start,
        "peak_time": t_peak,
        "max_compression": delta_max,
    }


def loglog_slope(x, y) -> float:
    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return slope
