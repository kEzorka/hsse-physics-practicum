import numpy as np
from scipy.integrate import solve_ivp


def gravity_torque(angle, mass, com_distance, g):
    return -mass * g * com_distance * np.sin(angle)


def friction_torque(angular_velocity, viscous):
    return -viscous * angular_velocity


def equations_of_motion(t, y, inertia, mass, com_distance, g, viscous):
    angle, angular_velocity = y
    torque = gravity_torque(angle, mass, com_distance, g) + friction_torque(
        angular_velocity, viscous
    )
    return np.array([angular_velocity, torque / inertia])


def simulate(
    pendulum,
    angle0,
    angular_velocity0,
    t_span,
    g,
    viscous,
    solver_method,
    solver_rtol,
    solver_atol,
    t_eval=None,
    events=None,
):
    if viscous < 0:
        raise ValueError("Viscous friction coefficient must be non-negative.")

    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=[angle0, angular_velocity0],
        args=(
            pendulum.inertia,
            pendulum.mass,
            pendulum.com_distance,
            g,
            viscous,
        ),
        method=solver_method,
        rtol=solver_rtol,
        atol=solver_atol,
        dense_output=True,
        t_eval=t_eval,
        events=events,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed: {sol.message}. "
            f"t_span={t_span}, stopped at t={sol.t[-1]}"
        )

    return sol


def kinetic_energy(angular_velocity, inertia):
    return inertia * angular_velocity**2 / 2


def potential_energy(angle, mass, com_distance, g):
    return mass * g * com_distance * (1 - np.cos(angle))


def total_energy(angle, angular_velocity, pendulum, g):
    return kinetic_energy(angular_velocity, pendulum.inertia) + potential_energy(
        angle, pendulum.mass, pendulum.com_distance, g
    )


def _positive_to_negative_crossing(t, y, *args):
    return y[0]


_positive_to_negative_crossing.direction = -1


def _turning_point(t, y, *args):
    return y[1]


def measure_period(
    pendulum, amplitude, g, n_periods, solver_method, solver_rtol, solver_atol
):
    if not 0 < amplitude < np.pi:
        raise ValueError(
            "Amplitude must be in (0, pi): at pi the pendulum is in "
            f"unstable equilibrium and the period diverges. Got {amplitude}."
        )

    small_angle_period = (
        2
        * np.pi
        * np.sqrt(pendulum.inertia / (pendulum.mass * g * pendulum.com_distance))
    )
    t_end = (n_periods + 1) * small_angle_period

    while True:
        sol = simulate(
            pendulum,
            amplitude,
            0.0,
            (0, t_end),
            g,
            0.0,
            solver_method,
            solver_rtol,
            solver_atol,
            events=_positive_to_negative_crossing,
        )
        crossings = sol.t_events[0]
        if len(crossings) >= n_periods + 1:
            return float(np.mean(np.diff(crossings[: n_periods + 1])))
        t_end *= 2


def turning_points(sol):
    times = sol.t_events[0]
    angles = sol.y_events[0][:, 0]
    return times, angles


def simulate_with_turning_points(
    pendulum,
    angle0,
    angular_velocity0,
    t_span,
    g,
    viscous,
    solver_method,
    solver_rtol,
    solver_atol,
    t_eval=None,
):
    return simulate(
        pendulum,
        angle0,
        angular_velocity0,
        t_span,
        g,
        viscous,
        solver_method,
        solver_rtol,
        solver_atol,
        t_eval=t_eval,
        events=_turning_point,
    )


def fit_decay_rate(times, amplitudes):
    slope, _ = np.polyfit(times, np.log(np.abs(amplitudes)), 1)
    return -slope
