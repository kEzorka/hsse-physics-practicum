from math import pi, radians, sqrt

import config
import numpy as np
import pytest
from src import model


def test_mechanical_energy_decreases_with_viscous_friction():
    pendulum = config.PENDULUM
    time = np.linspace(0.0, 30.0, 3000)

    solution = model.simulate(
        pendulum,
        radians(30.0),
        0.0,
        (time[0], time[-1]),
        config.G,
        0.05,
        config.SOLVER_METHOD,
        config.SOLVER_RTOL,
        config.SOLVER_ATOL,
        t_eval=time,
    )
    energy = model.total_energy(solution.y[0], solution.y[1], pendulum, config.G)

    assert energy[-1] < energy[0]
    assert np.max(np.diff(energy)) < 1e-9 * energy[0]


def test_small_damped_oscillations_match_linear_solution():
    pendulum = config.PENDULUM
    amplitude = radians(1.0)
    viscous = 0.05
    time = np.linspace(0.0, 30.0, 3000)
    omega0 = sqrt(pendulum.mass * config.G * pendulum.com_distance / pendulum.inertia)
    gamma = viscous / (2 * pendulum.inertia)
    damped_frequency = sqrt(omega0**2 - gamma**2)
    expected_angle = (
        amplitude
        * np.exp(-gamma * time)
        * (
            np.cos(damped_frequency * time)
            + gamma / damped_frequency * np.sin(damped_frequency * time)
        )
    )

    solution = model.simulate(
        pendulum,
        amplitude,
        0.0,
        (time[0], time[-1]),
        config.G,
        viscous,
        config.SOLVER_METHOD,
        config.SOLVER_RTOL,
        config.SOLVER_ATOL,
        t_eval=time,
    )
    maximum_relative_error = np.max(np.abs(solution.y[0] - expected_angle)) / amplitude

    assert maximum_relative_error < 1e-3


def test_turning_points_follow_theoretical_decay_rate():
    pendulum = config.PENDULUM
    viscous = 0.05
    expected_decay_rate = viscous / (2 * pendulum.inertia)

    solution = model.simulate_with_turning_points(
        pendulum,
        radians(2.0),
        0.0,
        (0.0, 60.0),
        config.G,
        viscous,
        config.SOLVER_METHOD,
        config.SOLVER_RTOL,
        config.SOLVER_ATOL,
    )
    times, amplitudes = model.turning_points(solution)
    measured_decay_rate = model.fit_decay_rate(times, amplitudes)

    assert len(times) > 20
    assert measured_decay_rate == pytest.approx(expected_decay_rate, rel=5e-4)


def test_simulation_rejects_negative_viscous_friction():
    with pytest.raises(ValueError, match="must be non-negative"):
        model.simulate(
            config.PENDULUM,
            radians(10.0),
            0.0,
            (0.0, 2 * pi),
            config.G,
            -0.01,
            config.SOLVER_METHOD,
            config.SOLVER_RTOL,
            config.SOLVER_ATOL,
        )
