from math import pi, radians, sin, sqrt

import config
import numpy as np
import pytest
from scipy.special import ellipk
from src import model


def _measure_period(amplitude, n_periods=4):
    return model.measure_period(
        config.PENDULUM,
        amplitude,
        config.G,
        n_periods,
        config.SOLVER_METHOD,
        config.SOLVER_RTOL,
        config.SOLVER_ATOL,
    )


def test_energy_is_conserved_without_friction():
    pendulum = config.PENDULUM
    angle0 = radians(60.0)
    small_angle_period = (
        2
        * pi
        * sqrt(pendulum.inertia / (pendulum.mass * config.G * pendulum.com_distance))
    )
    time = np.linspace(0.0, 10 * small_angle_period, 4000)

    solution = model.simulate(
        pendulum,
        angle0,
        0.0,
        (time[0], time[-1]),
        config.G,
        0.0,
        config.SOLVER_METHOD,
        config.SOLVER_RTOL,
        config.SOLVER_ATOL,
        t_eval=time,
    )
    energy = model.total_energy(solution.y[0], solution.y[1], pendulum, config.G)
    relative_drift = np.max(np.abs(energy - energy[0])) / energy[0]

    assert relative_drift < 1e-8


def test_small_amplitude_period_matches_harmonic_formula():
    pendulum = config.PENDULUM
    expected_period = (
        2
        * pi
        * sqrt(pendulum.inertia / (pendulum.mass * config.G * pendulum.com_distance))
    )

    measured_period = _measure_period(radians(0.5))

    assert measured_period == pytest.approx(expected_period, rel=1e-5)


@pytest.mark.parametrize("amplitude_degrees", (30.0, 90.0, 150.0))
def test_nonlinear_period_matches_elliptic_integral(amplitude_degrees):
    pendulum = config.PENDULUM
    amplitude = radians(amplitude_degrees)
    parameter = sin(amplitude / 2) ** 2
    expected_period = (
        4
        * sqrt(pendulum.inertia / (pendulum.mass * config.G * pendulum.com_distance))
        * ellipk(parameter)
    )

    measured_period = _measure_period(amplitude)

    assert measured_period == pytest.approx(expected_period, rel=1e-7)


def test_period_increases_with_amplitude():
    periods = [_measure_period(radians(value)) for value in (5.0, 60.0, 120.0)]

    assert periods[0] < periods[1] < periods[2]


@pytest.mark.parametrize("amplitude", (-0.1, 0.0, pi, pi + 0.1))
def test_period_rejects_invalid_amplitude(amplitude):
    with pytest.raises(ValueError, match="Amplitude must be in"):
        _measure_period(amplitude)
