import pytest
import numpy as np
from dataclasses import replace

import config
import model

from state import Ball

GAP = 1e-3   # начальный зазор между поверхностями шаров, м


CASES = [
  pytest.param(
    Ball(position=(0.0, 0.0), velocity=(0.5, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    Ball(position=(0.1, 0.0), velocity=(-1.5, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    0.0,
    id="head-on-collision-with-equal-mass",
    marks=pytest.mark.xfail(reason="решатель перешагивает контакт, баг в model.py"),
  ),
  pytest.param(
    Ball(position=(0.0, 0.0), velocity=(1.0, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    Ball(position=(0.1, 0.0), velocity=(0.0, 0.0), radius=0.5, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    0.0,
    id="light-hits-resting-heavy"
  ),
  pytest.param(
    Ball(position=(0.0, 0.0), velocity=(0.0, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    Ball(position=(0.1, 0.0), velocity=(-1.0, 0.0), radius=0.5, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    0.0,
    id="heavy-hits-resting-light"
  ),
  pytest.param(
    Ball(position=(0.0, 0.0), velocity=(1.5, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    Ball(position=(0.1, 0.1), velocity=(-0.5, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    0.05,
    id="oblique-collision"
  ),
  pytest.param(
    Ball(position=(0.0, 0.0), velocity=(3.0, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    Ball(position=(0.1, 0.0), velocity=(1.0, 0.0), radius=0.05, density=7800, young_modulus=2e11, poisson_ratio=0.3),
    0.0,
    id="catch-up"
  ),    
]


def run(ball1, ball2, t_end):
  return model.simulate(ball1, ball2, (0, t_end), config.HERTZ_EXPONENT, config.MIN_SEPARATION, config.SOLVER_METHOD, config.SOLVER_RTOL, config.SOLVER_ATOL)


def close_balls(ball1, ball2, offset)-> tuple[Ball, Ball]:
  b1 = ball1
  b2 = ball2
  b2_close = replace(b2, position=(b1.position[0] + b1.radius + b2.radius + GAP, b1.position[1] + offset)) # сумма радиусов + зазор
  return b1, b2_close


@pytest.mark.parametrize("ball1, ball2, offset", CASES)
def test_collision_happened(ball1, ball2, offset):
  b1, b2 = close_balls(ball1, ball2, offset)
  sol = run(b1, b2, 0.01)
  vx_begin = sol.y[2, 0]
  vx_end = sol.y[2, -1]
  v_relative = np.hypot(sol.y[2, 0] - sol.y[6, 0], sol.y[3, 0] - sol.y[7, 0])
  assert abs(vx_begin - vx_end) > 0.01 * abs(v_relative) # скорость изменилась больше чем на 1%


@pytest.mark.parametrize("ball1, ball2, offset", CASES)
def test_momentum_conserved(ball1, ball2, offset):
  b1, b2 = close_balls(ball1, ball2, offset)
  sol = run(b1, b2, 0.01)

  p_x = b1.mass * sol.y[2, :] + b2.mass * sol.y[6, :]
  p_y = b1.mass * sol.y[3, :] + b2.mass * sol.y[7, :]

  scale = np.hypot(p_x[0], p_y[0])

  # Каждый элемент массива должен отличаться от начального значения не больше чем
  # на 1e-8 от масштаба. rtol=0 отключает скрытый относительный допуск.
  np.testing.assert_allclose(p_x, p_x[0], atol=1e-8 * scale, rtol=0)
  np.testing.assert_allclose(p_y, p_y[0], atol=1e-8 * scale, rtol=0)


@pytest.mark.parametrize("ball1, ball2, offset", CASES)
def test_kinetic_energy_conserved(ball1, ball2, offset):
  b1, b2 = close_balls(ball1, ball2, offset)
  sol = run(b1, b2, 0.01)

  ke = 0.5 * b1.mass * (sol.y[2, :]**2 + sol.y[3, :]**2) + 0.5 * b2.mass * (sol.y[6, :]**2 + sol.y[7, :]**2)

  assert_separated(sol, b1, b2)

  assert ke[-1] == pytest.approx(ke[0], rel=1e-7) # конечная энергия = начальной с точностью до 1e-7

def assert_separated(sol, ball1, ball2):
  dx = sol.y[0, -1] - sol.y[4, -1]
  dy = sol.y[1, -1] - sol.y[5, -1]
  distance = np.hypot(dx, dy)
  assert distance > ball1.radius + ball2.radius # шары не перекрываются = не в контакте
  
  dvx = sol.y[2, -1] - sol.y[6, -1]
  dvy = sol.y[3, -1] - sol.y[7, -1]
  assert dx*dvx + dy*dvy > 0 # скалярное произведение > 0: шары удаляются друг от друга
  

def test_A1_head_on_equal_masses_resting_target():
  b1, b2 = close_balls(config.FIRST_BALL, config.SECOND_BALL, 0.0)
  b2 = replace(b2, velocity=(0.0, 0.0)) # второй шар покоится
  v0 = b1.velocity[0]

  assert b1.mass == b2.mass
  assert b1.velocity[1] == 0

  sol = run(b1, b2, 0.01)

  assert_separated(sol, b1, b2)

  v1_expected = 0
  v2_expected = v0

  assert sol.y[2, -1] == pytest.approx(v1_expected, abs=1e-6 * abs(v0))
  assert sol.y[6, -1] == pytest.approx(v2_expected, abs=1e-6 * abs(v0))

  assert sol.y[3, -1] == pytest.approx(0.0, abs=1e-6 * abs(v0))
  assert sol.y[7, -1] == pytest.approx(0.0, abs=1e-6 * abs(v0))


@pytest.mark.parametrize("mass_ratio", [0.2, 0.5, 1.0, 2.0, 5.0])
def test_A2_head_on_arbitrary_masses(mass_ratio):
  b1, b2 = close_balls(config.FIRST_BALL, config.SECOND_BALL, 0.0)
  b2 = replace(b2, velocity=(0.0, 0.0), density=b2.density * mass_ratio)
  v0 = b1.velocity[0]

  assert b1.velocity[1] == 0

  sol = run(b1, b2, 0.01)

  assert_separated(sol, b1, b2)

  m1, m2 = b1.mass, b2.mass
  v1_expected = (m1 - m2) / (m1 + m2) * v0
  v2_expected = (2 * m1) / (m1 + m2) * v0

  v1_x = sol.y[2, -1]
  v1_y = sol.y[3, -1]
  v2_y = sol.y[7, -1]
  v2_x = sol.y[6, -1]

  assert v1_x == pytest.approx(v1_expected, abs=1e-6 * abs(v0))
  assert v1_y == pytest.approx(0.0, abs=1e-6 * abs(v0))
  assert v2_x == pytest.approx(v2_expected, abs=1e-6 * abs(v0))
  assert v2_y == pytest.approx(0.0, abs=1e-6 * abs(v0))
