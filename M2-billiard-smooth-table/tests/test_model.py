import pytest
import numpy as np
from dataclasses import replace

import config
import model

from state import Ball

GAP = 1e-3   # начальный зазор между поверхностями шаров, м

def run(ball1, ball2, t_end):
  return model.simulate(ball1, ball2, (0, t_end), config.HERTZ_EXPONENT, config.MIN_SEPARATION, config.SOLVER_METHOD, config.SOLVER_RTOL, config.SOLVER_ATOL)


def close_balls()-> tuple[Ball, Ball]:
  b1 = config.FIRST_BALL
  b2 = config.SECOND_BALL
  b2_close = replace(b2, position=(b1.position[0] + b1.radius + b2.radius + GAP, b1.position[1])) # сумма радиусов + зазор
  return b1, b2_close


def test_collision_happened():
  b1, b2 = close_balls()
  sol = run(b1, b2, 0.01)
  vx_begin = sol.y[2, 0]
  vx_end = sol.y[2, -1]
  assert abs(vx_begin - vx_end) > 0.01 * abs(vx_begin) # скорость изменилась больше чем на 1%


def test_momentum_conserved():
  b1, b2 = close_balls()
  sol = run(b1, b2, 0.01)

  p_x = b1.mass * sol.y[2, :] + b2.mass * sol.y[6, :]
  p_y = b1.mass * sol.y[3, :] + b2.mass * sol.y[7, :]

  scale = np.hypot(p_x[0], p_y[0])

  # Каждый элемент массива должен отличаться от начального значения не больше чем
  # на 1e-8 от масштаба. rtol=0 отключает скрытый относительный допуск.
  np.testing.assert_allclose(p_x, p_x[0], atol=1e-8 * scale, rtol=0)
  np.testing.assert_allclose(p_y, p_y[0], atol=1e-8 * scale, rtol=0)


def test_kinetic_energy_conserved():
  b1, b2 = close_balls()
  sol = run(b1, b2, 0.01)

  ke = 0.5 * b1.mass * (sol.y[2, :]**2 + sol.y[3, :]**2) + 0.5 * b2.mass * (sol.y[6, :]**2 + sol.y[7, :]**2)

  dx = sol.y[0, -1] - sol.y[4, -1]
  dy = sol.y[1, -1] - sol.y[5, -1]
  distance = np.hypot(dx, dy)

  assert distance > b1.radius + b2.radius # шары не перекрываются = не в контакте

  dvx = sol.y[2, -1] - sol.y[6, -1]
  dvy = sol.y[3, -1] - sol.y[7, -1]

  assert dx*dvx + dy*dvy > 0 # скалярное произведение > 0: шары удаляются друг от друга

  assert ke[-1] == pytest.approx(ke[0], rel=1e-7)      # конечная энергия = начальной с точностью до 1e-7