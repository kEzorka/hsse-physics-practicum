import pytest
import numpy as np
from dataclasses import replace

import config
import model

from state import Ball

def run(ball1, ball2, t_end):
  return model.simulate(ball1, ball2, (0, t_end), config.HERTZ_EXPONENT, config.MIN_SEPARATION, config.SOLVER_METHOD, config.SOLVER_RTOL, config.SOLVER_ATOL)

def close_balls()-> tuple[Ball, Ball]:
  b1 = config.FIRST_BALL
  b2 = config.SECOND_BALL
  b2_close = replace(b2, position=(0.058, 0.0))
  return b1, b2_close

def test_collision_happened():
  b1, b2 = close_balls()
  sol = run(b1, b2, 0.01)
  vx_begin = sol.y[2, 0]
  vx_end = sol.y[2, -1]
  assert abs(vx_begin - vx_end) > 0.01 * abs(vx_begin)

def test_momentum_conserved():
  b1, b2 = close_balls()
  sol = run(b1, b2, 0.01)

  p_x = b1.mass * sol.y[2, :] + b2.mass * sol.y[6, :]
  p_y = b1.mass * sol.y[3, :] + b2.mass * sol.y[7, :]

  scale = np.hypot(p_x[0], p_y[0])

  np.testing.assert_allclose(p_x, p_x[0], atol=1e-8 * scale, rtol=0)
  np.testing.assert_allclose(p_y, p_y[0], atol=1e-8 * scale, rtol=0)
