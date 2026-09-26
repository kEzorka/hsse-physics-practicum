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


def time_to_touch(ball1, ball2) -> float | None:
  d0 = np.array(ball2.position) - np.array(ball1.position)
  v_rel = np.array(ball2.velocity) - np.array(ball1.velocity)

  a = np.dot(v_rel, v_rel)
  b = 2 * np.dot(d0, v_rel)
  c = np.dot(d0, d0) - (ball1.radius + ball2.radius) ** 2

  if a == 0:
    return None  # шары не сближаются: относительная скорость равна 0
  
  discriminant = b**2 - 4*a*c
  if discriminant < 0:
    return None  # шары не пересекутся

  t1 = (-b - np.sqrt(discriminant)) / (2*a)
  t2 = (-b + np.sqrt(discriminant)) / (2*a) 

  if t1 >= 0:
    return t1
  elif t2 >= 0:
    return t2
  else:
    return None  # шары не пересекутся




def find_t_end(ball1, ball2) -> float | None:
  t_touch = time_to_touch(ball1, ball2)
  if t_touch is None:
    return None

  # множитель 3 и добавка 0,005 с подобраны эмпирически: нужно досчитать
  # до касания, пережить контакт и дать шарам разойтись.
  # После решения B4 заменить на t_touch + время контакта + время разлёта.

  t_end = 3 * t_touch + 0.005
  return t_end



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

  t_end = find_t_end(b1, b2)

  sol = run(b1, b2, t_end)

  vx_begin = sol.y[2, 0]
  vx_end = sol.y[2, -1]
  v_relative = np.hypot(sol.y[2, 0] - sol.y[6, 0], sol.y[3, 0] - sol.y[7, 0])
  
  # скорость изменилась больше чем на 1%
  assert abs(vx_begin - vx_end) > 0.01 * abs(v_relative)



@pytest.mark.parametrize("ball1, ball2, offset", CASES)
def test_momentum_conserved(ball1, ball2, offset):
  b1, b2 = close_balls(ball1, ball2, offset)

  t_end = find_t_end(b1, b2)
  sol = run(b1, b2, t_end)

  p_x = b1.mass * sol.y[2, :] + b2.mass * sol.y[6, :]
  p_y = b1.mass * sol.y[3, :] + b2.mass * sol.y[7, :]

  scale = np.hypot(p_x[0], p_y[0])

  # Каждый элемент массива должен отличаться от начального значения не 
  # больше чем на 1e-8 от масштаба. rtol=0 отключает скрытый относительный 
  # допуск.
  np.testing.assert_allclose(p_x, p_x[0], atol=1e-8 * scale, rtol=0)
  np.testing.assert_allclose(p_y, p_y[0], atol=1e-8 * scale, rtol=0)



@pytest.mark.parametrize("ball1, ball2, offset", CASES)
def test_kinetic_energy_conserved(ball1, ball2, offset):
  b1, b2 = close_balls(ball1, ball2, offset)
  t_end = find_t_end(b1, b2)

  sol = run(b1, b2, t_end)

  ke = 0.5 * b1.mass * (sol.y[2, :]**2 + sol.y[3, :]**2) + 0.5 * b2.mass * (sol.y[6, :]**2 + sol.y[7, :]**2)

  assert_separated(sol, b1, b2)

  # конечная энергия = начальной с точностью до 1e-7
  assert ke[-1] == pytest.approx(ke[0], rel=1e-7) 



def assert_separated(sol, ball1, ball2):
  dx = sol.y[0, -1] - sol.y[4, -1]
  dy = sol.y[1, -1] - sol.y[5, -1]
  distance = np.hypot(dx, dy)

  # шары не перекрываются = не в контакте  
  assert distance > ball1.radius + ball2.radius

  dvx = sol.y[2, -1] - sol.y[6, -1]
  dvy = sol.y[3, -1] - sol.y[7, -1]


  # скалярное произведение > 0: шары удаляются друг от друга
  assert dx*dvx + dy*dvy > 0 



def test_A1_head_on_equal_masses_resting_target():
  b1, b2 = close_balls(config.FIRST_BALL, config.SECOND_BALL, 0.0)
  b2 = replace(b2, velocity=(0.0, 0.0)) # второй шар покоится
  v0 = b1.velocity[0]

  assert b1.mass == b2.mass
  assert b1.velocity[1] == 0

  t_end = find_t_end(b1, b2)
  sol = run(b1, b2, t_end)

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

  t_end = find_t_end(b1, b2)
  sol = run(b1, b2, t_end)

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



@pytest.mark.parametrize("mass_ratio", [0.2, 1.0, 5.0])
@pytest.mark.parametrize("v1, v2", [(2.0, 0.5), (1.0, -0.5), (3.0, 1.0)])
def test_A3_head_on_arbitrary_velocities(v1, v2, mass_ratio):
  b1, b2 = close_balls(config.FIRST_BALL, config.SECOND_BALL, 0.0)
  b1 = replace(b1, velocity=(v1, 0.0))
  b2 = replace(b2, velocity=(v2, 0.0), density=b2.density * mass_ratio)

  assert b1.velocity[1] == 0

  t_end = find_t_end(b1, b2)
  sol = run(b1, b2, t_end)

  assert_separated(sol, b1, b2)

  m1, m2 = b1.mass, b2.mass
  v1_expected = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
  v2_expected = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

  v1_x = sol.y[2, -1]
  v1_y = sol.y[3, -1]
  v2_y = sol.y[7, -1]
  v2_x = sol.y[6, -1]

  assert v1_x == pytest.approx(v1_expected, abs=1e-6 * abs(v1 - v2))
  assert v1_y == pytest.approx(0.0, abs=1e-6 * abs(v1 - v2))
  assert v2_x == pytest.approx(v2_expected, abs=1e-6 * abs(v1 - v2))
  assert v2_y == pytest.approx(0.0, abs=1e-6 * abs(v1 - v2))



@pytest.mark.parametrize("mass_ratio", [0.2, 1.0, 5.0])
@pytest.mark.parametrize("v1, v2", [(0.5, 1.0), (1.0, 1.0)])
def test_A4_no_collision_when_not_catching_up(v1, v2, mass_ratio):

  assert v1 <= v2

  b1, b2 = close_balls(config.FIRST_BALL, config.SECOND_BALL, 0.0)
  b1 = replace(b1, velocity=(v1, 0.0))
  b2 = replace(b2, velocity=(v2, 0.0), density=b2.density * mass_ratio)

  dist_begin = np.hypot(b1.position[0] - b2.position[0], b1.position[1] - b2.position[1])

  tol_d = 1e-9 * GAP
  tol_v = 1e-6 * max(abs(v1), abs(v2))
  
  assert b1.velocity[1] == 0

  t_end = find_t_end(b1, b2)

  assert t_end is None

  # Касания нет, поэтому время берётся произвольным, лишь бы хватило убедиться, что шары не сближаются.
  sol = run(b1, b2, 0.01)

  x1 = sol.y[0, -1]
  x2 = sol.y[4, -1]
  y1 = sol.y[1, -1]
  y2 = sol.y[5, -1]

  dist_end = np.hypot(x1 - x2, y1 - y2)

  assert dist_end >= dist_begin - tol_d
  v1_x = sol.y[2, -1]
  v1_y = sol.y[3, -1]
  v2_x = sol.y[6, -1]
  v2_y = sol.y[7, -1]

  assert v1_x == pytest.approx(v1, abs=tol_v)
  assert v1_y == pytest.approx(0.0, abs=tol_v)
  assert v2_x == pytest.approx(v2, abs=tol_v)
  assert v2_y == pytest.approx(0.0, abs=tol_v)



GRAZING_BUG = pytest.mark.xfail(reason="решатель перешагивает короткий контакт при скользящем ударе, баг в model.py")

@pytest.mark.parametrize("v0", [2.0, 1.0, 3.0])
@pytest.mark.parametrize("b_fraction", [
    0.0, 0.25, 0.5,
    pytest.param(0.75, marks=GRAZING_BUG),
    pytest.param(0.99, marks=GRAZING_BUG),
])
def test_A5_oblique_equal_masses(v0, b_fraction):
  b1 = replace(config.FIRST_BALL, velocity=(v0, 0.0))
  b2 = replace(config.SECOND_BALL, velocity=(0.0, 0.0))

  b = b_fraction * (b1.radius + b2.radius)

  assert b1.mass == b2.mass
  
  b1, b2 = close_balls(b1, b2, b)

  sina = b / (b1.radius + b2.radius)
  cosa = np.sqrt(1 - sina**2)

  v1_x_expected = v0 * sina * sina
  v1_y_expected = v0 * sina * (-cosa)
  v2_x_expected = v0 * cosa * cosa  
  v2_y_expected = v0 * cosa * sina

  t_end = find_t_end(b1, b2)

  sol = run(b1, b2, t_end)

  assert_separated(sol, b1, b2)

  v1_x = sol.y[2, -1]
  v1_y = sol.y[3, -1]
  v2_x = sol.y[6, -1]
  v2_y = sol.y[7, -1]

  # допуск 1e-2: формулы выведены для абсолютно твёрдых шаров,
  # а в модели шары деформируются, поэтому в момент контакта
  # расстояние между центрами меньше суммы радиусов на величину сжатия
  assert v1_x == pytest.approx(v1_x_expected, abs=1e-2 * abs(v0))
  assert v1_y == pytest.approx(v1_y_expected, abs=1e-2 * abs(v0))
  assert v2_x == pytest.approx(v2_x_expected, abs=1e-2 * abs(v0))
  assert v2_y == pytest.approx(v2_y_expected, abs=1e-2 * abs(v0))

  # скалярное произведение = 0
  assert v1_x * v2_x + v1_y * v2_y == pytest.approx(0.0, abs=1e-6 * abs(v0)**2) 



# Известный баг модели: результат зависит от длины интервала интегрирования.
# Решатель с адаптивным шагом подбирает размер шага от ширины окна, поэтому
# при широком окне он перешагивает контакт: шары либо пролетают насквозь,
# либо получают завышенные скорости.
# Тест начнёт проходить, когда в model.py появится ограничение шага
# (max_step) около момента касания.
@pytest.mark.xfail(reason="решатель перешагивает контакт при широком окне интегрирования, баг в model.py")
@pytest.mark.parametrize("t_end", [0.02, 0.05, 0.1])
def test_result_does_not_depend_on_integration_window(t_end):
  ball1, ball2, offset = CASES[0].values
  b1, b2 = close_balls(ball1, ball2, offset)

  sol = run(b1, b2, t_end)

  ke = 0.5 * b1.mass * (sol.y[2, :]**2 + sol.y[3, :]**2) + 0.5 * b2.mass * (sol.y[6, :]**2 + sol.y[7, :]**2)

  assert_separated(sol, b1, b2)
  assert ke[-1] == pytest.approx(ke[0], rel=1e-7)



@pytest.mark.parametrize("mass_ratio", [0.2, 1.0, 5.0])
@pytest.mark.parametrize("b_fraction", [
    0.0, 0.25, 0.5,
    pytest.param(0.75, marks=GRAZING_BUG),
    pytest.param(0.99, marks=GRAZING_BUG),
])
def test_A6_oblique_arbitrary_masses(mass_ratio, b_fraction):
  v0 = 1.0
  b1 = replace(config.FIRST_BALL, velocity=(v0, 0.0))
  b2 = replace(config.SECOND_BALL, velocity=(0.0, 0.0), density=config.SECOND_BALL.density * mass_ratio)

  b = b_fraction * (b1.radius + b2.radius)
  
  b1, b2 = close_balls(b1, b2, b)

  sina = b / (b1.radius + b2.radius)
  cosa = np.sqrt(1 - sina**2)

  A = cosa * (b1.mass - b2.mass) / (b1.mass + b2.mass)
  B = cosa * 2 * b1.mass / (b1.mass + b2.mass)

  v1_x_expected = v0 * (A * cosa + sina * sina)
  v1_y_expected = v0 * (A * sina - sina * cosa)
  v2_x_expected = v0 * B * cosa  
  v2_y_expected = v0 * B * sina

  t_end = find_t_end(b1, b2)

  sol = run(b1, b2, t_end)

  assert_separated(sol, b1, b2)

  v1_x = sol.y[2, -1]
  v1_y = sol.y[3, -1]
  v2_x = sol.y[6, -1]
  v2_y = sol.y[7, -1]

  # допуск 1e-2: формулы выведены для абсолютно твёрдых шаров,
  # а в модели шары деформируются, поэтому в момент контакта
  # расстояние между центрами меньше суммы радиусов на величину сжатия
  assert v1_x == pytest.approx(v1_x_expected, abs=1e-2 * abs(v0))
  assert v1_y == pytest.approx(v1_y_expected, abs=1e-2 * abs(v0))
  assert v2_x == pytest.approx(v2_x_expected, abs=1e-2 * abs(v0))
  assert v2_y == pytest.approx(v2_y_expected, abs=1e-2 * abs(v0))

  v1 = np.array([v1_x, v1_y])
  v2 = np.array([v2_x, v2_y])


  K_before = (b1.mass * np.dot(b1.velocity, b1.velocity)) / 2 + b2.mass * np.dot(b2.velocity, b2.velocity) / 2
  K_new = (b1.mass * np.dot(v1, v1)) / 2 + (b2.mass * np.dot(v2, v2)) / 2
  assert K_before == pytest.approx(K_new, rel=1e-2)
