from scenarios.modes import free
from scenarios.pendulums import rod_with_sphere

# =============================================================================
# 1. ВЫБРАННЫЕ МАЯТНИК И РЕЖИМ
# =============================================================================

PENDULUM = rod_with_sphere.PENDULUM

INITIAL_ANGLE = free.INITIAL_ANGLE
INITIAL_ANGULAR_VELOCITY = free.INITIAL_ANGULAR_VELOCITY
VISCOUS_DAMPING = free.VISCOUS_DAMPING
SIMULATION_TIME = free.SIMULATION_TIME

G = 9.81
"""Ускорение свободного падения, м/с²."""


# =============================================================================
# 2. ЧИСЛЕННАЯ СХЕМА
# =============================================================================

SOLVER_METHOD = "DOP853"

SOLVER_RTOL = 1e-10

SOLVER_ATOL = 1e-12
