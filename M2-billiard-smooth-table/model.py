import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult


def effective_radius(radius1, radius2) -> float:
    return 1 / (1 / radius1 + 1 / radius2)


def reduced_mass(m1, m2) -> float:
    return 1 / (1 / m1 + 1 / m2)


def hertz_stiffness(
    young_modulus1, poisson_ratio1, radius1,
    young_modulus2, poisson_ratio2, radius2) -> float:

    E_eff = (1 / 
            ((1 - poisson_ratio1 ** 2) / young_modulus1
            + 
            (1 - poisson_ratio2 ** 2) / young_modulus2)
            )
    R_eff = effective_radius(radius1, radius2)
    stiffness = (4/3) * E_eff * np.sqrt(R_eff)

    return stiffness


def contact_force(compression_depth, stiffness, exponent) -> float:
    compression_depth = max(compression_depth, 0.0)
    return stiffness * compression_depth ** exponent


def max_compression(
    reduced_mass_value, normal_relative_speed, stiffness, exponent) -> float:
    power = exponent + 1
    return (
        (reduced_mass_value * normal_relative_speed ** 2 * power)
            / (2 * stiffness)
        ) ** (1 / power)


def contact_timescale(
    m1: float, m2: float,
    v1: tuple[float, float], v2: tuple[float, float],
    stiffness: float, exponent: float, min_relative_speed: float) -> np.float64:

    relative_speed = np.linalg.norm(np.subtract(v1, v2))
    if relative_speed < min_relative_speed:
        max_x = np.inf
    else:
        max_x = max_compression(
            reduced_mass(m1, m2),
            relative_speed,
            stiffness,
            exponent)
    
    return 2 * max_x / relative_speed


def equations_of_motion(
    t,
    y, 
    m1, m2, radius1, radius2,
    stiffness, compression_exponent, min_separation) -> np.ndarray:

    r1 = y[0:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]


    separation_vector = r2 - r1

    separation = np.linalg.norm(separation_vector)
    if separation <= min_separation:
        raise ValueError(
            "Balls are at the same position,"
            " which is physically impossible."
            f" separation = {separation}, t = {t}")
    
    direction = separation_vector / separation

    compression_depth = radius1 + radius2 - separation
    
    compression_force = contact_force(
        compression_depth, stiffness, compression_exponent)
    
    a1 = - compression_force / m1 * direction
    a2 = compression_force / m2 * direction

    return np.concatenate([v1, a1, v2, a2])


def simulate(
    ball1, ball2, t_span, 
    compression_exponent,
    min_separation, min_relative_speed,
    solver_method, solver_rtol, solver_atol) -> OptimizeResult:

    y0 = np.concatenate([
        ball1.position, ball1.velocity,
        ball2.position, ball2.velocity
    ])

    stiffness = hertz_stiffness(
        ball1.young_modulus, ball1.poisson_ratio, ball1.radius,
        ball2.young_modulus, ball2.poisson_ratio, ball2.radius
    )
    max_step = contact_timescale(
        ball1.mass, ball2.mass,
        ball1.velocity, ball2.velocity,
        stiffness, compression_exponent, min_relative_speed)

    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        args=(
            ball1.mass, ball2.mass,
            ball1.radius, ball2.radius,
            stiffness, compression_exponent, min_separation),
        method=solver_method,
        rtol=solver_rtol,
        atol=solver_atol,
        dense_output=True,
        max_step=max_step
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed: {sol.message}. "
            f"Status: {sol.status}, t_span={t_span}, stopped at t={sol.t[-1]}")

    return sol
