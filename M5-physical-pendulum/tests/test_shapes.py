import pytest
from scenarios.pendulums.rod_with_sphere import PENDULUM
from src.shapes import Disk, Pendulum, Plate, Ring, Rod, Sphere


@pytest.mark.parametrize(
    ("part", "expected_inertia"),
    (
        (Sphere(mass=5.0, radius=2.0, center=(0.0, 0.0)), 8.0),
        (Disk(mass=5.0, radius=2.0, center=(0.0, 0.0)), 10.0),
        (Ring(mass=5.0, radius=2.0, center=(0.0, 0.0)), 20.0),
        (Rod(mass=3.0, start=(0.0, 0.0), end=(4.0, 0.0)), 4.0),
        (
            Plate(
                mass=3.0,
                width=4.0,
                height=2.0,
                center=(0.0, 0.0),
            ),
            5.0,
        ),
    ),
)
def test_primitive_moment_of_inertia_about_center(part, expected_inertia):
    assert part.inertia_cm == pytest.approx(expected_inertia)


def test_composite_pendulum_uses_parallel_axis_theorem():
    assert PENDULUM.mass == pytest.approx(2.3)
    assert PENDULUM.center_of_mass == pytest.approx((0.0, -0.8))
    assert PENDULUM.inertia == pytest.approx(1.54608)


def test_pendulum_requires_at_least_one_part():
    with pytest.raises(ValueError, match="at least one part"):
        Pendulum(parts=())


@pytest.mark.parametrize("mass", (0.0, -1.0))
def test_pendulum_rejects_non_positive_part_mass(mass):
    part = Sphere(mass=mass, radius=0.1, center=(0.0, -1.0))

    with pytest.raises(ValueError, match="mass must be positive"):
        Pendulum(parts=(part,))


def test_pendulum_rejects_center_of_mass_at_pivot():
    lower_part = Sphere(mass=1.0, radius=0.1, center=(0.0, -1.0))
    upper_part = Sphere(mass=1.0, radius=0.1, center=(0.0, 1.0))

    with pytest.raises(ValueError, match="Center of mass coincides with the pivot"):
        Pendulum(parts=(lower_part, upper_part))
