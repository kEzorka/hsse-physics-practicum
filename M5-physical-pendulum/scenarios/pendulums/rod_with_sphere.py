from src.shapes import Pendulum, Rod, Sphere

PENDULUM = Pendulum(
    parts=(
        Rod(mass=0.3, start=(0.0, 0.0), end=(0.0, -0.8)),
        Sphere(mass=2.0, radius=0.06, center=(0.0, -0.86)),
    )
)
"""Маятник из тонкого стержня и сплошного шара."""
