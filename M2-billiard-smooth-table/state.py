from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class Ball:
    position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    density: float
    young_modulus: float
    poisson_ratio: float

    @property
    def mass(self):
        volume = 4/3 * pi * self.radius**3
        return self.density * volume
