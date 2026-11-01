from dataclasses import dataclass
from math import atan2, hypot, pi

import numpy as np


@dataclass(frozen=True)
class Sphere:
    mass: float
    radius: float
    center: tuple[float, float]

    @property
    def center_of_mass(self):
        return self.center

    @property
    def inertia_cm(self):
        return 2 / 5 * self.mass * self.radius**2

    def outline(self, n=60):
        phi = np.linspace(0, 2 * pi, n)
        return np.column_stack(
            [
                self.center[0] + self.radius * np.cos(phi),
                self.center[1] + self.radius * np.sin(phi),
            ]
        )


@dataclass(frozen=True)
class Disk:
    mass: float
    radius: float
    center: tuple[float, float]

    @property
    def center_of_mass(self):
        return self.center

    @property
    def inertia_cm(self):
        return 1 / 2 * self.mass * self.radius**2

    def outline(self, n=60):
        return Sphere(self.mass, self.radius, self.center).outline(n)


@dataclass(frozen=True)
class Ring:
    mass: float
    radius: float
    center: tuple[float, float]

    @property
    def center_of_mass(self):
        return self.center

    @property
    def inertia_cm(self):
        return self.mass * self.radius**2

    def outline(self, n=60):
        return Sphere(self.mass, self.radius, self.center).outline(n)


@dataclass(frozen=True)
class Rod:
    mass: float
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self):
        return hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def center_of_mass(self):
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    @property
    def inertia_cm(self):
        return 1 / 12 * self.mass * self.length**2

    def outline(self, n=2):
        return np.array([self.start, self.end])


@dataclass(frozen=True)
class Plate:
    mass: float
    width: float
    height: float
    center: tuple[float, float]

    @property
    def center_of_mass(self):
        return self.center

    @property
    def inertia_cm(self):
        return 1 / 12 * self.mass * (self.width**2 + self.height**2)

    def outline(self, n=5):
        cx, cy = self.center
        w, h = self.width / 2, self.height / 2
        return np.array(
            [
                (cx - w, cy - h),
                (cx + w, cy - h),
                (cx + w, cy + h),
                (cx - w, cy + h),
                (cx - w, cy - h),
            ]
        )


@dataclass(frozen=True)
class Pendulum:
    parts: tuple

    def __post_init__(self):
        if not self.parts:
            raise ValueError("Pendulum must contain at least one part.")
        for part in self.parts:
            if part.mass <= 0:
                raise ValueError(f"Part mass must be positive: {part}")
        if self.com_distance <= 0:
            raise ValueError(
                "Center of mass coincides with the pivot: "
                "the pendulum is in neutral equilibrium and does not oscillate."
            )

    @property
    def mass(self):
        return sum(part.mass for part in self.parts)

    @property
    def center_of_mass(self):
        weighted = sum(part.mass * np.array(part.center_of_mass) for part in self.parts)
        return tuple(weighted / self.mass)

    @property
    def com_distance(self):
        return hypot(*self.center_of_mass)

    @property
    def inertia(self):
        return sum(
            part.inertia_cm
            + part.mass * (part.center_of_mass[0] ** 2 + part.center_of_mass[1] ** 2)
            for part in self.parts
        )

    @property
    def drawing_angle(self):
        x, y = self.center_of_mass
        return atan2(x, -y)

    @property
    def reduced_length(self):
        return self.inertia / (self.mass * self.com_distance)

    def describe(self):
        x, y = self.center_of_mass
        return (
            f"масса m = {self.mass:.4g} кг\n"
            f"центр масс = ({x:.4g}, {y:.4g}) м, d = {self.com_distance:.4g} м\n"
            f"момент инерции относительно подвеса I = {self.inertia:.4g} кг*м^2\n"
            f"приведённая длина l = I/(m d) = {self.reduced_length:.4g} м"
        )
