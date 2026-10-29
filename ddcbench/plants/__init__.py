from .base import Plant, rk4_step
from .msd import MassSpringDamper
from .pendulum import Pendulum
from .quadrotor2d import PlanarQuadrotor

PLANTS = {"msd": MassSpringDamper, "pendulum": Pendulum, "quadrotor": PlanarQuadrotor}


def make_plant(name):
    return PLANTS[name]()


__all__ = ["Plant", "rk4_step", "MassSpringDamper", "Pendulum", "PlanarQuadrotor",
           "PLANTS", "make_plant"]
