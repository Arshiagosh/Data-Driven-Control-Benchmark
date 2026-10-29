import casadi as ca
import numpy as np
import pytest

from ddcbench.plants import make_plant, rk4_step


@pytest.mark.parametrize("name", ["msd", "pendulum"])
def test_equilibrium_input_holds_state(name):
    p = make_plant(name)
    for params in (p.true_params, p.prior_params):
        xr = np.array([0.3, 0.0])
        u = p.equilibrium_input(xr, params)
        assert np.allclose(p.f(xr, u, params), 0, atol=1e-12)


def test_quadrotor_hover_without_wind():
    p = make_plant("quadrotor")
    x = np.array([0.5, 1.0, 0, 0, 0, 0])
    u = p.equilibrium_input(x, p.prior_params)
    assert np.allclose(p.f(x, u, p.prior_params), 0, atol=1e-12)


@pytest.mark.parametrize("name", ["msd", "pendulum", "quadrotor"])
def test_numpy_and_casadi_agree(name):
    p = make_plant(name)
    rng = np.random.default_rng(1)
    x = rng.normal(size=p.nx) * 0.3
    u = (p.u_min + p.u_max) / 2 + 0.1
    xs, us = ca.SX.sym("x", p.nx), ca.SX.sym("u", p.nu)
    F = ca.Function("F", [xs, us], [rk4_step(p, xs, us, p.true_params, p.dt, 3, lib=ca)])
    a = rk4_step(p, x, u, p.true_params, p.dt, 3)
    b = np.array(F(x, u)).ravel()
    assert np.allclose(a, b, atol=1e-12)
