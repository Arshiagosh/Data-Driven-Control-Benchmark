import numpy as np

from ddcbench.data import collect
from ddcbench.models import GPModel, ResidualGP
from ddcbench.tasks import make_task


def test_gp_fits_smooth_function_and_covers():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-3, 3, (200, 1))
    Y = np.sin(Z) + 0.05 * rng.standard_normal((200, 1))
    gp = GPModel(n_max=100, seed=0).fit(Z, Y)
    Zt = np.linspace(-2.5, 2.5, 50)[:, None]
    m, v, nz = gp.predict(Zt)
    assert np.max(np.abs(m - np.sin(Zt))) < 0.1
    Yt = np.sin(Zt) + 0.05 * rng.standard_normal((50, 1))
    cover = np.mean(np.abs(Yt - m) <= 2 * np.sqrt(v + nz))
    assert cover > 0.85


def test_casadi_mean_matches_numpy():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(80, 3))
    Y = np.stack([Z[:, 0] * Z[:, 1], np.cos(Z[:, 2])], 1)
    gp = GPModel(n_max=60, seed=0).fit(Z, Y)
    f = gp.casadi_mean()
    Zt = rng.normal(size=(5, 3))
    m, _, _ = gp.predict(Zt)
    mc = np.array([np.array(f(z)).ravel() for z in Zt])
    assert np.allclose(m, mc, atol=1e-9)


def test_residual_step_function_matches_batch():
    t = make_task("msd")
    ds = collect(t, n_episodes=4, ep_len=50, seed=0)
    res = ResidualGP(t.plant, n_max=60, seed=0).fit(ds)
    F = res.step_function()
    X, U, _ = ds.transitions()
    mean, _ = res.predict_step(X[:5], U[:5])
    mc = np.array([np.array(F(x, u)).ravel() for x, u in zip(X[:5], U[:5])])
    assert np.allclose(mean, mc, atol=1e-9)
