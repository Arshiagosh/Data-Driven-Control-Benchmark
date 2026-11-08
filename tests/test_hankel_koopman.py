import numpy as np

from ddcbench.data import collect
from ddcbench.models.hankel import HankelData
from ddcbench.models.koopman import KoopmanModel
from ddcbench.simulate import true_step
from ddcbench.tasks import make_task


def noise_free_msd():
    t = make_task("msd")
    t.noise_std = np.zeros(2)
    return t


def test_fundamental_lemma_noise_free_lti():
    """With noise-free LTI data the Hankel predictor is exact."""
    t = noise_free_msd()
    ds = collect(t, n_episodes=6, ep_len=100, seed=0)
    H = HankelData(ds, Tini=2, N=20)
    assert H.ncol >= H.min_columns(2)
    rng = np.random.default_rng(3)
    x = np.array([0.4, -0.2])
    U = rng.uniform(-2, 2, (22, 1))
    Y = []
    for u in U:
        x = true_step(t.plant, x, u)
        Y.append(x)
    Y = np.array(Y)
    pred = H.predict(U[:2], Y[:2], U[2:], lam=1e-10)
    assert np.max(np.abs(pred - Y[2:])) < 1e-5


def test_dmd_recovers_linear_map():
    t = noise_free_msd()
    ds = collect(t, n_episodes=4, ep_len=100, seed=1)
    km = KoopmanModel(t.plant, n_rbf=0, ridge=0.0).fit(ds)
    X, U, X1 = ds.transitions()
    Z = km.lift(X)
    pred = (Z @ km.A.T + U @ km.B.T) @ km.C.T
    assert np.max(np.abs(pred - X1)) < 1e-8
