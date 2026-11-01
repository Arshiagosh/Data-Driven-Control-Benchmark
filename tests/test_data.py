import numpy as np

from ddcbench.data import collect
from ddcbench.tasks import make_task


def test_collect_size_and_determinism():
    t = make_task("pendulum")
    a = collect(t, n_episodes=3, ep_len=40, seed=5)
    b = collect(t, n_episodes=3, ep_len=40, seed=5)
    assert a.n_samples == 120
    Xa, Ua, _ = a.transitions()
    Xb, Ub, _ = b.transitions()
    assert np.array_equal(Xa, Xb) and np.array_equal(Ua, Ub)


def test_inputs_within_limits():
    t = make_task("quadrotor")
    ds = collect(t, n_episodes=2, ep_len=50, seed=0)
    _, U, _ = ds.transitions()
    assert np.all(U >= t.plant.u_min - 1e-12) and np.all(U <= t.plant.u_max + 1e-12)
