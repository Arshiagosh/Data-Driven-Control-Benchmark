import numpy as np

from ddcbench.controllers import nominal_mpc, oracle_mpc
from ddcbench.controllers.deepc import DeePC
from ddcbench.controllers.koopman_mpc import KoopmanMPC
from ddcbench.data import collect
from ddcbench.metrics import evaluate
from ddcbench.models.koopman import KoopmanModel
from ddcbench.simulate import run_closed_loop
from ddcbench.tasks import make_task


def noise_free(name):
    t = make_task(name)
    t.noise_std = np.zeros(t.plant.nx)
    return t


def test_oracle_has_no_offset_and_nominal_does():
    t = noise_free("msd")
    mo = evaluate(t, run_closed_loop(t, oracle_mpc(t)))
    mn = evaluate(t, run_closed_loop(t, nominal_mpc(t)))
    assert mo["ss_err"] < 2e-3
    assert mn["ss_err"] > 10 * mo["ss_err"]


def test_deepc_tracks_noise_free_lti():
    t = noise_free("msd")
    ds = collect(t, n_episodes=8, ep_len=100, seed=0)
    m = evaluate(t, run_closed_loop(t, DeePC(t, ds, lam_g=1e-4, Tini=2)))
    assert m["ss_err"] < 2e-3 and m["fails"] == 0


def test_koopman_mpc_runs_on_pendulum():
    t = make_task("pendulum")
    ds = collect(t, n_episodes=6, ep_len=100, seed=0)
    km = KoopmanModel(t.plant, n_rbf=20).fit(ds)
    m = evaluate(t, run_closed_loop(t, KoopmanMPC(t, km)))
    assert not m["diverged"] and m["ss_err"] < 0.05
