"""Shared settings for the experiments.

Seeds: evaluation seeds 0-9 draw the training data (seed s) and the
closed-loop measurement noise (500 + s). Seed 100 (data) / 600 (noise) is a
separate tuning seed used only to pick the DeePC regularisation weight.
Held-out identification data use seed 1000 + s.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ddcbench.controllers import nominal_mpc, oracle_mpc          # noqa: E402
from ddcbench.controllers.deepc import DeePC                       # noqa: E402
from ddcbench.controllers.gp_mpc import GPMPC                      # noqa: E402
from ddcbench.controllers.koopman_mpc import KoopmanMPC            # noqa: E402
from ddcbench.data import collect                                  # noqa: E402
from ddcbench.models import ResidualGP                             # noqa: E402
from ddcbench.models.koopman import KoopmanModel                   # noqa: E402

EVAL_SEEDS = list(range(10))
TUNE_DATA_SEED, TUNE_NOISE_SEED = 100, 600
N_EPISODES, EP_LEN = 15, 100          # 1500 samples per dataset
GP_POINTS = 150
N_RBF = 40
DEEPC_TINI = 3
DEEPC_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CONTROLLERS = ["Nominal MPC", "Oracle MPC", "GP-MPC", "DeePC", "Koopman MPC"]

FIG = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def dataset(task, seed):
    return collect(task, n_episodes=N_EPISODES, ep_len=EP_LEN, seed=seed)


def tuned_lambda(plant_name):
    path = os.path.join(RESULTS, "deepc_tuning.json")
    with open(path) as f:
        return json.load(f)["chosen"][plant_name]


def build_controllers(task, ds, seed, lam_g):
    res = ResidualGP(task.plant, n_max=GP_POINTS, seed=seed).fit(ds)
    km = KoopmanModel(task.plant, n_rbf=N_RBF, seed=seed).fit(ds)
    return [
        nominal_mpc(task),
        oracle_mpc(task),
        GPMPC(task, res),
        DeePC(task, ds, Tini=DEEPC_TINI, lam_g=lam_g, seed=seed),
        KoopmanMPC(task, km),
    ]


def save_json(obj, name):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)
