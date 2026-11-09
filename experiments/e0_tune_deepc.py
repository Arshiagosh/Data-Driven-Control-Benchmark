"""E0: pick the DeePC regularisation weight on the tuning seed only.

Rule (fixed before looking at the evaluation seeds): among the weights whose
run did not diverge and whose mean QP solve time is below the sampling time
(50 ms), take the one with the lowest tracking cost + 10 x the largest
constraint violation (an exact-penalty style score, so a weight that buys
tracking cost with violations is not picked).
"""
import common  # noqa: F401  (sets sys.path)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (DEEPC_LAMBDAS, DEEPC_TINI, FIG, TUNE_DATA_SEED, TUNE_NOISE_SEED,
                    dataset, save_json)
from ddcbench.controllers.deepc import DeePC
from ddcbench.metrics import evaluate
from ddcbench.simulate import run_closed_loop
from ddcbench.tasks import TASK_NAMES, make_task


def main():
    out = {"lambdas": DEEPC_LAMBDAS, "runs": {}, "chosen": {}}
    for name in TASK_NAMES:
        task = make_task(name)
        ds = dataset(task, TUNE_DATA_SEED)
        rows = []
        for lam in DEEPC_LAMBDAS:
            c = DeePC(task, ds, Tini=DEEPC_TINI, lam_g=lam, seed=TUNE_DATA_SEED)
            m = evaluate(task, run_closed_loop(task, c, seed=TUNE_NOISE_SEED))
            rows.append(m)
            print(name, lam, m)
        dt_ms = task.plant.dt * 1e3
        score = [np.inf if (m["diverged"] or m["t_mean_ms"] >= dt_ms)
                 else m["cost"] + 10 * m["viol_max"] for m in rows]
        out["runs"][name] = rows
        out["chosen"][name] = DEEPC_LAMBDAS[int(np.argmin(score))]
    save_json(out, "deepc_tuning.json")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, name in zip(axes, TASK_NAMES):
        cost = [m["cost"] for m in out["runs"][name]]
        vs = [m["viol_steps"] for m in out["runs"][name]]
        ax.semilogx(DEEPC_LAMBDAS, cost, "o-", label="tracking cost")
        ax.set_xlabel(r"$\lambda_g$")
        ax.set_title(name)
        ax.axvline(out["chosen"][name], color="0.6", ls="--", lw=1)
        ax2 = ax.twinx()
        ax2.semilogx(DEEPC_LAMBDAS, vs, "s--", color="C3", label="violation steps")
        ax2.set_ylabel("violation steps", color="C3")
        ax.set_ylabel("cost")
        ax.set_yscale("log")
    fig.suptitle("DeePC regularisation sweep (tuning seed; dashed = chosen)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/deepc_sweep.png", dpi=150)
    print("chosen", out["chosen"])


if __name__ == "__main__":
    main()
