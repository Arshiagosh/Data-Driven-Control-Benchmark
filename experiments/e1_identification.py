"""E1: open-loop prediction error of the learned models versus horizon.

For each plant and evaluation seed, the models are fitted on the training
data and predict held-out episodes (seed 1000 + s) from the recorded inputs.
Error: RMS over states of (prediction - true state) / std(true state),
averaged over start points, episodes and seeds. Also reported: the coverage
of the GP 2-sigma interval on held-out one-step velocity residuals.
"""
import common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (DEEPC_TINI, EVAL_SEEDS, FIG, GP_POINTS, N_RBF, dataset, save_json)
from ddcbench.data import collect
from ddcbench.models import ResidualGP
from ddcbench.models.hankel import HankelData
from ddcbench.models.koopman import KoopmanModel
from ddcbench.tasks import TASK_NAMES, make_task

H = 20
MODELS = ["Prior model", "Prior + GP", "Koopman (EDMD)", "Linear (DMDc)", "Hankel (DeePC)"]


def rollout_gp(res, x0, U, use_gp):
    x = x0[None, :]
    out = []
    for u in U:
        if use_gp:
            x, _ = res.predict_step(x, u[None, :])
        else:
            x = res.prior_batch(x, u[None, :])
        out.append(x[0])
    return np.array(out)


def main():
    out = {"horizon": H, "models": MODELS, "plants": {}}
    for name in TASK_NAMES:
        task = make_task(name)
        errs = {m: [] for m in MODELS}
        cover = []
        for s in EVAL_SEEDS:
            ds = dataset(task, s)
            hold = collect(task, n_episodes=5, ep_len=100, seed=1000 + s)
            scale = np.vstack([e["Xt"] for e in hold.episodes]).std(0) + 1e-9
            res = ResidualGP(task.plant, n_max=GP_POINTS, seed=s).fit(ds)
            km = KoopmanModel(task.plant, n_rbf=N_RBF, seed=s).fit(ds)
            lin = KoopmanModel(task.plant, n_rbf=0, seed=s).fit(ds)
            hk = HankelData(ds, DEEPC_TINI, H, max_cols=800, seed=s)
            Xk, Uk, Xk1 = hold.transitions()
            m, v, nz = res.gp.predict(res.features(Xk, Uk))
            tg = res.targets(Xk, Uk, Xk1)
            cover.append(float(np.mean(np.abs(tg - m) <= 2 * np.sqrt(v + nz))))
            for e in hold.episodes:
                for st in range(DEEPC_TINI, len(e["U"]) - H, 10):
                    U = e["U"][st:st + H]
                    x0 = e["Xm"][st]
                    truth = e["Xt"][st + 1:st + H + 1]
                    preds = {
                        "Prior model": rollout_gp(res, x0, U, False),
                        "Prior + GP": rollout_gp(res, x0, U, True),
                        "Koopman (EDMD)": km.predict(x0, U),
                        "Linear (DMDc)": lin.predict(x0, U),
                        "Hankel (DeePC)": hk.predict(e["U"][st - DEEPC_TINI:st],
                                                     e["Xm"][st - DEEPC_TINI + 1:st + 1],
                                                     U, lam=1e-2),
                    }
                    for k, p in preds.items():
                        errs[k].append(np.sqrt((((p - truth) / scale) ** 2).mean(1)))
        out["plants"][name] = {
            "rmse_vs_horizon": {k: np.mean(v, 0).tolist() for k, v in errs.items()},
            "gp_2sigma_coverage_mean": float(np.mean(cover)),
            "gp_2sigma_coverage_min": float(np.min(cover)),
        }
        print(name, {k: round(np.mean(v, 0)[-1], 4) for k, v in errs.items()},
              "coverage", round(np.mean(cover), 3))
    save_json(out, "identification.json")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    steps = np.arange(1, H + 1)
    for ax, name in zip(axes, TASK_NAMES):
        for k in MODELS:
            ax.semilogy(steps, out["plants"][name]["rmse_vs_horizon"][k], label=k)
        ax.set_title(name)
        ax.set_xlabel("prediction step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalised RMSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Open-loop prediction error on held-out data (mean over 10 seeds)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/identification.png", dpi=150)


if __name__ == "__main__":
    main()
