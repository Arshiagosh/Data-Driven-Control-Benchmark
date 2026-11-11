"""E3: figures and tables from the saved results (no simulation here)."""
import common  # noqa: F401
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CONTROLLERS, FIG, RESULTS, ROOT
from ddcbench.tasks import TASK_NAMES, make_task

COLORS = {"Nominal MPC": "C7", "Oracle MPC": "k", "GP-MPC": "C0", "DeePC": "C1",
          "Koopman MPC": "C2"}
STYLES = {"Oracle MPC": "--"}


def load_traj(name):
    d = np.load(f"{RESULTS}/traj_{name}.npz")
    return {c: (d[f"{c}|X"], d[f"{c}|U"]) for c in CONTROLLERS}


def plot_trajectories(name):
    task = make_task(name)
    p = task.plant
    tr = load_traj(name)
    ref = task.reference()[:task.T + 1]
    t = np.arange(task.T + 1) * p.dt
    shown = list(task.track_idx) + [c for c in task.con_idx if c not in task.track_idx]
    n = len(shown) + 1
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 1.9 * n), sharex=True)
    for ax, i in zip(axes, shown):
        for c in CONTROLLERS:
            ax.plot(t, tr[c][0][:, i], STYLES.get(c, "-"), color=COLORS[c], lw=1.2, label=c)
        if i in task.track_idx:
            ax.plot(t, ref[:, i], ":", color="0.5", lw=1, label="reference")
        if i in task.con_idx:
            j = task.con_idx.index(i)
            for b in (task.x_min_c[j], task.x_max_c[j]):
                ax.axhline(b, color="C3", lw=0.8, ls="-.")
        ax.set_ylabel(p.state_names[i])
        ax.grid(alpha=0.3)
    ax = axes[-1]
    tu = t[:-1]
    for c in CONTROLLERS:
        ax.step(tu, tr[c][1][:, 0], where="post", color=COLORS[c], lw=1,
                ls=STYLES.get(c, "-"))
    ax.set_ylabel(p.input_names[0])
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="best")
    fig.suptitle(f"{name}: closed loop, evaluation seed 0 (true state)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/closed_loop_{name}.png", dpi=150)
    plt.close(fig)


def plot_quadrotor_plane():
    task = make_task("quadrotor")
    tr = load_traj("quadrotor")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in CONTROLLERS:
        X = tr[c][0]
        ax.plot(X[:, 0], X[:, 1], STYLES.get(c, "-"), color=COLORS[c], lw=1.6, label=c)
    for _, xr in task.ref_points:
        ax.plot(xr[0], xr[1], "*", ms=14, color="C3")
    ax.plot(0, 0, "o", color="0.3")
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_z$ [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Planar quadrotor with unknown mass, drag and wind\n"
                 "(start at origin, stars = set-points, seed 0)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/hero_quadrotor.png", dpi=150)
    plt.close(fig)


def plot_summary(res):
    metrics = [("cost", "tracking cost"), ("ss_err", "steady-state error"),
               ("viol_steps", "steps with violation"), ("t_mean_ms", "solve time [ms]")]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(11, 8.5))
    x = np.arange(len(CONTROLLERS))
    for j, name in enumerate(TASK_NAMES):
        S = res["plants"][name]["summary"]
        for i, (k, lab) in enumerate(metrics):
            ax = axes[i, j]
            mu = [S[c][k]["mean"] for c in CONTROLLERS]
            sd = [S[c][k]["std"] for c in CONTROLLERS]
            ax.bar(x, mu, yerr=sd, color=[COLORS[c] for c in CONTROLLERS], capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace(" MPC", "\nMPC") for c in CONTROLLERS] if i == 3
                               else [], fontsize=7)
            if k in ("t_mean_ms", "ss_err"):
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(name)
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Closed-loop metrics, mean and std over 10 seeds")
    fig.tight_layout()
    fig.savefig(f"{FIG}/summary.png", dpi=150)
    plt.close(fig)


def fmt(m, k, digits=3):
    return f"{m[k]['mean']:.{digits}f} ± {m[k]['std']:.{digits}f}"


def tables(res, ident):
    md = []
    tex = []
    for name in TASK_NAMES:
        S = res["plants"][name]["summary"]
        md.append(f"\n**{name}** (DeePC weight λ_g = {res['plants'][name]['deepc_lambda_g']:g})\n")
        md.append("| Controller | Tracking cost | Steady-state error | Max violation | "
                  "Steps violated | Solve time mean / max [ms] |")
        md.append("|---|---|---|---|---|---|")
        tex.append(r"\multicolumn{6}{l}{\textbf{" + name + r"}} \\")
        for c in CONTROLLERS:
            m = S[c]
            div = f" ({m['diverged_runs']} diverged)" if m["diverged_runs"] else ""
            md.append(f"| {c}{div} | {fmt(m, 'cost')} | {fmt(m, 'ss_err', 4)} | "
                      f"{m['viol_max']['mean']:.3f} | {m['viol_steps']['mean']:.1f} | "
                      f"{m['t_mean_ms']['mean']:.1f} / {m['t_max_ms']['max']:.0f} |")
            tex.append(f"{c} & ${m['cost']['mean']:.3f} \\pm {m['cost']['std']:.3f}$ & "
                       f"${m['ss_err']['mean']:.4f}$ & ${m['viol_max']['mean']:.3f}$ & "
                       f"${m['viol_steps']['mean']:.1f}$ & ${m['t_mean_ms']['mean']:.1f}$ \\\\")
        tex.append(r"\midrule")
    md.append("\n**Open-loop prediction error at step 20 (normalised RMSE, held-out data)**\n")
    md.append("| Model | " + " | ".join(TASK_NAMES) + " |")
    md.append("|---|" + "---|" * len(TASK_NAMES))
    for k in ident["models"]:
        md.append(f"| {k} | " + " | ".join(
            f"{ident['plants'][n]['rmse_vs_horizon'][k][-1]:.3f}" for n in TASK_NAMES) + " |")
    md.append("| GP 2σ coverage (one-step) | " + " | ".join(
        f"{ident['plants'][n]['gp_2sigma_coverage_mean']:.3f}" for n in TASK_NAMES) + " |")
    with open(f"{RESULTS}/tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(f"{ROOT}/docs/results_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex[:-1]) + "\n\\bottomrule\n")
    tex2 = []
    for k in ident["models"]:
        tex2.append(f"{k} & " + " & ".join(
            f"{ident['plants'][n]['rmse_vs_horizon'][k][-1]:.3f}" for n in TASK_NAMES) + r" \\")
    tex2.append(r"\midrule")
    tex2.append(r"GP $2\sigma$ coverage & " + " & ".join(
        f"{ident['plants'][n]['gp_2sigma_coverage_mean']:.3f}" for n in TASK_NAMES) + r" \\")
    with open(f"{ROOT}/docs/ident_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex2) + "\n\\bottomrule\n")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with open(f"{RESULTS}/closed_loop.json") as f:
        res = json.load(f)
    with open(f"{RESULTS}/identification.json") as f:
        ident = json.load(f)
    for name in TASK_NAMES:
        plot_trajectories(name)
    plot_quadrotor_plane()
    plot_summary(res)
    tables(res, ident)
    print(open(f"{RESULTS}/tables.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
