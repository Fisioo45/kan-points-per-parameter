"""
kan_optimizer_swap.py — control experiment for the KAN noise-robustness paper.

Question:
    In the first pilot, KANs memorized label noise at low N/P while same-size
    MLPs did not. But the KAN was trained with LBFGS and the MLP with AdamW.
    Is the memorization caused by the spline basis or by the optimizer?

Design (2 x 2): {KAN, MLP} x {LBFGS, Adam}
    - Same parameter count for KAN and MLP (as in the pilot).
    - Same optimizer settings for both models:
        LBFGS: pykan's own LBFGS, exactly as KAN.fit uses it (lr=1,
               history 10, strong Wolfe line search), 200 steps.
        Adam:  AdamW, lr 3e-3, cosine decay, no weight decay, 16,000 steps.
    - KANs get pykan's grid update (every 5 steps during the first 50) under
      BOTH optimizers, so the only difference is the optimizer.
    - Checkpoints along training, so a slow optimizer is not mistaken for a
      robust one: if Adam-KAN memorizes later, the curves show it.
    - Configurations: the low-N/P cases where the pilot saw memorization,
      plus one higher-N/P control. Label noise 0.3 x std(y). 3 seeds.

How to read the result:
    KAN memorizes under BOTH optimizers, MLP under NEITHER -> the spline basis
        causes it (the strong version of the paper).
    KAN-Adam behaves like MLP-Adam and MLP-LBFGS like KAN-LBFGS -> the
        optimizer causes it (the paper becomes about training, not KANs).
    Anything in between -> both contribute; the curves say how much.

Usage (same folder as kan_noise_pilot.py, which it imports):
    python kan_optimizer_swap.py --quick    # ~2 minute smoke test
    python kan_optimizer_swap.py            # the control run (~1.5-2.5 h on a laptop CPU)

Outputs (in ./results_optswap/):
    curves.csv        train / clean-test RMSE at every checkpoint
    final.csv         last checkpoint per run
    summary.csv       final values averaged over seeds
    absorption.png    noise absorbed at the end, 4 conditions vs N/P
    curves_<cfg>.png  noise absorbed during training, per configuration
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from kan import KAN
from kan.LBFGS import LBFGS

from kan_noise_pilot import (FUNCTIONS, MLP, count_kan_params, count_params,
                             make_data, mlp_hidden_for_budget, rmse)

# (function, N, G): the pilot's memorizing cases plus one higher-N/P control
CONFIGS = [
    ("f1_expsin", 200, 3),
    ("f1_expsin", 200, 10),
    ("f1_expsin", 200, 30),
    ("f3_damped_sine", 200, 10),
    ("f3_damped_sine", 200, 30),
    ("f1_expsin", 1000, 30),
]
NOISE = 0.3
LBFGS_STEPS, LBFGS_CKPTS = 200, [10, 25, 50, 100, 200]
ADAM_STEPS, ADAM_CKPTS = 16000, [250, 500, 1000, 2000, 4000, 8000, 16000]
GRID_UPDATE_EVERY, GRID_UPDATE_UNTIL = 5, 50  # pykan's defaults in KAN.fit


def build(kind, width, grid, n_var, seed, device):
    if kind == "KAN":
        m = KAN(width=width, grid=grid, k=3, seed=seed, device=device,
                auto_save=False, save_act=False)
        m.symbolic_enabled = False  # KAN.fit does the same when lamb=0
        return m, count_kan_params(m)
    raise ValueError(kind)


def train(model, is_kan, ds, opt_name, y_std, noise_sigma, log):
    """One training loop for every model/optimizer pair. Returns checkpoint rows."""
    x, y = ds["train_input"], ds["train_label"]
    params = model.get_params() if is_kan else list(model.parameters())
    if opt_name == "LBFGS":
        steps, ckpts = LBFGS_STEPS, LBFGS_CKPTS
        opt = LBFGS(params, lr=1, history_size=10, line_search_fn="strong_wolfe",
                    tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        sched = None
    else:
        steps, ckpts = ADAM_STEPS, ADAM_CKPTS
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def closure():
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        return loss

    rows = []
    for step in range(1, steps + 1):
        if is_kan and (step - 1) % GRID_UPDATE_EVERY == 0 and (step - 1) < GRID_UPDATE_UNTIL:
            with torch.no_grad():
                model.update_grid(x)
        if opt_name == "LBFGS":
            opt.step(closure)
        else:
            closure()
            opt.step()
            sched.step()
        if step in ckpts:
            tr = rmse(model, x, y)
            te = rmse(model, ds["test_input"], ds["test_label"])
            rows.append({**log, "step": step,
                         "train_rmse": tr / y_std,
                         "test_rmse_clean": te / y_std,
                         "absorb": tr / noise_sigma})
    return rows


def make_plots(final, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {("KAN", "LBFGS"): ("tab:blue", "o", "-"),
              ("KAN", "Adam"): ("tab:blue", "s", "--"),
              ("MLP", "LBFGS"): ("tab:gray", "o", "-"),
              ("MLP", "Adam"): ("tab:gray", "s", "--")}

    # 1) noise absorbed at the end vs N/P, four conditions
    s = final.groupby(["model", "opt", "func", "n_train", "grid", "np_ratio"]).absorb.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (m, o), (c, mk, ls) in styles.items():
        d = s[(s.model == m) & (s.opt == o)].sort_values("np_ratio")
        ax.plot(d.np_ratio, d.absorb, marker=mk, ls="none", color=c,
                mfc=c if o == "LBFGS" else "white", label=f"{m} + {o}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("training points per spline parameter, N/P")
    ax.set_ylabel("noise absorbed = train RMSE / noise sigma")
    ax.set_title("Optimizer swap: end of training (mean of seeds)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "absorption.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) training curves per configuration
    for (f, n, g), d in curves.groupby(["func", "n_train", "grid"]):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for (m, o), (c, mk, ls) in styles.items():
            dd = d[(d.model == m) & (d.opt == o)].groupby("step").absorb.mean()
            # put both optimizers on one axis: fraction of the training budget
            budget = LBFGS_STEPS if o == "LBFGS" else ADAM_STEPS
            ax.plot(dd.index / budget, dd.values, marker=mk, ls=ls, color=c,
                    mfc=c if o == "LBFGS" else "white", label=f"{m} + {o}")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("fraction of training budget")
        ax.set_ylabel("noise absorbed")
        ax.set_title(f"{f}, N={n}, G={g}")
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(out, f"curves_{f}_N{n}_G{g}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    global LBFGS_STEPS, LBFGS_CKPTS, ADAM_STEPS, ADAM_CKPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="~2 minute smoke test")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="results_optswap")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        args.device = "cpu"

    configs, seeds = CONFIGS, [0, 1, 2]
    if args.quick:
        configs, seeds = [("f3_damped_sine", 200, 30)], [0]
        LBFGS_STEPS, LBFGS_CKPTS = 50, [10, 50]
        ADAM_STEPS, ADAM_CKPTS = 2000, [500, 2000]

    os.makedirs(args.out, exist_ok=True)
    rows, t0 = [], time.time()
    total, done = len(configs) * len(seeds) * 4, 0

    for fname, n, grid in configs:
        n_var, fn, width = FUNCTIONS[fname]
        for seed in seeds:
            ds, y_std = make_data(fn, n_var, n, 2000, NOISE, seed, args.device)
            noise_sigma = NOISE * y_std
            # size the MLP to the KAN once per configuration
            p_kan = count_kan_params(KAN(width=width, grid=grid, k=3, seed=seed,
                                         auto_save=False, save_act=False))
            hidden = mlp_hidden_for_budget(n_var, p_kan)
            for kind in ["KAN", "MLP"]:
                for opt_name in ["LBFGS", "Adam"]:
                    if kind == "KAN":
                        model, p = build("KAN", width, grid, n_var, seed, args.device)
                    else:
                        torch.manual_seed(seed)
                        model = MLP(n_var, hidden).to(args.device)
                        p = count_params(model)
                    log = dict(func=fname, n_train=n, grid=grid, seed=seed, model=kind,
                               opt=opt_name, params=p, np_ratio=n / p_kan, noise=NOISE)
                    r = train(model, kind == "KAN", ds, opt_name, y_std, noise_sigma, log)
                    rows += r
                    done += 1
                    print(f"[{done}/{total}] {fname} N={n} G={grid} seed={seed} {kind:3s}+{opt_name:5s} "
                          f"| absorbed {r[-1]['absorb']:.2f}  test {r[-1]['test_rmse_clean']:.4f}  "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(args.out, "curves.csv"), index=False)
    final = curves.loc[curves.groupby(["func", "n_train", "grid", "seed", "model", "opt"]).step.idxmax()]
    final.to_csv(os.path.join(args.out, "final.csv"), index=False)
    summ = (final.groupby(["func", "n_train", "grid", "np_ratio", "model", "opt"])
                 .agg(absorb=("absorb", "mean"), absorb_sd=("absorb", "std"),
                      test=("test_rmse_clean", "mean"), test_sd=("test_rmse_clean", "std"))
                 .reset_index())
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nSaved curves.csv, final.csv and summary.csv in", args.out)
    print("\nNoise absorbed at the end (1.0 = averaged out, 0 = memorized):")
    print(summ.pivot_table(index=["func", "n_train", "grid"], columns=["model", "opt"],
                           values="absorb").round(2).to_string())
    try:
        make_plots(final, curves, args.out)
        print("\nSaved plots in", args.out)
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
