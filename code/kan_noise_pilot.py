"""
kan_noise_pilot.py — pilot experiment for the KAN noise-robustness paper.

Hypothesis this pilot tests (H1):
    KAN's sensitivity to label noise is governed mainly by data density per
    spline interval, rho = N / G (training points per grid interval), not by
    the KAN architecture as such. Small rho -> splines interpolate the noise
    (fragile); large rho -> splines average it out (robust). Different papers
    used different N and G, which would explain their contradicting results.

Design (~45-60 min on a laptop CPU; GPU is not faster for these tiny models):
    target function x training size N x grid size G x noise level x seed
    - Noise is added to TRAINING labels only (std = noise * std(y_clean)).
    - Test error is measured against the CLEAN function, so it measures how
      much noise the model absorbed, not how well it fits noisy labels.
    - Each KAN is compared with an MLP of the same parameter count
      (plain, and with weight decay).

Usage (Anaconda prompt, inside your environment):
    pip install pykan            # once
    python kan_noise_pilot.py --quick     # ~1 minute smoke test
    python kan_noise_pilot.py             # the pilot
    python kan_noise_pilot.py --device cuda   # optional

Outputs (in ./results_pilot/):
    runs.csv              one row per trained model
    summary.csv           mean and std over seeds
    heatmap_<f>_N<n>.png  test RMSE over grid size x noise, KAN vs MLPs
    collapse_<f>.png      KAN test RMSE against rho = N/G (the H1 test)
"""

import argparse
import contextlib
import io
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from kan import KAN

# ---------------------------------------------------------------------------
# Target functions (KAN-paper style; inputs uniform in [-1, 1])
# name: (n_var, f(x) for x of shape [N, n_var] -> [N, 1], KAN width)
# ---------------------------------------------------------------------------
FUNCTIONS = {
    "f1_expsin": (2, lambda x: torch.exp(torch.sin(math.pi * x[:, [0]]) + x[:, [1]] ** 2), [2, 5, 1]),
    "f3_damped_sine": (1, lambda x: torch.sin(3 * math.pi * x[:, [0]]) * torch.exp(-x[:, [0]] ** 2), [1, 3, 1]),
}


def make_data(fn, n_var, n_train, n_test, noise_rel, seed, device):
    g = torch.Generator().manual_seed(seed)
    x_tr = torch.rand(n_train, n_var, generator=g) * 2 - 1
    x_te = torch.rand(n_test, n_var, generator=g) * 2 - 1
    y_tr_clean, y_te_clean = fn(x_tr), fn(x_te)
    y_std = y_tr_clean.std()
    y_tr = y_tr_clean + noise_rel * y_std * torch.randn(y_tr_clean.shape, generator=g)
    ds = {"train_input": x_tr.to(device), "train_label": y_tr.to(device),
          "test_input": x_te.to(device), "test_label": y_te_clean.to(device)}  # clean on purpose
    return ds, float(y_std)


def rmse(model, x, y):
    with torch.no_grad():
        return torch.sqrt(torch.mean((model(x) - y) ** 2)).item()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_kan_params(model):
    """Numeric spline-branch parameters only (coef, scale_base, scale_sp).
    pykan also registers symbolic-branch 'affine' parameters that are unused
    in numeric training; counting them would make the MLP match unfair."""
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and n.startswith("act_fun."))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def train_kan(ds, width, grid, k, steps, seed, device, lamb=0.0):
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device,
                auto_save=False, save_act=False)
    # pykan draws a progress bar every step; silence it to keep the console readable
    with contextlib.redirect_stderr(io.StringIO()):
        model.fit(ds, opt="LBFGS", steps=steps, lamb=lamb, log=steps + 1)
    return model


class MLP(nn.Module):
    def __init__(self, n_in, hidden, depth=2):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp_hidden_for_budget(n_in, target_params, depth=2):
    """Smallest hidden width whose parameter count reaches the KAN's."""
    for h in range(2, 1024):
        if count_params(MLP(n_in, h, depth)) >= target_params:
            return h
    return 1024


def train_mlp(ds, n_in, hidden, steps, seed, weight_decay=0.0, lr=3e-3):
    torch.manual_seed(seed)
    device = ds["train_input"].device
    model = MLP(n_in, hidden).to(device)
    # AdamW = decoupled weight decay; plain Adam's coupled L2 crippled these small MLPs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    x, y = ds["train_input"], ds["train_label"]
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    return model


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (fname, n), sub in summ.groupby(["func", "n_train"]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        piv_all = [sub[sub.model == m].pivot(index="grid", columns="noise", values="test_mean")
                   for m in ["KAN", "MLP", "MLP-WD"]]
        vmin = np.log10(min(p.values.min() for p in piv_all))
        vmax = np.log10(max(p.values.max() for p in piv_all))
        for ax, mname, piv in zip(axes, ["KAN", "MLP", "MLP-WD"], piv_all):
            im = ax.imshow(np.log10(piv.values), aspect="auto", origin="lower",
                           cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
            ax.set_xlabel("label noise (x std of y)")
            ax.set_title(mname if mname == "KAN" else f"{mname} (same #params)")
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                            fontsize=7, color="w")
        axes[0].set_ylabel("KAN grid size G")
        fig.suptitle(f"{fname}, N={n}: test RMSE vs clean truth (normalised; colour = log10)")
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(os.path.join(out, f"heatmap_{fname}_N{n}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # H1 test: does KAN error collapse onto rho = N/G?
    kan = summ[(summ.model == "KAN") & (summ.noise > 0)]
    for fname, sub in kan.groupby("func"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for noise, s in sub.groupby("noise"):
            s = s.sort_values("rho")
            ax.plot(s["rho"], s["test_mean"], "o", label=f"noise {noise}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("rho = N / G  (training points per spline interval)")
        ax.set_ylabel("KAN test RMSE vs clean truth")
        ax.set_title(f"{fname}: H1 collapse check")
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(out, f"collapse_{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="~1 minute smoke test")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="results_pilot")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        args.device = "cpu"

    if args.quick:
        funcs, n_trains, grids, noises, seeds = ["f3_damped_sine"], [200], [3, 30], [0.0, 0.3], [0]
        kan_steps, mlp_steps = 50, 3000
    else:
        funcs = list(FUNCTIONS)
        n_trains = [200, 1000]
        grids = [3, 10, 30]
        noises = [0.0, 0.1, 0.3]
        seeds = [0, 1]
        kan_steps, mlp_steps = 200, 8000
    n_test, k = 2000, 3

    os.makedirs(args.out, exist_ok=True)
    rows, t0 = [], time.time()
    total = len(funcs) * len(n_trains) * len(grids) * len(noises) * len(seeds)
    done = 0

    for fname in funcs:
        n_var, fn, width = FUNCTIONS[fname]
        for n in n_trains:
            for noise in noises:
                for seed in seeds:
                    ds, y_std = make_data(fn, n_var, n, n_test, noise, seed, args.device)
                    for grid in grids:
                        base = dict(func=fname, n_train=n, grid=grid, rho=n / grid, noise=noise, seed=seed)
                        km = train_kan(ds, width, grid, k, kan_steps, seed, args.device)
                        p = count_kan_params(km)
                        res = {"KAN": (km, p)}
                        h = mlp_hidden_for_budget(n_var, p)
                        for tag, wd in [("MLP", 0.0), ("MLP-WD", 1e-2)]:
                            m = train_mlp(ds, n_var, h, mlp_steps, seed, weight_decay=wd)
                            res[tag] = (m, count_params(m))
                        for tag, (m, pc) in res.items():
                            rows.append({**base, "model": tag, "params": pc,
                                         "train_rmse": rmse(m, ds["train_input"], ds["train_label"]) / y_std,
                                         "test_rmse_clean": rmse(m, ds["test_input"], ds["test_label"]) / y_std})
                        done += 1
                        r = {row["model"]: row["test_rmse_clean"] for row in rows[-3:]}
                        print(f"[{done}/{total}] {fname} N={n} noise={noise} seed={seed} G={grid} | "
                              f"KAN {r['KAN']:.4f}  MLP {r['MLP']:.4f}  MLP-WD {r['MLP-WD']:.4f}  "
                              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    summ = (df.groupby(["func", "model", "n_train", "grid", "rho", "noise"])
              .agg(params=("params", "first"),
                   test_mean=("test_rmse_clean", "mean"), test_std=("test_rmse_clean", "std"),
                   train_mean=("train_rmse", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nSaved runs.csv and summary.csv in", args.out)
    try:
        make_plots(summ, args.out)
        print("Saved plots in", args.out)
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
