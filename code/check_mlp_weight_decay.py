"""
check_mlp_weight_decay.py — does weight decay change the MLP baseline? (Section 3.2)

Trains the parameter-matched MLP of kan_sweep.py with and without AdamW weight
decay (1e-2) on 96 configurations: four functions x N in {100, 1000} x
G in {3, 20} (which sets the MLP size) x noise in {0, 0.1, 0.4} x 2 seeds.
Data, schedule and checkpoints are exactly those of kan_sweep.py.

Usage (from code/, ~30 min with 2 workers)
    python check_mlp_weight_decay.py --out mlp_weight_decay_check.json --workers 4
"""

import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np
import torch

import kan_sweep as S


def run(job):
    f, n, g, noise, seed = job
    torch.set_num_threads(1)
    d = S.make_data(f, n, noise, seed)
    n_in, width = S.FUNCTIONS[f][0], S.FUNCTIONS[f][2]
    p = S.kan_param_count(width, g)
    h = S.mlp_hidden_for(n_in, p)
    out = {}
    for wd in (0.0, 1e-2):
        torch.manual_seed(seed)
        m = S.MLP(n_in, h)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=S.MLP_STEPS)
        best_val, best_test = np.inf, np.inf
        for step in range(1, S.MLP_STEPS + 1):
            opt.zero_grad()
            loss = torch.mean((m(d["x"]) - d["y"]) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            if step % S.MLP_EVERY == 0:
                cp = S._checkpoint(m, d, step)
                if cp["val"] < best_val:
                    best_val, best_test = cp["val"], cp["test"]
        out[wd] = (cp["test"], best_test)
    return dict(func=f, n=n, grid=g, noise=noise, seed=seed, np_ratio=n / p,
                final_no_wd=out[0.0][0], final_wd=out[1e-2][0],
                early_no_wd=out[0.0][1], early_wd=out[1e-2][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mlp_weight_decay_check.json")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = list(itertools.product(["exp_sin_sq", "damped_sine", "radial", "four_d"],
                                  [100, 1000], [3, 20], [0.0, 0.1, 0.4], [0, 1]))
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs)
    json.dump(res, open(args.out, "w"), indent=1)
    rf = np.array([r["final_wd"] / r["final_no_wd"] for r in res])
    re = np.array([r["early_wd"] / r["early_no_wd"] for r in res])
    print(f"{len(res)} configurations. Median |change| in MLP test RMSE from weight decay 1e-2: "
          f"final {np.median(abs(rf - 1)) * 100:.1f}%, early-stopped {np.median(abs(re - 1)) * 100:.1f}%")


if __name__ == "__main__":
    main()
