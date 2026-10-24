"""
kan_sweep.py — main experiment for the KAN noise-robustness paper (H1, H4).

What it runs
    For every cell  function x N x G x noise x seed  it trains
      - a KAN  (pykan, LBFGS 200 steps, pykan's grid updates), and
      - an MLP with the same number of trainable parameters (AdamW, 8,000 steps).
    Both are evaluated at checkpoints during training. From the same run we
    report two results per model:
      - "final":  the last checkpoint
      - "early":  the checkpoint with the lowest loss on a separate noisy
                  validation set (early stopping, no extra training needed)

Stages
    smoke   a few jobs, to check the installation (~2-3 min)
    core    the main factorial sweep, Gaussian label noise (~6-8 h on 6 cores)
    width   varies the KAN hidden width at fixed G, to separate N/P from N/G

Resuming
    Every finished job is appended to results_<stage>.jsonl. If the run stops
    (sleep, crash, Witcher), start the same command again: finished jobs are
    skipped. You can also stop it on purpose with Ctrl+C and resume later.

Usage (Anaconda prompt, in the folder of this file; use a folder outside OneDrive)
    python kan_sweep.py --stage smoke
    python kan_sweep.py --stage core --workers 6
    python kan_sweep.py --stage width --workers 6
    python kan_sweep_analyze.py results_core.jsonl results_width.jsonl

Workers: one job per CPU core works best. Leave 1-2 cores free if you want
to use the computer meanwhile (e.g. --workers 6 on an 8-core CPU).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Target functions: name -> (n_var, f(u) for u in [-1,1]^n_var, KAN width, formula)
# Formulas are kept as text for the symbolic-recovery study (H3) later.
# ---------------------------------------------------------------------------
PI = math.pi


def _j0(u):
    return torch.special.bessel_j0(20 * u[:, [0]])


def _damped_sine(u):
    return torch.sin(3 * PI * u[:, [0]]) * torch.exp(-u[:, [0]] ** 2)


def _exp_sin_sq(u):
    return torch.exp(torch.sin(PI * u[:, [0]]) + u[:, [1]] ** 2)


def _product(u):
    return u[:, [0]] * u[:, [1]]


def _radial(u):
    return torch.exp(torch.sin(u[:, [0]] ** 2 + u[:, [1]] ** 2))


def _four_d(u):
    return torch.exp(0.5 * (torch.sin(PI * (u[:, [0]] ** 2 + u[:, [1]] ** 2))
                            + torch.sin(PI * (u[:, [2]] ** 2 + u[:, [3]] ** 2))))


def _feyn_I_6_2(u):
    # Gaussian density: theta in [-3, 3], sigma in [1, 3]
    theta, sigma = 3 * u[:, [0]], 2 + u[:, [1]]
    return torch.exp(-((theta / sigma) ** 2) / 2) / (math.sqrt(2 * PI) * sigma)


def _feyn_I_18_4(u):
    # centre of mass: m1, m2 in [1, 3], r1, r2 in [1, 5]
    m1, m2 = 2 + u[:, [0]], 2 + u[:, [1]]
    r1, r2 = 3 + 2 * u[:, [2]], 3 + 2 * u[:, [3]]
    return (m1 * r1 + m2 * r2) / (m1 + m2)


FUNCTIONS = {
    "j0_bessel":    (1, _j0,          [1, 3, 1],    "J0(20*x1)"),
    "damped_sine":  (1, _damped_sine, [1, 3, 1],    "sin(3*pi*x1)*exp(-x1**2)"),
    "exp_sin_sq":   (2, _exp_sin_sq,  [2, 5, 1],    "exp(sin(pi*x1) + x2**2)"),
    "product":      (2, _product,     [2, 2, 1],    "x1*x2"),
    "radial":       (2, _radial,      [2, 3, 1],    "exp(sin(x1**2 + x2**2))"),
    "four_d":       (4, _four_d,      [4, 4, 2, 1], "exp((sin(pi*(x1**2+x2**2)) + sin(pi*(x3**2+x4**2)))/2)"),
    "feyn_I_6_2":   (2, _feyn_I_6_2,  [2, 5, 1],    "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)"),
    "feyn_I_18_4":  (4, _feyn_I_18_4, [4, 5, 1],    "(m1*r1 + m2*r2)/(m1 + m2)"),
}

K_SPLINE = 3
N_TEST = 2000
KAN_STEPS, KAN_EVERY = 200, 10        # LBFGS steps, checkpoint interval
MLP_STEPS, MLP_EVERY = 8000, 250      # AdamW steps, checkpoint interval
GRID_UPDATE_EVERY, GRID_UPDATE_UNTIL = 5, 50   # pykan defaults


def kan_param_count(width, grid, k=K_SPLINE):
    """Trainable spline parameters: per edge (G + k) coefficients + 2 scales."""
    edges = sum(a * b for a, b in zip(width[:-1], width[1:]))
    return edges * (grid + k + 2)


# ---------------------------------------------------------------------------
# Job lists
# ---------------------------------------------------------------------------
def seeds_for(np_ratio):
    return list(range(10)) if np_ratio < 2 else list(range(5))


def make_jobs(stage):
    jobs = []
    if stage == "smoke":
        for f, n, g, noise in [("damped_sine", 200, 30, 0.2), ("exp_sin_sq", 300, 5, 0.1),
                               ("four_d", 300, 5, 0.1)]:
            w = FUNCTIONS[f][2]
            jobs.append(dict(func=f, n=n, grid=g, noise=noise, seed=0, width=w))
    elif stage == "core":
        for f, (nv, _, w, _) in FUNCTIONS.items():
            for n in [100, 300, 1000, 3000]:
                for g in [3, 5, 10, 20, 50, 100]:
                    if n * g > 50000:
                        continue  # heaviest corner: same N/P range is covered by smaller N and G
                    npr = n / kan_param_count(w, g)
                    for noise in [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]:
                        for s in seeds_for(npr):
                            jobs.append(dict(func=f, n=n, grid=g, noise=noise, seed=s, width=w))
    elif stage == "width":
        for f in ["exp_sin_sq", "radial"]:
            for h in [2, 5, 10, 20]:
                w = [2, h, 1]
                for n in [300, 1000]:
                    for g in [5, 20]:
                        npr = n / kan_param_count(w, g)
                        for noise in [0.1, 0.4]:
                            for s in seeds_for(npr):
                                jobs.append(dict(func=f, n=n, grid=g, noise=noise, seed=s, width=w))
    else:
        raise ValueError(stage)
    for j in jobs:
        j["id"] = f"{j['func']}|w{'-'.join(map(str, j['width']))}|N{j['n']}|G{j['grid']}|s{j['noise']}|seed{j['seed']}"
        j["np_ratio"] = j["n"] / kan_param_count(j["width"], j["grid"])
    return jobs


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------
def make_data(func, n, noise, seed):
    nv, fn = FUNCTIONS[func][0], FUNCTIONS[func][1]
    g = torch.Generator().manual_seed(1000 * seed + 7)
    n_val = max(50, n // 4)
    u_tr = torch.rand(n, nv, generator=g) * 2 - 1
    u_va = torch.rand(n_val, nv, generator=g) * 2 - 1
    u_te = torch.rand(N_TEST, nv, generator=g) * 2 - 1
    y_tr, y_va, y_te = fn(u_tr), fn(u_va), fn(u_te)
    mu, sd = y_tr.mean(), y_tr.std()
    # standardise with clean-train statistics, so noise is in units of std(y)
    y_tr, y_va, y_te = (y_tr - mu) / sd, (y_va - mu) / sd, (y_te - mu) / sd
    y_tr_noisy = y_tr + noise * torch.randn(y_tr.shape, generator=g)
    y_va_noisy = y_va + noise * torch.randn(y_va.shape, generator=g)
    return dict(x=u_tr, y=y_tr_noisy, x_val=u_va, y_val=y_va_noisy, x_te=u_te, y_te=y_te)


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


def mlp_params(n_in, h, depth=2):
    return (n_in * h + h) + (depth - 1) * (h * h + h) + (h + 1)


def mlp_hidden_for(n_in, target):
    h = 2
    while mlp_params(n_in, h) < target:
        h += 1
    return h


def _rmse(model, x, y):
    with torch.no_grad():
        return torch.sqrt(torch.mean((model(x) - y) ** 2)).item()


def _checkpoint(model, d, step):
    return dict(step=step, train=_rmse(model, d["x"], d["y"]),
                val=_rmse(model, d["x_val"], d["y_val"]), test=_rmse(model, d["x_te"], d["y_te"]))


# ---------------------------------------------------------------------------
# Safe least squares for pykan's grid updates
#
# pykan's spline.curve2coef calls torch.linalg.lstsq inside a bare try/except
# that only prints "lstsq failed" and then crashes on an undefined variable.
# On Windows, Intel MKL's gelsy driver sometimes rejects systems that other
# builds solve ("Parameter 6 was incorrect on entry to SGELSY"). This wrapper
# tries exactly the call pykan makes first, so results are unchanged whenever
# that call works, and only on failure falls back to an SVD solver (gelsd),
# then to a pseudo-inverse. Non-finite inputs mean training has diverged; that
# is raised as KANDiverged and recorded as an outcome, not hidden as a crash.
# ---------------------------------------------------------------------------
class KANDiverged(FloatingPointError):
    pass


LSTSQ_FALLBACKS = {"gelsd": 0, "pinv": 0}


def _safe_curve2coef(x_eval, y_eval, grid, k):
    from kan.spline import B_batch
    batch, in_dim, out_dim = x_eval.shape[0], x_eval.shape[1], y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    if not (torch.isfinite(mat).all() and torch.isfinite(y).all()):
        raise KANDiverged("non-finite values reached the spline grid update")
    try:
        return torch.linalg.lstsq(mat, y).solution[:, :, :, 0]   # pykan's own call
    except Exception:
        pass
    try:
        sol = torch.linalg.lstsq(mat.contiguous(), y.contiguous(), driver="gelsd").solution
        LSTSQ_FALLBACKS["gelsd"] += 1
        return sol[:, :, :, 0]
    except Exception:
        LSTSQ_FALLBACKS["pinv"] += 1
        return (torch.linalg.pinv(mat) @ y)[:, :, :, 0]


def _patch_pykan():
    """Install the safe solver everywhere pykan imported curve2coef by name."""
    import importlib
    # look the modules up by name: the package exports a class called KANLayer
    # that hides the kan.KANLayer module behind the same attribute
    for name in ("kan.spline", "kan.KANLayer", "kan.MultKAN"):
        mod = importlib.import_module(name)
        if getattr(mod, "curve2coef", None) is not _safe_curve2coef:
            mod.curve2coef = _safe_curve2coef


def train_kan(d, width, grid, seed):
    from kan import KAN
    from kan.LBFGS import LBFGS
    _patch_pykan()
    # pykan rewrites the width list in place; give it a copy so job records stay correct
    model = KAN(width=list(width), grid=grid, k=K_SPLINE, seed=seed, auto_save=False, save_act=False)
    model.symbolic_enabled = False
    opt = LBFGS(model.get_params(), lr=1, history_size=10, line_search_fn="strong_wolfe",
                tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    x, y = d["x"], d["y"]

    def closure():
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        return loss

    curve, diverged_at = [], None
    for step in range(1, KAN_STEPS + 1):
        try:
            if (step - 1) % GRID_UPDATE_EVERY == 0 and (step - 1) < GRID_UPDATE_UNTIL:
                with torch.no_grad():
                    model.update_grid(x)
            opt.step(closure)
        except KANDiverged:
            diverged_at = step
            break
        if step % KAN_EVERY == 0:
            cp = _checkpoint(model, d, step)
            if not all(math.isfinite(v) for v in (cp["train"], cp["val"], cp["test"])):
                diverged_at = step
                break
            curve.append(cp)
    n_params = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("act_fun."))
    return curve, n_params, diverged_at


def train_mlp(d, n_in, hidden, seed):
    torch.manual_seed(seed)
    model = MLP(n_in, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_STEPS)
    x, y = d["x"], d["y"]
    curve = []
    for step in range(1, MLP_STEPS + 1):
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if step % MLP_EVERY == 0:
            curve.append(_checkpoint(model, d, step))
    return curve, sum(p.numel() for p in model.parameters())


def run_job(job):
    torch.set_num_threads(1)  # one core per job; parallelism comes from workers
    random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
    t0 = time.time()
    try:
        d = make_data(job["func"], job["n"], job["noise"], job["seed"])
        before = dict(LSTSQ_FALLBACKS)
        kan_curve, p_kan, kan_diverged_at = train_kan(d, job["width"], job["grid"], job["seed"])
        fallbacks = {k: LSTSQ_FALLBACKS[k] - before[k] for k in before}
        n_in = FUNCTIONS[job["func"]][0]
        mlp_curve, p_mlp = train_mlp(d, n_in, mlp_hidden_for(n_in, p_kan), job["seed"])
        return dict(job, ok=True, p_kan=p_kan, p_mlp=p_mlp, seconds=round(time.time() - t0, 1),
                    kan_diverged_at=kan_diverged_at, lstsq_fallbacks=fallbacks,
                    curves={"KAN": kan_curve, "MLP": mlp_curve})
    except Exception as e:  # keep the sweep alive; failed jobs are retried on resume
        return dict(job, ok=False, error=repr(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "core", "width"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None, help="results file (default results_<stage>.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="only count jobs")
    ap.add_argument("--limit", type=int, default=0, help="run only the N cheapest jobs (for testing)")
    args = ap.parse_args()

    out = args.out or f"results_{args.stage}.jsonl"
    jobs = make_jobs(args.stage)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done.add(r["id"])
                except json.JSONDecodeError:
                    pass  # a line cut off by a crash; that job is simply redone
    todo = [j for j in jobs if j["id"] not in done]
    # big jobs first, so the last hours are not spent waiting on one slow job
    todo.sort(key=lambda j: -(j["n"] * j["grid"]))
    if args.limit:
        todo = todo[-args.limit:]

    print(f"Stage '{args.stage}': {len(jobs)} jobs, {len(done)} already done, "
          f"{len(todo)} to run on {args.workers} workers. Results -> {out}")
    if args.dry_run or not todo:
        return

    t0, finished, failed = time.time(), 0, 0
    with Pool(args.workers) as pool, open(out, "a") as fh:
        for r in pool.imap_unordered(run_job, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            finished += 1
            if not r["ok"]:
                failed += 1
                print(f"  FAILED {r['id']}: {r['error']}", flush=True)
            if finished % 10 == 0 or finished == len(todo) or args.stage == "smoke":
                el = time.time() - t0
                eta = el / finished * (len(todo) - finished)
                print(f"[{finished}/{len(todo)}] {el/3600:.2f} h elapsed, ~{eta/3600:.1f} h left"
                      + (f", {failed} failed" if failed else ""), flush=True)
    print("Done." + (f" {failed} jobs failed; run the same command again to retry them." if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
