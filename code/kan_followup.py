"""
kan_followup.py — follow-up experiments for the KAN noise-robustness paper.

Stages (run in this order; each resumes where it stopped, like kan_sweep.py)
    smoke   one quick job of every stage, to check the installation (~5 min)

Expected run times on a 6-core desktop with --workers 10 (rough):
    ckpt ~1.5 h, effkan ~0.5 h, h3 ~4.5 h, h2 ~2 h
    ckpt    early stopping at full resolution: KANs at N/P < 2 with a checkpoint
            after EVERY LBFGS step (the core sweep checkpointed every 10)
    effkan  second implementation: efficient-kan (B-splines, same parameter
            count) trained exactly like the pykan runs, on a subset of cells
    h2      grid extension under noise: pykan's accuracy recipe, refining the
            grid 3 -> 5 -> 10 -> 20 -> 50 -> 100 with 200 LBFGS steps per stage
    h3      symbolic recovery under noise: pykan's discovery workflow (sparsity
            penalty, prune, snap edges to library functions, refit), scored
            against the true formula on clean test data

All stages use the same data generator and seeds as kan_sweep.py, so their
results join cell by cell with results_core.jsonl.

Usage (same folder as kan_sweep.py and efficient_kan_v010.py; outside OneDrive)
    python kan_followup.py --stage smoke --workers 4
    python kan_followup.py --stage ckpt --workers 10
    python kan_followup.py --stage effkan --workers 10
    python kan_followup.py --stage h2 --workers 10
    python kan_followup.py --stage h3 --workers 10
    python kan_followup_analyze.py --core results_core.jsonl

Results go to followup_<stage>.jsonl. Stop any time by closing the window;
run the same command again to continue.
"""

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch

import kan_sweep as S

PI = math.pi

# ---------------------------------------------------------------------------
# Extra target functions for H3. Each is exactly representable by a pruned
# two-layer KAN whose edges are library functions.
# ---------------------------------------------------------------------------
def _norm3(u):
    return torch.sqrt(1 + u[:, [0]] ** 2 + u[:, [1]] ** 2)


def _additive3(u):
    return torch.sin(PI * u[:, [0]]) + u[:, [1]] ** 2 + torch.exp(u[:, [2]])


S.FUNCTIONS.setdefault("norm3", (2, _norm3, [2, 5, 1], "sqrt(1 + x1**2 + x2**2)"))
S.FUNCTIONS.setdefault("additive3", (3, _additive3, [3, 5, 1], "sin(pi*x1) + x2**2 + exp(x3)"))

H3_FUNCS = {  # function -> starting width for the discovery workflow
    "exp_sin_sq": [2, 5, 1],
    "product": [2, 5, 1],
    "norm3": [2, 5, 1],
    "additive3": [3, 5, 1],
}
H3_LIB = ["x", "x^2", "x^3", "x^4", "1/x", "sqrt", "exp", "log", "sin", "cos", "tanh", "abs", "0"]
SUBSET_FUNCS = ["damped_sine", "exp_sin_sq", "radial", "feyn_I_6_2"]
H2_GRIDS = [3, 5, 10, 20, 50, 100]
MAX_NG = 50000  # same compute cap as the core sweep


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        yield


def _id(stage, **kw):
    w = "-".join(map(str, kw["width"]))
    return f"{stage}|{kw['func']}|w{w}|N{kw['n']}|G{kw.get('grid', 'ext')}|s{kw['noise']}|seed{kw['seed']}"


def _np_ratio(n, width, grid):
    return n / S.kan_param_count(width, grid)


# ---------------------------------------------------------------------------
# Job lists
# ---------------------------------------------------------------------------
def make_jobs(stage):
    jobs = []
    if stage in ("ckpt", "smoke"):
        for f, (nv, _, w, _) in S.FUNCTIONS.items():
            if f in ("norm3", "additive3"):
                continue
            for n in [100, 300, 1000]:
                for g in [3, 5, 10, 20, 50, 100]:
                    if n * g > MAX_NG or _np_ratio(n, w, g) >= 2:
                        continue
                    for noise in [0.1, 0.4]:
                        for s in range(5):
                            jobs.append(dict(kind="ckpt", func=f, n=n, grid=g, noise=noise, seed=s, width=w))
    if stage in ("effkan", "smoke"):
        for f in SUBSET_FUNCS:
            w = S.FUNCTIONS[f][2]
            for n in [100, 300, 1000]:
                for g in [3, 10, 50]:
                    for noise in [0.0, 0.1, 0.4]:
                        for s in range(5):
                            jobs.append(dict(kind="effkan", func=f, n=n, grid=g, noise=noise, seed=s, width=w))
    if stage in ("h2", "smoke"):
        for f in SUBSET_FUNCS:
            w = S.FUNCTIONS[f][2]
            for n in [100, 300, 1000, 3000]:
                for noise in [0.0, 0.05, 0.1, 0.2, 0.4]:
                    for s in range(5):
                        jobs.append(dict(kind="h2", func=f, n=n, noise=noise, seed=s, width=w,
                                         grids=[g for g in H2_GRIDS if n * g <= MAX_NG]))
    if stage in ("h3", "smoke"):
        for f, w in H3_FUNCS.items():
            for n in [100, 300, 1000]:
                for g in [3, 10, 20]:  # symbolic fitting is slow: 3 grids x 3 seeds
                    if n * g > MAX_NG:
                        continue
                    for noise in [0.0, 0.05, 0.1, 0.2]:
                        for s in range(3):
                            jobs.append(dict(kind="h3", func=f, n=n, grid=g, noise=noise, seed=s, width=w))
    for j in jobs:
        j["id"] = _id(j["kind"], **j)
        if "grid" in j:
            j["np_ratio"] = _np_ratio(j["n"], j["width"], j["grid"])
    if stage == "smoke":  # one small job of each kind
        pick = {}
        for j in sorted(jobs, key=lambda j: j["n"] * j.get("grid", 10)):
            pick.setdefault(j["kind"], j)
        jobs = list(pick.values())
    return jobs


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
def _cp(model, d, step):
    return S._checkpoint(model, d, step)


def _finite(cp):
    return all(math.isfinite(cp[k]) for k in ("train", "val", "test"))


def _pykan(width, grid, seed, ckpt_path=None, symbolic=False, save_act=False):
    from kan import KAN
    S._patch_pykan()
    kw = dict(width=list(width), grid=grid, k=S.K_SPLINE, seed=seed, auto_save=False, save_act=save_act)
    if ckpt_path:
        kw["ckpt_path"] = ckpt_path
    m = KAN(**kw)
    if not symbolic:
        m.symbolic_enabled = False
    return m


def _lbfgs_pykan(params):
    from kan.LBFGS import LBFGS
    return LBFGS(params, lr=1, history_size=10, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


def _train_pykan(model, d, steps, every, grid_updates=True):
    """The core sweep's KAN loop, with a configurable checkpoint interval."""
    opt = _lbfgs_pykan(model.get_params())
    x, y = d["x"], d["y"]

    def closure():
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        return loss

    curve = []
    for step in range(1, steps + 1):
        try:
            if grid_updates and (step - 1) % S.GRID_UPDATE_EVERY == 0 and (step - 1) < S.GRID_UPDATE_UNTIL:
                with torch.no_grad():
                    model.update_grid(x)
            opt.step(closure)
        except S.KANDiverged:
            return curve, step
        if step % every == 0:
            cp = _cp(model, d, step)
            if not _finite(cp):
                return curve, step
            curve.append(cp)
    return curve, None


# ---------------------------------------------------------------------------
# Stage: ckpt
# ---------------------------------------------------------------------------
def job_ckpt(job, d):
    model = _pykan(job["width"], job["grid"], job["seed"])
    curve, div = _train_pykan(model, d, S.KAN_STEPS, every=1)
    return dict(curve=curve, kan_diverged_at=div)


# ---------------------------------------------------------------------------
# Stage: effkan (second implementation)
# ---------------------------------------------------------------------------
def _patch_effkan():
    import efficient_kan_v010 as E
    if getattr(E.KANLinear.curve2coeff, "_safe", False):
        return E
    original = E.KANLinear.curve2coeff

    def safe(self, x, y):
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise S.KANDiverged("non-finite values reached the efficient-kan grid update")
        try:
            return original(self, x, y)
        except Exception:
            A = self.b_splines(x).transpose(0, 1)
            B = y.transpose(0, 1)
            sol = torch.linalg.lstsq(A.contiguous(), B.contiguous(), driver="gelsd").solution
            return sol.permute(2, 0, 1).contiguous()

    safe._safe = True
    E.KANLinear.curve2coeff = safe
    return E


def job_effkan(job, d):
    E = _patch_effkan()
    torch.manual_seed(job["seed"])
    model = E.KAN(list(job["width"]), grid_size=job["grid"], spline_order=S.K_SPLINE)
    p = sum(q.numel() for q in model.parameters())
    opt = torch.optim.LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-32, tolerance_change=1e-32)
    x, y = d["x"], d["y"]

    def closure():
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        return loss

    curve, div = [], None
    for step in range(1, S.KAN_STEPS + 1):
        try:
            if (step - 1) % S.GRID_UPDATE_EVERY == 0 and (step - 1) < S.GRID_UPDATE_UNTIL:
                with torch.no_grad():
                    model(x, update_grid=True)
            opt.step(closure)
        except S.KANDiverged:
            div = step
            break
        if step % S.KAN_EVERY == 0:
            cp = _cp(model, d, step)
            if not _finite(cp):
                div = step
                break
            curve.append(cp)
    return dict(curve=curve, kan_diverged_at=div, p_effkan=p)


# ---------------------------------------------------------------------------
# Stage: h2 (grid extension)
# ---------------------------------------------------------------------------
def job_h2(job, d):
    stages, prev = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for g in job["grids"]:
            model = _pykan(job["width"], g, job["seed"], ckpt_path=tmp)
            if prev is not None:
                # what pykan's refine() does: fit the new, finer splines to the old model
                prev.save_act = True  # the hand-over reads the old model's recorded activations
                with torch.no_grad(), quiet():
                    try:
                        model.initialize_from_another_model(prev, d["x"])
                    except S.KANDiverged:
                        stages.append(dict(grid=g, diverged_at=0, curve=[]))
                        break
            curve, div = _train_pykan(model, d, S.KAN_STEPS, every=S.KAN_EVERY)
            stages.append(dict(grid=g, np_ratio=_np_ratio(job["n"], job["width"], g),
                               diverged_at=div, curve=curve))
            if div is not None:
                break
            prev = model
    return dict(stages=stages)


# ---------------------------------------------------------------------------
# Stage: h3 (symbolic recovery)
# ---------------------------------------------------------------------------
def job_h3(job, d):
    import sympy
    nv = S.FUNCTIONS[job["func"]][0]
    ds = dict(train_input=d["x"], train_label=d["y"], test_input=d["x_val"], test_label=d["y_val"])
    out = dict(stage_reached="start")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            model = _pykan(job["width"], job["grid"], job["seed"], ckpt_path=tmp, symbolic=True, save_act=True)
            with quiet():
                model.fit(ds, opt="LBFGS", steps=50, lamb=0.001)   # pykan tutorial: sparsify,
                model = model.prune()                               # prune,
                model.fit(ds, opt="LBFGS", steps=50, lamb=0.0)      # retrain
            out["stage_reached"] = "pruned"
            out["width_pruned"] = [w[0] if isinstance(w, list) else w for w in model.width]
            out["spline_train"] = S._rmse(model, d["x"], d["y"])
            out["spline_test"] = S._rmse(model, d["x_te"], d["y_te"])
            with quiet():
                model.auto_symbolic(lib=H3_LIB, verbose=0)
            out["stage_reached"] = "symbolic"
            out["snapped_test"] = S._rmse(model, d["x_te"], d["y_te"])
            # pykan's tutorial refits the formula's constants; LBFGS can diverge here,
            # so keep the snapped constants whenever the refit does not stay finite
            import copy
            state = copy.deepcopy(model.state_dict())
            with quiet():
                try:
                    model.fit(ds, opt="LBFGS", steps=50, lamb=0.0, update_grid=False)
                    refit_ok = math.isfinite(S._rmse(model, d["x"], d["y"]))
                except Exception:
                    refit_ok = False
            if not refit_ok:
                model.load_state_dict(state)
            out["refit"] = "kept" if refit_ok else "reverted"
            out["edge_functions"] = [model.symbolic_fun[l].funs_name for l in range(len(model.symbolic_fun))]
            xs = [sympy.Symbol(f"x_{i + 1}") for i in range(nv)]
            with quiet():
                f = model.symbolic_formula(var=xs)[0][0]
            fn = sympy.lambdify(xs, f, "numpy")
            xt = d["x_te"].numpy()
            with np.errstate(all="ignore"):
                pred = np.broadcast_to(np.asarray(fn(*[xt[:, i] for i in range(nv)]), dtype=float),
                                       (xt.shape[0],))
            err = float(np.sqrt(np.mean((pred - d["y_te"].numpy()[:, 0]) ** 2)))
            out["symbolic_test"] = err if math.isfinite(err) else None
            out["formula"] = str(sympy.nsimplify(f, tolerance=1e-2, rational=False))[:400]
            out["stage_reached"] = "formula"
        except S.KANDiverged:
            out["diverged"] = True
        except Exception as e:  # a failed discovery is a result, not a crash
            out["failure"] = repr(e)[:300]
    # recovered = R^2 >= 0.999 on clean test data (y is standardised, so RMSE < 0.0316),
    # the accuracy criterion of SRBench; exact = RMSE < 0.01
    st = out.get("symbolic_test")
    out["recovered"] = bool(st is not None and st < 0.0316)
    out["exact"] = bool(st is not None and st < 0.01)
    return out


# ---------------------------------------------------------------------------
# Runner (same resume logic as kan_sweep.py)
# ---------------------------------------------------------------------------
JOBS = {"ckpt": job_ckpt, "effkan": job_effkan, "h2": job_h2, "h3": job_h3}


def run_job(job):
    torch.set_num_threads(1)
    random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
    t0 = time.time()
    try:
        d = S.make_data(job["func"], job["n"], job["noise"], job["seed"])
        res = JOBS[job["kind"]](job, d)
        return dict(job, ok=True, seconds=round(time.time() - t0, 1), **res)
    except Exception as e:
        return dict(job, ok=False, error=repr(e)[:500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "ckpt", "effkan", "h2", "h3"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run only the N cheapest jobs (testing)")
    args = ap.parse_args()

    out = args.out or f"followup_{args.stage}.jsonl"
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
                    pass
    todo = [j for j in jobs if j["id"] not in done]
    cost = lambda j: j["n"] * max(j.get("grids", [j.get("grid", 3)]))
    todo.sort(key=lambda j: -cost(j))
    if args.limit:
        todo = todo[-args.limit:]
    print(f"Stage '{args.stage}': {len(jobs)} jobs, {len(done)} already done, "
          f"{len(todo)} to run on {args.workers} workers. Results -> {out}", flush=True)
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
                print(f"[{finished}/{len(todo)}] {el / 3600:.2f} h elapsed, ~{el / finished * (len(todo) - finished) / 3600:.1f} h left"
                      + (f", {failed} failed" if failed else ""), flush=True)
    print("Done." + (f" {failed} jobs failed; run the same command again to retry them." if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
