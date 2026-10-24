"""
kan_sweep_analyze.py — turns kan_sweep.py results into the H1 evidence.

Usage
    python kan_sweep_analyze.py results_core.jsonl [results_width.jsonl ...] [--out analysis]

Outputs (in ./analysis/)
    runs.csv               one row per job x model x variant (final / early-stopped)
    cells.csv              medians over seeds per cell
    h1_regression.txt      the central test: does N/P alone explain KAN error as
                           well as N, G and width entered separately?
    threshold.txt          N/P where the KAN's median noise absorption reaches 0.9
    early_stopping.csv     how much early stopping rescues each model, by N/P bin
    fig_absorption.png     noise absorbed vs N/P, KAN vs MLP, final and early-stopped
    fig_test_error.png     KAN and MLP test error vs N/P, one panel per noise level

Metrics (all in units of std(y), because kan_sweep.py standardises y)
    test      RMSE against the clean function on 2,000 test points
    absorb    train RMSE / noise level: ~1 = noise averaged out, ~0 = memorized
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load(paths):
    """Returns (rows, jobs). rows: one per job x model x variant. jobs: one per
    finished job, with divergence and solver-fallback flags for the health report."""
    rows, jobs, failed_ids, seen = [], [], set(), set()
    for p in paths:
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("ok"):
                    failed_ids.add(r.get("id"))
                    continue
                if r["id"] in seen:  # a job recorded twice: keep the first
                    continue
                seen.add(r["id"])
                # pykan may have rewritten a width like [2, 5, 1] as [[2, 0], [5, 0], [1, 0]]
                w = [x[0] if isinstance(x, list) else x for x in r["width"]]
                edges = sum(a * b for a, b in zip(w[:-1], w[1:]))
                base = dict(func=r["func"], width="-".join(map(str, w)), edges=edges,
                            n=r["n"], grid=r["grid"], noise=r["noise"], seed=r["seed"],
                            p_kan=r["p_kan"], np_ratio=r["n"] / r["p_kan"], source=os.path.basename(p))
                fb = r.get("lstsq_fallbacks") or {}
                jobs.append(dict(base, kan_diverged=r.get("kan_diverged_at") is not None,
                                 lstsq_fallback=bool(sum(fb.values()))))
                for model, curve in r["curves"].items():
                    if not curve:  # KAN diverged before its first checkpoint
                        continue
                    if model == "KAN" and r.get("kan_diverged_at") is not None:
                        continue  # diverged runs are counted in the health report, not in the medians
                    final = curve[-1]
                    early = min(curve, key=lambda c: c["val"])
                    for variant, c in [("final", final), ("early", early)]:
                        rows.append(dict(base, model=model, variant=variant, step=c["step"],
                                         train=c["train"], test=c["test"]))
    df = pd.DataFrame(rows)
    df["absorb"] = np.where(df.noise > 0, df.train / df.noise.replace(0, np.nan), np.nan)
    jobs = pd.DataFrame(jobs)
    unresolved = failed_ids - seen
    return df, jobs, unresolved


def health_report(jobs, unresolved):
    lines = ["Run health"]
    lines.append(f"  finished jobs: {len(jobs)}")
    lines.append(f"  jobs that failed and were never rerun successfully: {len(unresolved)}"
                 + ("  -> run the same kan_sweep.py command again to retry them" if unresolved else ""))
    if len(jobs):
        n_fb = int(jobs.lstsq_fallback.sum())
        n_div = int(jobs.kan_diverged.sum())
        lines.append(f"  jobs where the safe least-squares fallback was used: {n_fb}")
        lines.append(f"  KAN runs that diverged (excluded from medians, reported here): {n_div}")
        if n_div:
            j = jobs.assign(np_bin=pd.cut(jobs.np_ratio, [0, 0.5, 1, 2, 5, 20, np.inf]))
            t = j.groupby("np_bin", observed=True).kan_diverged.agg(["sum", "count"])
            t["rate"] = (t["sum"] / t["count"]).round(3)
            lines.append("  KAN divergence by N/P bin:\n" + t.to_string())
    return "\n".join(lines) + "\n"


def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ b
    n, k = X1.shape
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    sigma2 = (resid ** 2).sum() / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    aic = n * np.log((resid ** 2).sum() / n) + 2 * k
    lines = [f"  n = {n}, R^2 = {r2:.3f}, AIC = {aic:.1f}"]
    for nm, bi, si in zip(["intercept"] + names, b, se):
        if not nm.startswith("func="):
            lines.append(f"  {nm:12s} {bi:+.3f}  (se {si:.3f})")
    return r2, aic, "\n".join(lines)


def h1_regression(cells):
    """Cell medians, KAN, final, noise > 0. Function fixed effects in both models."""
    d = cells[(cells.model == "KAN") & (cells.variant == "final") & (cells.noise > 0)].copy()
    d = d[d.test > 0]
    if len(d) < 10:
        return "Not enough cells for the regression yet."
    funcs = sorted(d.func.unique())
    fe = np.column_stack([(d.func == f).astype(float) for f in funcs[1:]]) if len(funcs) > 1 else np.empty((len(d), 0))
    fe_names = [f"func={f}" for f in funcs[1:]]
    y = np.log(d.test.values)
    full = np.column_stack([np.log(d.n), np.log(d.grid), np.log(d.edges), np.log(d.noise), fe])
    short = np.column_stack([np.log(d.np_ratio), np.log(d.noise), fe])
    r2f, aicf, tf = ols(y, full, ["log N", "log G", "log edges", "log noise"] + fe_names)
    r2s, aics, ts = ols(y, short, ["log N/P", "log noise"] + fe_names)
    verdict = ("N/P alone fits about as well as N, G and width separately (supports H1)."
               if r2f - r2s < 0.03 else
               "N, G and width carry information beyond N/P (H1 only partly holds).")
    return ("H1 test on KAN cell medians (final checkpoint, noise > 0), outcome = log test RMSE\n\n"
            "Full model: log N, log G, log edges, log noise, function fixed effects\n" + tf +
            "\n\nN/P model: log N/P, log noise, function fixed effects\n" + ts +
            f"\n\nR^2 lost by using N/P alone: {r2f - r2s:+.3f}; AIC difference (N/P - full): {aics - aicf:+.1f}\n"
            "Under H1, log N and log G get roughly equal and opposite coefficients in the full model.\n"
            + verdict + "\n")


def threshold(cells):
    """N/P where the KAN stops memorizing. Only cells where the KAN can learn the
    function at all count (clean-data test RMSE < 0.1 at the same N, G, width):
    an underfitting KAN has train error above the noise level, which would
    otherwise look like perfect robustness."""
    kan = cells[(cells.model == "KAN") & (cells.variant == "final")]
    clean = kan[kan.noise == 0].set_index(["func", "width", "n", "grid"]).test.rename("clean_test")
    d = kan[kan.noise > 0].join(clean, on=["func", "width", "n", "grid"])
    d = d[d.clean_test < 0.1]
    if d.empty:
        return "No learnable noisy KAN cells yet (needs the noise = 0 runs of the same N and G).\n"
    lines = ["N/P at which the KAN's median noise absorption first reaches 0.9",
             "(only cells where the KAN fits clean data to test RMSE < 0.1):", ""]
    bins = np.logspace(-2, 3, 26)
    found = []
    for f, s in [("all functions", d)] + list(d.groupby("func")):
        g = s.assign(b=pd.cut(s.np_ratio, bins)).groupby("b", observed=True).absorb.median().dropna()
        cross = next((iv.left for iv, a in g.items() if a >= 0.9), None)
        lines.append(f"  {f:16s} " + (f"N/P ~ {cross:.2f}" if cross is not None else "never reaches 0.9"))
        if f != "all functions" and cross is not None:
            found.append(cross)
    if found:
        lines.append(f"\n  median over functions: N/P ~ {np.median(found):.2f}")
    return "\n".join(lines) + "\n"


def early_table(cells):
    d = cells[cells.noise > 0].copy()
    d["np_bin"] = pd.cut(d.np_ratio, [0, 0.5, 1, 2, 5, 20, np.inf])
    t = d.pivot_table(index="np_bin", columns=["model", "variant"], values=["absorb", "test"],
                      aggfunc="median", observed=True)
    return t.round(3)


def plots(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"KAN": "tab:blue", "MLP": "tab:gray"}

    d = cells[cells.noise > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, variant in zip(axes, ["final", "early"]):
        for model in ["MLP", "KAN"]:
            s = d[(d.model == model) & (d.variant == variant)]
            ax.scatter(s.np_ratio, s.absorb, s=10, alpha=0.5, color=colors[model], label=model)
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("training points per spline parameter, N/P")
        ax.set_title("end of training" if variant == "final" else "early-stopped (best validation)")
    axes[0].set_ylabel("noise absorbed = train RMSE / noise")
    axes[0].legend()
    fig.savefig(os.path.join(out, "fig_absorption.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    noises = sorted(d.noise.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(3.2 * len(noises), 3.4), sharey=True, squeeze=False)
    for ax, nz in zip(axes[0], noises):
        for model in ["MLP", "KAN"]:
            s = d[(d.model == model) & (d.variant == "final") & (d.noise == nz)]
            ax.scatter(s.np_ratio, s.test, s=10, alpha=0.5, color=colors[model], label=model)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"noise {nz}")
        ax.set_xlabel("N/P")
    axes[0][0].set_ylabel("test RMSE vs clean function")
    axes[0][0].legend()
    fig.savefig(os.path.join(out, "fig_test_error.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, jobs, unresolved = load(args.files)
    health = health_report(jobs, unresolved)
    print(health)
    open(os.path.join(args.out, "health.txt"), "w").write(health)
    if runs.empty:
        print("No finished jobs found.")
        return
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    keys = ["func", "width", "edges", "n", "grid", "noise", "np_ratio", "model", "variant"]
    cells = runs.groupby(keys).agg(test=("test", "median"), absorb=("absorb", "median"),
                                   train=("train", "median"), seeds=("seed", "nunique"),
                                   step=("step", "median")).reset_index()
    cells.to_csv(os.path.join(args.out, "cells.csv"), index=False)

    reg, thr, early = h1_regression(cells), threshold(cells), early_table(cells)
    open(os.path.join(args.out, "h1_regression.txt"), "w").write(reg)
    open(os.path.join(args.out, "threshold.txt"), "w").write(thr)
    early.to_csv(os.path.join(args.out, "early_stopping.csv"))
    print(f"{runs[['func','n','grid','noise','seed','width']].drop_duplicates().shape[0]} jobs, "
          f"{len(cells)} cells.\n")
    print(reg)
    print(thr)
    print("Early stopping, medians by N/P bin:\n", early.to_string(), "\n")
    try:
        plots(cells, args.out)
        print("Saved figures and tables in", args.out)
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
