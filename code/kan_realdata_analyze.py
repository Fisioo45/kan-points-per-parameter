"""
kan_realdata_analyze.py — analysis of kan_realdata.py results.

Usage
    python kan_realdata_analyze.py realdata_real.jsonl --out realdata_analysis

Writes to the output folder
    realdata_runs.csv          one row per job with final and early-stopped errors
    realdata_report.txt        all tables below
    fig6_realdata.(pdf|png)    (a) KAN train/validation RMSE ratio, (b) KAN/MLP test ratio

Metrics (real labels carry unknown measurement noise, so absorption = train/sigma
cannot be computed; we use quantities that need no knowledge of the noise):
    fit ratio     train RMSE / validation RMSE, both on (noisy) labels drawn the
                  same way. About 1 when the model has not memorised its training
                  labels; near 0 when it has. This is the real-data analogue of
                  noise absorption.
    KAN/MLP       test RMSE ratio against the parameter-matched MLP, same split.
    degradation   test RMSE at injected sigma / test RMSE at sigma = 0, same split.
    G selection   test RMSE of a grid chosen (i) by the N/P >= 4 rule, (ii) by the
                  validation set, (iii) fixed at G = 5 or G = 20, relative to the
                  best grid in hindsight.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

BINS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 64]
NOISE_STYLE = {0.0: ("#000000", "o"), 0.1: ("#E69F00", "s"), 0.3: ("#CC79A7", "v")}
EDGES_PER_DIM = lambda w: sum(a * b for a, b in zip(w[:-1], w[1:]))


def early(curve):
    best = min(curve, key=lambda c: c["val"])
    return best["test"], best["step"]


def load(path):
    rows, seen = [], set()
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok"):
                continue
            h = r.get("hidden", r["width"][1])
            key = (r["dataset"], h, r["n"], r["grid"], r["noise"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            k, m = r["curves"]["KAN"], r["curves"]["MLP"]
            div = r.get("kan_diverged_at") is not None or not k
            row = dict(dataset=r["dataset"], hidden=h, d_in=r["width"][0], n=r["n"], grid=r["grid"],
                       noise=r["noise"], seed=r["seed"], np_ratio=r["np_ratio"], p_kan=r["p_kan"],
                       edges=EDGES_PER_DIM(r["width"]), diverged=div)
            if not div:
                ke, ks = early(k)
                row.update(kan_train=k[-1]["train"], kan_val=k[-1]["val"], kan_test=k[-1]["test"],
                           kan_early=ke, kan_best_step=ks)
            me, _ = early(m)
            row.update(mlp_train=m[-1]["train"], mlp_val=m[-1]["val"], mlp_test=m[-1]["test"], mlp_early=me)
            rows.append(row)
    d = pd.DataFrame(rows)
    d["bin"] = pd.cut(d.np_ratio, BINS)
    d["kan_fit"] = d.kan_train / d.kan_val
    d["mlp_fit"] = d.mlp_train / d.mlp_val
    d["ratio_final"] = d.kan_test / d.mlp_test
    d["ratio_early"] = d.kan_early / d.mlp_early
    return d


def crossing(x, y, level):
    """First N/P (log-interpolated) where the binned median y reaches level."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            t = (level - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
    return float("nan") if y[0] < level else float(x[0])


def binned(d, col, by="noise", how="median"):
    g = d.dropna(subset=[col]).groupby([by, "bin"], observed=True)[col]
    t = g.agg(how).unstack(0)
    t.index = [f"{iv.left:g}-{iv.right:g}" for iv in t.index]
    return t


def bin_centres(d):
    return {iv: float(np.sqrt(max(iv.left, 0.06) * iv.right)) for iv in d.bin.cat.categories}


def report(d):
    out = []
    ok = d[~d.diverged]
    out.append(f"Runs: {len(d)} ({int(d.diverged.sum())} KAN runs diverged, excluded from KAN metrics)")
    out.append("Runs per dataset and width:\n" + d.groupby(["dataset", "hidden"]).size().unstack().to_string())
    out.append("N/P range per dataset: " + ", ".join(
        f"{k} {v.np_ratio.min():.2f}-{v.np_ratio.max():.1f}" for k, v in d.groupby("dataset")))

    # 1. fit ratio (memorisation)
    out.append("\n1. Fit ratio train/val RMSE, median (1 = no memorisation, 0 = memorised)")
    out.append("KAN, final checkpoint\n" + binned(ok, "kan_fit").round(2).to_string())
    out.append("MLP, final checkpoint\n" + binned(d, "mlp_fit").round(2).to_string())
    cen = bin_centres(d)
    for nz in sorted(d.noise.unique()):
        for name, frame, col in [("KAN", ok, "kan_fit"), ("MLP", d, "mlp_fit")]:
            s = frame[frame.noise == nz].groupby("bin", observed=True)[col].median()
            out.append(f"  sigma {nz}: {name} fit ratio reaches 0.9 at N/P ~ "
                       f"{crossing([cen[b] for b in s.index], s.values, 0.9):.2g}")
    per = []
    for ds, g in ok[ok.noise > 0].groupby("dataset"):
        s = g.groupby("bin", observed=True).kan_fit.median()
        per.append(f"{ds} {crossing([cen[b] for b in s.index], s.values, 0.9):.2g}")
    out.append("  per dataset (injected noise pooled), KAN fit ratio 0.9 at N/P ~ " + ", ".join(per))

    # 2. KAN vs MLP
    out.append("\n2. KAN/MLP test RMSE ratio, median (<1: KAN better)")
    out.append("final checkpoint\n" + binned(ok, "ratio_final").round(2).to_string())
    out.append("early-stopped (both models at their validation minimum)\n" + binned(ok, "ratio_early").round(2).to_string())
    w = ok.assign(kan_wins=(ok.kan_early < ok.mlp_early).astype(float))
    out.append("fraction of runs where the early-stopped KAN beats the early-stopped MLP\n"
               + binned(w, "kan_wins", how="mean").round(2).to_string())
    out.append("by dataset, early-stopped ratio (all noise levels), N/P < 2 vs >= 2:")
    for ds, g in ok.groupby("dataset"):
        lo, hi = g[g.np_ratio < 2].ratio_early.median(), g[g.np_ratio >= 2].ratio_early.median()
        out.append(f"  {ds:9s} {lo:.2f}  {hi:.2f}")

    # 3. degradation from injected noise
    base = d[d.noise == 0].set_index(["dataset", "hidden", "n", "grid", "seed"])[["kan_test", "mlp_test", "kan_early", "mlp_early"]]
    nz = d[d.noise > 0].join(base, on=["dataset", "hidden", "n", "grid", "seed"], rsuffix="_0")
    for m in ["kan_test", "mlp_test", "kan_early", "mlp_early"]:
        nz[f"deg_{m}"] = nz[m] / nz[f"{m}_0"]
    nz = nz[~nz.diverged]
    out.append("\n3. Degradation: test RMSE with injected noise / without, median")
    for m in ["kan_test", "mlp_test", "kan_early", "mlp_early"]:
        out.append(f"{m}\n" + binned(nz, f"deg_{m}").round(2).to_string())

    # 4. does N/P collapse N and G? log KAN test on dataset fixed effects
    out.append("\n4. Regression of log KAN test RMSE (final), dataset x width fixed effects")
    for sigma in sorted(d.noise.unique()):
        g = ok[ok.noise == sigma].groupby(["dataset", "hidden", "n", "grid"], observed=True).agg(
            t=("kan_test", "median"), npr=("np_ratio", "first"), edges=("edges", "first")).reset_index()
        if len(g) < 6:
            out.append(f"  sigma {sigma}: too few cells ({len(g)})")
            continue
        fe = pd.get_dummies(g.dataset + "_" + g.hidden.astype(str), dtype=float).to_numpy()
        yv = np.log(g.t.to_numpy())

        def fit(X):
            X = np.column_stack([X, fe])
            b, *_ = np.linalg.lstsq(X, yv, rcond=None)
            r = yv - X @ b
            return b, 1 - r.var() / yv.var()

        b1, r1 = fit(np.log(g.npr.to_numpy())[:, None])
        b2, r2 = fit(np.column_stack([np.log(g.n.to_numpy(float)), np.log(g.grid.to_numpy(float) + 5)]))
        out.append(f"  sigma {sigma}: log N/P coef {b1[0]:+.2f} (R2 {r1:.3f}) | "
                   f"log N {b2[0]:+.2f}, log(G+5) {b2[1]:+.2f} (R2 {r2:.3f}); {len(g)} cells")
    out.append("\n4b. Same regression for memorisation: logit of the KAN fit ratio (clipped to 0.01-0.99)")
    for sigma in sorted(d.noise.unique()):
        g = ok[ok.noise == sigma].groupby(["dataset", "hidden", "n", "grid"], observed=True).agg(
            f=("kan_fit", "median"), npr=("np_ratio", "first")).reset_index()
        if len(g) < 6:
            continue
        fe = pd.get_dummies(g.dataset + "_" + g.hidden.astype(str), dtype=float).to_numpy()
        fr = g.f.clip(0.01, 0.99).to_numpy()
        yv = np.log(fr / (1 - fr))

        def fit(X):
            X = np.column_stack([X, fe])
            b, *_ = np.linalg.lstsq(X, yv, rcond=None)
            r = yv - X @ b
            return b, 1 - r.var() / yv.var()

        b1, r1 = fit(np.log(g.npr.to_numpy())[:, None])
        b2, r2 = fit(np.column_stack([np.log(g.n.to_numpy(float)), np.log(g.grid.to_numpy(float) + 5)]))
        out.append(f"  sigma {sigma}: log N/P coef {b1[0]:+.2f} (R2 {r1:.3f}) | "
                   f"log N {b2[0]:+.2f}, log(G+5) {b2[1]:+.2f} (R2 {r2:.3f})")

    # 5. choosing G
    out.append("\n5. Choosing the grid: median test RMSE relative to the best grid in hindsight")
    out.append("   ruleT = finest grid with N/P >= T (coarsest grid if none); val = lowest validation error;")
    out.append("   suffix _e = early-stopped KAN (relative to the best early-stopped grid)")
    sel = []
    for key, g in ok.groupby(["dataset", "hidden", "n", "noise", "seed"]):
        g = g.sort_values("grid")
        if len(g) < 3:
            continue
        best, best_e = g.kan_test.min(), g.kan_early.min()
        row = dict(dataset=key[0], noise=key[3], np_max=g.np_ratio.max())
        for T in (1, 2, 4, 8, 16):
            okT = g[g.np_ratio >= T]
            r = okT.iloc[-1] if len(okT) else g.iloc[0]
            row[f"rule{T}"], row[f"rule{T}_e"] = r.kan_test / best, r.kan_early / best_e
        val = g.loc[g.kan_val.idxmin()]
        row.update(val=val.kan_test / best, rule4_beats_val=float(g.loc[g.index[0]].kan_test >= 0) and float(
            (g[g.np_ratio >= 4].iloc[-1] if (g.np_ratio >= 4).any() else g.iloc[0]).kan_test <= val.kan_test),
            val_e_vs_mlp=val.kan_early / g.mlp_early.iloc[0], best_e_vs_mlp=best_e / g.mlp_early.iloc[0])
        for G in (3, 5, 20):
            if (g.grid == G).any():
                r = g[g.grid == G].iloc[0]
                row[f"G{G}"], row[f"G{G}_e"] = r.kan_test / best, r.kan_early / best_e
        sel.append(row)
    sel = pd.DataFrame(sel)
    if sel.empty:
        out.append("  no cell has three or more grids yet")
        return "\n".join(out) + "\n", nz, sel
    cols = [c for c in ["rule1", "rule2", "rule4", "rule8", "rule16", "val", "G3", "G5", "G20"] if c in sel]
    out.append("final checkpoint\n" + sel.groupby("noise")[cols].median().round(2).to_string())
    cols_e = [c for c in ["rule1_e", "rule2_e", "rule4_e", "rule8_e", "rule16_e", "G3_e", "G5_e", "G20_e"] if c in sel]
    out.append("early-stopped\n" + sel.groupby("noise")[cols_e].median().round(2).to_string())
    out.append(f"{len(sel)} cells (dataset, width, N, noise, seed) with three or more grids")
    out.append("fraction of cells where the N/P >= 4 rule is at least as good as validation selection: "
               + ", ".join(f"sigma {k}: {v:.2f}" for k, v in sel.groupby("noise").rule4_beats_val.mean().items()))
    out.append("early-stopped KAN at the validation-selected grid vs early-stopped MLP: median ratio "
               + ", ".join(f"sigma {k}: {v.median():.2f} (KAN better in {(v < 1).mean():.0%})"
                           for k, v in sel.groupby("noise").val_e_vs_mlp))
    return "\n".join(out) + "\n", nz, sel


def figure(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         # IOP asks for Helvetica-type fonts in vector figures, embedded as TrueType
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
                         "mathtext.bf": "sans:bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    ok = d[~d.diverged]
    cen = bin_centres(d)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    def band(ax, frame, col, color, marker, label, ls="-"):
        s = frame.groupby("bin", observed=True)[col]
        q = s.quantile([0.25, 0.5, 0.75]).unstack()
        q = q[s.size() >= 5]
        xs = [cen[b] for b in q.index]
        ax.plot(xs, q[0.5], color=color, marker=marker, ms=4.5, lw=1.3, ls=ls, label=label)
        ax.fill_between(xs, q[0.25], q[0.75], color=color, alpha=0.13, lw=0)

    # (a) memorisation: injected noise barely matters here, so all noise levels are pooled
    band(axes[0], ok, "kan_fit", "#0072B2", "o", "KAN")
    band(axes[0], d, "mlp_fit", "#D55E00", "s", "MLP", ls="--")
    axes[0].set_ylim(0, 1.15)
    axes[0].axhline(1, color="k", lw=0.7, ls=":")
    axes[0].set_ylabel("Train / validation RMSE")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    # (b) KAN vs MLP, both early-stopped, by injected noise
    for nz, (c, mk) in NOISE_STYLE.items():
        band(axes[1], ok[ok.noise == nz], "ratio_early", c, mk,
             "no added noise" if nz == 0 else f"added $\\sigma$ = {nz}")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(FixedLocator([0.8, 1, 1.25, 1.5, 2, 3]))
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].axhline(1, color="k", lw=0.7, ls=":")
    axes[1].set_ylabel("KAN / MLP test RMSE\n(both early-stopped)")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    for ax, lab in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 4, 16], ["0.1", "0.3", "1", "4", "16"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axvline(4, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("Training points per spline parameter, $N/P$")
        ax.text(-0.18, 1.02, f"({lab})", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig6_realdata.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="realdata_real.jsonl")
    ap.add_argument("--out", default="realdata_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = load(args.results)
    txt, _, sel = report(d)
    d.drop(columns="bin").to_csv(os.path.join(args.out, "realdata_runs.csv"), index=False)
    sel.to_csv(os.path.join(args.out, "realdata_gselect.csv"), index=False)
    open(os.path.join(args.out, "realdata_report.txt"), "w").write(txt)
    figure(d, args.out)
    print(txt)
    print("Tables and fig6_realdata written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
