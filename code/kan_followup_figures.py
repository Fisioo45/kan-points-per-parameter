"""
kan_followup_figures.py — figures and numbers for the follow-up experiments.

Reads the CSV tables written by kan_followup_analyze.py:
    h2_stages.csv  (grid extension)   -> fig4_grid_extension.(pdf|png)
    h3_runs.csv    (symbolic recovery) -> fig5_symbolic_recovery.(pdf|png)
    ckpt_runs.csv, effkan_runs.csv     -> numbers only (followup_numbers.txt)

Usage
    python kan_followup_figures.py followup_analysis --out figures
"""

import argparse
import os

import numpy as np
import pandas as pd

NOISE_STYLE = {
    0.0: ("#000000", "o"), 0.05: ("#009E73", "^"), 0.1: ("#E69F00", "s"),
    0.2: ("#D55E00", "D"), 0.4: ("#CC79A7", "v"),
}


def fig_h2(h2, out):
    import matplotlib.pyplot as plt
    d = h2.dropna(subset=["ext_test"]).copy()
    base = d[d.grid == 3].set_index(["func", "n", "noise", "seed"]).ext_test.rename("g3")
    d = d.join(base, on=["func", "n", "noise", "seed"])
    d["rel"] = d.ext_test / d.g3
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for nz, (col, mk) in NOISE_STYLE.items():
        s = d[d.noise == nz].groupby("grid").rel
        med = s.median()
        ax.plot(med.index, med.values, color=col, marker=mk, ms=4.5, lw=1.3,
                label="clean" if nz == 0 else f"{nz}")
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100], ["3", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Grid size $G$ after refinement")
    ax.set_ylabel("Test RMSE relative to $G = 3$")
    ax.legend(title="Noise level", fontsize=7, title_fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig4_grid_extension.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_h3(h3, out):
    import matplotlib.pyplot as plt
    d = h3[h3.func.isin(["additive3", "exp_sin_sq"])].copy()
    bins = [0, 0.5, 1, 2, 4, 100]
    d["bin"] = pd.cut(d.np_ratio, bins)
    centres = {iv: float(np.sqrt(max(iv.left, 0.2) * min(iv.right, 16))) for iv in d.bin.cat.categories}
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for nz, (col, mk) in NOISE_STYLE.items():
        if nz not in set(d.noise):
            continue
        s = d[d.noise == nz].groupby("bin", observed=True).recovered.mean()
        ax.plot([centres[b] for b in s.index], s.values, color=col, marker=mk, ms=4.5, lw=1.3,
                label="clean" if nz == 0 else f"{nz}")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 4, 8], ["0.3", "0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Training points per spline parameter, $N/P$")
    ax.set_ylabel("Formula recovered ($R^2 \\geq 0.999$)")
    ax.legend(title="Noise level", fontsize=7, title_fontsize=8, frameon=False, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig5_symbolic_recovery.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def numbers(folder):
    out = []
    p = os.path.join(folder, "ckpt_runs.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        c["bin"] = pd.cut(c.np_ratio, [0, 0.5, 1, 2])
        g = c.groupby("bin", observed=True).agg(n=("test1", "size"), best_step=("step1", "median"),
                                                kan_every1=("test1", "median"), kan_every10=("test10", "median"),
                                                mlp_early=("mlp_early_test", "median"))
        g["improvement_%"] = (1 - g.kan_every1 / g.kan_every10) * 100
        g["kan/mlp"] = g.kan_every1 / g.mlp_early
        out.append("Early stopping at every LBFGS step (KAN) vs early-stopped MLP\n" + g.round(3).to_string())
        out.append(f"  median best step {c.step1.median():.0f} of 200; KAN worse than MLP in "
                   f"{(c.test1 > c.mlp_early_test).mean() * 100:.1f}% of runs")
    p = os.path.join(folder, "effkan_runs.csv")
    if os.path.exists(p):
        e = pd.read_csv(p)
        out.append(f"\nefficient-kan: {len(e)} runs, {int(e.diverged.sum())} diverged "
                   f"({int(e[e.diverged].grid.eq(3).sum())} of them at G = 3)")
    p = os.path.join(folder, "h2_stages.csv")
    if os.path.exists(p):
        d = pd.read_csv(p).sort_values(["func", "n", "noise", "seed", "grid"])
        d["prev"] = d.groupby(["func", "n", "noise", "seed"]).ext_test.shift(1)
        d["helped"] = np.where(d.prev.isna() | d.ext_test.isna(), np.nan, (d.ext_test < d.prev).astype(float))
        d["ratio"] = d.ext_test / d.prev
        mid = d[(d.np_ratio > 0.5) & (d.np_ratio <= 16)]
        t = mid.groupby("noise").agg(helped=("helped", "mean"), step_ratio=("ratio", "median"))
        out.append("\nGrid extension, refinement steps landing at 0.5 < N/P <= 16\n" + t.round(2).to_string())
        best = d.dropna(subset=["ext_test"]).loc[lambda x: x.groupby(["func", "n", "noise", "seed"]).ext_test.idxmin()]
        out.append("  best grid = 3 (no refinement) in: " + ", ".join(
            f"noise {nz}: {(s.grid == 3).mean() * 100:.0f}%" for nz, s in best.groupby("noise")))
    p = os.path.join(folder, "h3_runs.csv")
    if os.path.exists(p):
        h = pd.read_csv(p)
        out.append("\nSymbolic recovery rate by function and noise\n"
                   + h.pivot_table(index="func", columns="noise", values="recovered", aggfunc="mean").round(2).to_string())
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with the CSV tables from kan_followup_analyze.py")
    ap.add_argument("--h2", default=None, help="h2_stages.csv if it is in another folder")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         # IOP asks for Helvetica-type fonts in vector figures, embedded as TrueType
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
                         "mathtext.bf": "sans:bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    h2p = args.h2 or os.path.join(args.folder, "h2_stages.csv")
    if os.path.exists(h2p):
        fig_h2(pd.read_csv(h2p), args.out)
    h3p = os.path.join(args.folder, "h3_runs.csv")
    if os.path.exists(h3p):
        fig_h3(pd.read_csv(h3p), args.out)
    txt = numbers(args.folder)
    if args.h2:
        import shutil
        shutil.copy(args.h2, os.path.join(args.folder, "h2_stages.csv"))
        txt = numbers(args.folder)
    open(os.path.join(args.out, "followup_numbers.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
