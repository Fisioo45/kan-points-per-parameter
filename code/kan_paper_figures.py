"""
kan_paper_figures.py — figures and headline numbers for the manuscript.

Reads the outputs of kan_sweep_analyze.py (analysis/cells.csv) and writes
publication figures (PDF for LaTeX, PNG for previews) plus a text file with
every number quoted in the paper, so the text can be checked against the data.

Usage
    python kan_paper_figures.py analysis/cells.csv --out figures

Figures follow MLST guidance: colour is never the only carrier of meaning
(each series also has its own marker and line style), and they are sized for a
single journal column (3.4 in) or the full page width (7 in).
"""

import argparse
import os

import numpy as np
import pandas as pd

BINS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, np.inf]
STYLE = {  # model -> (colour, marker, line style); colour-blind safe (Okabe-Ito)
    "KAN": ("#0072B2", "o", "-"),
    "MLP": ("#555555", "s", "--"),
}
NOISE_STYLE = {  # noise level -> (colour, marker)
    0.01: ("#56B4E9", "v"), 0.05: ("#009E73", "^"), 0.1: ("#E69F00", "o"),
    0.2: ("#D55E00", "s"), 0.4: ("#CC79A7", "D"),
}


def geo_centre(interval):
    lo, hi = interval.left, interval.right
    lo = max(lo, 0.125)
    hi = hi if np.isfinite(hi) else 64
    return float(np.sqrt(lo * hi))


def prepare(path):
    c = pd.read_csv(path)
    k = c[(c.model == "KAN") & (c.variant == "final")]
    clean = k[k.noise == 0].set_index(["func", "width", "n", "grid"]).test.rename("kan_clean")
    c = c.join(clean, on=["func", "width", "n", "grid"])
    c["bin"] = pd.cut(c.np_ratio, BINS)
    return c


def crossing(d, level):
    """N/P where the binned median of `absorb` first rises through `level` (log-linear interpolation)."""
    edges = np.arange(-1.5, 2.01, 0.25)
    g = (d.assign(lb=pd.cut(np.log10(d.np_ratio), edges))
           .groupby("lb", observed=True)
           .agg(x=("np_ratio", lambda s: np.exp(np.log(s).mean())), a=("absorb", "median"))
           .dropna())
    xs, ys = np.log10(g.x.values), g.a.values
    for i in range(1, len(xs)):
        if ys[i - 1] < level <= ys[i]:
            t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return 10 ** (xs[i - 1] + t * (xs[i] - xs[i - 1]))
    return np.nan


def ratio_crossing(r):
    """N/P where the KAN/MLP error ratio falls through 1, interpolated between bin centres."""
    xs = np.log10([geo_centre(b) for b in r.index])
    ys = np.log10(r.values)
    for i in range(1, len(xs)):
        if ys[i - 1] > 0 >= ys[i]:
            t = ys[i - 1] / (ys[i - 1] - ys[i])
            return 10 ** (xs[i - 1] + t * (xs[i] - xs[i - 1]))
    return np.nan


def numbers(c):
    out = []
    L = c[(c.noise > 0) & (c.kan_clean < 0.1)]
    KF = L[(L.model == "KAN") & (L.variant == "final")]
    per = KF.groupby("func").apply(lambda d: crossing(d, 0.9), include_groups=False)
    rng = np.random.default_rng(0)
    funcs = per.index.values
    boot = [np.nanmedian(per.loc[rng.choice(funcs, len(funcs))].values) for _ in range(5000)]
    out.append("Memorisation threshold (KAN, learnable cells, final checkpoint)")
    out.append(f"  N/P where median absorption = 0.9, pooled: {crossing(KF, 0.9):.2f}")
    out.append(f"  per function: " + ", ".join(f"{f} {v:.2f}" for f, v in per.items()))
    out.append(f"  median over functions {per.median():.2f}, bootstrap 95% CI "
               f"[{np.nanpercentile(boot, 2.5):.2f}, {np.nanpercentile(boot, 97.5):.2f}]")
    out.append(f"  N/P where median absorption = 0.5, pooled: {crossing(KF, 0.5):.2f}")
    MF = L[(L.model == "MLP") & (L.variant == "final")]
    out.append(f"  MLP: N/P where median absorption = 0.9, pooled: {crossing(MF, 0.9):.2f}")
    H = L[L.noise >= 0.1]
    out.append(f"  noise >= 0.1 only (as in figure 1): KAN 0.9 at N/P {crossing(H[(H.model=='KAN')&(H.variant=='final')], 0.9):.2f}, "
               f"0.5 at {crossing(H[(H.model=='KAN')&(H.variant=='final')], 0.5):.2f}; "
               f"MLP 0.9 at {crossing(H[(H.model=='MLP')&(H.variant=='final')], 0.9):.2f}")
    for v in ["final", "early"]:
        t = H[H.variant == v].pivot_table(index="bin", columns="model", values="absorb", aggfunc="median", observed=True)
        out.append(f"  median absorption, {v}, noise >= 0.1: " + ", ".join(f"{b}: KAN {r.KAN:.2f} / MLP {r.MLP:.2f}" for b, r in t.iterrows()))

    A = c[c.noise > 0]
    out.append("\nKAN/MLP median test-RMSE ratio by N/P bin (all noisy cells)")
    for v in ["final", "early"]:
        t = A[A.variant == v].pivot_table(index="bin", columns="model", values="test",
                                          aggfunc="median", observed=True)
        out.append(f"  {v}: " + ", ".join(f"{b}: {r:.2f}" for b, r in (t.KAN / t.MLP).items()))
    out.append("\nN/P where KANs start to beat same-size MLPs (final), by noise level")
    for nz in sorted(A.noise.unique()):
        t = A[(A.variant == "final") & (A.noise == nz)].pivot_table(
            index="bin", columns="model", values="test", aggfunc="median", observed=True)
        x = ratio_crossing(t.KAN / t.MLP)
        out.append(f"  noise {nz}: " + (f"N/P ~ {x:.1f}" if np.isfinite(x) else
                                        f"never in tested range (max N/P {c.np_ratio.max():.0f})"))
    C0 = c[(c.noise == 0) & (c.variant == "final")].pivot_table(
        index="bin", columns="model", values="test", aggfunc="median", observed=True)
    out.append("\nClean data, median test RMSE KAN / MLP by N/P bin")
    out.append("  " + ", ".join(f"{b}: {r.KAN:.4f}/{r.MLP:.4f}" for b, r in C0.iterrows()))
    return "\n".join(out) + "\n"


def fig_absorption(c, out):
    import matplotlib.pyplot as plt
    # Absorption measures memorisation only when noise dominates the fitting error:
    # train RMSE = sqrt(noise^2 + bias^2), so at noise 0.01 a small bias reads as
    # absorption >> 1. Use noise >= 0.1 and cells where the KAN can fit clean data.
    L = c[(c.noise >= 0.1) & (c.kan_clean < 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    for ax, variant, title in zip(axes, ["final", "early"], ["(a) End of training", "(b) Early-stopped"]):
        for model in ["MLP", "KAN"]:
            col, mk, ls = STYLE[model]
            d = L[(L.model == model) & (L.variant == variant)]
            ax.scatter(d.np_ratio, d.absorb, s=5, alpha=0.18, color=col, marker=mk, lw=0)
            g = d.groupby("bin", observed=True).absorb
            x = [geo_centre(b) for b in g.median().index]
            ax.plot(x, g.median().values, color=col, marker=mk, ls=ls, lw=1.6, ms=5, label=model)
            ax.fill_between(x, g.quantile(0.25).values, g.quantile(0.75).values, color=col, alpha=0.15, lw=0)
        ax.axhline(1, color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 70)
        ax.set_ylim(-0.05, 1.35)
        ax.set_xlabel("Training points per spline parameter, $N/P$")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("Noise absorbed\n(train RMSE / noise level)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig1_absorption.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ratio(c, out):
    import matplotlib.pyplot as plt
    A = c[c.noise > 0]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for nz, (col, mk) in NOISE_STYLE.items():
        t = A[(A.variant == "final") & (A.noise == nz)].pivot_table(
            index="bin", columns="model", values="test", aggfunc="median", observed=True)
        x = [geo_centre(b) for b in t.index]
        ax.plot(x, (t.KAN / t.MLP).values, color=col, marker=mk, lw=1.3, ms=4.5, label=f"{nz}")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.annotate("above: KAN worse\nbelow: KAN better", xy=(0.15, 0.3), fontsize=7.5, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training points per spline parameter, $N/P$")
    ax.set_ylabel("Test RMSE ratio, KAN / MLP")
    ax.legend(title="Noise level", fontsize=7, title_fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig2_kan_mlp_ratio.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_grid_vs_width(c, out):
    import matplotlib.pyplot as plt
    W = c[(c.func.isin(["exp_sin_sq", "radial"])) & (c.model == "KAN") & (c.variant == "final")
          & (c.noise.isin([0.1, 0.4])) & (c.grid.isin([5, 20])) & (c.n.isin([300, 1000]))
          & (c.width.isin(["2-2-1", "2-5-1", "2-10-1", "2-20-1"]))]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for g, (col, mk) in {5: ("#009E73", "o"), 20: ("#D55E00", "^")}.items():
        d = W[W.grid == g]
        ax.scatter(d.np_ratio, d.absorb, color=col, marker=mk, s=22, alpha=0.8,
                   label=f"G = {g}, widths 2 to 20")
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Training points per spline parameter, $N/P$")
    ax.set_ylabel("Noise absorbed")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"fig3_grid_vs_width.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", help="analysis/cells.csv from kan_sweep_analyze.py")
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
    c = prepare(args.cells)
    txt = numbers(c)
    open(os.path.join(args.out, "numbers.txt"), "w").write(txt)
    print(txt)
    fig_absorption(c, args.out)
    fig_ratio(c, args.out)
    fig_grid_vs_width(c, args.out)
    print("Saved figures in", args.out)


if __name__ == "__main__":
    main()
