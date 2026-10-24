"""
kan_followup_analyze.py — summaries of the follow-up experiments.

Usage
    python kan_followup_analyze.py --core results_core.jsonl [--dir .] [--out followup_analysis]

Reads followup_ckpt.jsonl, followup_effkan.jsonl, followup_h2.jsonl and
followup_h3.jsonl (whichever exist) and writes one text report per stage plus
CSV tables. The core results are needed to compare against pykan KANs and
same-size MLPs trained on exactly the same data (same seeds and generator).

Metrics as in the paper: test RMSE against the clean function in units of
std(y); noise absorption = train RMSE / noise level (only meaningful when
noise >= 0.1 and the model can fit the clean function).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

BINS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, np.inf]


def read(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                rows.append(r)
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def width_str(w):
    return "-".join(str(x[0] if isinstance(x, list) else x) for x in w)


def core_table(path):
    """One row per core job: KAN/MLP final and early-stopped (every-10) results."""
    rows = []
    for r in read(path):
        key = dict(func=r["func"], width=width_str(r["width"]), n=r["n"], grid=r["grid"],
                   noise=r["noise"], seed=r["seed"], np_ratio=r["n"] / r["p_kan"])
        for m in ["KAN", "MLP"]:
            c = r["curves"].get(m) or []
            if not c or (m == "KAN" and r.get("kan_diverged_at") is not None):
                continue
            best = min(c, key=lambda q: q["val"])
            rows.append(dict(key, model=m, final_test=c[-1]["test"], final_train=c[-1]["train"],
                             early_test=best["test"], early_train=best["train"], early_step=best["step"]))
    return pd.DataFrame(rows)


def absorb(train, noise):
    return np.where(noise > 0, train / np.where(noise > 0, noise, np.nan), np.nan)


# ---------------------------------------------------------------------------
def report_ckpt(rows, core):
    if not rows:
        return "ckpt: no results yet.\n"
    out = []
    for r in rows:
        c = r["curve"]
        if not c or r.get("kan_diverged_at") is not None:
            continue
        best1 = min(c, key=lambda q: q["val"])
        c10 = [q for q in c if q["step"] % 10 == 0]
        best10 = min(c10, key=lambda q: q["val"]) if c10 else best1
        out.append(dict(func=r["func"], width=width_str(r["width"]), n=r["n"], grid=r["grid"], noise=r["noise"],
                        seed=r["seed"], np_ratio=r["np_ratio"], step1=best1["step"], test1=best1["test"],
                        train1=best1["train"], test10=best10["test"], train10=best10["train"], final_test=c[-1]["test"]))
    d = pd.DataFrame(out)
    d["absorb1"] = absorb(d.train1, d.noise)
    d["absorb10"] = absorb(d.train10, d.noise)
    if not core.empty:
        mlp = core[core.model == "MLP"][["func", "width", "n", "grid", "noise", "seed", "early_test", "early_train"]]
        d = d.merge(mlp.rename(columns={"early_test": "mlp_early_test", "early_train": "mlp_early_train"}),
                    on=["func", "width", "n", "grid", "noise", "seed"], how="left")
        d["mlp_absorb"] = absorb(d.mlp_early_train, d.noise)
    d["bin"] = pd.cut(d.np_ratio, BINS)
    agg = {"step1": "median", "test1": "median", "test10": "median", "absorb1": "median", "absorb10": "median"}
    if "mlp_early_test" in d:
        agg.update({"mlp_early_test": "median", "mlp_absorb": "median"})
    t = d.groupby(["bin", "noise"], observed=True).agg(agg)
    if "mlp_early_test" in d:
        t["kan/mlp (every step)"] = t.test1 / t.mlp_early_test
    d.to_csv("ckpt_runs.csv", index=False)
    return ("ckpt: early stopping with a checkpoint at every LBFGS step\n"
            f"  {len(d)} runs. step1 = best step at full resolution; test1/absorb1 = early-stopped at every step;\n"
            "  test10/absorb10 = the same runs early-stopped on the core sweep's every-10 grid.\n\n"
            + t.round(3).to_string() + "\n")


def report_effkan(rows, core):
    if not rows:
        return "effkan: no results yet.\n"
    out = []
    for r in rows:
        c = r["curve"]
        div = r.get("kan_diverged_at") is not None or not c
        base = dict(func=r["func"], width=width_str(r["width"]), n=r["n"], grid=r["grid"], noise=r["noise"],
                    seed=r["seed"], np_ratio=r["np_ratio"], diverged=div)
        if not div:
            best = min(c, key=lambda q: q["val"])
            base.update(eff_final_test=c[-1]["test"], eff_final_train=c[-1]["train"], eff_early_test=best["test"])
        out.append(base)
    d = pd.DataFrame(out)
    if not core.empty:
        for m, tag in [("KAN", "pykan"), ("MLP", "mlp")]:
            sub = core[core.model == m][["func", "width", "n", "grid", "noise", "seed", "final_test", "final_train", "early_test"]]
            d = d.merge(sub.rename(columns={"final_test": f"{tag}_final_test", "final_train": f"{tag}_final_train",
                                            "early_test": f"{tag}_early_test"}),
                        on=["func", "width", "n", "grid", "noise", "seed"], how="left")
    for tag in ["eff", "pykan", "mlp"]:
        col = f"{tag}_final_train"
        if col in d:
            d[f"{tag}_absorb"] = absorb(d[col], d.noise)
    d["bin"] = pd.cut(d.np_ratio, BINS)
    noisy = d[d.noise >= 0.1]
    cols = [c for c in ["eff_absorb", "pykan_absorb", "mlp_absorb", "eff_final_test", "pykan_final_test", "mlp_final_test"] if c in d]
    t = noisy.groupby(["bin"], observed=True)[cols].median()
    rho = ""
    if "pykan_final_test" in d:
        ok = noisy.dropna(subset=["eff_final_test", "pykan_final_test"])
        if len(ok) > 5:
            rho = (f"\n  Spearman correlation of log test error, efficient-kan vs pykan, same cells: "
                   f"{ok[['eff_final_test', 'pykan_final_test']].apply(np.log).corr(method='spearman').iloc[0, 1]:.2f}"
                   f" (n = {len(ok)})")
    d.to_csv("effkan_runs.csv", index=False)
    return ("effkan: second implementation (efficient-kan, same parameter count and training)\n"
            f"  {len(d)} runs, {int(d.diverged.sum())} diverged. Medians over noisy runs (noise >= 0.1):\n\n"
            + t.round(3).to_string() + rho + "\n")


def report_h2(rows, core):
    if not rows:
        return "h2: no results yet.\n"
    out = []
    for r in rows:
        for s in r["stages"]:
            c = s.get("curve") or []
            base = dict(func=r["func"], width=width_str(r["width"]), n=r["n"], noise=r["noise"], seed=r["seed"],
                        grid=s["grid"], np_ratio=s.get("np_ratio", np.nan), diverged=s.get("diverged_at") is not None or not c)
            if c:
                base.update(ext_test=c[-1]["test"], ext_train=c[-1]["train"])
            out.append(base)
    d = pd.DataFrame(out)
    if not core.empty:
        k = core[core.model == "KAN"][["func", "width", "n", "grid", "noise", "seed", "final_test"]]
        d = d.merge(k.rename(columns={"final_test": "scratch_test"}), on=["func", "width", "n", "grid", "noise", "seed"], how="left")
        d["ext_vs_scratch"] = d.ext_test / d.scratch_test
    # per run: is each refinement step an improvement?
    d = d.sort_values(["func", "n", "noise", "seed", "grid"])
    d["prev_test"] = d.groupby(["func", "n", "noise", "seed"]).ext_test.shift(1)
    # the first stage has nothing to compare with
    d["refine_helped"] = np.where(d.prev_test.isna() | d.ext_test.isna(), np.nan, (d.ext_test < d.prev_test).astype(float))
    d["bin"] = pd.cut(d.np_ratio, BINS)
    t1 = d.pivot_table(index="bin", columns="noise", values="refine_helped", aggfunc="mean", observed=True)
    t2 = d.pivot_table(index="grid", columns="noise", values="ext_test", aggfunc="median", observed=True)
    best = d.dropna(subset=["ext_test"]).loc[lambda x: x.groupby(["func", "n", "noise", "seed"]).ext_test.idxmin()]
    t3 = best.pivot_table(index="n", columns="noise", values="grid", aggfunc="median")
    t4 = best.pivot_table(index="n", columns="noise", values="np_ratio", aggfunc="median")
    extra = ""
    if "ext_vs_scratch" in d:
        extra = ("\n\nMedian test error of grid extension relative to training at the same G from scratch "
                 "(below 1 = extension better):\n"
                 + d.pivot_table(index="grid", columns="noise", values="ext_vs_scratch", aggfunc="median").round(2).to_string())
    d.to_csv("h2_stages.csv", index=False)
    return ("h2: grid extension (3 -> 5 -> 10 -> 20 -> 50 -> 100) under noise\n\n"
            "Share of refinement steps that lowered test error, by N/P after the step:\n" + t1.round(2).to_string()
            + "\n\nMedian test error after each stage:\n" + t2.round(4).to_string()
            + "\n\nMedian grid size at which test error was lowest (the right place to stop refining):\n" + t3.to_string()
            + "\n\n... and the N/P at that best grid:\n" + t4.round(2).to_string() + extra + "\n")


def report_h3(rows):
    if not rows:
        return "h3: no results yet.\n"
    out = []
    for r in rows:
        out.append(dict(func=r["func"], n=r["n"], grid=r["grid"], noise=r["noise"], seed=r["seed"], np_ratio=r["np_ratio"],
                        recovered=r.get("symbolic_test") is not None and r["symbolic_test"] < 0.0316,
                        exact=r.get("symbolic_test") is not None and r["symbolic_test"] < 0.01,
                        symbolic_test=r.get("symbolic_test"),
                        spline_test=r.get("spline_test"), stage=r.get("stage_reached"), refit=r.get("refit"),
                        width_pruned=width_str(r["width_pruned"]) if r.get("width_pruned") else None,
                        failed=bool(r.get("failure") or r.get("diverged"))))
    d = pd.DataFrame(out)
    d["bin"] = pd.cut(d.np_ratio, BINS)
    t1 = d.pivot_table(index="bin", columns="noise", values="recovered", aggfunc="mean", observed=True)
    t2 = d.pivot_table(index="func", columns="noise", values="recovered", aggfunc="mean")
    t3 = d.pivot_table(index="bin", columns="noise", values="spline_test", aggfunc="median", observed=True)
    clean_hi = d[(d.noise == 0) & (d.np_ratio >= 4)].groupby("func").recovered.mean()
    d.to_csv("h3_runs.csv", index=False)
    t_exact = d.pivot_table(index="bin", columns="noise", values="exact", aggfunc="mean", observed=True)
    return ("h3: symbolic recovery (prune, snap to library, refit)\n"
            "  recovered = formula R^2 >= 0.999 on clean test data (RMSE < 0.0316); exact = RMSE < 0.01\n"
            f"  {len(d)} runs; pipeline failures {int(d.failed.sum())}; refits reverted {int((d.refit == 'reverted').sum())}\n\n"
            "Recovery rate by N/P (rows) and noise (columns):\n" + t1.round(2).to_string()
            + "\n\nExact recovery rate (RMSE < 0.01) by N/P and noise:\n" + t_exact.round(2).to_string()
            + "\n\nRecovery rate by function and noise:\n" + t2.round(2).to_string()
            + "\n\nBaseline: recovery rate on clean data with N/P >= 4 (what each function can reach at best):\n"
            + clean_hi.round(2).to_string()
            + "\n\nMedian test error of the pruned spline model before snapping:\n" + t3.round(3).to_string() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="results_core.jsonl")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="followup_analysis")
    args = ap.parse_args()
    core = core_table(args.core) if os.path.exists(args.core) else pd.DataFrame()
    if core.empty:
        print("Note: core results not found; comparisons with pykan and MLPs are skipped.")
    os.makedirs(args.out, exist_ok=True)
    here = os.getcwd()
    rows = {s: read(os.path.join(args.dir, f"followup_{s}.jsonl")) for s in ["ckpt", "effkan", "h2", "h3"]}
    os.chdir(args.out)
    try:
        reports = [report_ckpt(rows["ckpt"], core), report_effkan(rows["effkan"], core),
                   report_h2(rows["h2"], core), report_h3(rows["h3"])]
    finally:
        os.chdir(here)
    text = "\n" + ("\n" + "=" * 78 + "\n").join(reports)
    open(os.path.join(args.out, "followup_report.txt"), "w").write(text)
    print(text)
    print("Saved report and CSV tables in", args.out)


if __name__ == "__main__":
    main()
