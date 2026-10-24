"""
kan_realdata.py — does the N/P threshold hold on real measured data?

Five public UCI regression datasets from physics and engineering:
    airfoil   NASA airfoil self-noise, wind-tunnel measurements      1503 x 5
    ccpp      combined-cycle power plant, hourly sensor averages     9568 x 4
    concrete  concrete compressive strength, lab tests               1030 x 8
    energy    building heating load (simulated, nearly noise-free)    768 x 8
    yacht     yacht hull residuary resistance, towing tank (log y)    308 x 6

For every dataset, training-set size N and grid G are varied so that N/P spans
the same range as the synthetic sweep. Real labels already carry measurement
noise of unknown size; on top of that we inject Gaussian label noise of
sigma = 0, 0.1 and 0.3 (units of std(y)) into the training and validation
labels. The test set always keeps the original labels.

Each job trains a pykan KAN of width [d, h, 1] with h = 2 or 5 (the narrow
network reaches higher N/P on the small datasets) and a parameter-matched MLP,
exactly as in kan_sweep.py, except that KANs are checkpointed after every LBFGS
step (so early stopping is as favourable to KANs as possible).

Usage (same folder as kan_sweep.py)
    pip install ucimlrepo                       # once, for the download
    python kan_realdata.py --stage fetch        # downloads the five datasets to data/
    python kan_realdata.py --stage smoke --workers 4
    python kan_realdata.py --stage real --workers 10
    python kan_realdata_analyze.py realdata_real.jsonl --out realdata_analysis

Results go to realdata_<stage>.jsonl. Stop any time by closing the window; run
the same command again to continue. Expected run time with --workers 10 on a
6-core desktop: about 4-6 hours for 'real' (an overnight run).
"""

import argparse
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import torch

import kan_sweep as S

# checkpoint KANs after every LBFGS step, MLPs every 100 AdamW steps
S.KAN_EVERY = 1
S.MLP_EVERY = 100

DATA_DIR = "data"
HIDDENS = [2, 5]          # KAN width [d, h, 1]
GRIDS = [3, 5, 10, 20, 50]
NOISES = [0.0, 0.1, 0.3]
SEEDS = list(range(5))
MAX_NG = 50000

# name: (UCI id, target column, log-transform target?, training sizes)
DATASETS = {
    "airfoil":  (291, "scaled-sound-pressure", False, [100, 200, 400, 800]),
    "ccpp":     (294, "PE", False, [100, 200, 400, 800, 1600, 3200]),
    "concrete": (165, "Concrete compressive strength", False, [100, 200, 400]),
    "energy":   (242, "Y1", False, [100, 200, 400]),
    "yacht":    (243, "residuary_resistance", True, [75, 150]),
}
YACHT_ZIP = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
YACHT_COLS = ["longitudinal_position", "prismatic_coefficient", "length_displacement",
              "beam_draught", "length_beam", "froude_number", "residuary_resistance"]


# ---------------------------------------------------------------------------
# Download once, then read the cached CSV
# ---------------------------------------------------------------------------
def fetch_all():
    import pandas as pd
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, (uci_id, target, _, _) in DATASETS.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            print(f"  {name}: already in {path}")
            continue
        if name == "yacht":
            # not importable through ucimlrepo; read the archive's whitespace-separated file
            import urllib.request
            raw = urllib.request.urlopen(YACHT_ZIP, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                txt = z.read([n for n in z.namelist() if n.endswith(".data")][0]).decode()
            df = pd.read_csv(io.StringIO(txt), sep=r"\s+", header=None, names=YACHT_COLS)
        else:
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=uci_id)
            df = d.data.features.copy()
            df[target] = d.data.targets[target].values
        df = df.dropna().astype(float)
        df = df[[c for c in df.columns if c != target] + [target]]   # target last
        df.to_csv(path, index=False)
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1] - 1} features -> {path}")


@lru_cache(maxsize=None)
def load(name):
    import pandas as pd
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    x = df.iloc[:, :-1].to_numpy(np.float64)
    y = df.iloc[:, -1].to_numpy(np.float64)
    if DATASETS[name][2]:
        y = np.log(y)
    # features to [-1, 1] with the full-data range (features only, no labels)
    lo, hi = x.min(0), x.max(0)
    x = 2 * (x - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
    return x, y


def split_sizes(name):
    total = len(load(name)[1])
    n_test = min(2000, round(0.25 * total))
    return total, n_test


def make_data(name, n, noise, seed):
    x, y = load(name)
    total, n_test = split_sizes(name)
    n_val = max(50, n // 4)
    rng = np.random.default_rng(1000 * seed + 11)
    perm = rng.permutation(total)
    te, tr, va = perm[:n_test], perm[n_test:n_test + n], perm[n_test + n:n_test + n + n_val]
    mu, sd = y[tr].mean(), y[tr].std()
    ys = (y - mu) / sd                     # original labels, in units of train std(y)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    y_tr = ys[tr] + noise * rng.standard_normal(len(tr))
    y_va = ys[va] + noise * rng.standard_normal(len(va))
    return dict(x=t(x[tr]), y=t(y_tr)[:, None], x_val=t(x[va]), y_val=t(y_va)[:, None],
                x_te=t(x[te]), y_te=t(ys[te])[:, None])


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def make_jobs(stage):
    jobs = []
    if stage == "smoke":
        for name, n, g, nz in [("airfoil", 100, 5, 0.1), ("yacht", 75, 3, 0.0), ("ccpp", 200, 10, 0.3)]:
            jobs.append(dict(dataset=name, hidden=5, n=n, grid=g, noise=nz, seed=0))
    elif stage == "real":
        for name, (_, _, _, sizes) in DATASETS.items():
            for h in HIDDENS:
                for n in sizes:
                    for g in GRIDS:
                        if n * g > MAX_NG:
                            continue
                        for nz in NOISES:
                            for s in SEEDS:
                                jobs.append(dict(dataset=name, hidden=h, n=n, grid=g, noise=nz, seed=s))
    else:
        raise ValueError(stage)
    for j in jobs:
        d_in = len(load(j["dataset"])[0][0])
        j["width"] = [d_in, j["hidden"], 1]
        j["np_ratio"] = j["n"] / S.kan_param_count(j["width"], j["grid"])
        j["id"] = f"{j['dataset']}|h{j['hidden']}|N{j['n']}|G{j['grid']}|s{j['noise']}|seed{j['seed']}"
    return jobs


def run_job(job):
    torch.set_num_threads(1)
    random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
    t0 = time.time()
    try:
        d = make_data(job["dataset"], job["n"], job["noise"], job["seed"])
        before = dict(S.LSTSQ_FALLBACKS)
        kan_curve, p_kan, kan_div = S.train_kan(d, job["width"], job["grid"], job["seed"])
        fallbacks = {k: S.LSTSQ_FALLBACKS[k] - before[k] for k in before}
        n_in = job["width"][0]
        mlp_curve, p_mlp = S.train_mlp(d, n_in, S.mlp_hidden_for(n_in, p_kan), job["seed"])
        return dict(job, ok=True, p_kan=p_kan, p_mlp=p_mlp, seconds=round(time.time() - t0, 1),
                    kan_diverged_at=kan_div, lstsq_fallbacks=fallbacks,
                    curves={"KAN": kan_curve, "MLP": mlp_curve})
    except Exception as e:
        return dict(job, ok=False, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "smoke", "real"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run only the N cheapest jobs (testing)")
    args = ap.parse_args()

    missing = [n for n in DATASETS if not os.path.exists(os.path.join(DATA_DIR, f"{n}.csv"))]
    if args.stage == "fetch" or missing:
        print("Downloading datasets to", os.path.abspath(DATA_DIR))
        fetch_all()
        if args.stage == "fetch":
            for n in DATASETS:
                total, n_test = split_sizes(n)
                print(f"  {n}: {total} rows, test {n_test}, N up to {max(DATASETS[n][3])}")
            return

    out = args.out or f"realdata_{args.stage}.jsonl"
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
    todo.sort(key=lambda j: -(j["n"] * j["grid"]))
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
                print(f"[{finished}/{len(todo)}] {el / 3600:.2f} h elapsed, "
                      f"~{el / finished * (len(todo) - finished) / 3600:.1f} h left"
                      + (f", {failed} failed" if failed else ""), flush=True)
    print("Done." + (f" {failed} jobs failed; run the same command again to retry them." if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
