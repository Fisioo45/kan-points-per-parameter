# Points per parameter: data density and the noise robustness of KANs

Code, raw results and figures for the paper

> **Points per parameter: data density governs the noise robustness of Kolmogorov–Arnold networks**
> [Authors], submitted to *Machine Learning: Science and Technology* (2026).
> [Zenodo DOI badge and journal DOI to be added]

When Kolmogorov–Arnold networks (KANs) are trained on noisy labels, how much of the noise they memorise depends mainly on **N/P**, the number of training points per trainable spline parameter. In the paper's synthetic experiments, KANs average out 90% of the label noise only above N/P ≈ 4. Parameter-matched MLPs reach the same level at N/P ≈ 1.3. This repository has everything needed to reproduce each table and figure of the paper.

## Repository layout

```
code/                     all experiment and analysis scripts (run them from this folder)
results/raw/              raw results, one JSON line per training run (gzip-compressed)
results/tables/core/      analysis outputs of the core sweep (cell medians, regression, threshold)
results/tables/followup/  analysis outputs of the follow-up experiments
results/tables/realdata/  analysis outputs of the real-data experiment
figures/                  Figures 1–6 of the paper (PDF and PNG)
```

## Installation

Python 3.10 or newer. Every experiment runs on a CPU; a GPU does not speed up networks this small.

```bash
conda create -n kan-np python=3.11
conda activate kan-np
pip install -r requirements.txt
```

The core, follow-up and measured-data experiments ran with PyTorch 2.6.0 and pykan 0.2.8 on Windows, using an AMD Ryzen 5 5600 with 10 parallel workers. The pilot, the optimiser swap (Table 3) and the MLP weight-decay check ran with PyTorch 2.14.1 and pykan 0.2.8 under Python 3.13 on a Linux cloud CPU. Other versions of pykan may change the results, because its training defaults have changed between releases.

## Reproducing the analyses from the raw results (minutes)

Decompress the raw results into `code/`, then run the analysis scripts:

```bash
cd code
python unpack_results.py                            # writes the .jsonl files into code/

python kan_sweep_analyze.py results_core.jsonl results_width.jsonl --out analysis
python kan_paper_figures.py analysis/cells.csv --out figures                  # Figures 1-3
python kan_followup_analyze.py --core results_core.jsonl --out followup_analysis
python kan_followup_figures.py followup_analysis --out figures                 # Figures 4-5
python kan_realdata_analyze.py realdata_real.jsonl --out realdata_analysis     # Figure 6, Tables 6-7
```

The pilot and the optimiser swap write small CSV files rather than JSON lines; their outputs are in `results/tables/pilot/` and `results/tables/optswap/` (Table 3 is the `absorb` column of `optswap/summary.csv`, rounded). The per-stage table of the grid-extension runs is also kept in `results/tables/followup/h2_stages.csv`, and Figure 4 can be redrawn from it directly:

```bash
python kan_followup_figures.py followup_analysis --h2 ../results/tables/followup/h2_stages.csv --out figures
```

## Re-running the experiments (hours)

The sweep scripts (`kan_sweep.py`, `kan_followup.py`, `kan_realdata.py`) run their jobs in parallel (`--workers`) and append one line per finished job to a `.jsonl` file. You can stop a run at any time by closing the window. Running the same command again continues where it stopped. Start each of these with its `smoke` stage (`--stage smoke`), which checks the installation in a few minutes; the pilot and the optimiser swap have `--quick` for the same purpose.

| Paper | Script and command | Jobs | Time with 10 workers |
| --- | --- | --- | --- |
| Pilot (not in paper) | `python kan_noise_pilot.py` | — | ~1 h (single process) |
| Table 3 (optimiser swap) | `python kan_optimizer_swap.py` | 72 | ~2 h (single process) |
| Sections 4.1–4.7, Figures 1–3 | `python kan_sweep.py --stage core --workers 10` then `--stage width` | 7,200 + 480 | ~13 h |
| Section 4.3 (every-step early stopping) | `python kan_followup.py --stage ckpt --workers 10` | 800 | ~1.5 h |
| Section 4.8, Table 4 (efficient-kan) | `python kan_followup.py --stage effkan --workers 10` | 540 | ~0.5 h |
| Section 5.1, Figure 4 (grid extension) | `python kan_followup.py --stage h2 --workers 10` | 400 | ~2 h |
| Section 5.2, Table 5, Figure 5 (symbolic recovery) | `python kan_followup.py --stage h3 --workers 10` | 432 | ~4.5 h |
| Section 3.2 (MLP weight-decay check) | `python check_mlp_weight_decay.py --workers 4` | 96 | ~0.5 h |
| Section 6, Tables 6–7, Figure 6 (measured data) | `python kan_realdata.py --stage real --workers 10` | 2,610 | ~4 h |

The measured-data experiment first downloads five public datasets from the UCI Machine Learning Repository (`python kan_realdata.py --stage fetch`): airfoil self-noise, combined-cycle power plant, concrete compressive strength, energy efficiency and yacht hydrodynamics. They are not redistributed here.

### Notes

- **Windows:** use `--workers` no larger than the number of logical cores. Ctrl+C can hang with multiprocessing on Windows. Closing the window is safe, because finished jobs are already saved.
- **pykan patch:** pykan 0.2.8 can crash inside its grid update when `torch.linalg.lstsq` fails, which happens with some Intel MKL builds on Windows. `kan_sweep.py` replaces that solver with one that calls pykan's original solver first and falls back to SVD-based solvers only if that call fails. Non-finite values are recorded as divergence, not hidden. The number of fallbacks is stored with every job (`lstsq_fallbacks`).
- **Seeds:** each job draws its data from a random generator seeded by its seed index, and seeds the network initialisation the same way, so a rerun reproduces the same data and the same starting network. The follow-up experiments reuse the core sweep's data generator, so their runs pair with the core runs one to one.

## Raw result format

Each line of a `.jsonl` file is one job, holding its settings (`func` or `dataset`, `width`, `n`, `grid`, `noise`, `seed`, `np_ratio`), the parameter counts (`p_kan`, `p_mlp`), any divergence (`kan_diverged_at`), and the training curves of both models (`curves.KAN` and `curves.MLP`). Each curve is a list of checkpoints with training, validation and test RMSE in units of std(y). In the synthetic experiments, test RMSE is measured against the clean function. In the measured-data experiment, it is measured against the original labels.

## Third-party code

- `code/efficient_kan_v010.py` is a copy of `kan.py` from [efficient-kan](https://github.com/Blealtan/efficient-kan) v0.1.0 by Huanqi Cao (MIT licence; the original notice is kept at the top of the file).
- [pykan](https://github.com/KindXiaoming/pykan) (MIT licence) is installed from PyPI and not included here.

## Citation

If you use this code or these results, please cite the paper (see `CITATION.cff`).

## Licence

MIT; see `LICENSE`.
