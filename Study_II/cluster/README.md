# RQ2-bis on the Leonardo cluster (DCGP, Intel CPUs)

Final sweeps run on Leonardo's Data Centric General Purpose partition
(CPU-only blades, 2× Intel Sapphire Rapids per node). The laptop stays for
development and smoke tests. Laptop and cluster timings are not comparable
(recorded per run in `hardware.json`).

## 1. Copy the repo and the data

`data/` (7.9 GB) is git-ignored, so it does not travel with the clone:

```bash
# on Leonardo (login node)
git clone -b cluster_run https://github.com/gdonninelli/LDNets.git
rsync -av --progress <laptop>:Dev/LDNets/data/ LDNets/data/
```

The notebooks address the data as `../../data`, so the repo layout must stay
as cloned (run from `Study_II/Case_B_2D-Stokes/` and `Study_II/Case_C_AP1D/`).

## 2. Python 3.11.7 environment (do once)

The laptop pins (`requirements.txt`) use TensorFlow 2.8, which has no Python
3.11 wheels. On the cluster install `requirements-cluster.txt` instead
(TF 2.13.1: newest TF with the Keras-2 API this repo uses, and the last whose
`tf.config.experimental.enable_op_determinism` matches the notebook code):

```bash
module purge
module avail python   # pick the 3.11.7 module available on the system
module load python/3.11.7
python3.11 -m venv "$WORK/venvs/ldnets311"
source "$WORK/venvs/ldnets311/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-cluster.txt
python -c "import tensorflow, numpy, scipy, pandas; print(tensorflow.__version__, numpy.__version__, scipy.__version__, pandas.__version__)"
```

The sweep script loads this module and uses `$WORK/venvs/ldnets311` by
default. Set `LDNETS_VENV` before submitting if you created the environment
elsewhere.

## 3. Accuracy sweeps (Pass A)

One job per case folder; split a case with a Slurm array, one shard per task.
At ~50 min (B) / ~40 min (C) per L0 run at 10000 iterations, 4 shards of
~8 L0 runs fit in ~7 h each (inside the 24 h job limit):

```bash
cd Study_II/cluster
sbatch --array=0-3 sbatch_sweep.sh Case_B_2D-Stokes
sbatch --array=0-3 sbatch_sweep.sh Case_C_AP1D
```

Each shard writes `results_shard<i>/` inside its case folder. The settings
guard in the notebook refuses to mix these with the laptop's 1000-iteration
results: keep the old `results/` folders untouched (rename, do not merge).

Merge the shards of one case into `results/`:

```bash
cd ../Case_B_2D-Stokes   # or ../Case_C_AP1D
python ../cluster/merge_results.py results results_shard0 results_shard1 results_shard2 results_shard3
```

## 4. Timing (Pass T)

After merging, one exclusive node per case, nothing else on it:

```bash
cd Study_II/cluster
sbatch sbatch_timing.sh Case_B_2D-Stokes
sbatch sbatch_timing.sh Case_C_AP1D
```

This fills `timing.csv` + `hardware.json` in the merged `results/` via
`RQ2BIS_STAGE5`. Thread counts default to `1,2,4,6,12,18`; override with
`RQ2BIS_THREADS` up to the node's cores (the template uses
`1,2,4,8,16,32,64,112` — check `lscpu` first).

## 5. What changed for the cluster (branch `cluster_run`)

- `requirements-cluster.txt`: TF 2.13.1 + NumPy 1.24 + SciPy 1.11 + pandas 2 + Matplotlib 3.7 for Python 3.11.
- `optimization.py` (both case folders + `src/`): `np.product` → `np.prod`
  (removed in NumPy 2.x; identical otherwise). No pandas/numpy-2-only API is used elsewhere.
- Notebooks: `RQ2BIS_OUT` selects the results folder (per-shard folders);
  `RQ2BIS_SHARD=i/n` restricts `sweep()` to `configs[i::n]` (unset = everything,
  i.e. laptop behaviour is unchanged); `RQ2BIS_STAGE5=1` runs Stage 5 headlessly;
  setup comment no longer assumes Apple GPUs; `hardware_record()` reads the CPU
  model from `/proc/cpuinfo`/`lscpu` on Linux and records Slurm variables.
- `timing_worker.py`: also mirrors the thread count into `MKL_NUM_THREADS` /
  `OPENBLAS_NUM_THREADS` (Intel MKL / OpenBLAS under NumPy/SciPy).
- This folder: Slurm templates + `merge_results.py`.

## 6. Known limits

- No GPU use: DCGP nodes have none; the code hides GPUs and runs float64 on CPU.
- `plt.show()` under `MPLBACKEND=Agg` only warns; figures are not saved to disk
  (re-run the analysis cells on the laptop against merged results for plots).
- Do not run two processes (or two shards) in the same results folder.
