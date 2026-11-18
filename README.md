# Role of Nonlinear Spatial Reconstruction and Computational Scalability of Latent Dynamics Networks

[![Upstream Zenodo Data](https://zenodo.org/badge/DOI/10.5281/zenodo.10436827.svg)](https://doi.org/10.5281/zenodo.10436827)

This repository contains the compiled report ([`Report.pdf`](Report.pdf)), analysis scripts, experiment notebooks, cluster execution scripts, and stored numerical results investigating **when spatial nonlinearity is needed in Latent Dynamics Networks (LDNets)** and **how accuracy scales with network capacity, quasi-Newton iterations, FLOPs, and multi-core wall-clock time** across partial differential equation (PDE) benchmarks.

It builds upon the upstream LDNets codebase by Regazzoni et al. (2024) [[1]](#references).

---

## Report

The full technical report is provided at the root of this repository:

- **[`Report.pdf`](Report.pdf)** — *On the Role of Nonlinear Spatial Reconstruction and the Computational Scalability of Latent Dynamics Networks for Partial Differential Equations* (Giulio Donninelli, Politecnico di Milano).

---

## Repository Structure

```text
.
├── Report.pdf                      # Full compiled technical report
├── README.md
├── requirements.txt                # Workstation Python 3.9 pins (TensorFlow 2.8)
├── requirements-macos.txt          # Apple Silicon macOS pins
├── requirements-cluster.txt        # CINECA Leonardo Python 3.11.7 pins (TensorFlow 2.13.1)
│
├── Study_I/                        # Study I: Spatial reconstructor ablation (816 runs)
│   ├── README.md                   # Detailed Study I narrative and tables
│   ├── protocol.md                 # Experimental protocol
│   ├── report_analysis.py          # Pure-NumPy verification, metrics, and figure generator
│   ├── figures/                    # Study I figures (fig1..fig9) and tables.md
│   ├── Case_A_ADR-1b/              # 1D periodic linear advection–diffusion–reaction (Fourier basis)
│   ├── Case_B_2D-Stokes/           # 2D incompressible Navier–Stokes cavity flow (Vector POD basis)
│   └── Case_C_AP1D/                # 1D Aliev–Panfilov electrophysiology (Weighted-QR Legendre basis)
│
├── Study_II/                       # Study II: 10,000-iter BFGS capacity & HPC timing (126 cluster runs)
│   ├── README.md                   # Detailed Study II narrative and tables
│   ├── protocol.md                 # Experimental protocol
│   ├── report_analysis.py          # Pure-NumPy verification, rank/Jacobian SVD, and figure generator
│   ├── figures/                    # Study II figures (fig1..fig9) and tables.md
│   ├── Case_B_2D-Stokes/           # 2D cavity flow (results/, executed_timing.ipynb, notebook & workers)
│   ├── Case_C_AP1D/                # 1D Aliev–Panfilov fronts (results/, executed_timing.ipynb, notebook & workers)
│   └── cluster/                    # CINECA Leonardo Slurm scripts (sbatch_sweep.sh, sbatch_timing.sh, logs)
│
├── paper/                          # Post-processing scripts & FLOP/throughput figures used in Report.pdf
│   ├── generate_flops_plots.py     # Generates continuous FLOP/wall-time trajectories & throughput figures
│   └── figures/                    # fig_flops_trajectories.png, fig_flops_arithmetic_intensity.png
│
└── src/                            # Upstream LDNet baseline scripts and notebooks
```

---

## Summary of Benchmarks and Findings

| Study | Scope | Benchmarks | Key Result |
|---|---|---|---|
| **Study I** (`Study_I/`) | Four reconstructor architectures ($`\mathcal{R}_{\mathrm{MLP}}`$ [`L0`], $`\mathcal{R}_{\mathrm{id}}`$ [`L1`], $`\mathcal{R}_{\mathrm{aff}}`$ [`L2`], $`\mathcal{R}_{\mathrm{nl}}`$ [`L3`]) under Frozen-Latent (`R1`) and End-to-End (`R2`) training (816 runs) | Case A (`Case_A_ADR-1b`), Case B (`Case_B_2D-Stokes`), Case C (`Case_C_AP1D`) | Fixed-basis linear decoders match or beat $`\mathcal{R}_{\mathrm{MLP}}`$ when the Kolmogorov $`n`$-width decays rapidly (Cases A & B), while $`\mathcal{R}_{\mathrm{MLP}}`$ learns curved latent coordinates in smooth PDEs and is essential to break the Kolmogorov barrier on sharp moving fronts (Case C, intrinsic threshold $`d_s = 3`$). |
| **Study II** (`Study_II/`) | Pure BFGS up to 10,000 iterations on CINECA Leonardo DCGP (`dcgp_usr_prod`, dual-socket Intel Xeon Platinum 8480+, 112 cores/node) + exclusive-node Pass-T timing (126 cluster runs) | Case B (`Case_B_2D-Stokes`), Case C (`Case_C_AP1D`) | **Case B:** replacing $`\mathcal{R}_{\mathrm{MLP}}`$ with $`\mathcal{R}_{\mathrm{aff}}`$ (POD) and widening $`\mathrm{NN}_{\mathrm{dyn}}`$ ($`w=28, n=10, K=65`$) lowers test NRMSE 2.5× ($`2.32\cdot 10^{-3}`$ vs. $`5.86\cdot 10^{-3}`$), iso-accuracy FLOPs by 54×–150× (56×–167× training-only), wall-time by 10.2×–11.8×, and inference time by 78×, while preserving discrete divergence. **Case C:** $`\mathcal{R}_{\mathrm{MLP}}`$ with deeper $`\mathrm{NN}_{\mathrm{dyn}}`$ ($`L=3`$) reaches $`7.83\cdot 10^{-3}`$ (0.54 ms front arrival MAE) on a rank 2–5 curved manifold, whereas $`\mathcal{R}_{\mathrm{aff}}`$ stalls at ≈ 0.10 and overfits sparse spatial samples. |

Two structural rank bounds proved in `Report.pdf` are verified to machine precision:

1. **Reconstructor Rank Bound** (for $`\mathcal{R}_{\mathrm{aff}}`$):

$$\operatorname{rank}(\hat{y}-\bar{y}) \le \min(\dim \mathbf{z}, K)$$

2. **Dynamics Hidden-Width Bottleneck Bound**:

$$\operatorname{rank}(\{\mathbf{s}_s^i\}),\ \operatorname{rank}\!\left(\frac{\partial \dot{\mathbf{s}}}{\partial \mathbf{s}}\right) \le \min(n, w_{\mathrm{dyn}})$$

under exact rest-state equilibrium subtraction (and $`\le \min(n, w_{\mathrm{dyn}}+1)`$ without it).

---

## Setup and Reproducibility

### 1. Python Environment

For workstations (Python 3.9):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or requirements-macos.txt on Apple Silicon
```

For HPC clusters (Linux, Python 3.11.7, used on CINECA Leonardo):
```bash
pip install -r requirements-cluster.txt
```

### 2. Benchmark Dataset (`data/`)

Training, validation, and test datasets (`data/ADR/`, `data/NS/`, `data/AP1D/`) are not tracked in Git (~7.9 GB) and can be downloaded from the upstream Zenodo archive:

```bash
curl -L https://zenodo.org/records/10436827/files/data.zip -o data.zip
unzip data.zip -d data
rm data.zip
```

### 3. Regenerating Analysis Tables and Figures

All CSV metrics (`runs.csv`, `predictions.csv`, `curves.csv`, `timing.csv`, `modes.csv`) and cluster figures are included in `Study_I/` and `Study_II/`. Note that binary `.npy`/`.npz` model weights are excluded by `.gitignore`; with `data/` and saved weights present locally, the pure-NumPy verification and plotting pipelines can be re-run via:

```bash
python Study_I/report_analysis.py > Study_I/figures/tables.md
python Study_II/report_analysis.py > Study_II/figures/tables.md
python paper/generate_flops_plots.py
```

---

## References

[1] F. Regazzoni, S. Pagani, M. Salvador, L. Dedè, A. Quarteroni, *"Learning the intrinsic dynamics of spatio-temporal processes through Latent Dynamics Networks"*, *Nature Communications* **15**, 1834 (2024). DOI: [`10.1038/s41467-024-45323-x`](https://doi.org/10.1038/s41467-024-45323-x).
