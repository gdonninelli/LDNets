---
name: rq2-ldnet-testcase
description: Use BEFORE implementing or extending an RQ2 reconstructor-ablation notebook/script (variants L0-L3, regimes R1/R2, K and d_s sweep) for Test Case B (Navier-Stokes cavity, src/TestCase_2.py) or C (AP1D electrophysiology, src/TestCase_3.py), or when editing RQ2/ADR-1b/TestCase_A.ipynb in the LDNets repo. Gives environment, verified gotchas, a map of the Case A reference implementation, the results schema, metric definitions, the headless verification workflow (scripts/nbtool.py) and the per-case differences, so new work stays consistent with Case A and nothing is rediscovered.
---

# RQ2 on LDNets: consistency guide for the other test cases

Case A (ADR 1b) is **done and verified**. B and C must follow the same design. Facts below are marked **(verified)** when they were run, **(source)** when only read from the code or docs.

## 0. Start here

- Spec: `RQ2/RQ2_protocol.md` (draft v0.1: variants L0-L3, regimes R1/R2, sweep, metrics Sec. 8, stats Sec. 10).
- Reference implementation: `RQ2/ADR-1b/TestCase_A.ipynb`. **Copy and adapt its cells, do not redesign.** Read it as text with `scripts/nbtool.py topy` (see Sec. 6) instead of loading the notebook JSON.
- Case facts: `data/SELECTED_TESTCASES.md` (verified data layouts), original scripts `src/TestCase_2.py` (B) and `src/TestCase_3.py` (C), `src/TestCase_1b.ipynb` (A). **Never edit `src/*`** (`TestCase_2.py` has uncommitted whitespace/comment edits by the user: leave them). Put new code in `RQ2/<case>/` with its own copies of `utils.py` and `optimization.py` (identical to `src/`; if a case needs a patch, patch the copy and say so in the notebook).
- What the user asked for in Case A, and expects again: code as simple as possible; one section per L* (L0..L3) with R1 and R2 inside, over the protocol Sec. 7 levels; results in CSV with aggregated rows **and one row per prediction**; metrics of protocol Sec. 8; reproducible; resumable, so new analysis cells never need re-training; boxplots over seeds per L* section, a trend plot vs dimensionality, and a final section (table + boxplot across procedures with the best configuration of each + FOM-vs-prediction plots).
- The user runs the notebooks himself, possibly in a live VS Code kernel: do not run full sweeps, smoke-test with a reduced config in a temp folder, never write into or delete `RQ2/*/results/`, leave no test output in the repo. Follow the user's AGENTS.md (ask only consequential questions, keep research short).

## 1. Environment (verified)

- Always use `/Users/giuliodonninelli/Dev/LDNets/.venv/bin/python` (3.9.6): tensorflow 2.8.0, keras 2.8.0, numpy 1.22.3, scipy 1.7.3, pandas 1.4.2, matplotlib 3.5.1, ipykernel 6.31. **Not installed:** seaborn, nbformat, nbconvert, nbclient, a jupyter server. Do not pip install without asking: build notebook JSON with `nbtool.py`, plot with matplotlib/pandas only.
- float64 on CPU only (the Metal plugin has no double kernels; never install tensorflow-metal): `tf.keras.backend.set_floatx('float64')`, hide the GPU, create variables under `cpu_device()` (setup cell of Case A).
- Machine: Apple M5 Pro, 18 cores, 48 GB. TF uses every core: two heavy jobs together run ~4x slower, so never time things in parallel.
- A notebook's cwd is its own folder: `import utils, optimization` are the local copies and data paths are `../../data/...` from `RQ2/<case>/`.
- Notebooks are nbformat 4.2 with kernelspec `LDNets (3.9.6)` (`python3`); copy `metadata` from the Case A notebook.

## 2. Gotchas that already cost time (verified)

1. **Keras input rank:** `Sequential` built with `input_shape=(n,)` rejects higher-rank inputs. Declare the leading axes: `(None, None, n)` for `[S,T,P,n]`, `(None, n)` for `[S,T,n]` (helper `mlp(n_in, width, n_out, lead=())`).
2. **AutoGraph:** the chain `loss -> mse -> states_of -> evolve` (the `for i in tf.range` loop) must be plain `def`s, never lambdas. In a headless `exec` AutoGraph cannot read function sources ("Unable to locate the source code", then "iterating over tf.Tensor is not allowed"): `nbtool.py run` registers them in `linecache`. Real Jupyter or a real `.py` file is fine.
3. **Determinism:** `tf.config.experimental.enable_op_determinism()` exists in TF 2.8 and costs no speed, but random ops then raise unless `tf.random.set_seed` was called first. `set_seed(seed)` (NumPy + TF) must run right before the networks of every run are created. Even without it the runs were bitwise reproducible (CPU, float64); keep it as a guard.
4. **`optimization.OptimizationProblem`:** prints every 10 epochs (wrap in `contextlib.redirect_stdout(io.StringIO())`); every construction traces new `tf.function`s (seconds, counted in `train_time_s`; silence the retracing warnings with `tf.get_logger().setLevel('ERROR')`); fails on an empty variable list (skip training: R1 of L1 has no parameters); **has no early stopping** (final-epoch weights are used, validation loss is only logged); BFGS `maxiter` = epochs, `gtol=1e-100`.
5. **`utils.denormalize_output`** returns numpy for numpy input and a tf tensor for tf input: wrap with `np.asarray`, do not call `.numpy()`.
6. **Original-script order:** seed, then create `NNdyn` first and `NNrec` second. Case A gate (seed 0, d_s=2, L0): test NRMSE 2.440e-02 (2.439626e-02), train 1.914e-02, test 1-rho 1.052e-02 (1.051673e-02), train 1-rho 7.859e-03, about 13 s. The refactor reproduces it exactly.
7. R1 builds an unused `NNdyn` first so that the reconstructor starts from the same weights in R1 and R2 (checked).
8. `scipy.stats.pearsonr` on constant input gives NaN: watch degenerate small-K models.
9. Case A timing: L0 about 13 s/run, R2 of L1-L3 4-7 s, R1 0.3-4 s, the whole sweep about 1 h. B and C budgets are far larger (BFGS 10000 and 5000): **time one full-budget run alone before planning a sweep.**

## 3. Case A reference implementation: the contracts to keep

Cell order: intro -> Setup (imports, float64/CPU, determinism, logger) -> problem / normalization / dt / datasets (verbatim from the original, only the data path changes) -> Experiment configuration (`SEEDS`, `D_S`, `K_LEVELS`, `K_for`, `EPOCHS_*`, `ALPHA_REG`, `OUT`, `meta.json` and a guard that refuses other training settings in the same folder) -> Helpers (fixed ingredients, networks and dynamics, metrics and tables, reconstructors `build_L0..L3` + `BUILD`, runs `setup_run / run_one / sweep / predict_test`, plots `boxplot_variant / trend_variant`) -> sections L0..L3 (markdown, `sweep(...)`, boxplot cell, trend cell; L0 also has the reproducibility gate) -> final comparison.

- `build_Lx(d_s, K) -> (rec, rec_nets)`. `rec(states)` returns `[S,T,P,n_out]`; `rec_nets` are Keras `Sequential`s (their layers feed `weights_reg` and the trainable variables). `BUILD` lives in Helpers, not in the sections, so that after a kernel restart Setup + Configuration + Helpers is enough for any analysis cell.
- `setup_run(variant, regime, d_s, K, seed) -> (states_of(split), rec, trained_nets, rec_nets)`. R2: `states_of` = `evolve(NNdyn, ...)`, trained = `[NNdyn] + rec_nets`. R1: frozen latents `results/latents/L0_ds{d}_seed{s}.npz` (keys `train`, `valid`, `test`), trained = `rec_nets`, error message if L0 has not run.
- `run_one` saves in this order: weights, latents (L0 only), `predictions.csv`, `modes.csv`, and **`runs.csv` last** (a run is done only when its row exists). `load(name)` drops duplicated key rows (keeps the last). `sweep` skips done keys and iterates seed-major.
- Keys: `variant, regime, d_s, K, seed`. L0 has `K = 0` and only regime `R2` (R1 = R2: L0 is the latent source); L1 has `K = d_s`; `run_id = f'{variant}_{regime}_ds{d}_K{K}_s{seed}'`.
- `results/` layout: `runs.csv` (key + `rec_params, width, train_time_s, infer_time_s, nrmse_{train,valid,test}, one_minus_rho_{train,valid,test}, eps_total_test, eps_proj_test, eff_ratio_test`), `predictions.csv` (test samples only: key + `sample, nrmse, one_minus_rho, eps_total, eps_proj, eff_ratio`), `modes.csv` (key + `mode, err_abs, ref_norm, err_rel`), `weights/{run_id}.npy` (flat `VariablesStitcher.stitch()`, restored with `update_variables`), `latents/`, `meta.json` (versions, git of the first session, `training` dict).
- **Best configuration** of a procedure = (d_s, K) with the lowest median **validation** NRMSE over seeds, never test (`best_configs(select=...)`). A procedure is (variant, regime).
- Plots: boxplots over the 3 seeds per (K, regime), one panel per d_s, colour by regime, log y; trend = median + IQR vs K (x = d_s for L0/L1), one panel per regime; final section: table of medians, boxplot across procedures, `predict_test` + FOM plots with an assertion that the rebuilt model reproduces the stored NRMSE.

## 4. Metrics as implemented (protocol Sec. 8)

- NRMSE and 1-rho exactly as the original scripts, on denormalized fields. Per-prediction NRMSE divides by the range of the whole split, so the aggregated NRMSE is the root mean square of the per-prediction values (checked).
- eps_total = ||y_hat - y|| / ||y|| (norm over time and space). eps_proj(K) = ||(y - y_bar) - P_K (y - y_bar)|| / ||y||, with P_K the orthogonal projection on the first K basis functions of the basis (orthonormal for the discrete L2 product <f,g>_h = h sum f g) applied to the mean-subtracted field, because every L1-L3 variant adds the training mean field y_bar(x) (fixed, not trainable). `eff_ratio` = eps_total / eps_proj, empty for L0.
- Per-mode error: c_k(t) = <y(.,t) - y_bar, phi_k>_h for the FOM and the prediction, for **every** variant (also L0, by projecting the prediction); stored as `err_abs`, `ref_norm`, `err_rel` aggregated over test samples and times.
- Cost: `rec_params`, `width` (hidden width of the MLPs), `train_time_s` (all optimisation, tracing included; in R1 only the reconstructor fit), `infer_time_s` (reconstructor on the test latents, best of 3).
- L3 width = largest hidden width whose parameter count does not exceed that of the L0 reconstructor (`matched_width`, minimum 1). It collapses to 1 for large K (A: L3 from K=33), so large-K points measure a bottleneck, not the basis; the notebook says so and the user knows. The protocol's "best size" sweep and the statistics of Sec. 10 are **not** implemented in A.

## 5. Case A facts (verified)

- `data/ADR/data_1b.npy`: `t (101)`, `x (100)` on [-1, 0.98] with h = 0.02 (periodic, endpoint excluded), `output (1000,101,100)`, `forcing (1000,101,2)`. Splits 0-99 / 100-199 / 200-299, dt = 0.05, time constant 2.3; normalization of z is the identity.
- The solution lies exactly in span{cos(pi x), sin(pi x)}: the Fourier basis {1, cos, sin} has projection error ~6e-12, so d_s = 2 is exact and eps_proj is at round-off for K >= 3 (efficiency ratio ~1e10, uninformative). The mean field is not negligible (max |y_bar| ~ 0.27 against a field maximum ~ 1.18).
- On this uniform periodic grid the Fourier functions {1/sqrt(2), cos k pi x, sin k pi x} are exactly h-orthonormal up to 99 functions (checked by an assertion).

## 6. Tooling and verification workflow

`scripts/nbtool.py` (stdlib only, run with the repo venv; path relative to this skill folder, i.e. `.opencode/skills/rq2-ldnet-testcase/scripts/nbtool.py`):

```
nbtool.py topy NB.ipynb OUT.py                 notebook -> percent text ('# %%' / '# %% [markdown]'); this text also runs as a plain .py script
nbtool.py tonb IN.py OUT.ipynb --like NB.ipynb percent text -> notebook (round trip with topy is exact)
nbtool.py run  NB.ipynb OUTDIR [--sub OLD NEW]... [--skip REGEX] [--cells 0,1,2] [--extra FILE.py] [--cwd DIR]
```

`run` executes the code cells in one namespace like a kernel, saves figures as `OUTDIR/fig_NNN.png` (view them with the Read tool) and prints per-cell time and peak RSS. `--sub` must match exactly once (use it to shrink `SEEDS`, `D_S`, `K_LEVELS`, the `EPOCHS_*` line and to point `OUT` to a temp folder); `--skip '^sweep\(.*$'` defines everything without training. Workflow that worked for A:

1. Draft the new case from the Case A text, convert with `tonb`.
2. Smoke test headless with a tiny config and a temp `OUT` (all sections, all plots, the final section), look at the PNGs and CSVs. Include the extreme configurations (largest K, largest d_s).
3. **Gate:** run the original script's code path once in this venv (copy it verbatim into a temp script) to get reference metrics with seed 0, then check L0 reproduces them at full budget (A: `rtol=2e-3` against the 4 printed digits). For B and C also compare with the paper values below.
4. **Reproducibility:** two independent processes (not concurrent if timing matters) must give identical `runs/predictions/modes` (excluding `*_time_s`), weights and latents.
5. **Resume:** a second pass prints "0 runs to do" in every sweep; appended duplicate rows must be removed by `load`.
6. Remove temp folders and `__pycache__` you created; do not touch the user's `RQ2/*/results/`.

## 7. What changes in B and C (source, not run yet)

| | A (done) | B: Navier-Stokes cavity (`src/TestCase_2.py`) | C: AP1D (`src/TestCase_3.py`) |
|---|---|---|---|
| data | `data/ADR/data_1b.npy`, `utils.ADR_create_dataset` | `data/NS/T20_80samples.npy`, `T20_20samples.npy`, `T40_10samples.npy`, `utils.NS_create_dataset` | `data/AP1D/APsolution{k}.npy` + `APsetting{k}.csv` (400), `utils.AP_create_dataset` |
| splits | 0-99 / 100-199 / 200-299 | train 80 (T=20), valid 20 (T=20), **test 10 with T=40 (extrapolation)** | 0-99 / 100-199 / 200-399 (200 test) |
| grid | 100 pts, periodic | 100x100 Cartesian from `meshgrid`, flattened to 10000 pts (`points[:,0]=X.flat`) | `x[::8]` of 801 nodes: 101 pts on [0,100], h=1, endpoints included, Neumann |
| times, dt, dt_base | 101, 0.05, 2.3 | 101 (test 201), 0.2, 5.4 | 501, 1, 205 |
| inputs / outputs | (A(t),P(t)) / z | u(t) lid velocity / **vector (ux,uy)** | (Iapp1,Iapp2) sparse pulses / u |
| normalization | z identity | x in [0,1]^2, u and ux,uy in [-20,20] (divide by 20) | x in [0,100], Iapp in [-2.5,2.5], u in [0,1.2] -> [-1,1] |
| point subsampling | none | train/valid 200 random pts per (sample,t), test full grid | train/valid 20 random pts per (sample,t), test full grid |
| reconstructor input | (s, x) | **(s, u(t), x)**, so z = [s; u] | (s, x) |
| NNdyn | 10-10-d_s | 7-7-d_s | 8-d_s with `NNdyn(inp) = base(inp) - base(0)` (equilibrium) |
| L0 NNrec | 7-7-1 | 24^4 -> 2, output transform (o^3 + 0.05 o)/1.05 | 17^5 -> 1, no transform |
| original d_s | 2 | 1 (paper sweeps 1, 5, 10) | 12 |
| loss | MSE + 1e-5 (reg dyn + reg rec) | MSE_v + 0.1 MSE_direction, direction = v/(1e-4 + norm(v)) on axis 3, **no Tikhonov** | MSE + **4.7e-3** (reg(NNdyn_base) + reg(NNrec)) |
| optimizer | Adam 50 + BFGS 150 | Adam 200 + BFGS 10000 | Adam 200 + BFGS 5000 |
| protocol basis | Fourier | vector POD on (vx,vy) | Chebyshev/Legendre |
| extra metrics | none | direction error, vorticity error, extrapolation ratio eps[20,40]/eps[0,20] | front position or activation time error |
| FOM plot | `utils.plot_output_1D` | contours (see `src/TestCase_2.py` postprocessing) | `utils.plot_output_1D(..., 5, 4)` |
| paper reference | gate above | NRMSE ~4e-3 (small d_s) to ~1.2e-3 (Fig. 4b) | LDNet test NRMSE 7.37e-3 (d_s=12, 1708 params) |

Consequences for the code (resolved protocol open items: A = `TestCase_1b`, C = `TestCase_3.py`; C's NNrec takes (s, x) only with no output transform; B data are on a uniform Cartesian grid):

1. **Random point subsampling (B, C).** `utils.process_dataset(..., num_points_subsample=n)` draws `np.random.choice(num_points, n)` (with replacement) for every (sample, time), in that order, from the global NumPy RNG, and does **not** keep the indices. `points_full` and `out_fields` become `[S,T,n,.]` and `dataset['num_points']` becomes n; the test set uses the full grid. The originals call `np.random.seed(0); tf.random.set_seed(0)` **before** `process_dataset` (train, then valid, then test). To reproduce the original data call `np.random.seed(0)` right before the train call, in that order. Fixed bases evaluated at the sampled points need either analytic evaluation at `points_full` (Legendre, Chebyshev, B-splines) or the indices (POD): patch the **local copy** of `process_dataset` to store `dataset['idxs']`, then gather the basis, giving a per-point basis `[S,T,n,K]` and a `decode` with `tf.einsum`; at test time the basis is the shared `[P,K]`.
2. **B: z = [s; u(t)].** `states_of` must return `concat([s, u(t)])` (u comes from the dataset, also in R1 where s comes from the stored L0 latents); W and g_theta then take d_s + 1 inputs; L0 takes (z, x). L1/L2 drop the output transform (strictly linear) and B also needs the **L2+link** variant that keeps it. The vector output needs vector-valued modes (c_k multiplies phi_k(x) in R^2, vector POD on stacked (vx,vy)) and a mean field per component.
3. **C: equilibrium-subtracted NNdyn** is a lambda over `NNdyn_base` in the original: write it as a `def` closure, and use `NNdyn_base` for the trainable variables and the Tikhonov term. Fourier does not apply (non-periodic, Neumann): use Chebyshev/Legendre on [0,100] mapped to [-1,1] and decide the quadrature weights of the discrete product (the grid includes both endpoints).
4. **Loss and regularisation are case specific**: keep `loss_valid` as the same loss on the validation set; `ALPHA_REG` is 0 for B, 4.7e-3 for C.
5. **Normalization is not the identity in B and C**: the networks work in normalized units, but y_bar, projections, per-mode errors and every metric must use denormalized fields (`denorm`), as in A.
6. **Runtime:** the protocol says to sweep B with a reduced BFGS budget, check on a few configurations that the ranking does not change, then run the final configurations at 10000 epochs. Measure first and agree on the grid, seeds and budget.

## 8. Settle with the user before coding B or C

1. C: d_s levels (protocol {1,2,3,5}, original 12) and K levels.
2. B and C: keep the subsampled dataset fixed across training seeds (recommended, reproduces the original data) or redraw it per seed.
3. Budget: BFGS epochs during the sweep, number of seeds and size of the grid, given the measured time per run.
4. Quadrature weights of the discrete L2 product on non-periodic grids; definitions of the direction error, vorticity error and front position / activation time error.
5. Whether to add what A leaves out: "best size" sweep of g_theta, per-mode analysis of the trained L0 decoder (protocol Sec. 9), Wilcoxon/Holm statistics (Sec. 10).
