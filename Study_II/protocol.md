# RQ2-bis protocol: computational scalability of LDNets (BFGS-only, cases B and C, nonlinear L0 and linear L2 reconstructors)

Version: v0.4-draft. Date: 2026-10-14. Supersedes v0.3-draft.
Changes in v0.4: the iteration cap is raised from 1000 to 10000 for every run (the first laptop sweep at 1000 iterations
showed the error still falling 25-45% in the last 500 iterations for most L0 configurations, so a size effect could be hidden
by the budget); Stage 4 is therefore not needed; a run that stops before the cap is flagged `stalled`; the final sweeps run on
the Leonardo cluster (several nodes), the laptop is for development and smoke tests; Stage 5 no longer assumes macOS.
Changes in v0.3 (kept): Stage 7 (linear reconstructor arm, RQ2's L2 in R2); operation-count cost axis and Pass-T unit
(seconds per function/gradient evaluation); corrected timing, checkpoint, equilibrium-offset and failure handling.
Earlier decisions stay: scipy BFGS with 0 Adam epochs, 3 seeds for both cases, integer grids, RQ2-style per-folder layout,
Stage 6 out.
Hardware: development on a MacBook Pro (Apple M5 Pro, 18 CPU cores, 48 GB unified memory); final sweeps and timing on
Leonardo nodes (node type and cores recorded in `hardware.json`; laptop and cluster timings are not comparable).

## 1. Purpose and scope change

RQ2 asked which reconstructor variant (L0-L3) is needed. RQ2-bis asks a computational question:

> How does the test error of an LDNet decrease as parameters and computational cost increase,
> under a fixed BFGS budget? Where is the knee, what limits accuracy beyond it, and does a linear
> reconstructor pay off when the comparison is made on compute and not only on error?

Changes with respect to RQ2:

| Item | RQ2 | RQ2-bis |
|---|---|---|
| Question | Is the nonlinear reconstructor needed? | Error vs parameters vs cost, for L0 and for the linear L2 |
| Optimizer | Adam, then BFGS | BFGS only, 0 Adam + at most 10000 BFGS (same scipy implementation as RQ2) |
| Test cases | A, B, C | B and C only |
| Factors swept | Reconstructor variant, $K$, $n$ | $\mathrm{NN}_{\mathrm{dyn}}$ width/depth, $n$, iteration budget (checkpoints); for L2 also $K$ |
| Reconstructor | Varied (L0-L3) | Two fixed arms: L0 (main) and L2 (Stage 7); L1, L3 not run |
| Regimes | R1 (frozen latents) and R2 (end-to-end) | R2 only (R1 needs an Adam-trained $\mathrm{L0}$) |

Out of scope: comparison with other methods, hyper-parameter tuning, Adam, case A, L1/L3, L2 with a depth sweep.
Case A is not run, so its fast-convergence regime is not covered by this study.

## 2. Test cases and canonical configurations

Both cases use the same data, splits, normalization and loss as in `Study_I/protocol.md`. Do not change them.
Case C (AP1D) is implemented in the `Study_II/Case_C_AP1D/` folder (`TestCase_C_bis.ipynb`).

| | Case B | Case C |
|---|---|---|
| Script / dataset | `src/TestCase_2.py` + `utils.NS_create_dataset` on `data/NS/T20_80samples.npy` / `T20_20samples.npy` / `T40_10samples.npy` | `src/TestCase_3.py` (confirmed) + `utils.AP_create_dataset` on `data/AP1D/APsolution*.npy` + `APsetting*.csv` |
| Train/val/test | 80 / 20 / 10 (`T20_80` idx 0-79 / `T20_20` idx 0-19 / `T40_10` idx 0-9) | 100 / 100 / 200 (idx 0-99 / 100-199 / 200-399) |
| Time steps, dt | $\mathit{dt} = 0.2$, $\mathit{dt}_{\mathrm{ref}} = 5.4$; train/valid $T = 20$, $N_t = 101$; test $T = 40$, $N_t = 201$; 200 random pts per (sample,$t$) train/valid, full 10000-pt grid test | $\mathit{dt} = 1\,\mathrm{ms}$, $\mathit{dt}_{\mathrm{ref}} = 205$, $T = 500\,\mathrm{ms}$, $N_t = 501$; 101 pts (801 raw, rate 8); 20 random pts per (sample,$t$) train/valid, full 101-pt grid test |
| Original optimizer budget | Adam 200 + BFGS 10000 | Adam 200 + BFGS 5000 |
| RQ2 reduced budget | 200 + 500 | 200 + 500 |
| RQ2-bis budget | 0 Adam + max 10000 BFGS (the original BFGS budget) | 0 Adam + max 10000 BFGS (twice the original 5000) |
| Original latent size $n^*$ | 1 (`num_latent_states = 1` in script; paper sweeps $\{1, 5, 10\}$) | 12 |
| $\mathrm{NN}_{\mathrm{dyn}}$ (width $\times$ depth), canonical | $7$-$7$-$d_s$ on $(\mathbf{s}, u)$, in-dim $2$ at $n^*$ ($85$ params at $n^* = 1$) | $8$-$d_s$ on $(\mathbf{s}, \mathbf{u})$ with equilibrium offset $\mathrm{NN}_{\mathrm{dyn}}(\mathbf{s},\mathbf{u}) = \mathrm{NN}_{\mathrm{base}}(\mathbf{s},\mathbf{u}) - \mathrm{NN}_{\mathrm{base}}(\mathbf{0},\mathbf{0})$, **re-evaluated with the current weights at every call** (as in the original script, so the dynamics vanish at rest throughout training), in-dim $14$ at $n^*$ ($228$ params at $n^* = 12$) |
| $\mathrm{NN}_{\mathrm{rec}}$ ($\mathrm{L0}$), canonical | $24^4 \to 2$ outputs on $(\mathbf{s}, u, \mathbf{x})$, in-dim $4$ at $n^*$ ($1970$ params at $n^* = 1$; $\sim 2000$) | $17^5 \to 1$ output on $(\mathbf{s}, \mathbf{x})$, in-dim $13$ at $n^*$ ($1480$ params at $n^* = 12$; $\sim 1400$; $1708$ total with $\mathrm{NN}_{\mathrm{dyn}}$) |
| Input $u$ to $\mathrm{NN}_{\mathrm{dyn}}$ / $\mathrm{NN}_{\mathrm{rec}}$ | yes / yes (rec input is $[\mathbf{s}; u; \mathbf{x}]$) | yes / no (dyn input $[\mathbf{s}; \mathbf{u}]$, rec input $[\mathbf{s}; \mathbf{x}]$ only) |
| Output transform (L0) | yes: $(o^3 + 0.05\,o) / 1.05$ | none |
| $\mathrm{L2}$ reconstructor (Stage 7) | $\bar y + \sum_{k \le K}(W[\mathbf{s};u] + \mathbf{b})_k \varphi_k$, vector POD basis, no output transform, $(n+1)K + K$ params | $\bar y + \sum_{k \le K}(W\mathbf{s} + \mathbf{b})_k \varphi_k$, Legendre basis, $nK + K$ params |

The table is confirmed from the scripts (parameter counts verified by constructing the models:
B $85$/$1970$, C $228$/$1480$). Canonical numbers and the grids are fixed in the notebook configuration cells
and the training settings in `results/meta.json` (no `configs/` tree: RQ2-style layout, Section 12).
Note for comparison with RQ2: the RQ2 Case C notebook computes the equilibrium offset once at initialisation,
which differs from the original script; RQ2-bis follows the original script.

## 3. Hypotheses (frozen before running; numeric thresholds in Section 11)

- **H1, capacity law.** At fixed iterations, test $\mathrm{NRMSE}$ decreases approximately as a power law
  in $P_{\mathrm{dyn}}$, then plateaus. Falsified if the error is flat in $P_{\mathrm{dyn}}$ from the smallest width,
  or if the fitted exponent $\alpha$ has a $95\%$ bootstrap interval whose lower end is below $0.1$ (a flat curve).
  Fitted on the width sweep of each arm; with an endpoint log-log slope fallback if the fit fails.
- **H2, cost split.** Cost per iteration is approximately $T \cdot P_{\mathrm{dyn}} + T \cdot N_x \cdot P_{\mathrm{rec}}$ per trajectory
  ($T$ time steps, $N_x$ query points), because $\mathrm{NN}_{\mathrm{rec}}$ is evaluated at every query point and $\mathrm{NN}_{\mathrm{dyn}}$ once per step.
  So widening $\mathrm{NN}_{\mathrm{dyn}}$ is cheap relative to the reconstructor. Falsified if the dynamics share of the
  per-iteration time exceeds $50\%$ at the widest $\mathrm{NN}_{\mathrm{dyn}}$.
  Implemented as the share from the linear fit $t(P_{\mathrm{dyn}}) = t_0 + t_1 \cdot P_{\mathrm{dyn}}$ on the sweep training
  times, not a profiler; the fixed cost of the sequential time loop falls into $t_0$, so the share is a lower bound.
  Reported for both arms (L2 is expected to move the share up).
- **H3, iteration-limited regime.** With a fixed iteration budget, large networks are under-converged,
  so error vs size becomes non-monotone or flattens. Falsified if the error is monotone decreasing
  up to the largest $P_{\mathrm{dyn}}$ at the largest iteration budget. Implemented as the monotonicity check at
  $10000$ iterations (value at $500$ printed); no verdict is given if some width has no checkpoint there
  (scipy stopped early).
- **H4, latent dimension.** Error stops improving once $n$ is above the intrinsic dimension of the data,
  and extra latents only add cost. Falsified if the error keeps decreasing for $n \ge 2\,n^*$.
  Implemented as the $n^* \to 2n^*$ improvement of the median test error against the $5\%$ threshold, for both arms.
- **H5, no free lunch in wall-time.** The configuration with the lowest error at a given wall-time budget
  is not the largest one. At budgets below the time of the largest network it cannot be selected, so the testable
  form is: falsified if the largest network ($\max P_{\mathrm{total}}$ of the arm) has the lowest median test error
  of the arm. The cheapest configuration within $10\%$ of the best is printed next to it.
- **H6, compute value of the linear reconstructor.** L2 reaches a given accuracy at lower cost than L0.
  Operationalised by `cost_to_reach`: for the target levels $1, 1.25, 1.5, 2, 3 \times$ the best L0 median test
  $\mathrm{NRMSE}$, the cheapest configuration of each arm that reaches it, in operations and in time.
  Falsified at a level if no L2 configuration reaches it, or if the L2 cost is not below the L0 cost; the decision
  level is $1.5\times$ (Section 11). The question is what the accuracy costs, not only whether L2 is accurate.

## 4. Fixed across all runs

- Data, splits, normalization, loss (including any direction term in B and the Tikhonov term in C), time step,
  initialization scheme.
- **Reconstructor:** two arms, each with a fixed architecture. **L0** = the original MLP at its canonical size
  (main arm; because it depends on $n$ through its input layer its size changes slightly with $n$). **L2** = the
  linear reconstruction of RQ2 (Stage 7). Report $P_{\mathrm{rec}}$ for every run.
- **Optimizer:** scipy BFGS, the exact RQ2 implementation and settings (`method = 'BFGS'`,
  `tol = gtol = 1e-100`, so the iteration count is fixed; 0 Adam epochs). There are no history-size
  or line-search settings to tune. The RQ2-bis copy of `optimization.py` adds only an observation hook
  (`callback_extra`, used for the Stage 3 checkpoints; default `None` reproduces RQ2 behaviour exactly)
  and an evaluation counter (`n_fev`, one joint function/gradient evaluation per scipy call, line-search
  steps included). Float64, CPU, no tensorflow-metal. Completed iterations are stored per run (`n_iters`);
  a run that scipy stops before the cap (line search gives up) is flagged `stalled` in `runs.csv`, a warning is printed,
  it stays in the statistics and is counted (`n_stalled`, `failure_map`); the gate warns and H3 gives no verdict when the
  budgets are unequal. Early stops are more likely at 10000 iterations than at 1000.
- **Initialization and seeds:** same initialization scheme and the same seed list $[0, 1, 2]$ for every
  configuration and both cases (seed $0$ is the original script seed). The reconstructor is created after
  $\mathrm{NN}_{\mathrm{dyn}}$ as in the original script, so its initial weights are identical across the width sweep
  but not across the depth sweep (a different number of layers consumes a different number of random draws);
  this is accepted and stated.
- **Time measurement inside the sweeps (Pass A):** `train_time_s` is the BFGS loop only. It excludes tf.function
  tracing (`trace_time_s`) and the checkpoint evaluations (each predicts the whole test set); the periodic loss
  evaluation of `iteration_callback` stays in it (small, identical for all runs).
- **Failures:** a run with non-finite predictions is stored with `failed = True` and NaN metrics (weights kept) and
  the sweep continues. Statistics skip failed runs and report `n_failed`. A `stalled` run (above) is not a failure: its
  error is reported as it is.
- **Gate (Stage 0, adapted: the RQ2 exact-match gate cannot apply with a different optimizer):**
  the canonical BFGS-only configuration with $3$ seeds must finish with finite test NRMSE, and its median
  must be within $5$ times the RQ2 reduced-budget value (B: $2.257\cdot 10^{-2}$ at $d_s = 1$, seed $0$;
  C: $1.765\cdot 10^{-2}$ at $d_s = 12$, seed $0$). The comparison against the paper full-budget references
  (B $\sim 4\cdot 10^{-3}$, C $7.37\cdot 10^{-3}$) is printed. Do not proceed until it passes.
- Do not modify the original TestCase scripts. New code lives next to the notebooks, RQ2-style:
  per-folder copies of `utils.py` (identical to `src/`, plus the stored subsample indices), `optimization.py`
  (see above) and `timing_worker.py` (Stage 5). Each run appends rows to `results/runs.csv`,
  `results/predictions.csv` and `results/curves.csv`, and stores weights in `results/weights/`; versions, git
  hash and training settings go in `results/meta.json`; Stage 5 writes `results/timing.csv` and
  `results/hardware.json`. There is no `latents/` folder (R2 only) and no per-run `config.json`
  (the grid is fixed in the notebook).

## 5. Design

Every sweep is run on both B and C. All sizes are relative to the canonical configuration
($w^*$ = canonical $\mathrm{NN}_{\mathrm{dyn}}$ hidden width, $L^*$ = canonical number of hidden tanh layers,
$n^*$ = canonical latent size; B: $L^* = 2$, C: $L^* = 1$). Fractional widths are rounded half-up
(minimum $1$) and duplicate grid points are removed, giving the fixed integer grids below.
A configuration is `(variant, K, width, depth, d_s)` with $K = 0$ for L0.

### Stage 0. BFGS-only feasibility (gate)

Run the canonical L0 configuration of B and of C with BFGS only, $3$ seeds, $I = 10000$.
Compare with the paper's reference metrics and with the RQ2 reduced-budget results (Section 4 gate).
- If the run diverges or stalls far above the RQ2 reduced-budget value, tune once, on validation data,
  with at most $3$ settings (optimizer algorithm and tolerances stay fixed). Then freeze the settings.
- If the error is still unusable, stop and discuss with the supervisor before continuing.

### Stage 1. Dynamics capacity (main sweep, L0)

- Width sweep at depth $L^*$, $n = n^*$: B $w \in \{2, 4, 7, 14, 28\}$, C $w \in \{2, 4, 8, 16, 32\}$.
- Depth sweep: $L \in \{1, 2, 3\}$ hidden tanh layers at width $w^*$, $n = n^*$
  (the canonical depth is already covered by the width sweep, so $2$ extra configs per case).
- Extend the grid by one point on each side if the plateau or the turn-up is not visible.

### Stage 2. Latent dimension (L0)

- B $n \in \{1, 2, 5, 10\}$ (the $\{n^*/4, n^*/2, n^*, 2n^*\}$ rule gives $\{1, 2\}$ after rounding and dedup;
  the paper's $5, 10$ are added so that L0 meets L2 at equal $n$), C $n \in \{3, 6, 12, 24\}$;
  $\mathrm{NN}_{\mathrm{dyn}}$ at the canonical size.
- Report $P_{\mathrm{dyn}}$, $P_{\mathrm{rec}}$ and $P_{\mathrm{total}}$ separately because $n$ changes both networks.

### Stage 3. Iteration budget (no extra runs)

BFGS has no learning-rate schedule, so one long run with checkpoints is equivalent to many short runs.
Every run logs loss, function evaluations, training time (checkpoint time excluded) and validation/test NRMSE after
$\{25, 50, 100, 200, 500, 1000, 2000, 5000, 10000\}$ completed BFGS iterations via the observation hook (the loss histories logged every
$10$ iterations are not stored). This gives the surface $\mathrm{error}(P, \mathrm{iterations})$.

### Stage 4. Long-budget anchor — NOT NEEDED (2026-10-14)

It existed to check whether the trends hold at the original budgets (B: $10000$, C: $5000$ iterations). Every run now
uses a budget of $10000$ iterations, so the stage is absorbed into the main sweeps. The checkpoint at $1000$ iterations still
shows what the shorter budget of the first laptop sweep would have given.

### Stage 5. Thread scaling and serial timing (Pass T only)

The unit is **seconds per joint function/gradient evaluation** on a stored run (median of $3$ repeats of $20$
evaluations after a warm-up), measured in a **fresh process per thread count** (TensorFlow only accepts its thread
counts before its first operation; `timing_worker.py` re-executes the tagged setup cells of the notebook).
- Thread scaling: the canonical L0 run with $1$, $2$, $4$, $6$, $12$ and $18$ threads by default (`thread_scaling(counts)`;
  on the cluster choose counts up to the cores of one node); speedup and parallel efficiency; the suggested thread count (best speedup per $\sqrt{\text{thread}}$) is used next.
- Spot timing: every stored configuration of both arms (seed $0$) at that thread count, one process, random order;
  the first configuration is timed again at the end (throttling check).
- The training time of a configuration is then estimated as `n_fev` $\times$ `s_per_eval`.

### Stage 6 (optional). Training-set size scaling — OUT (user decision).

### Stage 7. Linear reconstructor arm (L2, R2) — IN

RQ2's L2: $\bar y + \sum_{k \le K}(W\mathbf{z} + \mathbf{b})_k \varphi_k$ with $\mathbf{z} = [\mathbf{s}; u]$ in B and
$\mathbf{z} = \mathbf{s}$ in C, strictly linear (no output transform), $W,\mathbf{b}$ trainable, trained end-to-end with
$\mathrm{NN}_{\mathrm{dyn}}$ (R2) under the same BFGS budget. The latent size $n$ and the number of modes $K$ are
separate: $n$ latents are mapped affinely to $K$ coefficients (the v0.2 text with $n = K$ described RQ2's L1,
which has nothing to train, and is not used). Basis and mean field: B vector POD, C Legendre, from training data
only as in RQ2. An affine map of $n$ latents spans at most an $n$-dimensional set of fields, so $n$ is expected
to limit L2; hence $n$ is a main factor of the linear arm. Main $K = 65$ (the largest basis computed, so the
representation is least likely to be the bottleneck). Per case, $3$ seeds each:
- $n$ sweep at canonical dynamics, $K = 65$: B $n \in \{1, 2, 5, 10\}$, C $n \in \{3, 6, 12, 24\}$;
- width sweep with the Stage 1 widths at $n = n_{L2}$ (B $10$, C $12$; fixed by hand, to be revisited after the
  $n$ sweep), $K = 65$;
- $K$ sweep $K \in \{5, 17, 33, 65\}$ at $n = n_{L2}$, canonical width.
That is $11$ configurations after dedup. The L2 parameter count is $P_{\mathrm{rec}} = (\dim\mathbf{z} + 1)K$.

### Seeds

$3$ seeds for every grid point of both cases and both arms. Report the median and the min-max range,
not only the mean. Diverged or NaN runs are logged as failures (`failed` column) and counted, never dropped.

## 6. Iteration budget and run budget

$I_{\max} = 10000$ BFGS iterations for both cases (hard cap, user decision 2026-10-14; it was $1000$ in v0.3).

Run count (fixed grid): L0 Stage 1: $7$ configs; Stage 2: B $3$ ($n = 2, 5, 10$), C $3$ ($n = 3, 6, 24$) extra
configs; so $10$ L0 configs $\times$ $3$ seeds $= 30$ runs per case. L2 (Stage 7): $11$ configs $\times$ $3$ seeds
$= 33$ runs per case. Cost from the first laptop sweep (1000 iterations, both notebooks running at once): L0 about
$5$ min per $1000$ iterations in B ($4.6$–$6.7$) and about $4$ min in C ($3.6$–$5.1$); L2 about $0.7$–$0.8$ min. At
$10000$ iterations that is about $50$ min (B) or $40$ min (C) per L0 run and about $7$ min per L2 run, so about one day
of serial compute per case (L0 about $25$ h, L2 about $4$ h). The sweeps are meant to be split over the cluster nodes
at the level of single runs (every run is independent and takes at most about $1$ h).
**Concurrency rule:** two processes must never append to the same `results/` folder (concurrent appends to the CSV files
corrupt them): every process or shard gets its own results folder, and the folders are merged afterwards (the row
keys make the merge a concatenation). The training-settings guard of `meta.json` refuses to mix results made with other
settings: the first laptop sweep (1000 iterations) must be kept in a different folder.
Sweep cells skip finished runs, so `runs.csv` doubles as the planned-vs-finished manifest and the sweep resumes after
interruption. Budget estimates live in the notebook introductions.
Memory: the L2 arm keeps the basis gathered at the sampled training points (B about $1.7$ GB for $K = 65$,
less for smaller $K$; C about $0.5$ GB).

## 7. Two measurement passes (hardware discipline)

Timing is contaminated if runs share the machine, accuracy is not. So:

- **Pass A (accuracy):** the sweep cells run the grid (serially as written; parallel execution across
  kernels or machines is manual). Log error, iterations, function/gradient evaluations. Wall-times from
  this pass are orientation only.
- **Pass T (timing):** no retraining. Seconds per function/gradient evaluation of the stored runs (Stage 5), in
  fresh processes, serially, with the thread count fixed from the thread scaling. `medians()` uses
  `n_fev` $\times$ `s_per_eval` as the time of a configuration where `timing.csv` has it (`time_src = T`) and the
  Pass-A time otherwise (`time_src = A`).

Hygiene for both passes:
- Laptop: plugged in, fixed power mode, no other heavy load, display sleep off. Cluster: a whole node (`--exclusive`) for
  Stage 5, nothing else on it; thread counts above the cores allocated to the job are meaningless.
- Each timing rebuilds and retraces and does a warm-up evaluation before the $3$ timed repeats;
  randomized run order, `time.perf_counter()`.
- Thread counts are set explicitly in the fresh process before TensorFlow runs, and recorded.
- `hardware.json` records the chip (`machdep.cpu.brand_string` on macOS, else the platform string), the
  performance/efficiency core split (`hw.perflevel0.logicalcpu`, `hw.perflevel1.logicalcpu`; empty on Linux), the CPU
  count and the cores this job may use (`sched_getaffinity`), the node name and Slurm job id, OS version, Python,
  TensorFlow and NumPy versions and the thread count. Power mode is fixed manually, not auto-recorded.
- Check for throttling: the first timed config is repeated at the end of the session; if the time differs by more
  than $10\%$, discard and repeat the session.

## 8. Metrics

### Accuracy
- Test $\mathrm{NRMSE}$ and Pearson dissimilarity ($1 - \rho$), as in RQ2.
- Efficiency ratio $\varepsilon_{\mathrm{total}} / \varepsilon_{\mathrm{proj}}$: the floor uses $K_{\mathrm{ref}} = 17$ POD modes
  (B) or $33$ Legendre modes (C) for L0 and the run's own $K$ for L2. Values close to $1$ mean that the dynamics
  are no longer the bottleneck.
- Per-case extras: direction error, vorticity error and extrapolation ratio for B; for C the activation-time MAE
  over the points where both the FOM and the prediction activate, and `act_miss` = fraction of FOM-activating
  points that the prediction never activates (so a model that fails to activate cannot look good).
- Selection: the knee is chosen on validation error (Section 10); trade-off plots show test medians.

### Cost
- $P_{\mathrm{dyn}}$, $P_{\mathrm{rec}}$, $P_{\mathrm{total}}$ (counted from the model, not by hand).
- Function/gradient evaluations in total (`n_fev`; line search makes it more than one per iteration). This is
  hardware-independent but **not comparable across arms** (L0 and L2 do different work per evaluation).
- **Operations:** `flops_per_eval`, a formula count of multiply-adds of one loss+gradient evaluation on the training
  set: $3 \times (\text{dynamics rollout} + \text{reconstructor})$ with the backward pass counted as $2\times$ forward and
  activations ignored. Dynamics $2 P_{\mathrm{dyn}} S (T-1)$; L0 $2 P_{\mathrm{rec}} S T N_x$; L2 $2 P_{\mathrm{rec}} S T
  + 2 K n_{\mathrm{comp}} S T N_x$. `flops_total` $=$ `flops_per_eval` $\times$ `n_fev` is the cross-arm cost axis.
  The formula is checked against Pass T (`plot_flops_check`: measured seconds per operation should be similar for
  the two arms); where it is not, time is the reference.
- Wall-time per iteration, total training time, time-to-target (median over seeds of the first checkpoint time with
  validation $\mathrm{NRMSE}$ below each threshold of Section 11; a seed that never reaches it counts as infinite and
  `n_hit` reports how many did).
- No profiler breakdown and no peak-memory tracking: replaced by counts, the formula, measured seconds per evaluation
  and the linear time fit of Section 9.
- Inference time on the test set (`infer_time_s`, best of $3$).

## 9. Analysis and figures

One notebook per case (no cross-case panels); every plot shows the median line with the min-max seed band.

1. Test $\mathrm{NRMSE}$ vs $P_{\mathrm{total}}$ and vs $P_{\mathrm{dyn}}$ (log-log), per arm. Fit
   $E = E_{\infty} + a \cdot P^{-\alpha}$ with a bootstrap over seeds; report $\alpha$, $E_{\infty}$, $95\%$ interval,
   and the knee (Section 10 rule).
2. Test $\mathrm{NRMSE}$ vs function evaluations (headline for one arm) and vs training time (Pass T where available).
3. Test $\mathrm{NRMSE}$ vs iterations, one curve per width (H3), per arm.
4. Pareto front of (median time, median test error) per arm, with the best and the cheapest-within-$10\%$
   configurations printed (not overplotted).
5. Efficiency ratio: tabulated per run in `runs.csv`/`predictions.csv` (no dedicated figure).
6. Linear time model $t(P_{\mathrm{dyn}}) = t_0 + t_1 \cdot P_{\mathrm{dyn}}$ and dynamics share at $4w^*$, per arm (H2).
7. Thread scaling: speedup and efficiency vs threads (Stage 5).
8. Singular-value decay of the training snapshots (Kolmogorov-width proxy), as in RQ2.
9. Failure map: diverged (`failed`) and early-stopped (`stalled`) runs per configuration.
10. Stage 7: (a) error vs total operations, function evaluations, time and parameters for L0 and L2 on the same axes
    (`plot_arms`); (b) error vs $n$, vs $P_{\mathrm{dyn}}$ and vs $K$ with the projection floor
    (`plot_linear_sweeps`); (c) the `cost_to_reach` table (H6); (d) the operation-count check (`plot_flops_check`).

### Explainability add-on
- For each $n$ and each arm: effective rank of the latent trajectories $\mathbf{s}(t)$ (SVD, number of singular values
  above $10^{-3}$ of the first) on the test set.
- Jacobian rank of $\mathrm{NN}_{\mathrm{dyn}}$ along the trajectories (singular values of $\mathrm{d}\dot{\mathbf{s}} / \mathrm{d}\mathbf{s}$,
  averaged over 20 sampled trajectory points).
- If extra latents collapse (rank below $n$), this explains the plateau of H4 and links to the interpretability
  question of the project.

## 10. Decision rules

- The "knee" configuration is the smallest width in the width sweep of an arm (canonical depth, its $n$ and $K$)
  whose median validation $\mathrm{NRMSE}$ is within $10\%$ of the best width-sweep median at $I_{\max}$
  (the smallest $P_{\mathrm{total}}$ on that slice).
- The "compute-optimal" configuration for a time budget is the lowest-error point on the Pareto front of the arm
  under that budget. Deviation (as implemented): the front uses test medians; switching it to validation medians
  is a one-line change (`medians('nrmse_valid', variant)` in `pareto_front`), as is the H4/H5 test-vs-validation choice.
- Whether L2 is worth using is decided by `cost_to_reach` at the $1.5\times$ level (H6), on operations and on
  Pass-T time; if the two disagree the time is the reference.
- If H1 and H3 are both rejected, the study reports that accuracy is not limited by size
  or by optimization in this range, and the next step is to look at the reconstructor
  or at the data (floors).
- All conclusions are stated for BFGS-only at the 10000-iteration budget, never as asymptotic claims.

## 11. Thresholds to freeze (before running Stage 1)

| Item | Default, as implemented (to be confirmed after Stage 0) |
|---|---|
| $\mathrm{NRMSE}$ targets for time-to-target | $5\times10^{-2}$, $2\times10^{-2}$, $1\times10^{-2}$ (adjust to the Stage 0 level) |
| Plateau criterion | local log-log slope above $-0.1$ over two consecutive grid points (implemented as: the bootstrap interval of $\alpha$ reaches below $0.1$, plus the knee) |
| Seed-sensitivity warning | max/min test $\mathrm{NRMSE}$ across seeds above $2$ |
| Throttling check | repeat time differs by more than $10\%$ |
| H2 threshold | dynamics share of per-iteration time at $4w^*$ below $50\%$ |
| H4 threshold | improvement below $5\%$ from $n^*$ to $2n^*$ |
| H6 decision level | target $= 1.5 \times$ best L0 median test $\mathrm{NRMSE}$ |

Freeze means: commit the file, tag `rq2bis-protocol-v0.3`, and do not edit it afterwards.
Any deviation is written in `deviations.md` with the date and the reason.

## 12. Reproducibility

- Repository layout per case folder (`Study_II/Case_B_2D-Stokes/`, `Study_II/Case_C_AP1D/` holding Case C):
  `TestCase_{B,C}_bis.ipynb`, `utils.py`, `optimization.py`, `timing_worker.py`, `results/` with `runs.csv`,
  `predictions.csv`, `curves.csv`, `weights/`, `timing.csv`, `hardware.json`, `meta.json`.
  No `ldnet_ext/`, `configs/`, `runs/<case>/<config>/<seed>/` tree, `analysis/` folder,
  `budget.md` or `environment.txt` (package versions live in `meta.json`).
- One `sweep()` call per stage; `runs.csv` is the resume manifest (finished runs are skipped).
- Cells of the notebooks tagged `worker` are the setup that `timing_worker.py` re-executes: keep them free of
  side effects other than creating `results/`.
- Random seeds fixed and stored. Float64 everywhere. On the cluster, record the module/conda environment used
  (`meta.json` stores the library versions of the first session).
- Raw per-run rows are append-only and never overwritten (`timing.csv` is rewritten per timing session).
  The analysis is regenerated from them.

## 13. Open items (updated 2026-10-11; resolved items kept for the record)

- [x] Scripts and canonical configurations confirmed (Section 2 table; counts verified by construction).
- [x] $\mathrm{NN}_{\mathrm{dyn}}$/$\mathrm{NN}_{\mathrm{rec}}$ inputs and output transform confirmed (Section 2 table).
- [x] Optimizer chosen: scipy BFGS, 0 Adam + max BFGS iterations (user decision; not L-BFGS).
- [x] $I_{\max}$ raised from 1000 to 10000 for both cases (user decision 2026-10-14, after the laptop sweep at 1000).
- [x] Stage 6 out; Stage 4 not needed (every run has the 10000 budget); Stage 7 in as RQ2's L2 in R2.
- [x] Linear arm design decided: $K = 65$, $n$ / width / $K$ sweeps, B L0 latent grid extended to $\{1,2,5,10\}$,
  operation-count cost axis plus Pass-T timing.
- [ ] Revisit $n_{L2}$ (B $10$, C $12$) after the L2 $n$ sweep. The laptop sweep at 1000 iterations suggests that in B the
  error of L2 stops improving at $n = 2$–$5$ (flat or slightly worse at $n = 10$): check this at 10000.
- [ ] Freeze the numeric thresholds in Section 11 after Stage 0.
- [ ] Supervisor sign-off on the BFGS-for-L-BFGS swap.
- [ ] Cluster layout: how the runs are split over nodes (per-run or per-shard results folders) and merged;
  not built into the notebooks yet.
- [ ] RQ2 Case C notebook still computes the equilibrium offset once at initialisation (differs from the original
  script); decide whether to re-run RQ2 Case C.
- [ ] Optional code follow-ups: Pareto/H5/H4 on validation medians; knee/canonical markers and an
  efficiency-ratio figure; local two-point slope tabulation for the plateau criterion; a tolerance in the H3
  monotonicity check.
