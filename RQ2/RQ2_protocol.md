# RQ2_protocol.md

**Status:** draft v0.1. Freeze it (git tag `rq2-protocol-v1`) before running any L1–L3 experiment.

## 1. Question

**RQ2:** In a Latent Dynamics Network, how much of the reconstructor's nonlinearity is needed? Can $\mathrm{NN}_{\mathrm{rec}}$ be replaced by a linear spectral expansion without losing accuracy?

Design: an ablation that changes only the reconstructor. $\mathrm{NN}_{\mathrm{dyn}}$, the data, the loss, the optimizer and the splits stay fixed.

## 2. Test cases

| Case | Source | Setup (from `SELECTED_TESTCASES.md`) | Role |
|---|---|---|---|
| A | Notebook | train/valid/test $=100/100/100$, $\Delta t=0.05$, $d_s=2$, $\mathrm{NN}_{\mathrm{rec}}$ 7–7–1 (92 params), inputs $(\mathbf{s},x)$, Adam 50 + BFGS 150 [1] | Smooth case, where a linear decoder should work |
| B | `TestCase_2.py` | train/valid/test $=80/20/10$, test horizon $T=40$, $\Delta t=0.2$, $\mathrm{NN}_{\mathrm{rec}}$ $24^4$, inputs $(\mathbf{s},u,\mathbf{x})$, 200 random points per $(\text{sample},t)$, Adam 200 + BFGS 10000 [1] | 2D vector field with time extrapolation |
| C | `AP_create_dataset` | 400 samples, $501$ time steps, $101$ points after $\times 8$ subsampling, 2 input signals [1] | Sharp fronts, where a linear decoder should struggle |

The MD gives no canonical LDNet config for case C. Take it from the notebook or script and record it here before freezing.

## 3. Hypotheses and falsification criteria

Let $\varepsilon_{\mathrm{proj}}(K)$ be the projection floor (Section 6). Numeric thresholds are provisional. Freeze them after the projection floors are computed and before any training.

- **H1 (case A):** a linear reconstructor with $K \le 5$ Fourier modes matches L0. *Falsified if* $\mathrm{NRMSE}_{L2} > 1.5\,\mathrm{NRMSE}_{L0}$ at $K \le 5$.
- **H2 (case C):** linear variants (L1, L2) follow $\varepsilon_{\mathrm{proj}}(K)$, which decays slowly. L3 does not close the gap. Only the original reconstructor (L0), with its implicit learned basis, does. *Falsified if* $\mathrm{NRMSE}_{L3}(K{=}65) \le 2\,\mathrm{NRMSE}_{L0}$.
- **H3 (case B):** POD-based variants capture the bulk flow, but the direction error near corners and in small vortices dominates. The effect on extrapolation is open in sign: a linear decoder could extrapolate better or worse.
- **H4 (modality of the latents):** if R2 (end-to-end) is much better than R1 (frozen latents) for the same linear decoder, then the latents of L0 are not modal on their own. *Criterion:* $\mathrm{NRMSE}_{R1} > 2\,\mathrm{NRMSE}_{R2}$ at equal $K$.

## 4. Reconstructor variants

Define $\mathbf{z}=\mathbf{s}$ in case A and $\mathbf{z}=[\mathbf{s};u]$ in case B, matching the current $\mathrm{NN}_{\mathrm{rec}}$ inputs [1]. Every variant includes a mean field $\bar y(\mathbf{x})$.

| Variant | Reconstructor | Trainable |
|---|---|---|
| L0 | $\mathrm{NN}_{\mathrm{rec}}(\mathbf{z},\mathbf{x})$, original | MLP |
| L1 | $\bar y+\sum_{k=1}^{d_s}s_k\varphi_k(\mathbf{x})$ | none |
| L2 | $\bar y+\sum_{k=1}^{K}(W\mathbf{z}+\mathbf{b})_k\varphi_k$ | $W,\mathbf{b}$ |
| L3 | $\bar y+\sum_{k=1}^{K}g_\theta(\mathbf{z})_k\varphi_k$ | $g_\theta$ |

Rules:
- L1 and L2 drop the output transform that `TestCase_2.py` applies, $o\mapsto(o^3+\alpha o)/(1+\alpha)$ with $\alpha=0.05$ [1]. This makes them strictly linear. Add an **L2+link** variant that keeps the transform, to isolate its effect.
- Size $g_\theta$ to match the L0 reconstructor parameter count, and also report results at the best size.

## 5. Training regimes

- **R1, frozen latents:** train L0, save $\mathbf{s}(t)$, then fit only the new reconstructor on those latents.
- **R2, end-to-end:** train $\mathrm{NN}_{\mathrm{dyn}}$ and the reconstructor together from scratch with the original pipeline (Adam then BFGS).

**Fixed across variants:** splits, $\Delta t$, normalization, loss (including the direction term in case B and the $10^{-5}$ Tikhonov term in case A [1]), optimizer schedule, and the early-stopping rule on validation loss.

## 6. Bases and projection floor

| Case | Basis |
|---|---|
| A | Fourier |
| B | Vector POD on $(v_x,v_y)$ |
| C | Chebyshev or Legendre |

- **POD from training data only**, on the full grid, to avoid leakage.
- Orthonormalize every basis on the full grid with the discrete $L^2$ inner product.
- Compute the projection floor before any training:
$$\varepsilon_{\mathrm{proj}}(K)=\frac{\|y-P_Ky\|}{\|y\|},\qquad P_Ky=\sum_{k=1}^{K}\langle y,\varphi_k\rangle_h\,\varphi_k$$
- Also plot the singular-value decay $\sigma_k/\sigma_1$ of the training snapshot matrix for each case, as an empirical proxy for the Kolmogorov $n$-width.

## 7. Sweep

| Factor | Levels |
|---|---|
| $K$ | $\{d_s,3,5,9,17,33,65\}$ |
| $d_s$ | $\{1,2,3,5\}$ ($\{1,5,10\}$ for B, as in the paper [1]) |
| Regime | R1, R2 |
| Seeds | 3 |

Order: A, then C, then B. For case B, run a single reduced-budget sweep (adopted 2026-10-04: 200 Adam + 500 BFGS; pilot: two L0 runs at 1000 BFGS took ~6-7 min/run with test NRMSE ~1.6e-2/1.4e-2 vs ~4e-3 at full budget, so the full 10000-epoch sweep of 195 runs is infeasible). No full-budget rerun is planned: all B numbers are conditional on this budget, and the L0 gate against the paper reference stays provisional.

## 8. Metrics

- **Primary:** $\mathrm{NRMSE}$ and Pearson dissimilarity $1-\rho$.
- **Efficiency ratio:** $\varepsilon_{\mathrm{total}}/\varepsilon_{\mathrm{proj}}$. A value near 1 means the dynamics are not the bottleneck.
- **Per-mode error:** $\varepsilon_k=|\hat c_k-c_k|/|c_k|$.
- **Case B:** direction error, vorticity error, and the extrapolation ratio $\varepsilon_{[20,40]}/\varepsilon_{[0,20]}$ [1].
- **Case C:** front position or activation time error.
- **Cost:** reconstructor parameters, training time, inference time.

## 9. Spectral analysis of the trained L0 decoder

1. Spectral coefficients $\hat c_k(\mathbf{z})=\langle\mathrm{NN}_{\mathrm{rec}}(\mathbf{z},\cdot),\varphi_k\rangle_h$.
2. **Linearizability index** $R^2_{\mathrm{lin}}(K)$ from the best fit $\hat{\mathbf{c}}\approx W\mathbf{z}+\mathbf{b}$.
3. **Effective spectral width** $K_{99}(t)$.
4. **Latent mode shapes** $\partial\mathrm{NN}_{\mathrm{rec}}/\partial z_k$, compared with POD by principal angles between subspaces.
5. **Cross-seed alignment** of latents (affine map or CCA) before any interpretation. Report only findings that hold across seeds.

## 10. Statistics and reporting

- Report median and IQR over 3 seeds. Never report a single run.
- Use paired comparisons with the Wilcoxon signed-rank test, Holm correction, and effect sizes. Give more weight to effect sizes and consistency across seeds than to $p$-values, because 3 seeds give little power.
- **Core figures:** error vs $K$ with the projection floor and the L0 band (the main RQ2 figure), $R^2_{\mathrm{lin}}$ vs $K$, mode shapes vs POD, per-mode error spectra, a Pareto plot of parameters vs error, and the R1 vs R2 gap.

## 11. Reproducibility

- **Gate:** the refactored L0 with the original seed must match the original script's predictions to $\le 10^{-10}$ (float64), plus the paper's reference metrics. Do not proceed until this passes.
- Do not modify the original TestCase scripts. Put new code in `ldnet_ext/`.
- Each run stores `config.json`, the git hash, `metrics.json`, weights and latents.
- Float64 runs on CPU on macOS. Do not install `tensorflow-metal`.

## 12. Open items (resolve before freezing)

- [ ] Read `TestCase_1a/1b/1c.ipynb` and `TestCase_3.py` to confirm which scripts correspond to cases A and C.
- [ ] Confirm whether the case C $\mathrm{NN}_{\mathrm{rec}}$ takes $\mathbf{u}$ as an input and whether it uses an output transform.
- [ ] Add the canonical config for case C to Section 2.
- [ ] Freeze the numeric thresholds in H1, H2 and H4 after computing the projection floors.
- [ ] Confirm that case B data are on a uniform Cartesian grid (`NS_create_dataset` builds the points with `meshgrid`), and correct any "triangular FEM mesh" claim in the ontology if so.

I can save this as a Notion page under HPC → Corsi High Performance Computing → Multidisciplinary project, or draft `bases.py` and `reconstructors.py` for the fork.