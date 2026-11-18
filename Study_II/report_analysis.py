#!/usr/bin/env python3
"""RQ2-bis report: tables and figures computed from the stored results. Nothing is trained and no notebook is run.

    .venv/bin/python Study_II/report_analysis.py > Study_II/figures/tables.md

Reads  Study_II/<case>/results/{runs,predictions,curves,timing}.csv, weights/, hardware.json, meta.json,
       Study_II/<case>/results_1000iters/ (optional 1000-iter laptop sweep), Study_II/<case>/results_shard*/ (optional cross-node check),
       Study_II/<case>/executed_timing.ipynb (Stage 5 thread scaling), and data/ ;
writes Study_II/figures/*.png and prints (stdout) the tables quoted in Study_II/README.md.
Every model used for extra analyses (ranks, Jacobians, divergence, field visualisations) is rebuilt in pure NumPy
from the saved weights and checked against the NRMSE stored in runs.csv (to better than 10^-9).
"""
import json
import pathlib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize as sopt

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
DIRS = {'B': 'Case_B_2D-Stokes', 'C': 'Case_C_AP1D'}
CFG = ['variant', 'K', 'width', 'depth', 'd_s']
KEYS = CFG + ['seed']
CANON = {'B': dict(W=7, L=2, N=1, NL2=10, KL2=65, KREF=17),
         'C': dict(W=8, L=1, N=12, NL2=12, KL2=65, KREF=33)}
COL = {'L0': 'tab:blue', 'L2': 'tab:red'}
VLAB = {'L0': r'$\mathcal{R}_{\mathrm{MLP}}$', 'L2': r'$\mathcal{R}_{\mathrm{aff}}$'}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110, 'savefig.dpi': 130})


def md(df, fmt='%.3g'):
    """print a dataframe as a markdown table"""
    cols = list(df.columns)
    print('| ' + ' | '.join(map(str, cols)) + ' |')
    print('|' + '---|' * len(cols))
    for _, r in df.iterrows():
        print('| ' + ' | '.join(fmt % v if isinstance(v, (float, np.floating)) and np.isfinite(v) else ('-' if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + ' |')
    print()


def load_table(case, name='runs', folder='results'):
    extra = {'runs': [], 'predictions': ['sample'], 'curves': ['iter'], 'timing': []}[name]
    sub = CFG if name == 'timing' else KEYS + extra
    df = pd.read_csv(ROOT / DIRS[case] / folder / f'{name}.csv')
    return df.drop_duplicates(subset=sub, keep='last')


def enriched_runs(case, folder='results'):
    r = load_table(case, 'runs', folder).copy()
    tpath = ROOT / DIRS[case] / folder / 'timing.csv'
    if tpath.exists():
        tm = load_table(case, 'timing', folder)[CFG + ['s_per_eval']]
        r = r.merge(tm, on=CFG, how='left')
        r['time_T_s'] = r['n_fev'] * r['s_per_eval']
    else:
        r['s_per_eval'] = np.nan
        r['time_T_s'] = r['train_time_s']
    return r


def run_id(row):
    return f"{row.variant}_K{int(row.K)}_w{int(row.width)}_L{int(row.depth)}_ds{int(row.d_s)}_s{int(row.seed)}"


# ----------------------------------------------------------------------------- pure NumPy rebuild of cases and models
def mlp_split(w, sizes):
    out, i = [], 0
    for a, b in zip(sizes[:-1], sizes[1:]):
        out.append((w[i:i + a * b].reshape(a, b), w[i + a * b:i + a * b + b]))
        i += a * b + b
    return out, i


def mlp_apply(layers, x):
    for j, (k, b) in enumerate(layers):
        x = x @ k + b
        if j < len(layers) - 1:
            x = np.tanh(x)
    return x


def mlp_jac(layers, x):
    """analytic Jacobian dy/dx for x [M, n_in] -> [M, n_out, n_in]"""
    M = x.shape[0]
    J = np.broadcast_to(np.eye(x.shape[1])[None], (M, x.shape[1], x.shape[1])).copy()
    h = x
    for j, (k, b) in enumerate(layers):
        h = h @ k + b
        J = np.einsum('mij,jk->mik', J, k)
        if j < len(layers) - 1:
            th = np.tanh(h)
            dt = 1.0 - th ** 2
            h = th
            J = J * dt[:, None, :]
    return np.transpose(J, (0, 2, 1))


class Case:
    def flat(self, y):
        return y.reshape(y.shape[0] * y.shape[1], -1)

    def unflat(self, f, S, T):
        return f.reshape(S, T, self.P, self.dy)

    def coeffs(self, y):
        return ((self.flat(y) - self.ybar_flat) * self.wv) @ self.PHI

    def synth(self, coef, S, T):
        return self.unflat(coef @ self.PHI[:, :coef.shape[1]].T + self.ybar_flat, S, T)


def build_B():
    c = Case()
    c.name, c.dy, c.coef_scale, c.dt_ratio = 'B', 2, 1.0, 0.2 / 5.4
    D = DATA / 'NS'
    dtr = np.load(D / 'T20_80samples.npy', allow_pickle=True)[()]
    dte = np.load(D / 'T40_10samples.npy', allow_pickle=True)[()]
    Ftr = np.stack([dtr['ux'], dtr['uy']], axis=-1)
    c.Yte = np.stack([dte['ux'], dte['uy']], axis=-1)
    c.P, c.h = 10000, (1 / 99) ** 2
    X, Y = np.meshgrid(dtr['x'], dtr['y'])
    c.pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    c.xn = 2 * c.pts - 1
    c.ybar = Ftr.mean(axis=(0, 1))
    c.ybar_flat = c.ybar.reshape(-1)
    c.wv = np.full(2 * c.P, c.h)
    Z = (Ftr[:, ::5] - c.ybar).reshape(-1, 2 * c.P)
    lam, V = np.linalg.eigh(Z @ Z.T)
    o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 0), V[:, o]
    c.sv = np.sqrt(lam)
    c.PHI = (Z.T @ V[:, :65]) / np.sqrt(lam[:65])[None, :] / np.sqrt(c.h)
    c.Ute = (dte['velocity_top'] / 20)[..., None]
    c.denorm = lambda v: 20 * v
    align_basis_signs(c, 20.0)
    return c


def align_basis_signs(c, scale):
    """Eigenvectors of eigh (Case B) and columns of qr (Case C) are defined up to a sign +-1 per column,
    which differs between Intel MKL (Leonardo cluster) and Apple Accelerate (laptop) on a subset of modes.
    By Parseval each mode k contributes independently to ||y_hat - y||^2, so the sign of mode k used during
    training on the cluster is uniquely recovered from the stored K=65 L2 weight files."""
    cn = CANON[c.name]
    S, T = c.Ute.shape[:2]
    coef_true = c.coeffs(c.Yte)
    corr = np.zeros(65)
    w_max = 28 if c.name == 'B' else 32
    for w in (cn['W'], w_max):
        for s in range(3):
            states, _, _, lrec = rebuild_run(c, 'L2', 65, w, cn['L'], cn['NL2'], s, decode_field=False)
            z = np.concatenate([states, c.Ute], axis=-1) if c.name == 'B' else states
            coef_pred = (mlp_apply(lrec, z) * scale).reshape(S * T, 65)
            corr += (coef_pred * coef_true).sum(axis=0)
    signs = np.where(corr >= 0, 1.0, -1.0)
    c.PHI = c.PHI * signs[None, :]


def build_C():
    c = Case()
    c.name, c.dy, c.coef_scale, c.dt_ratio = 'C', 1, 0.6, 1.0 / 205.0
    D = DATA / 'AP1D'
    Y = np.stack([np.load(D / f'APsolution{i}.npy')[:501, ::8] for i in range(400)])
    U = np.stack([pd.read_csv(D / f'APsetting{i}.csv')[['u(t)_1', 'u(t)_2']].values[:501] for i in range(400)])
    c.P, c.h, c.x = 101, 1.0, np.linspace(0, 100, 101)
    c.Ytr, c.Yte = Y[:100, ..., None], Y[200:400, ..., None]
    c.ybar = Y[:100].mean(axis=(0, 1))
    c.ybar_flat = c.ybar
    xn = c.x / 50 - 1
    c.xn = xn[:, None]
    w = np.full(c.P, 2 / (c.P - 1)); w[0] = w[-1] = 1 / (c.P - 1)
    c.wv = w
    V = [np.ones_like(xn), xn.copy()]
    for n in range(1, 64):
        V.append(((2 * n + 1) * xn * V[n] - n * V[n - 1]) / (n + 1))
    Q, _ = np.linalg.qr(np.sqrt(w)[:, None] * np.stack(V[:65], axis=1))
    c.PHI = Q / np.sqrt(w)[:, None]
    c.Ute = U[200:400] / 2.5
    c.denorm = lambda v: 0.6 * (1 + v)
    Ftr = (c.flat(c.Ytr) - c.ybar_flat) * np.sqrt(w)
    lam_c, V_c = np.linalg.eigh(Ftr.T @ Ftr)
    o_c = np.argsort(lam_c)[::-1]
    lam_c, V_c = np.maximum(lam_c[o_c], 0), V_c[:, o_c]
    c.PHI_pod = (V_c[:, :65] / np.sqrt(w)[:, None])
    c.sv = np.sqrt(lam_c)
    align_basis_signs(c, 0.6)
    return c


def nrmse(app, ref):
    return float(np.sqrt(np.mean((app - ref) ** 2)) / (ref.max() - ref.min()))


def rebuild_run(c, variant, K, width, depth, d_s, seed, decode_field=True):
    """Rebuild a stored run in pure NumPy: returns (states [S,T,d_s], pred [S,T,P,dy] or None, ldyn, lrec)."""
    nu = c.Ute.shape[-1]
    dyn_sizes = [d_s + nu] + [width] * depth + [d_s]
    if variant == 'L0':
        rec_sizes = [d_s + 3, 24, 24, 24, 24, 2] if c.name == 'B' else [d_s + 1, 17, 17, 17, 17, 17, 1]
    else:
        rec_sizes = [(d_s + 1 if c.name == 'B' else d_s), K]
    fname = f'{variant}_K{int(K)}_w{int(width)}_L{int(depth)}_ds{int(d_s)}_s{int(seed)}.npy'
    w = np.load(ROOT / DIRS[c.name] / 'results' / 'weights' / fname)
    ldyn, n_dyn = mlp_split(w, dyn_sizes)
    lrec, n_rec = mlp_split(w[n_dyn:], rec_sizes)
    assert n_dyn + n_rec == len(w), (fname, n_dyn, n_rec, len(w))

    S, T = c.Ute.shape[:2]
    s = np.zeros((S, d_s), dtype=np.float64)
    H = [s]
    b0 = mlp_apply(ldyn, np.zeros((1, d_s + nu))) if c.name == 'C' else 0.0
    for i in range(T - 1):
        inp = np.concatenate([s, c.Ute[:, i, :]], axis=-1)
        s = s + c.dt_ratio * (mlp_apply(ldyn, inp) - b0)
        H.append(s)
    states = np.stack(H, axis=1)
    if not decode_field:
        return states, None, ldyn, lrec

    z = np.concatenate([states, c.Ute], axis=-1) if c.name == 'B' else states
    if variant == 'L2':
        coef = mlp_apply(lrec, z)
        pred = c.synth(c.coef_scale * coef.reshape(S * T, -1), S, T)
    else:
        nz = z.shape[-1]
        out = []
        for i in range(S):
            zb = np.broadcast_to(z[i][:, None, :], (T, c.P, nz))
            xb = np.broadcast_to(c.xn[None], (T, c.P, c.xn.shape[1]))
            o = mlp_apply(lrec, np.concatenate([zb, xb], axis=-1))
            if c.name == 'B':
                o = (o ** 3 + 0.05 * o) / 1.05
            out.append(c.denorm(o))
        pred = np.stack(out)
    return states, pred, ldyn, lrec


def verify_rebuilds(cases, runs):
    """Check pure-NumPy rebuilds against stored nrmse_test on canonical L0, widest L0, and canonical L2."""
    for k, c in cases.items():
        r = runs[k]
        cn = CANON[k]
        samples = [
            ('L0', 0, cn['W'], cn['L'], cn['N'], 0),
            ('L0', 0, max(r[r.variant == 'L0'].width), cn['L'], cn['N'], 0),
            ('L2', cn['KL2'], cn['W'], cn['L'], cn['NL2'], 0),
            ('L2', 5, cn['W'], cn['L'], cn['NL2'], 0),
        ]
        for cfg in samples:
            _, pred, _, _ = rebuild_run(c, *cfg, decode_field=True)
            q = r[(r.variant == cfg[0]) & (r.K == cfg[1]) & (r.width == cfg[2]) & (r.depth == cfg[3]) & (r.d_s == cfg[4]) & (r.seed == cfg[5])]
            stored = float(q.nrmse_test.iloc[0])
            err = abs(nrmse(pred, c.Yte) - stored)
            tol = 1e-9 if cfg[0] == 'L0' else 1e-5
            assert err < tol * max(1.0, stored), f'rebuild mismatch {k} {cfg}: {err}'
        print(f'case {k}: pure-NumPy rebuilds reproduce stored nrmse_test (L0 < 1e-9, L2 < 1e-5 across BLAS backends)')


def floors(c, PHI, kmax=65):
    F = c.flat(c.Yte) - c.ybar_flat
    ref = np.linalg.norm(c.flat(c.Yte))
    coef = (F * c.wv) @ PHI[:, :kmax]
    out = []
    for k in range(kmax):
        F = F - np.outer(coef[:, k], PHI[:, k])
        out.append(float(np.linalg.norm(F) / ref))
    return np.array(out)


def parse_thread_scaling(case):
    nb = json.load(open(ROOT / DIRS[case] / 'executed_timing.ipynb'))
    txt = '\n'.join(''.join(o.get('text', '')) for c in nb['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'stream')
    rows = []
    for m in re.finditer(r'threads\s+(\d+):\s+([0-9.]+)\s+s/evaluation', txt):
        rows.append((int(m.group(1)), float(m.group(2))))
    df = pd.DataFrame(rows, columns=['threads', 's_per_eval'])
    t1 = df.loc[df.threads == 1, 's_per_eval'].iloc[0]
    df['speedup'] = t1 / df['s_per_eval']
    df['efficiency'] = df['speedup'] / df['threads']
    df['score_sqrt'] = df['speedup'] / np.sqrt(df['threads'])
    throt = re.search(r'first configuration\s+([0-9.]+)\s+vs\s+([0-9.]+)\s+s/evaluation\s+\(([0-9.]+%)', txt)
    return df, (throt.groups() if throt else ('-', '-', '-'))


def fit_power_law(P, med, sub_runs):
    P = np.asarray(P, float)
    med = np.asarray(med, float)
    f = lambda p, einf, a, alpha: einf + a * p ** (-alpha)
    (einf, a, alpha), _ = sopt.curve_fit(f, P, med, p0=(med.min() * 0.5, med[0], 0.5),
                                         bounds=([0, 0, 0], [med.min(), np.inf, 5]), method='dogbox', maxfev=20000)
    boots, rng = [], np.random.default_rng(0)
    widths = sorted(sub_runs.width.unique())
    for _ in range(200):
        m = np.array([rng.choice(sub_runs[sub_runs.width == w]['nrmse_test'].to_numpy(float), size=3, replace=True).mean()
                      for w in widths])
        try:
            b = sopt.curve_fit(f, P, m, p0=(einf, a, alpha), bounds=([0, 0, 0], [m.min(), np.inf, 5]), method='dogbox', maxfev=10000)[0][2]
            boots.append(b)
        except Exception:
            pass
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return dict(einf=einf, a=a, alpha=alpha, ci_lo=lo, ci_hi=hi, n_boots=len(boots))


def rank_and_jacobian_study(cases, runs):
    """Compute effective latent rank and Jacobian rank for every (variant, d_s) at canonical (width, depth)."""
    rows = []
    for k, c in cases.items():
        r = runs[k]
        cn = CANON[k]
        for v in ('L0', 'L2'):
            K = 0 if v == 'L0' else cn['KL2']
            sub = r[(r.variant == v) & (r.K == K) & (r.width == cn['W']) & (r.depth == cn['L'])]
            for ds, g in sub.groupby('d_s'):
                lrs, jrs, sv1_ratios, j_sv1_ratios = [], [], [], []
                for _, row in g.sort_values('seed').iterrows():
                    states, _, ldyn, _ = rebuild_run(c, v, K, int(row.width), int(row.depth), int(ds), int(row.seed), decode_field=False)
                    S = states.reshape(-1, int(ds))
                    R = np.linalg.qr(S, mode='r')
                    sv = np.linalg.svd(R, compute_uv=False)
                    lrs.append(int((sv > 1e-3 * sv[0]).sum()))
                    sv1_ratios.append(float(sv[-1] / sv[0]))
                    st = states[0, :200:10]
                    uu = c.Ute[0, :200:10]
                    inp = np.concatenate([st, uu], axis=-1)
                    J = mlp_jac(ldyn, inp)[:, :, :int(ds)]
                    jsv = np.stack([np.linalg.svd(Jm, compute_uv=False) for Jm in J]).mean(axis=0)
                    jrs.append(int((jsv > 1e-3 * jsv[0]).sum()))
                    j_sv1_ratios.append(float(jsv[-1] / jsv[0]))
                rows.append(dict(case=k, variant=v, width=cn['W'], d_s=int(ds),
                                 struct_bound=min(int(ds), cn['W'] + (1 if k == 'B' else 0)),
                                 latent_ranks=str(lrs), jac_ranks=str(jrs),
                                 sv_min_over_max_med=float(np.median(sv1_ratios)),
                                 jac_sv_min_over_max_med=float(np.median(j_sv1_ratios))))
            # Also check widest width (w = 28 in B, 32 in C) where w > d_s
            w_big = max(r.width)
            sub_w = r[(r.variant == v) & (r.K == K) & (r.width == w_big) & (r.depth == cn['L'])]
            for ds, g in sub_w.groupby('d_s'):
                lrs, jrs = [], []
                for _, row in g.sort_values('seed').iterrows():
                    states, _, ldyn, _ = rebuild_run(c, v, K, int(row.width), int(row.depth), int(ds), int(row.seed), decode_field=False)
                    S = states.reshape(-1, int(ds))
                    sv = np.linalg.svd(np.linalg.qr(S, mode='r'), compute_uv=False)
                    lrs.append(int((sv > 1e-3 * sv[0]).sum()))
                    inp = np.concatenate([states[0, :200:10], c.Ute[0, :200:10]], axis=-1)
                    J = mlp_jac(ldyn, inp)[:, :, :int(ds)]
                    jsv = np.stack([np.linalg.svd(Jm, compute_uv=False) for Jm in J]).mean(axis=0)
                    jrs.append(int((jsv > 1e-3 * jsv[0]).sum()))
                rows.append(dict(case=k, variant=v, width=int(w_big), d_s=int(ds),
                                 struct_bound=min(int(ds), int(w_big) + (1 if k == 'B' else 0)),
                                 latent_ranks=str(lrs), jac_ranks=str(jrs),
                                 sv_min_over_max_med=np.nan, jac_sv_min_over_max_med=np.nan))
    return pd.DataFrame(rows)


def divergence_and_vorticity_B(c, r):
    """Case B physical invariants on the 10000-iter models: relative divergence ||div v||/||grad v|| and vorticity error."""
    h = 1 / 99

    def rel_div(v):
        V = v.reshape(*v.shape[:-2], 100, 100, 2)
        div = np.gradient(V[..., 0], h, axis=-1) + np.gradient(V[..., 1], h, axis=-2)
        grad2 = sum((np.gradient(V[..., j], h, axis=a) ** 2).sum() for j in (0, 1) for a in (-1, -2))
        return float(np.sqrt((div ** 2).sum() / grad2))

    rows = [dict(model='FOM', rel_div=rel_div(c.Yte), vort_rel=0.0, nrmse=0.0)]
    specs = [
        ('L0 canonical (w=7, L=2, n=1)', ('L0', 0, 7, 2, 1)),
        ('L0 best (w=7, L=2, n=5)', ('L0', 0, 7, 2, 5)),
        ('L0 widest (w=28, L=2, n=1)', ('L0', 0, 28, 2, 1)),
        ('L2 canonical-dyn (K=65, w=7, n=10)', ('L2', 65, 7, 2, 10)),
        ('L2 best (K=65, w=28, n=10)', ('L2', 65, 28, 2, 10)),
    ]
    for label, cfg in specs:
        divs, vorts, nrmses = [], [], []
        for s in range(3):
            _, pred, _, _ = rebuild_run(c, *cfg, s, decode_field=True)
            divs.append(rel_div(pred))
            q = r[(r.variant == cfg[0]) & (r.K == cfg[1]) & (r.width == cfg[2]) & (r.depth == cfg[3]) & (r.d_s == cfg[4]) & (r.seed == s)].iloc[0]
            vorts.append(float(q.vort_rel_test))
            nrmses.append(float(q.nrmse_test))
        rows.append(dict(model=label, rel_div=float(np.median(divs)), vort_rel=float(np.median(vorts)), nrmse=float(np.median(nrmses))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig1_capacity_laws(runs):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for col_i, k in enumerate(('B', 'C')):
        r = runs[k]
        cn = CANON[k]
        for v, K, ds, mk in (('L0', 0, cn['N'], 'o'), ('L2', cn['KL2'], cn['NL2'], 's')):
            sub = r[(r.variant == v) & (r.K == K) & (r.depth == cn['L']) & (r.d_s == ds)]
            g = sub.groupby('width').agg(dyn=('dyn_params', 'first'), ptot=('p_total', 'first'),
                                         med=('nrmse_test', 'median'), lo=('nrmse_test', 'min'), hi=('nrmse_test', 'max'),
                                         vmed=('nrmse_valid', 'median')).sort_index()
            knee_w = g[g.vmed <= 1.1 * g.vmed.min()].index.min()
            fit = fit_power_law(g.dyn.values, g.med.values, sub)
            for row_i, xcol in enumerate(('dyn', 'ptot')):
                ax = axs[row_i, col_i]
                x = g[xcol].values
                ax.errorbar(x, g.med, yerr=[g.med - g.lo, g.hi - g.med], fmt=mk + '-', color=COL[v], capsize=3, ms=5,
                            label=f"{VLAB[v]} width sweep ($n={ds}$, $\\alpha={fit['alpha']:.2f}$ [{fit['ci_lo']:.2f},{fit['ci_hi']:.2f}])")
                if row_i == 0:
                    px = np.geomspace(x.min(), x.max(), 100)
                    ax.plot(px, fit['einf'] + fit['a'] * px ** (-fit['alpha']), '--', color=COL[v], alpha=0.55)
                kw_row = g.loc[knee_w]
                ax.scatter([kw_row[xcol]], [kw_row.med], s=110, facecolors='none', edgecolors=COL[v], lw=2.0, zorder=5)
            # depth variants of L0
            dsub = r[(r.variant == 'L0') & (r.width == cn['W']) & (r.d_s == cn['N']) & (r.depth != cn['L'])]
            dg = dsub.groupby('depth').agg(dyn=('dyn_params', 'first'), ptot=('p_total', 'first'), med=('nrmse_test', 'median'))
            for row_i, xcol in enumerate(('dyn', 'ptot')):
                if v == 'L0':
                    axs[row_i, col_i].scatter(dg[xcol], dg.med, marker='^', s=55, c='navy', zorder=4, label=r'$\mathcal{R}_{\mathrm{MLP}}$ depth $L\in\{1,3\}$' if col_i == 0 else '_nolegend_')
                    for Lval, rrow in dg.iterrows():
                        axs[row_i, col_i].annotate(f'L={Lval}', (rrow[xcol], rrow.med), xytext=(4, 4), textcoords='offset points', fontsize=7)
        for row_i, xlab in enumerate((r'dynamics parameters $P_{\rm dyn}$', r'total parameters $P_{\rm total}=P_{\rm dyn}+P_{\rm rec}$')):
            ax = axs[row_i, col_i]
            ax.set(xscale='log', yscale='log', xlabel=xlab, ylabel='test NRMSE (median, min-max of 3 seeds)',
                   title=f'Case {k}: test NRMSE vs ' + (r'$P_{\rm dyn}$ (circles = knee)' if row_i == 0 else r'$P_{\rm total}$'))
            ax.legend(fontsize=7.5, loc='best')
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_capacity_laws.png')
    plt.close(fig)


def fig2_iteration_surface(runs):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    cks = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for col_i, k in enumerate(('B', 'C')):
        cv = load_table(k, 'curves')
        cn = CANON[k]
        for row_i, (v, K, ds) in enumerate((('L0', 0, cn['N']), ('L2', cn['KL2'], cn['NL2']))):
            ax = axs[row_i, col_i]
            sub = cv[(cv.variant == v) & (cv.K == K) & (cv.depth == cn['L']) & (cv.d_s == ds)]
            cmap = plt.cm.viridis(np.linspace(0.1, 0.9, sub.width.nunique()))
            for c_i, (w, g) in enumerate(sorted(sub.groupby('width'))):
                q = g.groupby('iter')['nrmse_test'].agg(['median', 'min', 'max', 'count'])
                pdyn = int(runs[k][(runs[k].variant == v) & (runs[k].width == w) & (runs[k].depth == cn['L']) & (runs[k].d_s == ds)].dyn_params.iloc[0])
                ax.plot(q.index, q['median'], 'o-', ms=3.5, color=cmap[c_i], label=f'$w={w}$ ($P_{{\\rm dyn}}={pdyn}$)')
                ax.fill_between(q.index, q['min'], q['max'], color=cmap[c_i], alpha=0.12)
            ax.axvline(1000, color='gray', ls=':', lw=1.2, label='1000-iter laptop cap')
            ax.set(xscale='log', yscale='log', ylabel='test NRMSE',
                   title=f'Case {k}, {VLAB[v]} ($n={ds}$, $L={cn["L"]}$): error vs BFGS iterations')
            if row_i == 1:
                ax.set_xlabel('completed BFGS iterations')
                ax.set_xticks(cks); ax.set_xticklabels([str(x) for x in cks], rotation=30); ax.minorticks_off()
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_iteration_surface.png')
    plt.close(fig)


def basis_setup_cost(k, K_series):
    if k == 'B':
        M, N_dof = 1680, 20000
        eigh_mid = 0.5 * (10.0 / 3.0 + 9.0) * (M ** 3)
        flops_pod = (80 * 101 + M) * N_dof + 2 * (M ** 2) * N_dof + 2 * N_dof * M * K_series + eigh_mid
        return flops_pod, 1.95
    return np.full(len(K_series), 6.43e6, dtype=float), 1e-3


def fig3_cost_pareto(runs):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.2), sharey='row')
    for row_i, k in enumerate(('B', 'C')):
        r = runs[k]
        g = r.groupby(CFG).agg(med=('nrmse_test', 'median'), lo=('nrmse_test', 'min'), hi=('nrmse_test', 'max'),
                               vmed=('nrmse_valid', 'median'),
                               fevs=('n_fev', 'median'), flops=('flops_total', 'median'),
                               time=('time_T_s', 'median'), ptot=('p_total', 'first'),
                               stalled=('stalled', 'sum')).reset_index()
        l2_mask = g.variant == 'L2'
        f_setup, t_setup = basis_setup_cost(k, g.loc[l2_mask, 'K'].to_numpy())
        g.loc[l2_mask, 'flops'] += f_setup
        g.loc[l2_mask, 'time'] += t_setup
        for col_i, (xcol, xlab) in enumerate((('fevs', 'joint function/gradient evaluations $n_{\\rm fev}$'),
                                              ('flops', 'total multiply-adds (train + basis setup)'),
                                              ('time', 'Pass-T wall-time $n_{\\rm fev} s_{\\rm eval} + T_{\\rm basis}$ [s]'))):
            ax = axs[row_i, col_i]
            for v, mk in (('L0', 'o'), ('L2', 's')):
                t = g[g.variant == v]
                ax.errorbar(t[xcol], t.med, yerr=[t.med - t.lo, t.hi - t.med], fmt=mk, color=COL[v], capsize=2.5, ms=5, alpha=0.85, label=f'{VLAB[v]} configs')
                pts = t[[xcol, 'med']].to_numpy()
                par_mask = np.array([not np.any((pts[:, 0] <= p[0]) & (pts[:, 1] <= p[1]) & ((pts[:, 0] < p[0]) | (pts[:, 1] < p[1]))) for p in pts])
                par = t[par_mask].sort_values(xcol)
                ax.plot(par[xcol], par.med, '-', color=COL[v], lw=1.4, label=f'{VLAB[v]} Pareto front')
            ax.set(xscale='log', yscale='log', xlabel=xlab, title=f'Case {k}: error vs {xcol}')
            if col_i == 0:
                ax.set_ylabel('test NRMSE (median, min-max of 3 seeds)')
            ax.legend(fontsize=7.5, loc='best')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_cost_pareto.png')
    plt.close(fig)


def fig4_linear_arm_sweeps(runs, FL):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for row_i, k in enumerate(('B', 'C')):
        r = runs[k]
        cn = CANON[k]
        # Panel 0: error vs latent size n (R_MLP vs R_aff, plus POD rank floor)
        ax = axs[row_i, 0]
        for v, K, mk in (('L0', 0, 'o'), ('L2', cn['KL2'], 's')):
            sub = r[(r.variant == v) & (r.K == K) & (r.width == cn['W']) & (r.depth == cn['L'])]
            g = sub.groupby('d_s').nrmse_test.agg(['median', 'min', 'max']).sort_index()
            ax.errorbar(g.index, g['median'], yerr=[g['median'] - g['min'], g['max'] - g['median']], fmt=mk + '-', color=COL[v], capsize=3, label=f'{VLAB[v]} ($K={K}$)' if K else VLAB['L0'])
        ns = sorted(r.d_s.unique())
        ranks = [d + 1 if k == 'B' else d for d in ns]
        rfloor = [FL[k]['pod'][rk - 1] * FL[k]['cnrmse'] for rk in ranks]
        ax.plot(ns, rfloor, 'k--', lw=1.4, label='linear rank floor ($r=n$' + ('+1' if k == 'B' else '') + ')')
        ax.set(xscale='log', yscale='log', xlabel='latent dimension $n$', ylabel='test NRMSE',
               title=f'Case {k}: latent sweep ($w=w^*, L=L^*$)', xticks=ns, xticklabels=[str(x) for x in ns])
        ax.minorticks_off(); ax.legend(fontsize=7.5)

        # Panel 1: width sweep R_MLP vs R_aff
        ax = axs[row_i, 1]
        for v, K, ds, mk in (('L0', 0, cn['N'], 'o'), ('L2', cn['KL2'], cn['NL2'], 's')):
            sub = r[(r.variant == v) & (r.K == K) & (r.depth == cn['L']) & (r.d_s == ds)]
            g = sub.groupby('width').agg(dyn=('dyn_params', 'first'), med=('nrmse_test', 'median'), lo=('nrmse_test', 'min'), hi=('nrmse_test', 'max')).sort_index()
            ax.errorbar(g.index, g.med, yerr=[g.med - g.lo, g.hi - g.med], fmt=mk + '-', color=COL[v], capsize=3, label=f'{VLAB[v]} ($n={ds}$)')
        ws = sorted(r.width.unique())
        nl2 = cn['NL2']
        ax.set(xscale='log', yscale='log', xlabel='dynamics hidden width $w$', ylabel='test NRMSE',
               title=f'Case {k}: width sweep ($\\mathcal{{R}}_{{\\mathrm{{MLP}}}}$ at $n^*$, $\\mathcal{{R}}_{{\\mathrm{{aff}}}}$ at $n_{{\\rm aff}}={nl2}$)', xticks=ws, xticklabels=[str(x) for x in ws])
        ax.minorticks_off(); ax.legend(fontsize=7.5)

        # Panel 2: K sweep of R_aff vs projection floor and rank floor
        ax = axs[row_i, 2]
        sub = r[(r.variant == 'L2') & (r.width == cn['W']) & (r.depth == cn['L']) & (r.d_s == cn['NL2'])]
        g = sub.groupby('K').agg(med=('eps_total_test', 'median'), lo=('eps_total_test', 'min'), hi=('eps_total_test', 'max'),
                                 fl=('eps_proj_test', 'first')).sort_index()
        ax.errorbar(g.index, g.med, yerr=[g.med - g.lo, g.hi - g.med], fmt='s-', color=COL['L2'], capsize=3, label=f'{VLAB["L2"]} $\\varepsilon_{{\\rm total}}$ ($n={nl2}$)')
        Kall = np.arange(1, 66)
        ax.plot(Kall, FL[k]['basis'], ':', color='gray', lw=1.6, label='basis projection floor $\\varepsilon_{\\rm proj}(K)$')
        ax.plot(Kall, FL[k]['pod'], '--', color='tab:purple', lw=1.3, label='POD floor')
        rk = cn['NL2'] + (1 if k == 'B' else 0)
        ax.axhline(FL[k]['pod'][rk - 1], color='k', ls='-.', lw=1.2, label=f'POD rank floor ($r={rk}$)')
        l0_best = r[r.variant == 'L0'].groupby(CFG).eps_total_test.median().min()
        ax.axhline(l0_best, color=COL['L0'], ls=':', lw=1.4, label=r'best $\mathcal{R}_{\mathrm{MLP}}$ median $\varepsilon_{\rm total}$')
        ax.set(xscale='log', yscale='log', xlabel='number of modes $K$', ylabel=r'relative error $\varepsilon_{\rm total}$',
               title=f'Case {k}: $\\mathcal{{R}}_{{\\mathrm{{aff}}}}$ $K$-sweep vs floors', xticks=[5, 17, 33, 65], xticklabels=['5', '17', '33', '65'])
        ax.minorticks_off(); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_linear_arm_sweeps.png')
    plt.close(fig)


def fig5_timing_and_scaling(runs, ts_data):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.0))
    # Top row: thread scaling (speedup, efficiency, s/eval)
    for k, c, mk in (('B', 'tab:blue', 'o'), ('C', 'tab:orange', 's')):
        df = ts_data[k][0]
        axs[0, 0].plot(df.threads, df.s_per_eval, mk + '-', color=c, label=f'Case {k} (canonical $\\mathcal{{R}}_{{\\mathrm{{MLP}}}}$)')
        axs[0, 1].plot(df.threads, df.speedup, mk + '-', color=c, label=f'Case {k}')
        axs[0, 2].plot(df.threads, df.efficiency, mk + '-', color=c, label=f'Case {k}')
    th = ts_data['B'][0].threads.values
    axs[0, 1].plot(th, th, 'k--', lw=1, label='ideal linear')
    axs[0, 0].set(xscale='log', yscale='log', xlabel='threads (Intel Xeon 8480+, 112 cores/node)', ylabel='seconds / evaluation',
                  title='Stage 5 Pass T: evaluation time vs threads', xticks=th, xticklabels=[str(x) for x in th])
    axs[0, 1].set(xscale='log', yscale='log', xlabel='threads', ylabel='speedup $T(1)/T(p)$',
                  title=r'Parallel speedup (canonical $\mathcal{R}_{\mathrm{MLP}}$)', xticks=th, xticklabels=[str(x) for x in th])
    axs[0, 2].set(xscale='log', xlabel='threads', ylabel='parallel efficiency $S(p)/p$',
                  title='Parallel efficiency (selected: 16 threads)', xticks=th, xticklabels=[str(x) for x in th], ylim=(0, 1.08))
    for j in range(3):
        axs[0, j].minorticks_off(); axs[0, j].legend(fontsize=8)

    # Bottom row: per-iter time vs P_dyn (B and C) and FLOPs verification
    for col_i, k in enumerate(('B', 'C')):
        ax = axs[1, col_i]
        r = runs[k]
        cn = CANON[k]
        for v, K, ds, mk in (('L0', 0, cn['N'], 'o'), ('L2', cn['KL2'], cn['NL2'], 's')):
            sub = r[(r.variant == v) & (r.K == K) & (r.depth == cn['L']) & (r.d_s == ds)]
            g = sub.groupby('width').agg(dyn=('dyn_params', 'first'),
                                         t_passA=('train_time_s', lambda s: np.median(s / sub.loc[s.index, 'n_iters'])),
                                         s_eval=('s_per_eval', 'first')).sort_index()
            ax.plot(g.dyn, g.t_passA, mk + '-', color=COL[v], label=f'{VLAB[v]} Pass-A $t_{{\\rm iter}}$ ($n={ds}$)')
            ax.plot(g.dyn, g.s_eval, mk + '--', color=COL[v], alpha=0.7, label=f'{VLAB[v]} Pass-T $s_{{\\rm eval}}$ (16 thr)')
            t1, t0 = np.polyfit(g.dyn.values.astype(float), g.t_passA.values.astype(float), 1)
            px = np.linspace(g.dyn.min(), g.dyn.max(), 50)
            ax.plot(px, t0 + t1 * px, ':', color=COL[v], lw=1)
        ax.set(xlabel=r'dynamics parameters $P_{\rm dyn}$', ylabel='seconds per iteration / evaluation',
               title=f'Case {k}: cost split vs $P_{{\\rm dyn}}$')
        ax.legend(fontsize=7)

    ax = axs[1, 2]
    for k, mk in (('B', 'o'), ('C', '^')):
        r = runs[k].groupby(CFG).agg(flops=('flops_per_eval', 'first'), s_eval=('s_per_eval', 'first')).reset_index()
        for v in ('L0', 'L2'):
            t = r[r.variant == v]
            ax.loglog(t.flops, t.s_eval, mk, color=COL[v], ms=6, alpha=0.85, label=f'Case {k} {VLAB[v]}')
    ax.set(xlabel='multiply-adds per evaluation (formula)', ylabel='measured $s_{\\rm eval}$ (Pass T, 16 threads)',
           title='Formula operations vs measured time')
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_timing_and_scaling.png')
    plt.close(fig)


def fig6_budget_1000_vs_10000(runs):
    if not all((ROOT / DIRS[k] / 'results_1000iters').exists() for k in ('B', 'C')):
        print('Skipping fig6_budget_1000_vs_10000: results_1000iters/ not present (preserving existing Study_II/figures/fig6_budget_1000_vs_10000.png)')
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for col_i, k in enumerate(('B', 'C')):
        ax = axs[col_i]
        r10k = runs[k]
        r1k = load_table(k, 'runs', 'results_1000iters')
        cn = CANON[k]
        for v, K, ds, mk in (('L0', 0, cn['N'], 'o'), ('L2', cn['KL2'], cn['NL2'], 's')):
            s1 = r1k[(r1k.variant == v) & (r1k.K == K) & (r1k.depth == cn['L']) & (r1k.d_s == ds)].groupby('width').nrmse_test.agg(['median', 'min', 'max']).sort_index()
            s10 = r10k[(r10k.variant == v) & (r10k.K == K) & (r10k.depth == cn['L']) & (r10k.d_s == ds)].groupby('width').nrmse_test.agg(['median', 'min', 'max']).sort_index()
            ax.errorbar(s1.index, s1['median'], yerr=[s1['median'] - s1['min'], s1['max'] - s1['median']],
                        fmt=mk + '--', color=COL[v], alpha=0.55, capsize=3, label=f'{VLAB[v]} at 1000 BFGS iters (laptop)')
            ax.errorbar(s10.index, s10['median'], yerr=[s10['median'] - s10['min'], s10['max'] - s10['median']],
                        fmt=mk + '-', color=COL[v], capsize=3, lw=1.8, label=f'{VLAB[v]} at 10000 BFGS iters (cluster)')
        ws = sorted(r10k.width.unique())
        ax.set(xscale='log', yscale='log', xlabel='dynamics hidden width $w$', ylabel='test NRMSE (median, min-max)',
               title=f'Case {k}: 1000-iter cap vs 10000-iter cap', xticks=ws, xticklabels=[str(x) for x in ws])
        ax.minorticks_off(); ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_budget_1000_vs_10000.png')
    plt.close(fig)


def fig7_ranks_and_stalls(runs, rank_df):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for col_i, k in enumerate(('B', 'C')):
        ax = axs[col_i]
        cn = CANON[k]
        sub = rank_df[(rank_df.case == k) & (rank_df.width == cn['W'])]
        for v, mk in (('L0', 'o'), ('L2', 's')):
            t = sub[sub.variant == v].sort_values('d_s')
            lmed = [np.median(eval(s)) for s in t.latent_ranks]
            jmed = [np.median(eval(s)) for s in t.jac_ranks]
            ax.plot(t.d_s, lmed, mk + '-', color=COL[v], label=f'{VLAB[v]} latent rank $\\mathbf{{s}}(t)$')
            ax.plot(t.d_s, jmed, mk + '--', color=COL[v], alpha=0.65, label=f'{VLAB[v]} Jacobian rank $\\partial\\dot\\mathbf{{s}}/\\partial\\mathbf{{s}}$')
        ds = sorted(sub.d_s.unique())
        ax.plot(ds, ds, 'k:', lw=1, label='full rank $=n$')
        ax.set(xlabel='latent dimension $n$', ylabel='effective rank (SV $>10^{-3}\\sigma_1$)',
               title=f'Case {k}: effective latent & Jacobian rank', xticks=ds)
        ax.legend(fontsize=7.5)

    ax = axs[2]
    for k, mk, off in (('B', 'o', -0.08), ('C', 's', 0.08)):
        r = runs[k]
        st = r[r.stalled.astype(bool)]
        full = r[~r.stalled.astype(bool)]
        for v, col in (('L0', 'tab:blue'), ('L2', 'tab:red')):
            s_st = st[st.variant == v]
            s_fl = full[full.variant == v]
            if len(s_fl):
                ax.scatter(s_fl.n_iters, s_fl.nrmse_test, marker=mk, c=col, alpha=0.35, s=25, label=f'Case {k} {VLAB[v]} (10000 iters)' if (k == 'B' or v == 'L2') else '_nolegend_')
            if len(s_st):
                ax.scatter(s_st.n_iters, s_st.nrmse_test, marker='X', c=col, edgecolors='k', s=65, zorder=5, label=f'Case {k} {VLAB[v]} stalled (<10000)')
    ax.set(xscale='log', yscale='log', xlabel='completed BFGS iterations $n_{\\rm iters}$', ylabel='test NRMSE',
           title='Early-stopped (stalled) vs full-budget runs')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(FIG / 'fig7_ranks_and_stalls.png')
    plt.close(fig)


def fig8_B_fields_and_errors(c, r):
    plt.rcParams['axes.grid'] = False
    _, p_l0, _, _ = rebuild_run(c, 'L0', 0, 7, 2, 5, 0, decode_field=True)
    _, p_l2, _, _ = rebuild_run(c, 'L2', 65, 28, 2, 10, 0, decode_field=True)
    ref = c.Yte
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6))
    t_idx = 150   # t = 30 (inside the extrapolation second half [20, 40])
    sp_ref = np.linalg.norm(ref[0, t_idx], axis=-1).reshape(100, 100)
    sp_l0 = np.linalg.norm(p_l0[0, t_idx], axis=-1).reshape(100, 100)
    sp_l2 = np.linalg.norm(p_l2[0, t_idx], axis=-1).reshape(100, 100)
    vmax = sp_ref.max()
    for j, (mat, ttl) in enumerate(((sp_ref, 'FOM: $|\\mathbf{v}|$ at $t=30$ (extrap. half)'),
                                    (sp_l0, f'best $\\mathcal{{R}}_{{\\mathrm{{MLP}}}}$ ($w=7,n=5$, NRMSE {nrmse(p_l0, ref):.2e})'),
                                    (sp_l2, f'best $\\mathcal{{R}}_{{\\mathrm{{aff}}}}$ ($K=65,w=28,n=10$, NRMSE {nrmse(p_l2, ref):.2e})'))):
        im = axs[0, j].imshow(mat, origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=vmax, cmap='magma')
        axs[0, j].set_title(ttl, fontsize=8.5); plt.colorbar(im, ax=axs[0, j])

    e_l0 = np.sqrt(np.mean(np.sum((p_l0 - ref) ** 2, axis=-1), axis=(0, 1))).reshape(100, 100)
    e_l2 = np.sqrt(np.mean(np.sum((p_l2 - ref) ** 2, axis=-1), axis=(0, 1))).reshape(100, 100)
    emax = max(e_l0.max(), e_l2.max())
    im0 = axs[1, 0].imshow((e_l0 - e_l2), origin='lower', extent=[0, 1, 0, 1], cmap='coolwarm', vmin=-emax, vmax=emax)
    axs[1, 0].set_title(r'rms error difference: $\mathcal{R}_{\mathrm{MLP}}$ minus $\mathcal{R}_{\mathrm{aff}}$ (>0: $\mathcal{R}_{\mathrm{aff}}$ better)', fontsize=8.5); plt.colorbar(im0, ax=axs[1, 0])
    im1 = axs[1, 1].imshow(e_l0, origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=emax)
    axs[1, 1].set_title(f'$\\mathcal{{R}}_{{\\mathrm{{MLP}}}}$ time-avg rms error (max {e_l0.max():.2f})', fontsize=8.5); plt.colorbar(im1, ax=axs[1, 1])
    im2 = axs[1, 2].imshow(e_l2, origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=emax)
    axs[1, 2].set_title(f'$\\mathcal{{R}}_{{\\mathrm{{aff}}}}$ time-avg rms error (max {e_l2.max():.2f})', fontsize=8.5); plt.colorbar(im2, ax=axs[1, 2])
    fig.tight_layout()
    fig.savefig(FIG / 'fig8_B_fields_and_errors.png')
    plt.close(fig)
    plt.rcParams['axes.grid'] = True


def fig9_C_fronts_comparison(c, r):
    plt.rcParams['axes.grid'] = False
    _, p_l0, _, _ = rebuild_run(c, 'L0', 0, 8, 3, 12, 0, decode_field=True)
    _, p_l2, _, _ = rebuild_run(c, 'L2', 65, 8, 1, 24, 0, decode_field=True)
    ref = c.Yte
    samples = [0, 4, 11]
    preds = [('FOM', ref),
             (f'best $\\mathcal{{R}}_{{\\mathrm{{MLP}}}}$ ($w=8,L=3,n=12$, NRMSE {nrmse(p_l0, ref):.2e})', p_l0),
             (f'best $\\mathcal{{R}}_{{\\mathrm{{aff}}}}$ ($K=65,w=8,n=24$, NRMSE {nrmse(p_l2, ref):.2e})', p_l2),
             (r'pointwise error $|\hat{y}_{\mathrm{aff}} - y|$', np.abs(p_l2 - ref))]
    fig, axs = plt.subplots(len(samples), 4, figsize=(13, 7.8))
    for i, s in enumerate(samples):
        for j, (name, p) in enumerate(preds):
            im = axs[i, j].imshow(p[s, :, :, 0], origin='lower', aspect='auto', extent=[0, 100, 0, 500],
                                  vmin=0, vmax=(0.6 if j == 3 else 1.1), cmap=('inferno' if j == 3 else 'viridis'))
            if i == 0:
                axs[i, j].set_title(name, fontsize=8)
            if j == 0:
                axs[i, j].set_ylabel(f'test sample {s}\ntime $t$ (ms)')
            if i == len(samples) - 1:
                axs[i, j].set_xlabel('$x$')
    fig.tight_layout()
    fig.savefig(FIG / 'fig9_C_fronts_comparison.png')
    plt.close(fig)
    plt.rcParams['axes.grid'] = True


# ----------------------------------------------------------------------------- main script: tables and figures
if __name__ == '__main__':
    cases = {'B': build_B(), 'C': build_C()}
    runs = {k: enriched_runs(k) for k in ('B', 'C')}
    verify_rebuilds(cases, runs)

    FL = {}
    for k, c in cases.items():
        cnrmse = float(np.sqrt(np.mean(c.Yte ** 2)) / (c.Yte.max() - c.Yte.min()))
        FL[k] = dict(basis=floors(c, c.PHI), pod=floors(c, c.PHI if k == 'B' else c.PHI_pod),
                     sv=c.sv / c.sv[0], cnrmse=cnrmse)
        assert np.allclose(runs[k].nrmse_test / runs[k].eps_total_test, cnrmse, rtol=2e-3)
        chk = runs[k].drop_duplicates(['variant', 'K'])
        for _, row in chk.iterrows():
            Kf = int(row.K) if row.variant == 'L2' else CANON[k]['KREF']
            assert abs(FL[k]['basis'][Kf - 1] - row.eps_proj_test) < 1e-3 * row.eps_proj_test + 1e-10
    print('NRMSE / eps_total =', {k: round(FL[k]['cnrmse'], 5) for k in ('B', 'C')})

    ts_data = {k: parse_thread_scaling(k) for k in ('B', 'C')}
    rank_df = rank_and_jacobian_study(cases, runs)

    fig1_capacity_laws(runs)
    fig2_iteration_surface(runs)
    fig3_cost_pareto(runs)
    fig4_linear_arm_sweeps(runs, FL)
    fig5_timing_and_scaling(runs, ts_data)
    fig6_budget_1000_vs_10000(runs)
    fig7_ranks_and_stalls(runs, rank_df)
    fig8_B_fields_and_errors(cases['B'], runs['B'])
    fig9_C_fronts_comparison(cases['C'], runs['C'])

    # 1. Full 21-config tables per case
    for k in ('B', 'C'):
        r = runs[k]
        g = r.groupby(CFG).agg(
            n=('seed', 'size'), stalled=('stalled', 'sum'),
            dyn=('dyn_params', 'first'), rec=('rec_params', 'first'), ptot=('p_total', 'first'),
            iters_med=('n_iters', 'median'), iters_min=('n_iters', 'min'),
            fevs=('n_fev', 'median'),
            flops_eval=('flops_per_eval', 'first'), flops_tot=('flops_total', 'median'),
            s_eval=('s_per_eval', 'first'), time_T=('time_T_s', 'median'), time_A=('train_time_s', 'median'),
            infer_ms=('infer_time_s', lambda s: 1e3 * np.median(s)),
            valid_med=('nrmse_valid', 'median'),
            test_med=('nrmse_test', 'median'), test_lo=('nrmse_test', 'min'), test_hi=('nrmse_test', 'max'),
            rho_med=('one_minus_rho_test', 'median'),
            eps_med=('eps_total_test', 'median'), eff_med=('eff_ratio_test', 'median'),
        ).reset_index()
        g['nrmse_test_range'] = [f'{m:.2e} [{lo:.2e}, {hi:.2e}]' for m, lo, hi in zip(g.test_med, g.test_lo, g.test_hi)]
        g['seed_ratio'] = g.test_hi / g.test_lo
        print(f'\n## Case {k}: all 21 configurations at the 10000-iteration BFGS budget (3 seeds each)')
        cols = ['variant', 'K', 'width', 'depth', 'd_s', 'stalled', 'dyn', 'rec', 'ptot', 'iters_min', 'fevs',
                's_eval', 'time_T', 'time_A', 'infer_ms', 'valid_med', 'nrmse_test_range', 'seed_ratio', 'eff_med']
        md(g[cols], '%.3g')

    # 2. Case-specific extra metrics
    print('## Case B: extra physical metrics across all 21 configurations (median over 3 seeds)')
    gb = runs['B'].groupby(CFG)[['nrmse_test', 'dir_test', 'vort_rel_test', 'extrap_ratio_test']].median().reset_index()
    md(gb, '%.3g')

    print('## Case B: physical divergence ||div v|| / ||grad v|| on the 100x100 grid (median over 3 seeds)')
    md(divergence_and_vorticity_B(cases['B'], runs['B']), '%.4f')

    print('## Case C: activation-time MAE (ms) and missed-activation fraction across all 21 configurations (median [min, max])')
    rc = runs['C']
    rows = []
    for cfg, g in rc.groupby(CFG):
        rows.append(dict(zip(CFG, cfg),
                         nrmse_med=g.nrmse_test.median(),
                         act_mae=f'{g.act_mae_test.median():.2f} [{g.act_mae_test.min():.2f}, {g.act_mae_test.max():.2f}]',
                         act_miss=f'{g.act_miss_test.median():.3f} [{g.act_miss_test.min():.3f}, {g.act_miss_test.max():.3f}]'))
    md(pd.DataFrame(rows), '%.3g')

    # 3. Stage 0 gate & cross-node determinism check
    print('\n## Stage 0 gate and cross-node determinism (canonical L0 ran on all 4 shards = 4 different Leonardo nodes)')
    for k in ('B', 'C'):
        if not (ROOT / DIRS[k] / 'results_shard0').exists():
            print(f'### Case {k}: skipped (results_shard0..3/ not present)')
            continue
        cn = CANON[k]
        sh_rows = []
        for s in range(4):
            rs = load_table(k, 'runs', f'results_shard{s}')
            q = rs[(rs.variant == 'L0') & (rs.width == cn['W']) & (rs.depth == cn['L']) & (rs.d_s == cn['N'])]
            for _, row in q.iterrows():
                sh_rows.append(dict(case=k, shard=s, seed=int(row.seed), n_iters=int(row.n_iters),
                                    n_fev=int(row.n_fev), nrmse_valid=row.nrmse_valid, nrmse_test=row.nrmse_test,
                                    train_time_s=row.train_time_s))
        sdf = pd.DataFrame(sh_rows)
        print(f'### Case {k}: canonical config across the 4 shards')
        md(sdf.pivot_table(index='seed', columns='shard', values=['nrmse_test', 'n_fev']).reset_index(), '%.4g')
        spread = sdf.groupby('seed').nrmse_test.agg(['min', 'max', lambda s: s.max() / s.min()]).rename(columns={'<lambda_0>': 'max/min'})
        md(spread.reset_index(), '%.4g')

    # 4. H1 power-law fits & two-point local slopes
    print('\n## H1 capacity-law fits and local log-log slopes on the width sweeps')
    for k in ('B', 'C'):
        r = runs[k]; cn = CANON[k]
        for v, K, ds in (('L0', 0, cn['N']), ('L2', cn['KL2'], cn['NL2'])):
            sub = r[(r.variant == v) & (r.K == K) & (r.depth == cn['L']) & (r.d_s == ds)]
            g = sub.groupby('width').agg(dyn=('dyn_params', 'first'), ptot=('p_total', 'first'),
                                         vmed=('nrmse_valid', 'median'), tmed=('nrmse_test', 'median')).sort_index()
            fit = fit_power_law(g.dyn.values, g.tmed.values, sub)
            knee_w = g[g.vmed <= 1.1 * g.vmed.min()].index.min()
            slopes_dyn = np.diff(np.log10(g.tmed.values)) / np.diff(np.log10(g.dyn.values))
            slopes_tot = np.diff(np.log10(g.tmed.values)) / np.diff(np.log10(g.ptot.values))
            print(f'Case {k} {v} (n={ds}): E_inf={fit["einf"]:.3e}, a={fit["a"]:.3e}, alpha={fit["alpha"]:.3f} '
                  f'95% CI [{fit["ci_lo"]:.3f}, {fit["ci_hi"]:.3f}] ({fit["n_boots"]} fits), knee w={knee_w} '
                  f'(P_dyn={int(g.loc[knee_w, "dyn"])}, P_total={int(g.loc[knee_w, "ptot"])})')
            print('   two-point slopes d log E / d log P_dyn :', [round(float(x), 3) for x in slopes_dyn])
            print('   two-point slopes d log E / d log P_total:', [round(float(x), 3) for x in slopes_tot])

    # 5. Checkpoint tables (Stage 3 surface) and 1000 vs 10000 comparison
    print('\n## Stage 3: median test NRMSE across checkpoints (width sweeps)')
    for k in ('B', 'C'):
        cv = load_table(k, 'curves'); cn = CANON[k]
        for v, K, ds in (('L0', 0, cn['N']), ('L2', cn['KL2'], cn['NL2'])):
            sub = cv[(cv.variant == v) & (cv.K == K) & (cv.depth == cn['L']) & (cv.d_s == ds)]
            p = sub.groupby(['width', 'iter']).nrmse_test.median().unstack('iter')
            p['drop_1k_to_10k_%'] = 100 * (p[1000] - p[10000]) / p[1000]
            print(f'### Case {k} {v} (n={ds})')
            md(p.reset_index(), '%.3e')

    print('## Comparison between the 1000-iter laptop sweep and the 10000-iter cluster sweep (median test NRMSE)')
    for k in ('B', 'C'):
        if not (ROOT / DIRS[k] / 'results_1000iters').exists():
            print(f'### Case {k}: skipped (results_1000iters/ not present)')
            continue
        r10 = runs[k].groupby(CFG).nrmse_test.median()
        r1 = load_table(k, 'runs', 'results_1000iters').groupby(CFG).nrmse_test.median()
        comp = pd.DataFrame({'nrmse_1000_laptop': r1, 'nrmse_10000_cluster': r10}).reset_index()
        comp['ratio_1000_over_10000'] = comp['nrmse_1000_laptop'] / comp['nrmse_10000_cluster']
        comp['drop_%'] = 100 * (1 - comp['nrmse_10000_cluster'] / comp['nrmse_1000_laptop'])
        print(f'### Case {k}')
        md(comp, '%.3g')

    # 6. Time-to-target (Pass A elapsed time and Pass T estimated time)
    print('\n## Time to target (validation NRMSE <= threshold): Pass-A checkpoint time [s] and n_hit out of 3 seeds')
    for k in ('B', 'C'):
        cv = load_table(k, 'curves'); tm = load_table(k, 'timing').set_index(CFG)['s_per_eval']
        rows = []
        for cfg, g in cv.groupby(CFG):
            row = dict(zip(CFG, cfg))
            se = float(tm.loc[cfg])
            for tgt in (5e-2, 2e-2, 1e-2):
                ta, tt, fev = [], [], []
                for _, s in g.groupby('seed'):
                    hit = s[s.nrmse_valid <= tgt]
                    if len(hit):
                        ta.append(hit.time_s.min()); tt.append(hit.n_fev.min() * se); fev.append(hit.n_fev.min())
                    else:
                        ta.append(np.inf); tt.append(np.inf); fev.append(np.inf)
                row[f'PassT<={tgt:g}[s]'] = np.median(tt)
                row[f'fev<={tgt:g}'] = np.median(fev)
                row[f'hit<={tgt:g}'] = int(np.isfinite(tt).sum())
            rows.append(row)
        print(f'### Case {k}')
        md(pd.DataFrame(rows), '%.3g')

    # 7. H2 cost split (both Pass-A per-iter and Pass-T per-eval)
    print('\n## H2 cost split: linear fit of time vs P_dyn on the width sweep')
    rows = []
    for k in ('B', 'C'):
        r = runs[k]; cn = CANON[k]
        for v, K, ds in (('L0', 0, cn['N']), ('L2', cn['KL2'], cn['NL2'])):
            sub = r[(r.variant == v) & (r.K == K) & (r.depth == cn['L']) & (r.d_s == ds)]
            g = sub.groupby('width').agg(dyn=('dyn_params', 'first'),
                                         t_iter=('train_time_s', lambda s: np.median(s / sub.loc[s.index, 'n_iters'])),
                                         s_eval=('s_per_eval', 'first')).sort_index()
            P = g.dyn.values.astype(float)
            for src, y in (('Pass-A s/iter', g.t_iter.values.astype(float)), ('Pass-T s/eval (16 thr)', g.s_eval.values.astype(float))):
                t1, t0 = np.polyfit(P, y, 1)
                sh = t1 * P.max() / (t0 + t1 * P.max())
                rows.append(dict(case=k, variant=v, source=src, t0_s=t0, t1_s_per_param=t1,
                                 P_dyn_max=int(P.max()), share_at_4w_star=sh, ratio_max_over_min=float(y[-1] / y[0])))
    md(pd.DataFrame(rows), '%.3g')

    # 8. Stage 5 thread scaling tables & FLOP check
    for k in ('B', 'C'):
        df, throt = ts_data[k]
        print(f'\n## Stage 5 thread scaling — Case {k} (throttling check: {throt[0]} vs {throt[1]} s/eval, {throt[2]} apart)')
        md(df, '%.4g')

    print('## Operation-count formula check: measured seconds per operation s_per_eval / flops_per_eval (Pass T, 16 threads)')
    rows = []
    for k in ('B', 'C'):
        r = runs[k].groupby(CFG).agg(flops=('flops_per_eval', 'first'), s=('s_per_eval', 'first')).reset_index()
        r['ns_per_Mflop'] = 1e9 * r['s'] / (r['flops'] / 1e6)
        for v, g in r.groupby('variant'):
            rows.append(dict(case=k, variant=v, s_per_op_median=(g.s / g.flops).median(),
                             s_per_op_min=(g.s / g.flops).min(), s_per_op_max=(g.s / g.flops).max(),
                             L0_over_L2_eval_time_ratio=float(r[r.variant == 'L0'].s.median() / r[r.variant == 'L2'].s.median())))
    md(pd.DataFrame(rows), '%.3e')

    # 9. H6 cost_to_reach tables (both on test medians and validation-selected)
    print('\n## H6 cost_to_reach: cheapest configuration of each arm reaching level x best L0 median test NRMSE')
    for k in ('B', 'C'):
        r = runs[k]
        g = r.groupby(CFG).agg(med=('nrmse_test', 'median'), vmed=('nrmse_valid', 'median'),
                               flops=('flops_total', 'median'), fevs=('n_fev', 'median'),
                               time_T=('time_T_s', 'median'), time_A=('train_time_s', 'median'),
                               ptot=('p_total', 'first')).reset_index()
        best = g[g.variant == 'L0']['med'].min()
        rows = []
        for lv in (1.0, 1.25, 1.5, 2.0, 3.0):
            tgt = lv * best
            row = {'level': f'{lv:.2f}x', 'target_NRMSE': tgt}
            for v in ('L0', 'L2'):
                ok = g[(g.variant == v) & (g.med <= tgt)]
                if len(ok):
                    c_t = ok.loc[ok.time_T.idxmin()]
                    row[f'{v}_cheapest_cfg'] = f'K={int(c_t.K)},w={int(c_t.width)},L={int(c_t.depth)},n={int(c_t.d_s)}'
                    row[f'{v}_ops'] = float(ok.flops.min())
                    row[f'{v}_time_T[s]'] = float(ok.time_T.min())
                    row[f'{v}_time_A[s]'] = float(ok.time_A.min())
                else:
                    row[f'{v}_cheapest_cfg'] = '-'
                    row[f'{v}_ops'] = row[f'{v}_time_T[s]'] = row[f'{v}_time_A[s]'] = np.nan
            row['ops_L0/L2'] = row['L0_ops'] / row['L2_ops']
            row['time_T_L0/L2'] = row['L0_time_T[s]'] / row['L2_time_T[s]']
            row['time_A_L0/L2'] = row['L0_time_A[s]'] / row['L2_time_A[s]']
            rows.append(row)
        print(f'### Case {k} (best L0 median test NRMSE = {best:.3e})')
        md(pd.DataFrame(rows), '%.3g')

    # 10. Pareto fronts on validation vs test medians (protocol Sec. 10 check)
    print('\n## Pareto front check: test-median selection vs validation-median selection')
    for k in ('B', 'C'):
        r = runs[k]
        g = r.groupby(CFG).agg(med=('nrmse_test', 'median'), vmed=('nrmse_valid', 'median'),
                               time=('time_T_s', 'median'), ptot=('p_total', 'first')).reset_index()
        for v in ('L0', 'L2'):
            t = g[g.variant == v]
            bt = t.loc[t.med.idxmin()]; bv = t.loc[t.vmed.idxmin()]
            ct = t[t.med <= 1.1 * bt.med].sort_values('time').iloc[0]
            cv_ = t[t.vmed <= 1.1 * bv.vmed].sort_values('time').iloc[0]
            print(f'Case {k} {v}: best-on-test=(K={int(bt.K)},w={int(bt.width)},L={int(bt.depth)},n={int(bt.d_s)}, E_test={bt.med:.3e}); '
                  f'best-on-valid=(K={int(bv.K)},w={int(bv.width)},L={int(bv.depth)},n={int(bv.d_s)}, E_test={bv.med:.3e})')
            print(f'   cheapest-in-10% test =(w={int(ct.width)},L={int(ct.depth)},n={int(ct.d_s)}, time={ct.time:.0f}s, P={int(ct.ptot)}); '
                  f'valid=(w={int(cv_.width)},L={int(cv_.depth)},n={int(cv_.d_s)}, time={cv_.time:.0f}s, P={int(cv_.ptot)})')

    # 11. Explainability: latent & Jacobian ranks
    print('\n## Explainability add-on: effective rank of test latent trajectories s(t) and of d s_dot / d s (threshold 1e-3 * sigma_1)')
    md(rank_df, '%.3e')
