#!/usr/bin/env python3
"""RQ2 report: tables and figures computed from the stored results. Nothing is trained and no notebook is run.

    python RQ2/report_analysis.py > RQ2/figures/tables.md        (about 4 minutes: loading the data and decoding the L0 runs on the test sets)

Reads  RQ2/<case>/results/{runs,predictions,modes}.csv, weights/, latents/  and  data/ ;
writes RQ2/figures/*.png and prints (stdout) the tables quoted in RQ2/README.md.
The L0 decoders (and the R1 decoders on the frozen L0 latents) are rebuilt in numpy from the saved weights:
every rebuild is checked against the NRMSE stored in runs.csv.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
DIRS = {'A': 'ADR-1b', 'B': '2D-Stokes', 'C': 'AP1D'}
KEYS = ['variant', 'regime', 'd_s', 'K', 'seed']
CONFIG = ['variant', 'regime', 'd_s', 'K']
D_REF = {'A': 3, 'B': 10, 'C': 12}            # latent size used in the K-sweep figure of each case
COL = {'L0': 'k', 'L1': 'tab:green', 'L2': 'tab:blue', 'L3': 'tab:red'}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110, 'savefig.dpi': 130})


def md(df, fmt='%.3g'):
    """print a dataframe as a markdown table"""
    cols = list(df.columns)
    print('| ' + ' | '.join(map(str, cols)) + ' |')
    print('|' + '---|' * len(cols))
    for _, r in df.iterrows():
        print('| ' + ' | '.join(fmt % v if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + ' |')
    print()


# ----------------------------------------------------------------------------- stored results
def runs_of(case, name='runs'):
    extra = {'runs': [], 'predictions': ['sample'], 'modes': ['mode']}[name]
    df = pd.read_csv(ROOT / DIRS[case] / 'results' / f'{name}.csv')
    return df.drop_duplicates(subset=KEYS + extra, keep='last')


def best_configs(r, select='nrmse_valid'):
    """(d_s, K) with the lowest median validation NRMSE over the seeds, for every procedure (variant, regime): the rule of the notebooks"""
    med = r.groupby(CONFIG)[select].median().reset_index()
    return med.loc[med.groupby(['variant', 'regime'])[select].idxmin(), CONFIG]


def best_seed(r, variant, regime, d_s, K):
    q = r[(r.variant == variant) & (r.regime == regime) & (r.d_s == d_s) & (r.K == K)]
    row = q.loc[q.nrmse_valid.idxmin()]
    return int(row.seed), int(row.width)


# ----------------------------------------------------------------------------- the three cases, rebuilt in numpy (test set, bases, mean field)
def mlp_split(w, sizes):
    """flat weight vector -> [(kernel, bias), ...]  (C-order, kernel then bias, layer by layer: VariablesStitcher of the notebooks)"""
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


class Case:
    """test fields Y [S,T,P,dy] (physical), mean field ybar, orthonormal basis PHI [D,65] for the quadrature weights wv [D]
    (D = P*dy: the components of a vector field are interleaved), inputs of the dynamics, sizes of the L0 networks."""

    def flat(self, y):                     # [S,T,P,dy] -> [S*T, D]
        return y.reshape(y.shape[0] * y.shape[1], -1)

    def unflat(self, f, S, T):
        return f.reshape(S, T, self.P, self.dy)

    def coeffs(self, y):                   # coefficients of (y - ybar) on the basis: [S*T, 65]
        return ((self.flat(y) - self.ybar_flat) * self.wv) @ self.PHI

    def synth(self, coef, S, T):           # ybar + sum_k coef_k phi_k  (physical units): [S,T,P,dy]
        return self.unflat(coef @ self.PHI[:, :coef.shape[1]].T + self.ybar_flat, S, T)


def build_A():
    c = Case(); c.name, c.dy, c.coef_scale = 'A', 1, 1.0
    d = np.load(DATA / 'ADR/data_1b.npy', allow_pickle=True)[()]
    t, c.x = d['t'], d['x']
    tt = np.arange(t[0], t[-1] + 0.05 * 1e-10, step=0.05)              # process_dataset: linear resampling on the dt = 0.05 grid (201 instants)
    res = lambda a: interpolate.interp1d(t, a, axis=1)(tt)
    tr, te = res(d['output'][0:100])[..., None], res(d['output'][200:300])[..., None]
    c.P, c.h, c.Yte, c.Ytr = len(c.x), c.x[1] - c.x[0], te, tr
    c.ybar = tr.mean(axis=(0, 1))[:, 0]
    c.ybar_flat = c.ybar
    c.wv = np.full(c.P, c.h)
    cols, k = [np.full_like(c.x, 1 / np.sqrt(2))], 1                   # Fourier: {1/sqrt2, cos pi x, sin pi x, cos 2 pi x, ...}
    while len(cols) < 65:
        cols.append(np.cos(k * np.pi * c.x)); cols.append(np.sin(k * np.pi * c.x)); k += 1
    c.PHI = np.stack(cols[:65], axis=1)
    c.Ute = np.stack([(2 * res(d['forcing'][200:300])[..., 0] - 0.8) / 0.8, res(d['forcing'][200:300])[..., 1] / 4], axis=-1)
    c.xn = c.x[:, None]
    c.dyn_sizes = lambda ds: [ds + 2, 10, 10, ds]
    c.rec_sizes = lambda ds: [ds + 1, 7, 7, 1]
    c.denorm = lambda v: v
    c.train_flat = c.flat(tr) - c.ybar_flat
    c.train_w = c.wv
    return c


def build_B():
    c = Case(); c.name, c.dy, c.coef_scale = 'B', 2, 1.0
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
    Z = (Ftr[:, ::5] - c.ybar).reshape(-1, 2 * c.P)                   # vector POD of the training snapshots (every 5th step), snapshot method
    lam, V = np.linalg.eigh(Z @ Z.T)
    o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 0), V[:, o]
    c.sv = np.sqrt(lam)
    c.PHI = (Z.T @ V[:, :65]) / np.sqrt(lam[:65])[None, :] / np.sqrt(c.h)
    c.Ute = (dte['velocity_top'] / 20)[..., None]
    c.dyn_sizes = lambda ds: [ds + 1, 7, 7, ds]
    c.rec_sizes = lambda ds: [ds + 3, 24, 24, 24, 24, 2]
    c.denorm = lambda v: 20 * v
    c.train_flat = None
    return c


def build_C():
    c = Case(); c.name, c.dy, c.coef_scale = 'C', 1, 0.6                # network coefficients are in normalised units (range 1.2)
    D = DATA / 'AP1D'
    Y = np.stack([np.load(D / f'APsolution{i}.npy')[:501, ::8] for i in range(400)])
    U = np.stack([pd.read_csv(D / f'APsetting{i}.csv')[['u(t)_1', 'u(t)_2']].values[:501] for i in range(400)])
    c.P, c.h, c.x = 101, 1.0, np.linspace(0, 100, 101)
    c.Ytr, c.Yte = Y[:100, ..., None], Y[200:400, ..., None]
    c.ybar = Y[:100].mean(axis=(0, 1))
    c.ybar_flat = c.ybar
    xn = c.x / 50 - 1
    c.xn = xn[:, None]
    w = np.full(c.P, 2 / (c.P - 1)); w[0] = w[-1] = 1 / (c.P - 1)      # trapezoidal weights of the discrete L2 product
    c.wv = w
    V = [np.ones_like(xn), xn.copy()]                                  # Legendre polynomials (three-term recurrence), then weighted QR
    for n in range(1, 64):
        V.append(((2 * n + 1) * xn * V[n] - n * V[n - 1]) / (n + 1))
    Q, _ = np.linalg.qr(np.sqrt(w)[:, None] * np.stack(V[:65], axis=1))
    c.PHI = Q / np.sqrt(w)[:, None]
    c.Ute = U[200:400] / 2.5
    c.dyn_sizes = lambda ds: [ds + 2, 8, ds]
    c.rec_sizes = lambda ds: [ds + 1, 17, 17, 17, 17, 17, 1]
    c.denorm = lambda v: 0.6 * (1 + v)
    c.train_flat = c.flat(c.Ytr) - c.ybar_flat
    c.train_w = w
    return c


BUILD = {'A': build_A, 'B': build_B, 'C': build_C}


def nrmse(app, ref):
    return np.sqrt(np.mean((app - ref) ** 2)) / (ref.max() - ref.min())


# ----------------------------------------------------------------------------- stored runs, rebuilt
def l0_run(c, ds, seed):
    """test latents z (the decoder input: s, plus the lid velocity u in case B) and the L0 decoder layers of a stored run"""
    base = ROOT / DIRS[c.name] / 'results'
    w = np.load(base / 'weights' / f'L0_R2_ds{ds}_K0_s{seed}.npy')
    n_dyn = mlp_split(w, c.dyn_sizes(ds))[1]
    lrec, n_rec = mlp_split(w[n_dyn:], c.rec_sizes(ds))
    assert n_dyn + n_rec == len(w)
    s = np.load(base / 'latents' / f'L0_ds{ds}_seed{seed}.npz')['test']
    return (np.concatenate([s, c.Ute], axis=-1) if c.name == 'B' else s), lrec


def l0_decode(c, z, lrec):
    S, T, nz = z.shape
    out = []
    for i in range(S):                                                 # one sample at a time (memory)
        zb = np.broadcast_to(z[i][:, None, :], (T, c.P, nz))
        xb = np.broadcast_to(c.xn[None], (T, c.P, c.xn.shape[1]))
        o = mlp_apply(lrec, np.concatenate([zb, xb], axis=-1))
        if c.name == 'B':
            o = (o ** 3 + 0.05 * o) / 1.05                             # tail compression of the original script
        out.append(c.denorm(o))
    return np.stack(out)


def r1_decode(c, variant, ds, K, seed, width):
    """prediction of a stored R1 run (L1/L2/L3 reading the frozen L0 latents)"""
    base = ROOT / DIRS[c.name] / 'results'
    s = np.load(base / 'latents' / f'L0_ds{ds}_seed{seed}.npz')['test']
    z = np.concatenate([s, c.Ute], axis=-1) if c.name == 'B' else s
    if variant == 'L1':
        coef = s[..., :ds]
    else:
        w = np.load(base / 'weights' / f'{variant}_R1_ds{ds}_K{K}_s{seed}.npy')
        layers, n = mlp_split(w, [z.shape[-1], K] if variant == 'L2' else [z.shape[-1], width, width, K])
        assert n == len(w)
        coef = mlp_apply(layers, z)
    S, T = s.shape[:2]
    return c.synth(c.coef_scale * coef.reshape(S * T, -1), S, T)


def check_rebuild(c, r):
    """the numpy rebuild must reproduce the NRMSE stored by the notebooks (L0 and R1 runs)"""
    for ds in sorted(r[r.variant == 'L0'].d_s.unique())[:2]:
        z, lrec = l0_run(c, ds, 0)
        stored = r[(r.variant == 'L0') & (r.d_s == ds) & (r.seed == 0)].nrmse_test.iloc[0]
        assert abs(nrmse(l0_decode(c, z, lrec), c.Yte) - stored) < 1e-9 * max(1, stored), f'L0 rebuild mismatch ({c.name}, d_s={ds})'
    q = r[(r.regime == 'R1') & (r.variant == 'L3')].iloc[3]
    pred = r1_decode(c, 'L3', int(q.d_s), int(q.K), int(q.seed), int(q.width))
    assert abs(nrmse(pred, c.Yte) - q.nrmse_test) < 1e-9, f'R1 rebuild mismatch ({c.name})'


# ----------------------------------------------------------------------------- projection floors and singular values
def floors(c, PHI, kmax=65):
    """eps_proj(K) = ||(y - ybar) - P_K (y - ybar)|| / ||y|| for K = 1..kmax on the test set (Euclidean norms, as in runs.csv)"""
    F = c.flat(c.Yte) - c.ybar_flat
    ref = np.linalg.norm(c.flat(c.Yte))
    coef = (F * c.wv) @ PHI[:, :kmax]
    out = []
    for k in range(kmax):
        F = F - np.outer(coef[:, k], PHI[:, k])
        out.append(np.linalg.norm(F) / ref)
    return np.array(out)


def pod_basis(c):
    """optimal linear basis (POD) of the training fluctuations, orthonormal for the weights wv; singular values sigma_k"""
    if c.name == 'B':
        return c.PHI, c.sv
    sw = np.sqrt(c.train_w)
    _, s, Vt = np.linalg.svd(c.train_flat * sw, full_matrices=False)
    return (Vt.T / sw[:, None])[:, :65], s


def study_floors(cases, runs):
    out = {}
    for k, c in cases.items():
        pod, sv = pod_basis(c)
        out[k] = dict(basis=floors(c, c.PHI), pod=floors(c, pod), sv=sv / sv[0])
        r = runs[k]
        chk = r[r.eps_proj_test.notna()].drop_duplicates(['K'])
        for _, row in chk.iterrows():
            assert abs(out[k]['basis'][int(row.K) - 1] - row.eps_proj_test) < 1e-3 * row.eps_proj_test + 1e-10, f'floor mismatch {k} K={int(row.K)}'
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    col = {'A': 'tab:green', 'B': 'tab:blue', 'C': 'tab:red'}
    lab = {'A': 'A (Fourier)', 'B': 'B (vector POD)', 'C': 'C (Legendre)'}
    K = np.arange(1, 66)
    for k in 'ABC':
        axs[0].semilogy(np.arange(1, len(out[k]['sv'][:65]) + 1), np.maximum(out[k]['sv'][:65], 1e-13), 'o-', ms=2.5, c=col[k], label=lab[k])
        axs[1].semilogy(K, np.maximum(out[k]['basis'], 1e-13), '-', c=col[k], label=lab[k])
        axs[1].semilogy(K, np.maximum(out[k]['pod'], 1e-13), '--', c=col[k])
    axs[0].set(title='singular values of the training fluctuations', xlabel='k', ylabel=r'$\sigma_k/\sigma_1$', ylim=(1e-6, 2))
    axs[1].set(title='projection floor on the test set (solid: protocol basis, dashed: POD)', xlabel='K', ylabel=r'$\varepsilon_{\rm proj}(K)$', ylim=(1e-5, 1.2))
    axs[2].semilogy(np.arange(1, 13), np.maximum(out['A']['basis'][:12], 1e-13), 'o-', c=col['A'], label='A Fourier')
    axs[2].semilogy(np.arange(1, 13), np.maximum(out['A']['pod'][:12], 1e-13), 'o--', c=col['A'], label='A POD')
    axs[2].set(title='case A: the floor collapses at K = 3 (Fourier) / 2 (POD)', xlabel='K', ylabel=r'$\varepsilon_{\rm proj}(K)$', ylim=(1e-13, 2))
    for a in axs[:2]:
        a.legend(fontsize=8)
    axs[2].legend(fontsize=8)
    for a in axs:
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_spectra_floors.png')
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- spectral analysis of the trained L0 decoder (protocol Sec. 9)
KS = (1, 2, 3, 5, 9, 17, 33, 65)


def vorticity(c, vel):
    """dvy/dx - dvx/dy by central differences on the 100 x 100 grid (case B), vel [..., P, 2] -> [..., P]"""
    V = vel.reshape(*vel.shape[:-2], 100, 100, 2)
    h = 1 / 99
    return (np.gradient(V[..., 1], h, axis=-1) - np.gradient(V[..., 0], h, axis=-2)).reshape(*vel.shape[:-2], c.P)


def l0_analysis(c, ds_list):
    """For every stored L0 run: R2_lin(K) (best affine fit coefficients ~ W z + b), K_99 of the prediction and of the FOM,
    and the NRMSE of the L0 output projected on its first K modes."""
    rows, k99 = [], []
    cref = c.coeffs(c.Yte)
    vref = vorticity(c, c.Yte) if c.name == 'B' else None
    for ds in ds_list:
        for seed in range(3):
            z, lrec = l0_run(c, ds, seed)
            S, T, _ = z.shape
            pred = l0_decode(c, z, lrec)
            C = c.coeffs(pred)
            Z1 = np.concatenate([z.reshape(S * T, -1), np.ones((S * T, 1))], axis=1)
            res = C - Z1 @ np.linalg.lstsq(Z1, C, rcond=None)[0]
            for K in KS:
                row = dict(d_s=ds, seed=seed, K=K, R2=1 - (res[:, :K] ** 2).sum() / ((C[:, :K] - C[:, :K].mean(0)) ** 2).sum(),
                           nrmse_proj=nrmse(c.synth(C[:, :K], S, T), c.Yte))
                if c.name == 'B' and K in (3, 9, 17, 65):
                    row['vort_proj'] = np.linalg.norm(vorticity(c, c.synth(C[:, :K], S, T)) - vref) / np.linalg.norm(vref)
                rows.append(row)
            if c.name == 'B':
                rows.append(dict(d_s=ds, seed=seed, K=0, vort_proj=np.linalg.norm(vorticity(c, pred) - vref) / np.linalg.norm(vref)))
            for name, M in (('L0', C), ('FOM', cref)):
                e = np.cumsum(M ** 2, axis=1)
                ok = e[:, -1] > 1e-6 * np.median(e[:, -1])
                kk = (e[ok] >= 0.99 * e[ok, -1:]).argmax(axis=1) + 1
                k99.append(dict(d_s=ds, seed=seed, name=name, med=np.median(kk), q1=np.percentile(kk, 25), q3=np.percentile(kk, 75)))
        print(f'  case {c.name}: L0 analysis d_s = {ds} done', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(k99)


# ----------------------------------------------------------------------------- figures from the sweeps
def band(df, col, x='K'):
    g = df.groupby(x)[col]
    return g.median(), g.min(), g.max()


def fig_error_vs_K(runs, FL, CN, LIN):
    """The main RQ2 figure: test NRMSE of L2 and L3 against K (median and range over the 3 seeds), with the projection floors and the L0 band."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, k in zip(axs, 'ABC'):
        r, d = runs[k], D_REF[k]
        for v in ('L2', 'L3'):
            for reg, ls, mk in (('R2', '-', 'o'), ('R1', '--', 's')):
                q = r[(r.variant == v) & (r.regime == reg) & (r.d_s == d)]
                m, lo, hi = band(q, 'nrmse_test')
                ax.plot(m.index, m, ls=ls, marker=mk, ms=4, c=COL[v], label=f'{v}-{reg}')
                ax.fill_between(m.index, lo, hi, color=COL[v], alpha=0.12)
        for reg, mk in (('R2', 'o'), ('R1', 's')):
            q = r[(r.variant == 'L1') & (r.regime == reg) & (r.d_s == d)].nrmse_test
            ax.errorbar([d], [q.median()], yerr=[[q.median() - q.min()], [q.max() - q.median()]], fmt=mk, c=COL['L1'], mfc='none' if reg == 'R1' else COL['L1'],
                        ms=7, capsize=3, label=f'L1-{reg} ($K=d_s$)')
        q = r[(r.variant == 'L0') & (r.d_s == d)].nrmse_test
        ax.axhspan(q.min(), q.max(), color='k', alpha=0.15)
        ax.axhline(q.median(), c='k', lw=1.2, label='L0 (median, range of 3 seeds)')
        lin = LIN[k][0]
        lin = lin[lin.d_s == d].groupby('K').nrmse_proj.median()
        ax.plot(lin.index, lin, 'k-.', lw=1, label='L0 output projected on $K$ modes')
        Kall = np.arange(1, 66)
        ax.plot(Kall, np.maximum(FL[k]['basis'], 1e-13) * CN[k], ':', c='gray', lw=1.8, label='floor, protocol basis')
        ax.plot(Kall, np.maximum(FL[k]['pod'], 1e-13) * CN[k], ':', c='tab:purple', lw=1.8, label='floor, POD')
        ax.set(xscale='log', yscale='log', xlabel='$K$ (number of modes)', ylabel='test NRMSE', title=f'case {k}, $d_s={d}$')
        ax.set_xticks([1, 3, 5, 9, 17, 33, 65]); ax.set_xticklabels([1, 3, 5, 9, 17, 33, 65]); ax.minorticks_off()
        ax.set_ylim({'A': (3e-3, 0.3), 'B': (2e-4, 0.1), 'C': (2e-3, 0.6)}[k])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / 'fig2_error_vs_K.png')
    plt.close(fig)


def fig_rank_bound(runs, FL, CN):
    """eps_total against the latent size d_s: L0, L1, and L2/L3 (K chosen on validation), against the best possible linear decoder of rank d_s."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    rows = []
    for ax, k in zip(axs, 'ABC'):
        r = runs[k]
        ds = sorted(r.d_s.unique())
        for v, reg, lab in (('L0', 'R2', 'L0'), ('L1', 'R2', 'L1-R2'), ('L2', 'R2', 'L2-R2'), ('L3', 'R2', 'L3-R2')):
            y = []
            for d in ds:
                q = r[(r.variant == v) & (r.regime == reg) & (r.d_s == d)]
                med = q.groupby('K').nrmse_valid.median()
                q = q[q.K == med.idxmin()]
                y.append(q.eps_total_test.median())
                rows.append(dict(case=k, proc=lab, d_s=d, K=int(q.K.iloc[0]), eps_total=y[-1], nrmse=y[-1] * CN[k]))
            ax.plot(ds, y, 'o-', c=COL[v], label=lab)
        rank = [d + 1 if k == 'B' else d for d in ds]
        floor = [FL[k]['pod'][q - 1] for q in rank]
        ax.plot(ds, np.maximum(floor, 1e-3), 'k--', lw=1.6, label='best linear decoder of rank $d_s$' + (' (+1 for $u$)' if k == 'B' else ''))
        for d, f in zip(ds, floor):
            rows.append(dict(case=k, proc='rank floor', d_s=d, K=0, eps_total=f, nrmse=f * CN[k]))
        ax.set(yscale='log', xlabel='latent size $d_s$', ylabel=r'test $\varepsilon_{\rm total}$', title=f'case {k}', xticks=ds)
        ax.set_ylim({'A': (1e-3, 1.5), 'B': (5e-3, 1.0), 'C': (3e-2, 1.2)}[k])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / 'fig3_rank_bound.png')
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_L0_linearity(LIN):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, k in zip(axs, 'ABC'):
        df = LIN[k][0]
        df = df[df.K > 0]
        for d, g in df.groupby('d_s'):
            m = g.groupby('K').R2.median()
            ax.plot(m.index, m, 'o-', ms=3, label=f'$d_s={d}$')
        ax.set(xscale='log', ylim=(0, 1.02), xlabel='$K$ (modes included)', ylabel=r'$R^2_{\rm lin}(K)$', title=f'case {k}')
        ax.set_xticks([1, 3, 5, 9, 17, 33, 65]); ax.set_xticklabels([1, 3, 5, 9, 17, 33, 65]); ax.minorticks_off()
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_L0_linearity.png')
    plt.close(fig)


def fig_pareto(runs):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k in zip(axs, 'ABC'):
        g = runs[k].groupby(CONFIG).agg(p=('rec_params', 'median'), e=('nrmse_test', 'median')).reset_index()
        for (v, reg), q in g.groupby(['variant', 'regime']):
            ax.scatter(np.maximum(q.p, 1), q.e, s=22, c=COL[v], marker='o' if reg == 'R2' else 'x', alpha=0.8, label=f'{v}-{reg}')
        ax.set(xscale='log', yscale='log', xlabel='parameters of the reconstructor (L1: none, drawn at 1)', ylabel='test NRMSE (median of 3 seeds)', title=f'case {k}')
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_pareto.png')
    plt.close(fig)


def fig_modes(cases, runs):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    sty = {('L0', 'R2'): ('k', '-'), ('L1', 'R2'): ('tab:green', '-'), ('L2', 'R2'): ('tab:blue', '-'), ('L3', 'R2'): ('tab:red', '-'),
           ('L2', 'R1'): ('tab:blue', '--'), ('L3', 'R1'): ('tab:red', '--')}
    for ax, k in zip(axs, 'ABC'):
        r = runs[k]
        best = best_configs(r)
        m = runs_of(k, 'modes').merge(best, on=CONFIG)
        cref = cases[k].coeffs(cases[k].Yte)
        tot = (cref ** 2).sum()
        kmax = {'A': 8, 'B': 17, 'C': 33}[k]
        en = ((cref ** 2).sum(0) / tot)[:kmax]
        ax.fill_between(np.arange(1, 66)[:kmax], 1e-16, np.maximum(en, 1e-16), color='gray', alpha=0.25, label='FOM energy in mode $k$')
        ax.semilogy(np.arange(1, 66)[:kmax], np.maximum(en, 1e-16), c='gray', lw=1)
        for (v, reg, d, K), g in m.groupby(CONFIG):
            if (v, reg) not in sty:
                continue
            e = g.assign(e2=g.err_abs ** 2).groupby('mode').e2.mean().sort_index()
            col, ls = sty[(v, reg)]
            ax.semilogy(e.index[:kmax], np.maximum(e.values[:kmax] / tot, 1e-16), ls, c=col, marker='.', label=f'{v}-{reg} ($d_s$={d}' + (f', $K$={K})' if K else ')'))
        ax.set(xlabel='mode $k$', ylabel='squared error in mode $k$ / FOM energy', title=f'case {k}: where the error lives')
        ax.set_ylim({'A': (1e-12, 1.0), 'B': (1e-9, 1.0), 'C': (1e-7, 1.0)}[k])
        ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_modes.png')
    plt.close(fig)


def fig_C_decoders(c, r):
    """Case C: the same frozen L0 latents (d_s = 5) read by four decoders, on three test samples (rows: samples; columns: FOM, L0, L3-R1, L2-R1)."""
    d = 5
    seed, w3 = best_seed(r, 'L3', 'R1', d, 17)
    z, lrec = l0_run(c, d, seed)
    preds = {'FOM': c.Yte, 'L0': l0_decode(c, z, lrec),
             'L3-R1, $K$=17': r1_decode(c, 'L3', d, 17, seed, w3), 'L2-R1, $K$=33': r1_decode(c, 'L2', d, 33, seed, 0)}
    samples = [0, 4, 11]
    fig, axs = plt.subplots(len(samples), 4, figsize=(11, 8))
    for i, s in enumerate(samples):
        for j, (name, p) in enumerate(preds.items()):
            axs[i, j].imshow(p[s, :, :, 0], origin='lower', aspect='auto', extent=[0, 100, 0, 500], vmin=0, vmax=1.1, cmap='viridis')
            if i == 0:
                axs[i, j].set_title(name + (f' (NRMSE {nrmse(p, c.Yte):.1e})' if name != 'FOM' else ''), fontsize=9)
            axs[i, j].grid(False)
            if j == 0:
                axs[i, j].set_ylabel(f'test sample {s}\ntime $t$ (ms)')
            if i == len(samples) - 1:
                axs[i, j].set_xlabel('$x$')
    fig.tight_layout()
    fig.savefig(FIG / 'fig7_C_decoders.png')
    plt.close(fig)


def direction(v):
    return v / (1e-4 + np.linalg.norm(v, axis=-1, keepdims=True))


def fig_B_maps(c, r):
    """Case B: where the errors are. FOM mean speed, rms velocity error and mean direction error, L0 against L3-R1 (d_s = 10, K = 10)."""
    plt.rcParams['axes.grid'] = False                                  # (colour bars + grid trigger a matplotlib deprecation warning)
    seed, w3 = best_seed(r, 'L3', 'R1', 10, 10)
    z, lrec = l0_run(c, 10, seed)
    P = {'L0 ($d_s$=10)': l0_decode(c, z, lrec), 'L3-R1 ($d_s$=10, $K$=10)': r1_decode(c, 'L3', 10, 10, seed, w3)}
    ref = c.Yte
    dref = direction(ref)
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.6))
    im = axs[0, 0].imshow(np.linalg.norm(ref, axis=-1).mean(axis=(0, 1)).reshape(100, 100), origin='lower', extent=[0, 1, 0, 1])
    axs[0, 0].set_title('FOM: time-averaged speed $|\\mathbf{v}|$'); plt.colorbar(im, ax=axs[0, 0])
    mag = np.linalg.norm(ref, axis=-1)
    im = axs[1, 0].imshow((mag < 0.1).mean(axis=(0, 1)).reshape(100, 100), origin='lower', extent=[0, 1, 0, 1], cmap='magma'); plt.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_title('FOM: fraction of time with $|\\mathbf{v}|<0.1$')
    out = []
    for j, (name, p) in enumerate(P.items()):
        e = np.sqrt(np.mean(np.sum((p - ref) ** 2, axis=-1), axis=(0, 1))).reshape(100, 100)
        de2 = np.sum((direction(p) - dref) ** 2, axis=-1) / 2
        de = de2.mean(axis=(0, 1)).reshape(100, 100)
        im = axs[0, j + 1].imshow(e, origin='lower', extent=[0, 1, 0, 1]); plt.colorbar(im, ax=axs[0, j + 1])
        axs[0, j + 1].set_title(name + f': rms velocity error (max {e.max():.2f})', fontsize=9)
        im = axs[1, j + 1].imshow(de, origin='lower', extent=[0, 1, 0, 1], vmax=0.5); plt.colorbar(im, ax=axs[1, j + 1])
        axs[1, j + 1].set_title(name + ': mean direction error')
        share = lambda m: de2[m].sum() / de2.sum()
        out.append(dict(model=name, share_low_speed_0p1=share(mag < 0.1), frac_points_low_speed=float((mag < 0.1).mean()),
                        share_speed_below_0p01=share(mag < 0.01),
                        share_bottom_strip=de[:10].sum() / de.sum(), share_top_corners=de[90:, :10].sum() / de.sum() + de[90:, 90:].sum() / de.sum()))
    fig.tight_layout()
    fig.savefig(FIG / 'fig8_B_maps.png')
    plt.close(fig)
    plt.rcParams['axes.grid'] = True
    return pd.DataFrame(out)


def fig_A_latents(c):
    """Case A: Fourier coefficients of the FOM (centered) against the L0 latents of three seeds, coloured by the phase of the coefficient pair."""
    cr = c.coeffs(c.Yte)
    a, b = cr[:, 1], cr[:, 2]
    ph = np.arctan2(b, a)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    axs[0].scatter(a, b, s=1, alpha=0.3, c=ph, cmap='hsv'); axs[0].set_aspect('equal')
    axs[0].set(title='FOM: coefficients of $\\cos\\pi x$ and $\\sin\\pi x$', xlabel='$c_{\\cos}$', ylabel='$c_{\\sin}$')
    for j in range(3):
        s = np.load(ROOT / DIRS['A'] / 'results' / 'latents' / f'L0_ds2_seed{j}.npz')['test'].reshape(-1, 2)
        axs[j + 1].scatter(s[:, 0], s[:, 1], s=1, alpha=0.3, c=ph, cmap='hsv')
        axs[j + 1].set(title=f'L0 latents, seed {j}', xlabel='$s_1$', ylabel='$s_2$')
    for ax in axs:
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / 'fig9_A_latents.png')
    plt.close(fig)


# ----------------------------------------------------------------------------- latents across seeds, and the initial state
def latent_alignment(runs):
    """Test R^2 of the best affine map from the L0 latents of one seed to those of another (fitted on the training latents): is the latent space unique up to an affine change of coordinates?"""
    rows = []
    for k in 'ABC':
        for d in sorted(runs[k][runs[k].variant == 'L0'].d_s.unique()):
            L = [np.load(ROOT / DIRS[k] / 'results' / 'latents' / f'L0_ds{d}_seed{sd}.npz') for sd in range(3)]
            r2 = []
            for a in range(3):
                for b in range(3):
                    if a == b:
                        continue
                    Xa, Ya = L[a]['train'].reshape(-1, d), L[b]['train'].reshape(-1, d)
                    idx = np.arange(0, len(Xa), max(1, len(Xa) // 20000))
                    F = lambda x: np.concatenate([x, np.ones((len(x), 1))], axis=1)
                    W = np.linalg.lstsq(F(Xa[idx]), Ya[idx], rcond=None)[0]
                    Xt, Yt = L[a]['test'].reshape(-1, d), L[b]['test'].reshape(-1, d)
                    r2.append(1 - ((Yt - F(Xt) @ W) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum())
            rows.append(dict(case=k, d_s=d, median_R2=np.median(r2), min_R2=np.min(r2)))
    return pd.DataFrame(rows)


def initial_state_study(cases, runs):
    """All the linear variants add the mean field ybar(x), while every field starts at rest (y = 0) and the latent state starts at s = 0.
    Relative rms error (divided by the rms of the FOM) at t = 0, over the first steps and over the whole test horizon, for the frozen-latent (R1) runs."""
    rows = []
    for k, (d, K3, K2) in {'A': (3, 17, 5), 'B': (10, 10, 5), 'C': (12, 17, 33)}.items():
        c, r = cases[k], runs[k]
        ref = c.Yte
        rms = np.sqrt(np.mean(ref ** 2))
        seed, w3 = best_seed(r, 'L3', 'R1', d, K3)
        z, lrec = l0_run(c, d, seed)
        out = {'L0': l0_decode(c, z, lrec), 'L1-R1': r1_decode(c, 'L1', d, d, seed, 0),
               'L2-R1': r1_decode(c, 'L2', d, K2, seed, 0), 'L3-R1': r1_decode(c, 'L3', d, K3, seed, w3)}
        for name, p in out.items():
            e = np.sqrt(np.mean((p - ref) ** 2, axis=(0, 2, 3))) / rms
            rows.append(dict(case=k, model=name, rms_ybar_over_rms_y=np.sqrt(np.mean(c.ybar ** 2)) / rms, err_t0=e[0], err_t1_5=e[1:6].mean(), err_all=np.sqrt(np.mean(e ** 2))))
    # end-to-end (R2) runs of A and B: share of the squared error made in the first 20 time steps
    for k, dt_ratio, d, L in (('A', 0.05 / 2.3, 3, (('L1', 3), ('L2', 5), ('L3', 17))), ('B', 0.2 / 5.4, 5, (('L1', 5), ('L2', 17), ('L3', 9)))):
        c, r = cases[k], runs[k]
        ref = c.Yte
        rms = np.sqrt(np.mean(ref ** 2))
        for v, K in L:
            seed, width = best_seed(r, v, 'R2', d, K)
            w = np.load(ROOT / DIRS[k] / 'results' / 'weights' / f'{v}_R2_ds{d}_K{K}_s{seed}.npy')
            ldyn, n = mlp_split(w, c.dyn_sizes(d))
            S, T = ref.shape[:2]
            s, H = np.zeros((S, d)), []
            for t in range(T):                                              # forward Euler from s(0) = 0, as in the notebooks
                H.append(s)
                s = s + dt_ratio * mlp_apply(ldyn, np.concatenate([s, c.Ute[:, t]], axis=1))
            s = np.stack(H, axis=1)
            z = np.concatenate([s, c.Ute], axis=-1) if k == 'B' else s
            if v == 'L1':
                coef = s[..., :d]
            elif v == 'L2':
                lay, _ = mlp_split(w[n:], [z.shape[-1], K]); coef = z @ lay[0][0] + lay[0][1]
            else:
                lay, _ = mlp_split(w[n:], [z.shape[-1], width, width, K]); coef = mlp_apply(lay, z)
            p = c.synth(c.coef_scale * coef.reshape(S * T, -1), S, T)
            assert abs(nrmse(p, ref) - r[(r.variant == v) & (r.regime == 'R2') & (r.d_s == d) & (r.K == K) & (r.seed == seed)].nrmse_test.iloc[0]) < 1e-9
            e = np.sqrt(np.mean((p - ref) ** 2, axis=(0, 2, 3))) / rms
            rows.append(dict(case=k, model=f'{v}-R2 (d_s={d}, K={K})', rms_ybar_over_rms_y=np.sqrt(np.mean(c.ybar ** 2)) / rms, err_t0=e[0], err_t1_5=e[1:6].mean(),
                             err_all=np.sqrt(np.mean(e ** 2)), share_sq_err_first20=(e[:20] ** 2).sum() / (e ** 2).sum()))
    return pd.DataFrame(rows)


def divergence_study(c, r):
    """Case B: the velocity is divergence free. Relative divergence ||div v|| / ||grad v|| (central differences, 100 x 100 grid, whole test set)
    of the FOM, of L0 and of the POD-based decoders (a linear combination of divergence-free snapshots is divergence free)."""
    h = 1 / 99

    def ratio(v):
        V = v.reshape(*v.shape[:-2], 100, 100, 2)
        div = np.gradient(V[..., 0], h, axis=-1) + np.gradient(V[..., 1], h, axis=-2)
        grad2 = sum((np.gradient(V[..., j], h, axis=a) ** 2).sum() for j in (0, 1) for a in (-1, -2))
        return float(np.sqrt((div ** 2).sum() / grad2))

    rows = [dict(model='FOM', seed='-', rel_div=ratio(c.Yte))]
    for seed in range(3):
        z, lrec = l0_run(c, 10, seed)
        rows.append(dict(model='L0 (d_s=10)', seed=seed, rel_div=ratio(l0_decode(c, z, lrec))))
        _, w3 = best_seed(r, 'L3', 'R1', 10, 10)
        rows.append(dict(model='L3-R1 (d_s=10, K=10)', seed=seed, rel_div=ratio(r1_decode(c, 'L3', 10, 10, seed, w3))))
    return pd.DataFrame(rows)


def poly_study_A(c):
    """Case A: how nonlinear is the map from the L0 latents to the two Fourier coefficients (cos, sin)?  R^2 of polynomial fits of degree 1, 2, 3 in z,
    for the FOM coefficients and for the coefficients of the L0 prediction."""
    from itertools import combinations_with_replacement
    feats = lambda z, deg: np.stack([np.ones(len(z))] + [np.prod(z[:, list(idx)], axis=1) for g in range(1, deg + 1) for idx in combinations_with_replacement(range(z.shape[1]), g)], axis=1)
    cref = c.coeffs(c.Yte)[:, 1:3]
    rows = []
    for d in (2, 3, 5):
        for seed in range(3):
            z, lrec = l0_run(c, d, seed)
            S, T, _ = z.shape
            zz = z.reshape(S * T, -1)
            chat = c.coeffs(l0_decode(c, z, lrec))[:, 1:3]
            for name, tgt in (('FOM', cref), ('L0', chat)):
                for deg in (1, 2, 3):
                    F = feats(zz, deg)
                    rows.append(dict(d_s=d, target=name, degree=deg, R2=1 - ((tgt - F @ np.linalg.lstsq(F, tgt, rcond=None)[0]) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()))
    return pd.DataFrame(rows).groupby(['d_s', 'target', 'degree']).R2.median().unstack().reset_index()


def vorticity_floor(c):
    """Case B: relative vorticity error of the true flow itself after projection on its first K POD modes (the best a K-mode linear decoder can do)."""
    vref = vorticity(c, c.Yte)
    S, T = c.Yte.shape[:2]
    C = c.coeffs(c.Yte)
    return {K: float(np.linalg.norm(vorticity(c, c.synth(C[:, :K], S, T)) - vref) / np.linalg.norm(vref)) for K in (3, 10, 17, 33, 65)}


def test_rank_floors(c, ranks):
    """Rank floors computed from the SVD of the test fluctuations themselves (the best possible r-dimensional linear subspace on the test set, through the
    training mean): a check that the floor of the training POD evaluated on the test set, used in the report, is a tight proxy."""
    F = c.flat(c.Yte) - c.ybar_flat
    ref = np.linalg.norm(c.flat(c.Yte))
    sw = np.sqrt(c.wv)
    Vt = np.linalg.svd(F * sw, full_matrices=False)[2]
    out = {}
    for r in ranks:
        B = Vt[:r].T / sw[:, None]
        out[r] = float(np.linalg.norm(F - ((F * c.wv) @ B) @ B.T) / ref)
    return out


if __name__ == '__main__':
    runs = {k: runs_of(k) for k in 'ABC'}
    cases = {k: BUILD[k]() for k in 'ABC'}
    for k in 'ABC':
        check_rebuild(cases[k], runs[k])
        print(f'case {k}: numpy rebuild reproduces the stored NRMSE')
    FL = study_floors(cases, runs)
    # constant converting the relative L2 error into the NRMSE of the notebooks: NRMSE = c * eps_total (same test set for every run)
    CNRMSE = {k: float(np.sqrt(np.mean(cases[k].Yte ** 2)) / (cases[k].Yte.max() - cases[k].Yte.min())) for k in 'ABC'}
    print('NRMSE / eps_total =', {k: round(v, 5) for k, v in CNRMSE.items()})
    for k in 'ABC':
        r = runs[k]
        assert np.allclose(r.nrmse_test / r.eps_total_test, CNRMSE[k], rtol=2e-3)
    rows = []
    for k in 'ABC':
        for K in (1, 2, 3, 5, 9, 10, 12, 17, 33, 65):
            rows.append(dict(case=k, K=K, basis=FL[k]['basis'][K - 1], pod=FL[k]['pod'][K - 1]))
    print('## projection floors eps_proj(K) on the test set'); md(pd.DataFrame(rows), '%.3e')
    for k in 'ABC':
        for thr in (0.1, 0.05, 0.01):
            kb = int(np.argmax(FL[k]['basis'] <= thr)) + 1 if (FL[k]['basis'] <= thr).any() else None
            kp = int(np.argmax(FL[k]['pod'] <= thr)) + 1 if (FL[k]['pod'] <= thr).any() else None
            print(f'case {k}: K for eps_proj <= {thr}: protocol basis {kb}, POD {kp}')

    LIN = {k: l0_analysis(cases[k], sorted(runs[k][runs[k].variant == 'L0'].d_s.unique())) for k in 'ABC'}
    fig_error_vs_K(runs, FL, CNRMSE, LIN)
    RANK = fig_rank_bound(runs, FL, CNRMSE)
    fig_L0_linearity(LIN)
    fig_pareto(runs)
    fig_modes(cases, runs)
    fig_C_decoders(cases['C'], runs['C'])
    BMAP = fig_B_maps(cases['B'], runs['B'])
    fig_A_latents(cases['A'])

    # ------------------------------------------------------------------------- tables
    print('\n## best configuration of every procedure (selected on the validation NRMSE); median [min, max] over 3 seeds')
    for k in 'ABC':
        r = runs[k]
        br = r.merge(best_configs(r), on=CONFIG)
        l0 = br[br.variant == 'L0']
        rows = []
        for (v, reg, d, K), g in br.groupby(CONFIG):
            t = g.nrmse_test
            sep = '-' if v == 'L0' else ('yes' if (t.max() < l0.nrmse_test.min() or t.min() > l0.nrmse_test.max()) else 'no')
            rows.append(dict(procedure='L0' if v == 'L0' else f'{v}-{reg}', d_s=d, K=K or '-', params=int(g.rec_params.median()),
                             nrmse_test=f'{t.median():.2e} [{t.min():.2e}, {t.max():.2e}]', one_minus_rho=f'{g.one_minus_rho_test.median():.2e}',
                             eps_total=f'{g.eps_total_test.median():.3f}', ratio_to_L0=f'{t.median() / l0.nrmse_test.median():.2f}', separated_from_L0=sep,
                             train_s=f'{g.train_time_s.median():.0f}', infer_ms=f'{1e3 * g.infer_time_s.median():.1f}'))
        print(f'### case {k}'); md(pd.DataFrame(rows))

    print('## R1/R2 (same d_s, K, seed): ratio of the test NRMSE')
    rows = []
    for k in 'ABC':
        r = runs[k][runs[k].variant != 'L0']
        p = r.pivot_table(index=['variant', 'd_s', 'K', 'seed'], columns='regime', values='nrmse_test').reset_index()
        p['ratio'] = p.R1 / p.R2
        for v in ('L1', 'L2', 'L3'):
            q = p[(p.variant == v) & (p.K >= 3 if v != 'L1' else p.K >= 1)]
            qb = best_configs(runs[k])
            qb = qb[(qb.variant == v) & (qb.regime == 'R2')].iloc[0]
            at_best = p[(p.variant == v) & (p.d_s == qb.d_s) & (p.K == qb.K)].ratio
            cfg = q.groupby(['d_s', 'K']).ratio.median()
            rows.append(dict(case=k, variant=v, best_R2_config=f'd_s={qb.d_s}, K={qb.K}', ratio_at_best_R2=f'{at_best.median():.2f} [{at_best.min():.2f}, {at_best.max():.2f}]',
                             median_over_configs=f'{cfg.median():.2f}', range_over_configs=f'[{cfg.min():.2f}, {cfg.max():.2f}]', frac_configs_above_2=f'{(cfg > 2).mean():.2f}'))
    md(pd.DataFrame(rows))

    print('## L0 decoder: linearizability index R2_lin(K) (median over seeds) and spectral width K_99 (median [q25, q75] over test states, median over seeds)')
    rows = []
    for k in 'ABC':
        lin, k99 = LIN[k]
        for d in sorted(lin.d_s.unique()):
            g = lin[(lin.d_s == d) & (lin.K > 0)].groupby('K').R2.median()
            kk = k99[k99.d_s == d].groupby('name')[['med', 'q1', 'q3']].median()
            rows.append(dict(case=k, d_s=d, **{f'R2lin(K={K})': g[K] for K in (1, 3, 9, 33) if K in g.index},
                             K99_L0=f"{kk.loc['L0', 'med']:.0f} [{kk.loc['L0', 'q1']:.0f}, {kk.loc['L0', 'q3']:.0f}]",
                             K99_FOM=f"{kk.loc['FOM', 'med']:.0f} [{kk.loc['FOM', 'q1']:.0f}, {kk.loc['FOM', 'q3']:.0f}]"))
    md(pd.DataFrame(rows))

    print('## L0 output projected on its first K modes: test NRMSE (median over seeds) at the reference d_s')
    rows = []
    for k in 'ABC':
        lin = LIN[k][0]
        lin = lin[lin.d_s == D_REF[k]]
        raw = runs[k][(runs[k].variant == 'L0') & (runs[k].d_s == D_REF[k])].nrmse_test.median()
        rows.append(dict(case=k, d_s=D_REF[k], L0_raw=raw, **{f'K={K}': lin[lin.K == K].nrmse_proj.median() for K in (3, 9, 17, 33, 65)}))
    md(pd.DataFrame(rows), '%.3e')
    lb = LIN['B'][0]
    lb = lb[lb.d_s == 10]
    print('case B, d_s=10, relative vorticity error of L0 (median over seeds):', {int(K): round(g.vort_proj.median(), 3) for K, g in lb[lb.vort_proj.notna()].groupby('K')})

    print('\n## rank bound: eps_total against d_s (K of L2/L3 chosen on validation, R2)')
    md(RANK.pivot_table(index=['case', 'd_s'], columns='proc', values='eps_total').reset_index(), '%.3f')

    print('## case B: where the direction error lives'); md(BMAP, '%.3f')
    rb = runs['B'].merge(best_configs(runs['B']), on=CONFIG)
    print('## case B: extra metrics of the best configurations (median over seeds)')
    md(rb.groupby(CONFIG)[['dir_test', 'vort_rel_test', 'extrap_ratio_test']].median().reset_index(), '%.3f')
    print('## case C: activation-time error (ms) of the best configurations (median over seeds)')
    rc = runs['C'].merge(best_configs(runs['C']), on=CONFIG)
    md(rc.groupby(CONFIG).act_mae_test.agg(['median', 'min', 'max']).reset_index(), '%.2f')

    print('## generalisation indicators of the best configurations')
    rows = []
    for k in 'ABC':
        r = runs[k]
        b = best_configs(r)
        br = r.merge(b, on=CONFIG)
        pr = runs_of(k, 'predictions').merge(b, on=CONFIG)
        for (v, reg, d, K), g in br.groupby(CONFIG):
            q = pr[(pr.variant == v) & (pr.regime == reg) & (pr.d_s == d) & (pr.K == K)].nrmse
            rows.append(dict(case=k, procedure='L0' if v == 'L0' else f'{v}-{reg}', train=g.nrmse_train.median(), valid=g.nrmse_valid.median(), test=g.nrmse_test.median(),
                             test_over_valid=g.nrmse_test.median() / g.nrmse_valid.median(), seed_range=g.nrmse_test.max() / g.nrmse_test.min(),
                             sample_p90_over_median=q.quantile(0.9) / q.median(), sample_max_over_median=q.max() / q.median()))
    md(pd.DataFrame(rows), '%.3g')

    print('## regret: test NRMSE of each procedure divided by the best procedure of the case')
    reg = {}
    for k in 'ABC':
        r = runs[k]
        br = r.merge(best_configs(r), on=CONFIG)
        m = br.groupby(CONFIG).nrmse_test.median()
        for (v, rg, d, K), x in m.items():
            reg.setdefault('L0' if v == 'L0' else f'{v}-{rg}', {})[k] = x / m.min()
    md(pd.DataFrame(reg).T.reset_index().rename(columns={'index': 'procedure'}).assign(worst=lambda d: d[['A', 'B', 'C']].max(axis=1)), '%.2f')
    # cost: end-to-end (R2) training time and inference time of the best configurations
    print('## cost of the best R2 configurations (median)')
    rows = []
    for k in 'ABC':
        r = runs[k]
        br = r.merge(best_configs(r), on=CONFIG)
        br = br[br.regime == 'R2']
        for (v, d, K), g in br.groupby(['variant', 'd_s', 'K']):
            rows.append(dict(case=k, variant=v, d_s=d, K=K, params=int(g.rec_params.median()), train_s=g.train_time_s.median(), infer_ms=1e3 * g.infer_time_s.median()))
    md(pd.DataFrame(rows), '%.1f')

    # ------------------------------------------------------------------------- sweeps in detail
    print('\n## L0 and L1 against the latent size d_s: test NRMSE, median [min, max] over 3 seeds')
    rows = []
    for k in 'ABC':
        r = runs[k]
        for d in sorted(r.d_s.unique()):
            row = dict(case=k, d_s=d)
            for lab, v, reg in (('L0', 'L0', 'R2'), ('L1-R2', 'L1', 'R2'), ('L1-R1', 'L1', 'R1')):
                t = r[(r.variant == v) & (r.regime == reg) & (r.d_s == d)].nrmse_test
                row[lab] = f'{t.median():.2e} [{t.min():.2e}, {t.max():.2e}]'
            rows.append(row)
    md(pd.DataFrame(rows))

    print('## K-sweep at the reference d_s: test NRMSE (median of 3 seeds), efficiency ratio eps_total/eps_proj, width of g_theta')
    for k in 'ABC':
        r = runs[k]
        d = D_REF[k]
        rows = []
        for v in ('L2', 'L3'):
            for reg in ('R2', 'R1'):
                q = r[(r.variant == v) & (r.regime == reg) & (r.d_s == d)].groupby('K')
                nr, ef, wd = q.nrmse_test.median(), q.eff_ratio_test.median(), q.width.median()
                rows.append(dict(procedure=f'{v}-{reg}', **{f'K={K}': f'{nr[K]:.2e}' for K in nr.index}))
                if reg == 'R2':
                    rows.append(dict(procedure=f'  {v} eff. ratio', **{f'K={K}': f'{ef[K]:.2f}' if ef[K] < 100 else '>100' for K in ef.index}))
                    if v == 'L3':
                        rows.append(dict(procedure='  L3 width', **{f'K={K}': f'{wd[K]:.0f}' for K in wd.index}))
        print(f'### case {k}, d_s = {d}'); md(pd.DataFrame(rows).fillna(''))

    print('## H1 (case A): ratio of the L2 test NRMSE to the L0 test NRMSE at the same d_s, median [min, max] over seeds (paired by seed)')
    rA = runs['A']
    rows = []
    for d in (2, 3, 5):
        for K in (3, 5):
            row = dict(d_s=d, K=K)
            for reg in ('R2', 'R1'):
                a = rA[(rA.variant == 'L2') & (rA.regime == reg) & (rA.d_s == d) & (rA.K == K)].set_index('seed').nrmse_test
                b = rA[(rA.variant == 'L0') & (rA.d_s == d)].set_index('seed').nrmse_test
                q = a / b
                row[f'L2-{reg} / L0'] = f'{q.median():.2f} [{q.min():.2f}, {q.max():.2f}]'
            rows.append(row)
    md(pd.DataFrame(rows))

    print('## H2 (case C): test NRMSE of L1/L2/L3 divided by that of L0 at the same d_s (R2), median [min, max] over seeds')
    rC = runs['C']
    rows = []
    for d in (3, 5, 12):
        row = dict(d_s=d)
        b = rC[(rC.variant == 'L0') & (rC.d_s == d)].set_index('seed').nrmse_test
        for lab, v, K in (('L1 (K=d_s)', 'L1', d), ('L2 (K=33)', 'L2', 33), ('L3 (K=17)', 'L3', 17), ('L3 (K=33)', 'L3', 33)):
            a = rC[(rC.variant == v) & (rC.regime == 'R2') & (rC.d_s == d) & (rC.K == K)].set_index('seed').nrmse_test
            q = a / b
            row[lab] = f'{q.median():.2f} [{q.min():.2f}, {q.max():.2f}]'
        rows.append(row)
    md(pd.DataFrame(rows))

    print('\n## latents of different seeds: test R2 of the best affine map between them (all ordered pairs)')
    md(latent_alignment(runs), '%.3f')
    print('## initial state: relative rms error of the linear variants and L0 (reference d_s of each case)')
    md(initial_state_study(cases, runs), '%.3f')

    print('## case B: relative divergence of the predicted velocity')
    md(divergence_study(cases['B'], runs['B']).groupby('model').rel_div.agg(['median', 'min', 'max']).reset_index(), '%.4f')

    print('## case A: R2 of polynomial fits (degree 1, 2, 3 in z) of the (cos, sin) coefficients, median over seeds')
    md(poly_study_A(cases['A']), '%.3f')

    print('## case B: relative vorticity error of the true flow projected on its first K POD modes')
    print(vorticity_floor(cases['B']))

    print('## rank floors: training POD evaluated on the test set (used in the report) and best possible on the test set (SVD of the test fluctuations)')
    rows = []
    for k, ranks in {'A': (1, 2), 'B': (2, 6, 11), 'C': (1, 2, 3, 5, 12)}.items():
        best = test_rank_floors(cases[k], ranks)
        for r in ranks:
            rows.append(dict(case=k, rank=r, training_POD_on_test=FL[k]['pod'][r - 1], test_optimal=best[r], overestimate_percent=100 * (FL[k]['pod'][r - 1] / best[r] - 1) if best[r] > 1e-6 else np.nan))
    md(pd.DataFrame(rows), '%.4f')

