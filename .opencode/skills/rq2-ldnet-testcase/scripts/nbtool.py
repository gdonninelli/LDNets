#!/usr/bin/env python3
"""Author and verify the RQ2 notebooks without nbformat/nbconvert (not installed in the repo venv).

  nbtool.py topy NOTEBOOK.ipynb OUT.py             notebook -> "percent" text ('# %%' code cells, '# %% [markdown]' text cells)
  nbtool.py tonb IN.py OUT.ipynb [--like NB.ipynb] percent text -> notebook (kernelspec metadata copied from --like)
  nbtool.py run  NOTEBOOK.ipynb OUTDIR [options]   execute the code cells headlessly in ONE namespace, like a kernel

run options:
  --cwd DIR        working directory of the notebook (default: the notebook's folder: utils/optimization are imported from there)
  --sub OLD NEW    replace OLD (must occur exactly once in the code cells) by NEW; repeatable. Use it to shrink the sweep
                   (SEEDS, D_S, K_LEVELS, EPOCHS_*) and to point OUT to a temporary folder
  --skip REGEX     replace every code line matching REGEX (re.M) by `pass`, e.g. '^sweep\\(.*$' defines everything without training
  --cells 0,1,2    run only these code cells (0-based among the code cells)
  --extra FILE.py  executed in the same namespace after the cells
Figures are saved as OUTDIR/fig_NNN.png (plt.show is patched, Agg backend); per-cell time and peak memory are printed.
Run it with the repo venv: /Users/giuliodonninelli/Dev/LDNets/.venv/bin/python nbtool.py run ...
"""
import argparse
import json
import linecache
import os
import re
import sys
import time
import traceback

MD, CODE = '# %% [markdown]', '# %%'


def topy(nb_path, out):
    chunks = []
    for c in json.load(open(nb_path))['cells']:
        text = ''.join(c['source'])
        if c['cell_type'] == 'markdown':
            chunks.append(MD + '\n' + '\n'.join(('# ' + l) if l else '#' for l in text.split('\n')))
        else:
            chunks.append(CODE + '\n' + text)
    open(out, 'w').write('\n\n'.join(chunks) + '\n')


def parse(py_path):
    cells, kind, lines = [], None, []

    def flush():
        if kind is not None:
            text = '\n'.join(lines).strip('\n')
            if kind == 'markdown':
                text = '\n'.join(l[2:] if l.startswith('# ') else l[1:] if l.startswith('#') else l for l in text.split('\n'))
            cells.append((kind, text))

    for line in open(py_path).read().split('\n'):
        if line.startswith('# %%'):
            flush()
            kind, lines = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            lines.append(line)
    flush()
    return cells


def tonb(py_path, out, like=None):
    meta = json.load(open(like))['metadata'] if like else {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
    cells = []
    for kind, text in parse(py_path):
        lines = text.split('\n')
        cell = {'cell_type': kind, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    with open(out, 'w') as f:
        json.dump({'cells': cells, 'metadata': meta, 'nbformat': 4, 'nbformat_minor': 2}, f, indent=1, ensure_ascii=False)
        f.write('\n')


def run(nb_path, outdir, cwd=None, subs=(), skips=(), cells=None, extra=None):
    nb_path, outdir = os.path.abspath(nb_path), os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    code = [''.join(c['source']) for c in json.load(open(nb_path))['cells'] if c['cell_type'] == 'code']
    for old, new in subs:
        n = sum(src.count(old) for src in code)
        assert n == 1, f'--sub {old!r} occurs {n} times in the code cells (it must occur exactly once)'
        code = [src.replace(old, new) for src in code]
    for pattern in skips:
        code = [re.sub(pattern, 'pass', src, flags=re.M) for src in code]
    cwd = os.path.abspath(cwd or os.path.dirname(nb_path))
    os.chdir(cwd)
    sys.path.insert(0, cwd)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_fig = [0]

    def show(*args, **kwargs):
        for num in plt.get_fignums():
            plt.figure(num).savefig(f'{outdir}/fig_{n_fig[0]:03d}.png', dpi=60)
            n_fig[0] += 1
        plt.close('all')
    plt.show = show

    ns = {'__name__': '__main__'}
    for i, src in enumerate(code):
        if cells is not None and i not in cells:
            continue
        fname = f'<code cell {i}>'
        # AutoGraph (tf.function) needs the source of the functions: IPython provides it through linecache, a bare exec does not
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
        t0 = time.time()
        try:
            exec(compile(src, fname, 'exec'), ns)
        except BaseException:
            print(f'!!! code cell {i} FAILED')
            traceback.print_exc()
            sys.exit(1)
        print(f'--- code cell {i} done in {time.time() - t0:.1f}s', flush=True)
    if extra:
        exec(compile(open(extra).read(), extra, 'exec'), ns)
    import resource
    print('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)    # bytes on macOS
    print('ALL DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('topy'); p.add_argument('notebook'); p.add_argument('out')
    p = sub.add_parser('tonb'); p.add_argument('py'); p.add_argument('out'); p.add_argument('--like')
    p = sub.add_parser('run'); p.add_argument('notebook'); p.add_argument('outdir'); p.add_argument('--cwd')
    p.add_argument('--sub', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--skip', action='append', default=[], metavar='REGEX')
    p.add_argument('--cells'); p.add_argument('--extra')
    a = ap.parse_args()
    if a.cmd == 'topy':
        topy(a.notebook, a.out)
    elif a.cmd == 'tonb':
        tonb(a.py, a.out, a.like)
    else:
        run(a.notebook, a.outdir, a.cwd, a.sub, a.skip, None if a.cells is None else {int(i) for i in a.cells.split(',')}, a.extra)
