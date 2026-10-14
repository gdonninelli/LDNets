"""Merge per-shard RQ2-bis results into one folder.

usage: python merge_results.py MERGED_DIR SHARD_DIR [SHARD_DIR ...]
Concatenates runs/predictions/curves.csv with the same dedup rule as the
notebooks' load(), copies weights/*.npy, and copies timing.csv, hardware.json
and meta.json from the first shard that has them (warns if meta.json training
settings differ between shards).
"""
import json
import pathlib
import shutil
import sys

KEYS = {'runs': ['variant', 'K', 'width', 'depth', 'd_s', 'seed'],
        'predictions': ['variant', 'K', 'width', 'depth', 'd_s', 'seed', 'sample'],
        'curves': ['variant', 'K', 'width', 'depth', 'd_s', 'seed', 'iter']}

merged, shards = pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]]
assert len(shards) >= 1, 'give at least one shard dir'
merged.mkdir(parents=True, exist_ok=True)
(merged / 'weights').mkdir(exist_ok=True)

import pandas as pd

for name, key in KEYS.items():
    frames = [pd.read_csv(s / f'{name}.csv') for s in shards if (s / f'{name}.csv').exists()]
    assert frames, f'no {name}.csv in any shard'
    pd.concat(frames).drop_duplicates(subset=key, keep='last').to_csv(merged / f'{name}.csv', index=False)
    print(name, '->', len(pd.read_csv(merged / f'{name}.csv')), 'rows')

n = 0
for s in shards:
    for w in (s / 'weights').glob('*.npy'):
        shutil.copy(w, merged / 'weights' / w.name)
        n += 1
print('weights ->', n, 'files')

metas = [(s / 'meta.json').read_text() for s in shards if (s / 'meta.json').exists()]
if len(set(metas)) > 1:
    print('WARNING: meta.json differs between shards (mixed training settings?)')
for f in ('timing.csv', 'hardware.json', 'meta.json'):
    src = next((s / f for s in shards if (s / f).exists()), None)
    if src is not None:
        shutil.copy(src, merged / f)
        print(f, 'copied from', src.parent)
