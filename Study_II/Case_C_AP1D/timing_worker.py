"""Times configurations of an RQ2-bis notebook in a fresh process with a fixed thread count.

usage: python timing_worker.py THREADS NOTEBOOK '[[variant, K, width, depth, d_s, seed], ...]'
TensorFlow only accepts its thread counts before its first operation, so the counts are set before anything else runs.
The notebook cells tagged "worker" (setup, data, configuration, helpers, runs) are executed; time_eval comes from the Runs cell.
Prints one line: RESULT [seconds per function/gradient evaluation, ...]
"""
import json
import linecache
import os
import sys

threads, notebook, cfgs = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
os.environ['OMP_NUM_THREADS'] = str(threads)
for var in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):   # Intel MKL / OpenBLAS (NumPy/SciPy) follow OMP unless set
    os.environ.setdefault(var, str(threads))
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

namespace = {}
for i, cell in enumerate(json.load(open(notebook))['cells']):
    if cell['cell_type'] == 'code' and 'worker' in cell['metadata'].get('tags', []):
        src, name = ''.join(cell['source']), f'cell{i}'
        linecache.cache[name] = (len(src), None, src.splitlines(True), name)     # AutoGraph needs the source, as in Jupyter
        exec(compile(src, name, 'exec'), namespace)
print('RESULT', json.dumps([namespace['time_eval'](*cfg) for cfg in cfgs]))
