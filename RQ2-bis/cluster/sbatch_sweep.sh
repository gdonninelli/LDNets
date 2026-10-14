#!/bin/bash
# RQ2-bis accuracy sweep on Leonardo DCGP (Intel Sapphire Rapids, CPU-only).
#
# One job per case folder; split a case over nodes with a job array, one shard
# per task. Each shard MUST write its own results folder (concurrent appends to
# the same CSV files corrupt them); merge afterwards with merge_results.py.
#
# usage (after editing TODOs below):
#   sbatch --array=0-3 sbatch_sweep.sh 2D-Stokes   # 4 shards of Case B
#   sbatch --array=0-3 sbatch_sweep.sh ADR-1b      # 4 shards of Case C
#
# At ~50 min (B) / ~40 min (C) per L0 run at 10000 iterations, one shard of
# ~8 L0 runs fits in ~7 h; L2 runs take ~7 min each.

#SBATCH --job-name=rq2bis-sweep
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112            # TODO: check with `sinfo -Nel` / `lscpu` (2x Xeon 8480+)
#SBATCH --time=12:00:00                # TODO: adjust to shard size
#SBATCH --output=logs/sweep_%x_%A_%a.out
#SBATCH --error=logs/sweep_%x_%A_%a.err
#SBATCH --account=<PROJECT>            # TODO: your project account

set -euo pipefail
CASE=${1:?usage: sbatch_sweep.sh 2D-Stokes|ADR-1b}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# TODO: `module avail python` and put the 3.11.7 module provided on the system here
module purge
module load python/3.11.7
source "$WORK/venvs/ldnets311/bin/activate"   # TODO: path of the venv (see ../cluster README)

export MPLBACKEND=Agg                          # no display on compute nodes
export TF_CPP_MIN_LOG_LEVEL=2                  # quieter TensorFlow logs
export RQ2BIS_OUT="results_shard${SHARD}"
export RQ2BIS_SHARD="${SHARD}/${N_SHARDS}"

NB=$([ "$CASE" = "2D-Stokes" ] && echo TestCase_B_bis.ipynb || echo TestCase_C_bis.ipynb)
cd "$SLURM_SUBMIT_DIR/../$CASE"
mkdir -p logs
python -m jupyter nbconvert --to notebook --execute \
  --output "executed_shard${SHARD}.ipynb" \
  --ExecutePreprocessor.timeout=-1 "$NB"
