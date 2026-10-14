#!/bin/bash
# RQ2-bis accuracy sweep on Leonardo DCGP (Intel Sapphire Rapids, CPU-only).
#
# One job per case folder; split a case over nodes with a job array, one shard
# per task. Each shard MUST write its own results folder (concurrent appends to
# the same CSV files corrupt them); merge afterwards with merge_results.py.
#
# usage (from this directory):
#   sbatch --array=0-3 sbatch_sweep.sh 2D-Stokes   # 4 shards of Case B
#   sbatch --array=0-3 sbatch_sweep.sh ADR-1b      # 4 shards of Case C
#
# At ~50 min (B) / ~40 min (C) per L0 run at 10000 iterations, one shard of
# ~8 L0 runs fits in ~7 h; L2 runs take ~7 min each.

#SBATCH --job-name=rq2bis-sweep
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56            # TODO: check with `sinfo -Nel` / `lscpu` (2x Xeon 8480+)
#SBATCH --time=24:00:00                # TODO: adjust to shard size
#SBATCH --account=euhpc_d35_025
#SBATCH --chdir=/leonardo/home/userexternal/gdonnine/LDNets/RQ2-bis/cluster
# Slurm opens these files before the job starts, so the output directory must
# already exist. Keep submission logs in this directory.
#SBATCH --output=sweep_%x_%A_%a.out
#SBATCH --error=sweep_%x_%A_%a.err

set -euo pipefail
CASE=${1:?usage: sbatch_sweep.sh 2D-Stokes|ADR-1b}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

case "$CASE" in
  2D-Stokes) NB=TestCase_B_bis.ipynb ;;
  ADR-1b)    NB=TestCase_C_bis.ipynb ;;
  *) echo "Unsupported case: $CASE (expected 2D-Stokes or ADR-1b)" >&2; exit 2 ;;
esac

# Slurm runs a copied version of this script from /var/spool/slurmd. The
# --chdir directive above keeps the working directory tied to the repository.
CASE_DIR="$PWD/../$CASE"
VENV_DIR="${LDNETS_VENV:-$WORK/venvs/ldnets311}"

module purge
module load python/3.11.7
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python environment not found: $VENV_DIR" >&2
  echo "Create it with: python -m venv $VENV_DIR" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

export MPLBACKEND=Agg                          # no display on compute nodes
export TF_CPP_MIN_LOG_LEVEL=2                  # quieter TensorFlow logs
export RQ2BIS_OUT="results_shard${SHARD}"
export RQ2BIS_SHARD="${SHARD}/${N_SHARDS}"

cd "$CASE_DIR"
mkdir -p logs
python -m jupyter nbconvert --to notebook --execute \
  --output "executed_shard${SHARD}.ipynb" \
  --ExecutePreprocessor.timeout=-1 "$NB"
