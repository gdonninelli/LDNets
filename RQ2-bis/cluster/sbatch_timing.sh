#!/bin/bash
# RQ2-bis Pass-T timing on Leonardo DCGP. Run ONCE per case, AFTER merging the
# sweep shards, ALONE on a whole node (--exclusive): any other load on the node
# contaminates timing.csv and the thread-scaling numbers.
#
# usage (after editing TODOs below):
#   sbatch sbatch_timing.sh 2D-Stokes
#   sbatch sbatch_timing.sh ADR-1b
#
# RQ2BIS_OUT points at the MERGED results folder; the notebook's RQ2BIS_STAGE5
# block runs thread_scaling() then spot_timing() into timing.csv there.

#SBATCH --job-name=rq2bis-timing
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56            # TODO: check with `sinfo -Nel` / `lscpu`
#SBATCH --exclusive
#SBATCH --time=24:00:00                # TODO: adjust (dozens of configs x thread counts)
#SBATCH --output=logs/timing_%x_%j.out
#SBATCH --error=logs/timing_%x_%j.err
#SBATCH --account=euhpc_d35_025

set -euo pipefail
CASE=${1:?usage: sbatch_timing.sh 2D-Stokes|ADR-1b}

# TODO: same module + venv as sbatch_sweep.sh
module purge
module load python/3.11.7
source "$WORK/venvs/ldnets311/bin/activate"

export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2
export RQ2BIS_OUT="results"               # the MERGED folder (see merge_results.py)
export RQ2BIS_STAGE5=1
export RQ2BIS_THREADS="1,2,4,8,16,32,64,112"   # TODO: counts up to the node's cores

NB=$([ "$CASE" = "2D-Stokes" ] && echo TestCase_B_bis.ipynb || echo TestCase_C_bis.ipynb)
cd "$SLURM_SUBMIT_DIR/../$CASE"
mkdir -p logs
python -m jupyter nbconvert --to notebook --execute \
  --output executed_timing.ipynb \
  --ExecutePreprocessor.timeout=-1 "$NB"
