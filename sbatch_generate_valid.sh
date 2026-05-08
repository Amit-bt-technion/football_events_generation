#!/bin/bash

# Single generate_valid job with configurable tolerances.
#
# USAGE (submit via sbatch):
#   sbatch -c 2 --gres=gpu:1 -J <job_name> sbatch_generate_valid.sh <timestamp_tolerance> <max_time_violations> <output_dir>
#
# POSITIONAL ARGUMENTS:
#   $1  timestamp_tolerance  – per-event backward allowance in seconds (default: 0)
#   $2  max_time_violations  – max allowed time violations per sequence (default: 0)
#   $3  output_dir           – where to write results (default: outputs/generate_valid)
#
# EXAMPLE:
#   sbatch -c 2 --gres=gpu:1 -J gv_tt3_tv1 sbatch_generate_valid.sh 3 1 outputs/sweep_tt3_tv1

TIME_TOL=${1:-0}
TIME_VIO=${2:-0}
OUTPUT_DIR=${3:-"outputs/generate_valid"}

echo "running using sbatch - FOOTBALL EVENTS GENERATION (generate_valid)"
echo "  timestamp_tolerance : ${TIME_TOL}s"
echo "  max_time_violations : ${TIME_VIO}"
echo "  output_dir          : ${OUTPUT_DIR}"

cd /home/amit-ben/football_events_generation
source .venv/bin/activate

mkdir -p "${OUTPUT_DIR}"

python3 main.py \
    --step generate_valid \
    --checkpoint executions_log/2026-03-05-training/models/dit_best_model.pt \
    --num_valid_sequences 100 \
    --timestamp_tolerance "${TIME_TOL}" \
    --max_time_violations "${TIME_VIO}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/run.out" 2>&1

echo "Done – results in ${OUTPUT_DIR}"

rm -rf ./*slurm*
