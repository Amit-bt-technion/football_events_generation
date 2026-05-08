#!/bin/bash

# Generate a large number of unfiltered sequences and persist only the
# event-type column (feature 0) of each decoded event for downstream
# transition-matrix construction.
#
# USAGE (submit via sbatch):
#   sbatch -c 2 --gres=gpu:1 -J <job_name> sbatch_generate_for_matrix.sh \
#       <num_gen_samples> <output_dir>
#
# POSITIONAL ARGUMENTS:
#   $1  num_gen_samples      – total sequences to generate (default: 1000000)
#   $2  output_dir           – where to write results (default: outputs/gen_for_matrix)
#
# EXAMPLE:
#   sbatch -c 2 --gres=gpu:1 -J gfm_1m sbatch_generate_for_matrix.sh 1000000 outputs/gen_for_matrix

NUM_SAMPLES=${1:-1000000}
OUTPUT_DIR=${2:-"outputs/gen_for_matrix"}

echo "running using sbatch - FOOTBALL EVENTS GENERATION (generate_for_matrix)"
echo "  num_gen_samples     : ${NUM_SAMPLES}"
echo "  output_dir          : ${OUTPUT_DIR}"

cd /home/amit-ben/football_events_generation
source .venv/bin/activate

mkdir -p "${OUTPUT_DIR}"

python3 main.py \
    --step generate_for_matrix \
    --checkpoint executions_log/2026-03-05-training/models/dit_best_model.pt \
    --num_gen_samples "${NUM_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/run.out" 2>&1

echo "Done – results in ${OUTPUT_DIR}"

rm -rf ./*slurm*
