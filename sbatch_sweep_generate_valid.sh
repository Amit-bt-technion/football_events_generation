#!/bin/bash

# Submit 16 generate_valid jobs covering the full cross product of
#   timestamp_tolerance  : 0, 3, 6, 12  (seconds)
#   max_time_violations  : 0, 1, 2, 5
#
# USAGE:
#   bash sbatch_sweep_generate_valid.sh
#   sbatch -c 2 --gres=gpu:1 -J generate_valid_sweep sbatch_sweep_generate_valid.sh
#
# Each job is submitted independently so they can run in parallel on the cluster.
# Logs: outputs/sweep_tt<T>_tv<V>/run.out
# Jobs: named gv_tt<T>_tv<V>

cd /home/amit-ben/football_events_generation

TIME_TOLERANCES=(0 3 6 12)
TIME_VIOLATIONS=(0 1 2 5)

echo "Submitting generate_valid sweep – ${#TIME_TOLERANCES[@]} x ${#TIME_VIOLATIONS[@]} = $((${#TIME_TOLERANCES[@]} * ${#TIME_VIOLATIONS[@]})) jobs"
echo ""

for TIME_TOL in "${TIME_TOLERANCES[@]}"; do
    for TIME_VIO in "${TIME_VIOLATIONS[@]}"; do
        JOB_NAME="gv_tt${TIME_TOL}_tv${TIME_VIO}"
        OUT_DIR="outputs/sweep_tt${TIME_TOL}_tv${TIME_VIO}"

        mkdir -p "${OUT_DIR}"

        JOB_ID=$(sbatch \
            -c 2 \
            --gres=gpu:1 \
            -J "${JOB_NAME}" \
            --output="${OUT_DIR}/slurm-%j.out" \
            sbatch_generate_valid.sh "${TIME_TOL}" "${TIME_VIO}" "${OUT_DIR}" \
            | awk '{print $NF}')

        echo "  Submitted ${JOB_NAME}  ->  ${OUT_DIR}  (slurm job ${JOB_ID})"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
