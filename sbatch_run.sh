#!/bin/bash

# USAGE EXAMPLE:
# sbatch -c 2 --gres=gpu:1 -J initial_training sbatch_run.sh train

# Check if a parameter was provided
if [ -z "$1" ]; then
    echo "Error: No parameter provided."
    echo "Usage: $0 {train|test|process|deploy}"
    exit 1
fi

echo "running using sbatch - FOOTBALL EVENTS GENERATION"
cd /home/amit-ben/football_events_generation

# Common setup steps (runs for all valid cases)
# Only run this if you want these steps to happen every time regardless of the command
#rm -rf .venv
#rm -rf uv.lock
#uv venv --python 3.12
source .venv/bin/activate
#uv sync --resolution highest
#uv pip install --upgrade numba
#uv pip install -e .
#uv pip install tqdm
#uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Switch-case on the first parameter
case "$1" in
    train)
        echo "Starting Training..."
        python3 main.py --csv_dir ../match_csv/ --step train --autoencoder_path ../encoder > unified_time_training.out 2> unified_time_training.out
        ;;

    generate)
        echo "Starting generation..."
        python3 main.py --csv_dir ../match_csv/ --step generate --autoencoder_path ../encoder > generate.out 2> generate.out
        ;;

    *)
        echo "Error: Invalid parameter '$1'"
        echo "Usage: $0 {train|test|process|deploy}"
        exit 1
        ;;
esac

rm -rf ./*slurm*