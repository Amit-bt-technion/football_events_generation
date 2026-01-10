#!/bin/bash

###
# $ conda activate diffusion
# $ srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
#

unset XDG_RUNTIME_DIR

xvfb-run -a -s "-screen 0 1440x900x24" jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100

