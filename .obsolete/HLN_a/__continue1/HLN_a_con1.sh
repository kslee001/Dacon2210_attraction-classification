#!/bin/bash

#SBATCH --job-name=HLossnet_a_con1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00  # 12 hours timelimit
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=2

source /home/${USER}/dacon/.bashrc
conda activate tch

srun python -u /home/gyuseonglee/dacon/Dacon/HLN_a/__continue1/main_continue1.py