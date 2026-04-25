#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/common/home/users/s/samuel.sim.2024/SMU-Capstone/outputs/%u.%j.out
#SBATCH --requeue

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=msc
#SBATCH --account=msc
#SBATCH --qos=studentqos
#SBATCH --mail-user=samuel.sim.2024@msc.smu.edu.sg,samuelsimweixuan@gmail.com
#SBATCH --job-name=far-evaluate-profile-coherent

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

module purge
module load Python/3.13.1
module load CUDA/12.9.1

source .venv/bin/activate

# PC-LGCN ablation + lambda-sensitivity sweep at the winning LightGCN backbone
# (4 ablation cells + 3 lambda points = 7 trials, each a full 69-split eval).
srun --gres=gpu:1 uv run poe evaluate-profile-coherent --device cuda
