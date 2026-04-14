#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
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
#SBATCH --job-name=far-tuning

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.13.1
module load CUDA/12.9.1

# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source .venv/bin/activate

# Submit your job to the cluster
srun --gres=gpu:1 uv run poe tune --models "random_forest light_gcn sasrec tisasrec" --device cuda