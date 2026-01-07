#!/bin/bash
#
# SLURM job script for running on mcml gpus.

#SBATCH --job-name=collab_learn
#SBATCH --output=/dss/dsshome1/0F/di97pub/CollaborationLearning/logs/slurm/slurm.%j.%N.out
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1                 # change to 2/4 if a single run needs more GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --comment="chain"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jgrimm@cis.lmu.de
#SBATCH --chdir=/dss/dsshome1/0F/di97pub/CollaborationLearning/

# --- environment setup ---
source ~/.bashrc
conda activate collab_learn

srun python3 -u run_experiment.py -c ./configs/chain.json