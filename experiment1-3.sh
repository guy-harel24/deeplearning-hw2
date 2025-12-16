#!/bin/bash
#SBATCH --job-name=exp1_3_depth
#SBATCH --output=logs/exp1_3_%j.out
#SBATCH --error=logs/exp1_3_%j.err
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guy.harel24@gmail.com


# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw

# Safety: stop on error
 set -e

# Shared hyperparameters
P=4
H=100
BS=128
BATCHES=235

# ===== k = [64,128] =====
python -m hw2.experiments run-exp -n exp1_3  -K 64 128 -L 2  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_3  -K 64 128 -L 3  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_3  -K 64 128 -L 4  -P $P -H $H --batches $BATCHES --bs-train $BS
# Other runs already covered in experiment1-1.sh and experiment1-2.sh