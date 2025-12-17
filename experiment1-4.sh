#!/bin/bash
#SBATCH --job-name=exp1_4_depth
#SBATCH --output=logs/exp1_4_%j.out
#SBATCH --error=logs/exp1_4_%j.err
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
P=8
H=100
BS=128
BATCHES=235

# ===== K = 32 =====
python -m hw2.experiments run-exp -n exp1_4  -K 32  -L 8  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 32  -L 16 -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 32  -L 32 -P $P -H $H --batches $BATCHES --bs-train $BS
# ===== K = 64,128,256 =====
python -m hw2.experiments run-exp -n exp1_4  -K 64 128 256  -L 2  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 64 128 256  -L 4  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 64 128 256  -L 8  -P $P -H $H --batches $BATCHES --bs-train $BS

