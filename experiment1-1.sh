#!/bin/bash
#SBATCH --job-name=exp1_1_depth
#SBATCH --output=logs/exp1_1_%j.out
#SBATCH --error=logs/exp1_1_%j.err
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

# ===== K = 32 =====
python -m hw2.experiments run-exp -n exp1_1_L2_K32  -K 32 -L 2  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L4_K32  -K 32 -L 4  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L8_K32  -K 32 -L 8  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L16_K32 -K 32 -L 16 -P $P -H $H --batches $BATCHES --bs-train $BS
# ===== K = 64 =====
python -m hw2.experiments run-exp -n exp1_1_L2_K64  -K 64 -L 2  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L4_K64  -K 64 -L 4  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L8_K64  -K 64 -L 8  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_1_L16_K64 -K 64 -L 16 -P $P -H $H --batches $BATCHES --bs-train $BS