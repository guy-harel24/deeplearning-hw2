#!/bin/bash
#SBATCH --job-name=exp2_depth
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
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



python -m hw2.experiments run-exp -n exp1_4  -K 32 64 128  -L 3  -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 32 64 128 -L 6 -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 32 64 128 -L 9 -P $P -H $H --batches $BATCHES --bs-train $BS
python -m hw2.experiments run-exp -n exp1_4  -K 32 64 128 -L 12 -P $P -H $H --batches $BATCHES --bs-train $BS