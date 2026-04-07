#!/bin/bash
# Train the Style Encoder (Step 1)
# Run this before training the denoising network.
#
# SLURM users: uncomment and adjust the #SBATCH directives below.
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --mem=400G
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=32
# #SBATCH --output=log/se_stdout-%j.log
# #SBATCH --error=log/se_stderr-%j.log

# --- Environment ---
# Activate your environment before running, e.g.:
# source activate myenv
# conda activate msmd

# --- Configuration ---
DATA_ROOT="data/"          # Root directory containing your processed datasets
EXP_NAME="se_msmd_flmae_25fps"
DATASET_TYPE="celebv-text-full+ravdess-FLMAE"
FPS=25
NUM_WORKERS=4
MAX_ITER=1000000
BATCH_SIZE=32
STATS_FILE="data/coef_stats.npz"   # Path to your coefficient statistics file

# Uncomment to resume from a checkpoint:
# CONTINUE_FROM="experiments/SE/${EXP_NAME}-YYMMDD_HHMMSS"

python train_se.py \
    --data_root ${DATA_ROOT} \
    --exp_name ${EXP_NAME} \
    --dataset_type ${DATASET_TYPE} \
    --fps ${FPS} \
    --num_workers ${NUM_WORKERS} \
    --max_iter ${MAX_ITER} \
    --batch_size ${BATCH_SIZE} \
    --use_normalization \
    --stats_file ${STATS_FILE}
    # --continue_from ${CONTINUE_FROM}
