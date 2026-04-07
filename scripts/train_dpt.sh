#!/bin/bash
# Train DiffPoseTalk baseline (Step 2 - standard DPT)
# Requires a trained style encoder checkpoint from train_se.sh.
#
# SLURM users: uncomment and adjust the #SBATCH directives below.
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --mem=128G
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=32
# #SBATCH --output=log/dpt_stdout-%j.log
# #SBATCH --error=log/dpt_stderr-%j.log

# --- Environment ---
# source activate msmd

# --- Configuration ---
DATASET_TYPE="celebv-text-full+ravdess-FLMAE"
EXPNAME="msmd_dpt_run1"
STYLE_ENC_CKPT="experiments/SE/<YOUR_SE_EXP_DIR>/checkpoints/best.pt"  # <-- update this
DATA_ROOT="data/"
SCHEDULER="Warmup"
AUDIO_MODEL="hubert"
MAX_ITER=5000000
STYLE_ENC_MODEL_STYLE="diffposetalk"
GENERATOR_MODEL_STYLE="diffposetalk"
TRAINING_LOSS_STYLE="diffposetalk"
NUM_WORKERS=4
STATS_FILE="data/coef_stats.npz"

# Uncomment to resume:
# CONTINUE_FROM="experiments/dpt/${EXPNAME}-YYMMDD_HHMMSS"

python train_diffusion.py \
    --exp_name ${EXPNAME} \
    --data_root ${DATA_ROOT} \
    --use_indicator \
    --scheduler ${SCHEDULER} \
    --audio_model ${AUDIO_MODEL} \
    --style_enc_ckpt ${STYLE_ENC_CKPT} \
    --dataset_type ${DATASET_TYPE} \
    --max_iter ${MAX_ITER} \
    --num_workers ${NUM_WORKERS} \
    --use_vertex_space \
    --use_cross_style \
    --style_enc_model_style ${STYLE_ENC_MODEL_STYLE} \
    --generator_model_style ${GENERATOR_MODEL_STYLE} \
    --training_loss_style ${TRAINING_LOSS_STYLE} \
    --fps 25 \
    --use_normalization \
    --stats_file ${STATS_FILE}
    # --continue_from ${CONTINUE_FROM}
