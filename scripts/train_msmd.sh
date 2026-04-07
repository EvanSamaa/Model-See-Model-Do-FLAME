#!/bin/bash
# Train MSMD model (Step 2 - MSMD with static/dynamic decomposition + VAE style encoder)
# This is the main MSMD training script described in the paper.
#
# SLURM users: uncomment and adjust the #SBATCH directives below.
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --mem=400G
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=32
# #SBATCH --output=log/msmd_stdout-%j.log
# #SBATCH --error=log/msmd_stderr-%j.log

# --- Environment ---
# source activate msmd

# --- Configuration ---
EXPNAME="msmd_static_dynamic_vae_k4"
DATA_ROOT="data/"
SCHEDULER="Warmup"
AUDIO_MODEL="hubert"
STYLE_ENC_MODEL_STYLE="vae2"
GENERATOR_MODEL_STYLE="diffposetalk_static_dynamic_k_basis_no_head_alpha"
TRAINING_LOSS_STYLE="diffposetalk+vae"
DATASET_TYPE="celebv-text-full+ravdess-FLMAE"
NUM_WORKERS=4
D_STYLE=256
L_KL_DIV=1E-7
L_SMOOTH=1E1
MAX_ITER=2000000
BATCH_SIZE=16
NUM_OF_BASIS=4
PROB_CROSS_STYLE=0.5
STATS_FILE="data/coef_stats.npz"

# Uncomment to resume from a checkpoint:
# CONTINUE_FROM="experiments/dpt/${EXPNAME}-YYMMDD_HHMMSS"

python train_diffusion.py \
    --exp_name ${EXPNAME} \
    --data_root ${DATA_ROOT} \
    --use_indicator \
    --use_cross_style \
    --use_vertex_space \
    --batch_size ${BATCH_SIZE} \
    --num_of_basis ${NUM_OF_BASIS} \
    --scheduler ${SCHEDULER} \
    --audio_model ${AUDIO_MODEL} \
    --style_enc_model_style ${STYLE_ENC_MODEL_STYLE} \
    --generator_model_style ${GENERATOR_MODEL_STYLE} \
    --training_loss_style ${TRAINING_LOSS_STYLE} \
    --dataset_type ${DATASET_TYPE} \
    --num_workers ${NUM_WORKERS} \
    --d_style ${D_STYLE} \
    --l_kl_div ${L_KL_DIV} \
    --l_smooth ${L_SMOOTH} \
    --max_iter ${MAX_ITER} \
    --fps 25 \
    --prob_cross_style ${PROB_CROSS_STYLE} \
    --use_normalization \
    --stats_file ${STATS_FILE}
    # --continue_from ${CONTINUE_FROM}
