# MSMD: Motion Style and Motion Diffusion

Official code release for the MSMD paper.

MSMD is a speech-driven 3D facial animation system that generates stylistic, diverse FLAME motion coefficients from audio. It builds on [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk) with a static/dynamic motion decomposition and a VAE-based style encoder.

---

## Repository Structure

```
MSMD/
├── msmd/                    # Main package
│   ├── models/              # DiffTalkingHead, StyleEncoder, FLAME
│   ├── data/                # Dataset loaders and factory
│   ├── utils/               # Loss functions, utilities
│   └── options/             # Argument parsers
├── train_diffusion.py       # Train the denoising network (MSMD / DPT)
├── train_se.py              # Train the style encoder
├── inference.py             # Standalone inference script
├── scripts/
│   ├── train_msmd.sh        # Training script for MSMD model
│   ├── train_dpt.sh         # Training script for DiffPoseTalk baseline
│   └── train_se.sh          # Training script for style encoder
├── data/                    # Put your datasets here
└── pretrained_models/
    └── flame/               # FLAME model files (see below)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Install PyTorch separately following the [official guide](https://pytorch.org/get-started/locally/) to match your CUDA version.

---

### 0. Pretrained Models Setup
can download the pretrained model from here
https://drive.google.com/drive/folders/1rxyGPbro5uLAayhKOsSDhBH8dAq9W_20?usp=sharing

### 1. FLAME Model Files

Download the following files from the [FLAME official website](https://flame.is.tue.mpg.de/) (requires free registration):

- `generic_model.pkl` — from **FLAME 2020**
- `FLAME_albedo_from_BFM.npz` — from the FLAME texture model
- `landmark_embedding.npy` — from the FLAME landmark embedding

Download `FLAME_masks.pkl` from the [FLAME masks repository](https://github.com/TimoBolkart/FLAME_masks) (used by the evaluation scripts).

Place all files in `pretrained_models/flame/`:

```
pretrained_models/flame/
├── generic_model.pkl
├── FLAME_albedo_from_BFM.npz
├── landmark_embedding.npy
└── FLAME_masks.pkl
```

### 2. Audio Encoder (HuBERT)

The HuBERT encoder (`facebook/hubert-base-ls960`) is downloaded automatically from HuggingFace on first use and cached in `~/.cache/huggingface/`.

### 3. Trained MSMD Checkpoints

Download from [TODO: add link]. Place each experiment directory under `pretrained_models/checkpoints/`:

```
pretrained_models/checkpoints/
└── MSMD/
    ├── args.json
    └── checkpoints/
        └── iter_XXXXXX.pt
```

---

## Data Setup

Place your processed datasets under `data/`. The expected structure depends on the dataset type:

```
data/
├── coef_stats.npz                          # Coefficient statistics (mean/std)
├── FLAME_celebv-text/                      # CelebV-Text processed data
│   └── processed_flame_param/
│       └── flame_param_merged.lmdb/
└── FLAME_mead_ravdess/                     # MEAD + RAVDESS processed data
    ├── train_*.pickle
    └── val_*.pickle
```

See dataset preprocessing scripts in the original [DiffPoseTalk repository](https://github.com/DiffPoseTalk/DiffPoseTalk) for instructions on preparing `FLAME_celebv-text` and related datasets.

---

## Training

Training is a two-step process:

### Step 1: Train the Style Encoder

```bash
bash scripts/train_se.sh
```

Edit `scripts/train_se.sh` to set your `DATA_ROOT` and `STATS_FILE`. The checkpoint will be saved to `experiments/SE/<exp_name>-<timestamp>/`.

### Step 2: Train the Denoising Network

**MSMD model (paper):**
```bash
bash scripts/train_msmd.sh
```

**DiffPoseTalk baseline:**
```bash
bash scripts/train_dpt.sh
```

Edit the scripts to set `STYLE_ENC_CKPT` to the best checkpoint from Step 1. Checkpoints are saved to `experiments/dpt/<exp_name>-<timestamp>/`.

You can also run training directly:
```bash
python train_diffusion.py \
    --exp_name my_experiment \
    --data_root data/ \
    --dataset_type celebv-text-full+ravdess-FLMAE \
    --audio_model hubert \
    --fps 25 \
    --use_normalization \
    --stats_file data/coef_stats.npz \
    --help  # to see all options
```

---

## Inference

Run inference on a single audio file:

```bash
python inference.py \
    --model_dir pretrained_models/checkpoints/MSMD \
    --audio path/to/audio.wav \
    --output output_coefficients.npy \
    --device cuda
```

Optional arguments:
- `--checkpoint 0180000` — specify a checkpoint iteration (defaults to latest)
- `--shape_coef shape.npy` — FLAME shape coefficients [100,] (defaults to neutral)
- `--style_coef style_motion.npy` — motion sequence for style reference [T, D]
- `--cfg_scale 1.15` — classifier-free guidance scale

The output `.npy` file contains motion coefficients of shape `[n_frames, coef_dim]` where `coef_dim` is typically 67 (100 expression + 3 jaw, reduced for CelebV-Text models).

---

## Download Pretrained Model

You can download the pretrained model from here:


## Evaluation (Benchmark)

To run evaluation:

```bash
python benchmark_inference_for_eval.py \
    --model_dir experiments/dpt/<exp_name> \
    --checkpoint 0180000 \
    --dataset_type celebv-text-flmae-full \
    --data_root data/ \
    --stats_file data/coef_stats.npz \
    --output_dir results/
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{msmd2025,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2025},
}
```

This work builds on [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk):

```bibtex
@inproceedings{diffposetalk2024,
  title={DiffPoseTalk: Speech-Driven Stylistic 3D Facial Animation and Head Pose Generation via Diffusion Models},
  author={Sun, Zhiyao and Lv, Tian and Ye, Sheng and Lin, Matthieu Gaetan and Sheng, Jenny and Wen, Yu-Hui and Yu, Minjing and Liu, Yong-Jin},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  year={2024},
}
```

---

## License

See [LICENSE](LICENSE) for details. The FLAME model is subject to its own [license](https://flame.is.tue.mpg.de/modellicense.html).
