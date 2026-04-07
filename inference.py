"""
MSMD Inference Script

Run inference on a single audio file to generate FLAME motion coefficients.

Usage:
    python inference.py \
        --model_dir pretrained_models/checkpoints/MSMD \
        --audio path/to/audio.wav \
        --output output_coefficients.npy \
        --device cuda

The model_dir should contain:
    - args.json          (saved training arguments)
    - checkpoints/       (model checkpoint files, e.g. iter_XXXXXX.pt)
"""

import argparse
import copy
import gc
import json
import math
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import msmd.options.dpt as dpt_options
import msmd.utils as utils
from msmd.models.diff_talking_head import get_difftalkinghead_model
from msmd.models.style_encoder import get_style_encoder
from msmd.models.flame import FLAME, FLAMEConfig
from msmd.models.common import load_args


# ---------------------------------------------------------------------------
# Compat shims for older libs (chumpy etc.)
# ---------------------------------------------------------------------------
import inspect
from collections import namedtuple
import numpy as _np

_aliases = [
    ('bool', _np.bool_),
    ('int', int),
    ('float', float),
    ('complex', complex),
    ('object', object),
    ('unicode', str),
    ('str', str),
]
for _name, _target in _aliases:
    if not hasattr(_np, _name):
        setattr(_np, _name, _target)

if not hasattr(inspect, 'getargspec'):
    def _shim_getargspec(func):
        fas = inspect.getfullargspec(func)
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        return ArgSpec(fas.args, fas.varargs, fas.varkw, fas.defaults)
    inspect.getargspec = _shim_getargspec


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_style_code(style_enc, motion_coeff, model_args):
    """Extract style code from a motion coefficient sequence."""
    enc_style = getattr(model_args, 'style_enc_model_style', 'diffposetalk')
    if enc_style == 'vae2_with_lip_stats':
        return style_enc(motion_coeff[:, :100, :], {})
    elif enc_style is not None and enc_style[:3] == 'vae':
        result = style_enc(motion_coeff[:, :100, :])
        return result[0] if isinstance(result, tuple) else result
    else:
        return style_enc(motion_coeff[:, :100, :])


@torch.no_grad()
def infer_coefficients(
    model,
    args,
    audio: torch.Tensor,
    shape_coef: torch.Tensor,
    audio_unit: float = 640.0,
    style_feat=None,
    cfg_mode=None,
    cfg_cond=None,
    cfg_scale: float = 1.0,
    dynamic_threshold=None,
) -> torch.Tensor:
    """
    Run autoregressive inference over the full audio sequence.

    Args:
        model:        Loaded DiffTalkingHead model (eval mode, on device).
        args:         Namespace from load_args (contains fps, n_motions, etc.).
        audio:        Float tensor of shape [1, audio_samples] at 16 kHz.
        shape_coef:   Float tensor of shape [1, 100] (FLAME shape coefficients).
        audio_unit:   Audio samples per motion frame (640 for 25 fps at 16 kHz).
        style_feat:   Style feature tensor or list of tensors.
        cfg_mode:     Classifier-free guidance mode (None to disable).
        cfg_cond:     CFG condition list ([] to disable).
        cfg_scale:    CFG scale factor.
        dynamic_threshold: Dynamic thresholding tuple or None.

    Returns:
        Motion coefficient tensor of shape [1, n_frames, coef_dim].
    """
    clip_len = int(audio.shape[1] / 16000 * args.fps)
    stride = args.n_motions
    n_audio_samples = round(audio_unit * args.n_motions)
    n_subdivision = 1 if clip_len <= args.n_motions else math.ceil(clip_len / stride)
    n_padding_audio_samples = n_audio_samples * n_subdivision - audio.shape[1]
    n_padding_frames = math.ceil(n_padding_audio_samples / audio_unit) if n_padding_audio_samples > 0 else 0

    if n_padding_audio_samples > 0:
        audio = F.pad(audio, (0, n_padding_audio_samples), value=0)

    audio_feat = model.extract_audio_feature(audio, args.n_motions * n_subdivision)
    coef_list = []
    prev_motion_feat = None
    prev_audio_feat = None
    noise = None

    for i in range(n_subdivision):
        if isinstance(style_feat, list):
            sf = style_feat[i]
        else:
            sf = style_feat

        start_idx = i * stride
        indicator = (
            torch.ones((audio_feat.shape[0], args.n_motions)).to(model.device)
            if args.use_indicator else None
        )
        if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
            indicator[:, -n_padding_frames:] = 0

        audio_in = audio_feat[:, start_idx:start_idx + args.n_motions]

        if i == 0:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in, shape_coef, sf,
                indicator=indicator,
                cfg_mode=cfg_mode, cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold,
            )
        else:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in, shape_coef, sf,
                prev_motion_feat, prev_audio_feat, noise,
                indicator=indicator,
                cfg_mode=cfg_mode, cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold,
            )

        prev_motion_feat = motion_feat[:, -args.n_prev_motions:].clone()
        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]

        motion_coef = motion_feat
        if i == n_subdivision - 1 and n_padding_frames > 0:
            motion_coef = motion_coef[:, :-n_padding_frames]
        coef_list.append(motion_coef)

    return torch.cat(coef_list, dim=1)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_and_style_encoder(model_dir: Path, checkpoint: Optional[str], device: str):
    """
    Load DiffTalkingHead model and style encoder from an experiment directory.

    Args:
        model_dir:   Path to the experiment directory (contains args.json + checkpoints/).
        checkpoint:  Checkpoint iteration string (e.g. '0190000'). If None, loads latest.
        device:      'cuda' or 'cpu'.

    Returns:
        (model, style_enc, model_args)
    """
    model_args = load_args(model_dir)
    model = get_difftalkinghead_model(model_args).to(device)

    checkpoints_dir = model_dir / 'checkpoints'
    if checkpoint is not None:
        ckpt_path = checkpoints_dir / f'iter_{checkpoint}.pt'
    else:
        ckpt_files = sorted(checkpoints_dir.glob('iter_*.pt'))
        if not ckpt_files:
            raise FileNotFoundError(f'No checkpoints found in {checkpoints_dir}')
        ckpt_path = ckpt_files[-1]

    model_data = torch.load(ckpt_path, map_location=device, weights_only=False)

    enc_style = getattr(model_args, 'style_enc_model_style', 'diffposetalk')
    if enc_style is None:
        enc_style = 'diffposetalk'

    if enc_style == 'diffposetalk':
        style_enc_ckpt = getattr(model_args, 'style_enc_ckpt', None)
        if style_enc_ckpt and os.path.exists(style_enc_ckpt):
            enc_model_data = torch.load(style_enc_ckpt, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(
                f'Style encoder checkpoint not found: {style_enc_ckpt}\n'
                f'Pass --style_enc_ckpt explicitly or use a MSMD-style checkpoint.'
            )
        enc_model_args = utils.NullableArgs(enc_model_data['args'])
        style_enc = get_style_encoder(enc_model_args).to(device)
        style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
        model.load_state_dict(model_data['model'])
    else:
        style_enc = get_style_encoder(model_args, enc_style).to(device)
        style_enc.load_state_dict(model_data['style_enc'])
        model.load_state_dict(model_data['model'])

    model.eval()
    style_enc.eval()
    return model, style_enc, model_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MSMD inference: audio → FLAME motion coefficients')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to experiment directory (contains args.json and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint iteration string, e.g. "0190000". Defaults to latest.')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file (.wav at 16 kHz)')
    parser.add_argument('--shape_coef', type=str, default=None,
                        help='Path to .npy file containing shape coefficients [100,]. '
                             'If not provided, uses zeros (neutral shape).')
    parser.add_argument('--style_coef', type=str, default=None,
                        help='Path to .npy file with motion sequence to use as style reference '
                             '[T, coef_dim]. If not provided, infers from identity motion.')
    parser.add_argument('--output', type=str, default='output_coefficients.npy',
                        help='Output path for motion coefficients (.npy or .json)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale (1.0 = disabled)')
    parser.add_argument('--audio_unit', type=float, default=640.0,
                        help='Audio samples per motion frame (default: 640 = 16000/25)')
    args = parser.parse_args()

    device = args.device
    model_dir = Path(args.model_dir)

    # --- Load model ---
    print(f'Loading model from {model_dir}...')
    model, style_enc, model_args = load_model_and_style_encoder(model_dir, args.checkpoint, device)
    print('Model loaded.')

    # --- Load audio ---
    import librosa
    print(f'Loading audio from {args.audio}...')
    audio_np, sr = librosa.load(args.audio, sr=16000, mono=True)
    audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
    print(f'Audio: {audio.shape[1]} samples ({audio.shape[1]/16000:.2f} s)')

    # --- Load shape coefficients ---
    if args.shape_coef is not None:
        shape_np = np.load(args.shape_coef)
        shape_coef = torch.tensor(shape_np, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        print('No shape coefficients provided, using neutral (zeros).')
        shape_coef = torch.zeros(1, 100, dtype=torch.float32).to(device)

    # --- Compute style feature ---
    if args.style_coef is not None:
        style_np = np.load(args.style_coef)  # [T, coef_dim]
        style_motion = torch.tensor(style_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, D]
        print(f'Using provided style reference: {style_motion.shape}')
        style_feat = infer_style_code(style_enc, style_motion, model_args)
    else:
        # Use a zero motion sequence as neutral style
        n_motions = model_args.n_motions
        coef_dim = 67  # default for celebv-text models (100 exp + 3 jaw + head reduced)
        try:
            coef_dim = model_args.coef_dim
        except AttributeError:
            pass
        style_motion = torch.zeros(1, n_motions, coef_dim, dtype=torch.float32).to(device)
        style_feat = infer_style_code(style_enc, style_motion, model_args)
        print('Using neutral style (zero motion).')

    # --- Run inference ---
    print('Running inference...')
    with torch.no_grad():
        motion_coef = infer_coefficients(
            model=model,
            args=model_args,
            audio=audio,
            shape_coef=shape_coef,
            audio_unit=args.audio_unit,
            style_feat=style_feat,
            cfg_mode=None,
            cfg_cond=[],
            cfg_scale=args.cfg_scale,
            dynamic_threshold=None,
        )
    print(f'Generated motion coefficients: {motion_coef.shape}')

    # --- Save output ---
    output_path = Path(args.output)
    result = motion_coef.squeeze(0).cpu().numpy()  # [n_frames, coef_dim]

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump({'motion_coeff': result.tolist(), 'n_frames': result.shape[0]}, f)
    else:
        np.save(output_path, result)

    print(f'Saved to {output_path}  [{result.shape[0]} frames × {result.shape[1]} dims]')

    # Cleanup
    del audio, shape_coef, style_feat, motion_coef
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
