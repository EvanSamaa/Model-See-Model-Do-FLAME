import argparse
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import os
import gc
import numpy as np
import torch
import torch.optim as optim
import pickle as pkl
from torch.utils import data
from tqdm import tqdm
import math 
from scipy.io import wavfile
import json
import msmd.options.dpt as options
import msmd.utils as utils
from msmd.data import LmdbDataset, infinite_data_loader
from msmd.models import DiffTalkingHead, StyleEncoder
from msmd.models.diff_talking_head import DiffTalkingHead_celebv_text, get_difftalkinghead_model
from msmd.models.style_encoder import StyleEncoder_celebv, get_style_encoder, StyleEncoder_VAE
from msmd.data.datasets import PickleDataset_Evan, get_dataset, get_dataset_lmdb
from msmd.models.flame import FLAMEConfig
from msmd.models.flame_utils import FLAME
from msmd.models.common import load_args, load_pretrained_model, save_args, load_args_with_defaults
from msmd.utils.common import compute_loss_no_vert, compute_loss, StyleAdherenceLoss
from msmd.utils.common import compute_KL_loss, compute_loss_espnet
import librosa
# load the numpy to alembic pipeline

from msmd.data.dataset_factory import create_datasets, get_available_dataset_types


class TrainingLogger:
    """
    A flexible logging abstraction that supports TensorBoard (if available) 
    and falls back to JSON file logging.
    """
    def __init__(self, log_dir: Path, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scalars = defaultdict(list)
        self.texts = {}
        self.tensorboard_writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(str(self.log_dir))
                print(f"TensorBoard logging enabled at {self.log_dir}")
            except ImportError:
                print("TensorBoard not available, falling back to JSON logging only")
        
    def add_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.scalars[tag].append({'step': step, 'value': value})
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(tag, value, step)
    
    def add_text(self, tag: str, text: str):
        """Log text content."""
        self.texts[tag] = text
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag, text)
    
    def flush(self):
        """Flush any buffered data."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
        # Also save to JSON as backup
        self._save_json()
    
    def _save_json(self):
        """Save logged data to JSON file."""
        json_path = self.log_dir / 'training_log.json'
        log_data = {
            'scalars': {k: v for k, v in self.scalars.items()},
            'texts': self.texts
        }
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def close(self):
        """Close the logger and save all data."""
        self.flush()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()


def print_GPU_usage():
    device_id = torch.cuda.current_device()
    free_memory, total_memory = torch.cuda.mem_get_info(device_id)
    # Calculate used memory
    used_memory = total_memory - free_memory
    print(f"Total GPU memory: {total_memory / 1024**2:.2f} MB")
    print(f"Free GPU memory: {free_memory / 1024**2:.2f} MB")
    print(f"Used GPU memory: {used_memory / 1024**2:.2f} MB")

def clear_cuda_cache():
    """Helper function to clear CUDA cache and delete tensors"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def train(args, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], train_loader, val_loader, optimizer,
          save_dir, scheduler=None, logger=None, flame=None, out_abc_dir=None, start_iter=0):
    
    loss_weights = load_loss_weights(args)
    device = model.device
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    data_loader = infinite_data_loader(train_loader)
    if len(args.dataset_type.split("+")) <= 1:
        dataset = train_loader.dataset
    else:
        dataset = train_loader.dataset.datasets[0]
    coef_stats = dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    pbar = tqdm(range(start_iter, args.max_iter + 1), initial=start_iter, total=args.max_iter + 1, dynamic_ncols=True)
    
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    for it in pbar:
        audio_pair, coef_pair, audio_stats = next(data_loader)   
        clear_cuda_cache()
        audio_pair = [audio.to(device) for audio in audio_pair]
        # just cast all params to device (i.e. cuda)
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)] # the coefficients are [shape (T, 100), exp (T, 64), pose (T, 3)]

        if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess':
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]  # (N, L, 50+x)
        else:
            motion_coef_pair = [coef_pair[0]["motion"], coef_pair[1]["motion"]]
        # Use the shape coefficients from the first frame of the first clip as the condition
        if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
            shape_coef = coef_pair[0]['shape'].clone().to(device)
        elif coef_pair[0]['shape'].ndim == 1:  # (N, L, 100)
            shape_coef = coef_pair[0]['shape'].unsqueeze(0).clone().to(device)
        else:  # (N, L, 100)
            shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

        # Extract style features ( this is not frozen if we don't use diffposetalk style style encoder)
        if style_enc is not None:
            if args.style_enc_model_style == 'diffposetalk':
                with torch.no_grad():
                    style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
            elif args.style_enc_model_style[:18] == 'model_see_model_do':
                style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
            elif args.style_enc_model_style == "vae2_with_lip_stats":
                style_mu_logvar_pair = [style_enc(motion_coef_pair[i], coef_pair[i]) for i in range(2)]
                style_pair = [x[0] for x in style_mu_logvar_pair]
                mu_pair = [x[1] for x in style_mu_logvar_pair]
                logvar_pair = [x[2] for x in style_mu_logvar_pair]
            elif args.style_enc_model_style == 'vae2_with_audio_feat':
                input_audio_feature = model.extract_audio_768_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)
                input_audio_feature_pair = [input_audio_feature[:, i * args.n_motions:(i + 1) * args.n_motions] for i in range(2)]
                input_feature_pair = [torch.cat([motion_coef_pair[i], input_audio_feature_pair[i]], dim=2) for i in range(2)]
                style_mu_logvar_pair = [style_enc(input_feature_pair[i]) for i in range(2)]
                style_pair = [x[0] for x in style_mu_logvar_pair]
                mu_pair = [x[1] for x in style_mu_logvar_pair]
                logvar_pair = [x[2] for x in style_mu_logvar_pair]
            elif args.style_enc_model_style[:3] == 'vae':
                style_mu_logvar_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                style_pair = [x[0] for x in style_mu_logvar_pair]
                mu_pair = [x[1] for x in style_mu_logvar_pair]
                logvar_pair = [x[2] for x in style_mu_logvar_pair]
            elif (args.style_enc_model_style == 'basic_encoder' 
                  or args.style_enc_model_style == 'no_style' 
                  or args.style_enc_model_style == 'fft_encoder'
                  or args.style_enc_model_style[:25] == 'basic_encoder_transformer'
                  or args.style_enc_model_style == 'gst'):
                style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
            else:
                raise ValueError(f"Style Encoder Model style {args.style_enc_model_style} not recognized")
        if args.use_context_audio_feat:
            # Extract audio features
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

        # construct losses
        losses = {}
        for key in loss_weights:
            losses[key] = torch.tensor(0.0, device=device)

        # compute motion coefficients for each clip
        style_pair[0].shape

        for i in range(2):
            # i=0
            audio = audio_pair[i]  # (N, L_a)
            motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
            style = style_pair[i] if style_enc is not None else None
            if args.use_cross_style:
                if args.prob_cross_style == 1.0:
                    # we use cross style
                    style = style_pair[1 - i]
                else:
                    # we randomly choose the style to be crossed vs not crossed, not this is only in training, not in validation
                    style = []
                    for batch_sample_i in range(style_pair[i].shape[0]):
                        if np.random.rand() < args.prob_cross_style:
                            style.append(style_pair[1 - i][batch_sample_i])
                        else:
                            style.append(style_pair[i][batch_sample_i])
                    style = torch.stack(style, dim=0)
            batch_size = audio.shape[0]
            # truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode, expression_code_size=64)
                if args.use_context_audio_feat and i != 0:
                    # use contextualized audio feature for the second clip
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
            else:
                if args.use_context_audio_feat:
                    audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            if args.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(
                        1)
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # run model for the first frame
            if args.do_ignore_shape:
                input_shape_coef = torch.zeros_like(shape_coef)
            else:
                input_shape_coef = shape_coef
            
            use_CFG_during_training = True
            if args.do_ignore_cfg:
                use_CFG_during_training = False

            if i == 0:
                # if it's the first frame in the pair
                if args.use_static_loss:
                    noise, target, prev_motion_coef, prev_audio_feat, dynamic_features, static_features, alpha_t = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training, keep_separate=True)
                elif args.training_loss_style == "model_see_model_do":
                    noise, target, stylized_target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                elif args.training_loss_style == "model_see_model_do_ver2":
                    noise, target, training_guide, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                else:
                    noise, target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                # If it's second frame in the pair
                if args.use_static_loss:
                    noise, target, _, _, dynamic_features, static_features, alpha_t = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training, keep_separate=True)
                elif args.training_loss_style == "model_see_model_do":
                    noise, target, stylized_target, _, _ = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                elif args.training_loss_style == "model_see_model_do_ver2":
                    noise, target, training_guide, _, _ = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                else:
                    noise, target, _, _ = model(motion_coef_in, audio_in, input_shape_coef, style,
                                                prev_motion_coef, prev_audio_feat, indicator=indicator, train_with_CFG=use_CFG_during_training)
            torch.cuda.empty_cache()
            # 5GB of vram is used here (at a batch size of 16)


            # compute losses
            if args.training_loss_style == 'diffposetalk' or args.training_loss_style == 'diffposetalk+vae':
                
                if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == "flame_mead_ravdess" or args.dataset_type == "celebv-text-full+ravdess-FLMAE" or args.dataset_type == "celebv-text-medium+ravdess-FLMAE"):
                    loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                elif args.use_vertex_space:
                    # TODO: this needs implementation, compute_loss_espnet is also not implemented
                    # model = model.to("cpu")
                    torch.cuda.empty_cache()
                    seq_vertices = convert_sequence(target[:, args.n_prev_motions:], flame["neutral"], 
                                        (flame["expression2vert_mean"], flame["expression2vert_std"]), 
                                        flame["expression2vert_model"], device, 16)
                    with torch.no_grad():
                        gt_vertices = convert_sequence(motion_coef_in[:, :], flame["neutral"],
                                            (flame["expression2vert_mean"], flame["expression2vert_std"]),
                                            flame["expression2vert_model"], device, 16)
                    gt_vertices = gt_vertices.to(device)
                    seq_vertices = seq_vertices.to(device)
                    # print the memory size of flame["neutral"]
                    # model.to(device)
                    loss_dict = compute_loss_espnet(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, gt_vertices, seq_vertices, end_idx, return_dict=True)
                else:
                    loss_dict = compute_loss_no_vert(
                        args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
            elif args.training_loss_style == "model_see_model_do":
                if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                    loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    selector_loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, stylized_target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    for key in selector_loss_dict:
                        loss_dict["stylized_" + key] = selector_loss_dict[key]
                else:
                    raise ValueError(f"unknown config not recognized during loss computation")
            elif args.training_loss_style == "model_see_model_do_ver2":
                if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                    loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    style_guide_loss = torch.nn.functional.mse_loss(training_guide[:, args.n_prev_motions:], motion_coef_in)
                    loss_dict["style_guide_recon"] = style_guide_loss
                else:
                    raise ValueError(f"unknown config not recognized during loss computation")
            else:
                raise ValueError(f"Generator Model style {args.generator_model_style} not recognized during loss computation")

            if args.style_enc_model_style[:3] == 'vae':
                kl_loss = compute_KL_loss(mu_pair[i], logvar_pair[i])
                loss_dict['kl_div'] = kl_loss
            if args.use_style_adherence_loss:
                if args.use_cross_style:
                    style_adherence_loss = StyleAdherenceLoss()(target, motion_coef_pair[1-i])
                else:
                    style_adherence_loss = StyleAdherenceLoss()(target, motion_coef_pair[i])
                loss_dict['style_adherence'] = style_adherence_loss
            if args.use_static_loss:
                static_loss = torch.nn.functional.mse_loss(static_features[:, args.n_prev_motions:], motion_coef_in)
                loss_dict['static_loss'] = static_loss
            for key in loss_dict:
                # note that head_trans could be None in the case of a single frame
                if loss_weights[key] > 0 and loss_dict[key] is not None:
                    losses[key] += loss_dict[key]

        # aggregate losses
        if args.training_loss_style == "model_see_model_do_two_streams":
            loss1 = 0
            loss2 = 0
            for key in losses:
                if loss_weights[key] > 0:
                    loss_log[key].append(losses[key].item())
                    if key[:8] == "stylized":
                        loss2 += losses[key] * loss_weights[key]
                    else:
                        loss1 += losses[key] * loss_weights[key]
            scaling_factor = 0.05 
            loss2.backward(retain_graph=True) # this affects all components
            for param in model.denoising_net.parameters():
                if param.grad is not None and param.requires_grad:
                    param.grad *= scaling_factor
            for param in model.audio_encoder.parameters():
                if param.grad is not None and param.requires_grad:
                    param.grad *= scaling_factor
            loss1.backward() # this affects only the reconstruction loss
            loss_log["loss"].append(loss1.item() + loss2.item())
            print("we are processing two stream of losses")
            del loss1, loss2
        else:
            loss = 0
            for key in losses:
                if loss_weights[key] > 0:
                    loss_log[key].append(losses[key].item())
                    loss += losses[key] * loss_weights[key]
            model.to(device)
            loss.backward()
            loss_log['loss'].append(loss.item())
            del loss

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging in the progress bar
        description = 'Train loss: ['
        for key in loss_log:
            if key == "loss":
                description += f'{key}: {np.mean(loss_log[key]):.3e}, '
            if key != 'loss' and loss_weights[key] > 0:
                if loss_weights[key] > 0:
                    description += f'{key}: {np.mean(loss_log[key]):.3e}, '
        description += ']'
        pbar.set_description(description)

        # log to logger (TensorBoard or JSON)
        if it % args.log_iter == 0 and logger is not None:
            logger.add_scalar('train/loss', np.mean(loss_log['loss']), it)
            for key in loss_log:
                if key != 'loss' and loss_weights[key] > 0:
                    logger.add_scalar(f'train/{key}', np.mean(loss_log[key]), it)
            logger.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)
                       
        # update learning rate
        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # save model
        if (it % args.save_iter == 0 and it != 0 and it != start_iter) or it == args.max_iter or it == 10 or it == 50:
            if args.style_enc_model_style == 'diffposetalk' and args.generator_model_style[:12] == 'diffposetalk':
                torch.save({
                    'args': args,
                    'model': model.state_dict(),
                    'iter': it,
                }, save_dir / f'iter_{it:07}.pt')
            else:
                torch.save({
                    'args': args,
                    'model': model.state_dict(),
                    'style_enc': style_enc.state_dict(),
                    'iter': it,
                }, save_dir / f'iter_{it:07}.pt')
        
        del audio_pair, coef_pair, motion_coef_pair
        clear_cuda_cache()
        del losses, style_pair
        clear_cuda_cache()

        # validation
        if (it % args.val_iter == 0 and it != 0 and it != start_iter) or it == args.max_iter or it == 10 or it == 50:
            test(args, loss_weights, model, style_enc, val_loader, it, 1, 'val', logger, flame, out_abc_dir=out_abc_dir)

@torch.no_grad()
def infer_coeffs(
    model,
    args,
    audio,
    shape_coef,
    audio_unit, 
    style_feat=None,
    n_repetitions: int = 1,
    cfg_mode = None,
    cfg_cond = None,
    cfg_scale: float = 1.15,
    include_shape: bool = False,
):
    dynamic_threshold = (0, 1, 4)
    clip_len = int(len(audio) / 16000 * args.fps)
    stride = args.n_motions
    n_audio_samples = round(audio_unit * args.n_motions)
    n_subdivision = 1 if clip_len <= args.n_motions else math.ceil(clip_len / stride) # this is the number of windows to split everything into
    n_padding_audio_samples = n_audio_samples * n_subdivision - len(audio)
    n_padding_frames = math.ceil(n_padding_audio_samples / audio_unit)
    print(audio.shape, n_padding_frames)
    if n_padding_audio_samples > 0:
        padding_value = 0
        audio = torch.nn.functional.pad(audio, (0, n_padding_audio_samples), value=padding_value)    
    audio_feat = model.extract_audio_feature(audio.unsqueeze(0), args.n_motions * n_subdivision)
    coef_list = []

    # ############# test #############
    # n_repetitions = 1
    # cfg_mode = None
    # cfg_cond = None
    # cfg_scale = 1.15
    # ############# test #############
    
    for i in range(n_subdivision):
        start_idx = i * stride
        end_idx = start_idx + args.n_motions
        indicator = torch.ones((n_repetitions, args.n_motions)).to(model.device) if args.use_indicator else None
        if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
            indicator[:, -n_padding_frames:] = 0
        audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1)
        
        # Generate motion coefficients
        if i == 0:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in,
                shape_coef,
                style_feat,
                indicator=indicator,
                cfg_mode=cfg_mode,
                cfg_cond=cfg_cond,
                cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold
            )
        else:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in,
                shape_coef,
                style_feat,
                prev_motion_feat,
                prev_audio_feat,
                noise,
                indicator=indicator,
                cfg_mode=cfg_mode,
                cfg_cond=cfg_cond,
                cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold
            )
        prev_motion_feat = motion_feat[:, -args.n_prev_motions:].clone()
        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
        
        motion_coef = motion_feat
        if i == n_subdivision - 1 and n_padding_frames > 0:
            motion_coef = motion_coef[:, :-n_padding_frames]
        coef_list.append(motion_coef)    
    motion_coef = torch.cat(coef_list, dim=1)
    return motion_coef

@torch.no_grad()
def test(args, loss_weights, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], test_loader, current_iter, n_rounds=10,
         mode='val', logger=None, flame=None, out_abc_dir=None, do_render=False, do_save=False, do_save_path=None, do_ignore_style=False, do_compute_style_adherence_loss=False):
    # note that do_render will significantly slow down the process
    is_training = model.training
    device = model.device
    model.eval()
    if len(args.dataset_type.split("+")) <= 1:
        dataset = test_loader.dataset
    else:
        dataset = test_loader.dataset.datasets[0]
    coef_stats = dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    

    loss_log = defaultdict(list)
    for test_round in range(n_rounds):
        for audio_pair, coef_pair, audio_stats in test_loader:
            # audio_pair, coef_pair, audio_stats = next(infinite_data_loader(val_loader))
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess':
                motion_coef_pair = [
                    utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
                ]  # (N, L, 50+x)
            else:
                motion_coef_pair = [coef_pair[0]["motion"], coef_pair[1]["motion"]]
                
            # Use the shape coefficients from the first frame of the first clip as the condition
            if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
                shape_coef = coef_pair[0]['shape'].clone().to(device)
            else:  # (N, L, 100)
                shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

            # Extract style features
            if style_enc is not None:
                if args.style_enc_model_style == 'diffposetalk':
                    with torch.no_grad():
                        if do_ignore_style: # ignore the style, using mean style instead
                            mean_motion_coef_pair = [torch.zeros_like(motion_coef) for motion_coef in motion_coef_pair]
                            style_pair = [style_enc(mean_motion_coef_pair[i]) for i in range(2)]
                        else:
                            style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                elif args.style_enc_model_style[:18] == 'model_see_model_do':
                    style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                elif args.style_enc_model_style == "vae2_with_lip_stats":
                    style_mu_logvar_pair = [style_enc(motion_coef_pair[i], coef_pair[i]) for i in range(2)]
                    style_pair = [x[0] for x in style_mu_logvar_pair]
                    mu_pair = [x[1] for x in style_mu_logvar_pair]
                    logvar_pair = [x[2] for x in style_mu_logvar_pair]
                elif args.style_enc_model_style == 'vae2_with_audio_feat':
                    input_audio_feature = model.extract_audio_768_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)
                    input_audio_feature_pair = [input_audio_feature[:, i * args.n_motions:(i + 1) * args.n_motions] for i in range(2)]
                    input_feature_pair = [torch.cat([motion_coef_pair[i], input_audio_feature_pair[i]], dim=2) for i in range(2)]
                    style_mu_logvar_pair = [style_enc(input_feature_pair[i]) for i in range(2)]
                    style_pair = [x[0] for x in style_mu_logvar_pair]
                    mu_pair = [x[1] for x in style_mu_logvar_pair]
                    logvar_pair = [x[2] for x in style_mu_logvar_pair]
                elif args.style_enc_model_style[:3] == 'vae':
                    if do_ignore_style: # ignore the style, using mean style instead
                        mean_motion_coef_pair = [torch.zeros_like(motion_coef) for motion_coef in motion_coef_pair]
                        style_mu_logvar_pair = [style_enc(mean_motion_coef_pair[i]) for i in range(2)]
                    else:
                        style_mu_logvar_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                    style_pair = [x[0] for x in style_mu_logvar_pair]
                    mu_pair = [x[1] for x in style_mu_logvar_pair]
                    logvar_pair = [x[2] for x in style_mu_logvar_pair]
                elif args.style_enc_model_style == 'basic_encoder' or args.style_enc_model_style == 'gst' or args.style_enc_model_style == 'no_style' or args.style_enc_model_style == 'fft_encoder' or args.style_enc_model_style[:25] == 'basic_encoder_transformer':
                    if do_ignore_style: # ignore the style, using mean style instead
                        mean_motion_coef_pair = [torch.zeros_like(motion_coef) for motion_coef in motion_coef_pair]
                        style_pair = [style_enc(mean_motion_coef_pair[i]) for i in range(2)]
                    else:
                        style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                else:
                    raise ValueError(f"Style Encoder Model style {args.style_enc_model_style} not recognized")


            if args.use_context_audio_feat:
                # Extract audio features
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

            # construct losses        
            losses = {}
            for key in loss_weights:
                losses[key] = torch.tensor(0.0, device=device)
            if do_compute_style_adherence_loss:
                losses["style_adherence"] = torch.tensor(0.0, device=device)
            for i in range(2):
                audio = audio_pair[i]  # (N, L_a)
                motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
                style = style_pair[i] if style_enc is not None else None
                if args.use_cross_style:
                    style = style_pair[1 - i]
                batch_size = audio.shape[0]

                # truncate input audio and motion according to trunc_prob
                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:
                        # use contextualized audio feature for the second clip
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]

                else:
                    if args.use_context_audio_feat:
                        audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                    else:
                        audio_in = audio
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = torch.arange(args.n_motions, device=device).expand(batch_size,
                                                                                       -1) < end_idx.unsqueeze(1)
                    else:
                        indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None
                

                if args.do_ignore_shape:
                    input_shape_coef = torch.zeros_like(shape_coef)
                else:
                    input_shape_coef = shape_coef

                use_CFG_during_training = True
                if args.do_ignore_cfg:
                    use_CFG_during_training = False
                # Inference
                if i == 0:
                    if args.use_static_loss:
                        noise, target, prev_motion_coef, prev_audio_feat, dynamic_features, static_features, alpha_t = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training, keep_separate=True)
                    elif args.training_loss_style == "model_see_model_do":
                        noise, target, stylized_target, prev_motion_coef, prev_audio_feat = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                    elif args.training_loss_style == "model_see_model_do_ver2":
                        noise, target, training_guide, prev_motion_coef, prev_audio_feat = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                    else:
                        noise, target, prev_motion_coef, prev_audio_feat = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                    if end_idx is not None:  # was truncated, needs to use the complete feature
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            with torch.no_grad():
                                prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    if args.use_static_loss:
                        noise, target, _, _, dynamic_features, static_features, alpha_t = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training, keep_separate=True)
                    elif args.training_loss_style == "model_see_model_do":
                        noise, target, stylized_target, _, _ = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                    elif args.training_loss_style == "model_see_model_do_ver2":
                        noise, target, training_guide, _, _ = model(
                            motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                    else:
                        noise, target, _, _ = model(motion_coef_in, audio_in, input_shape_coef, style,
                                                    prev_motion_coef, prev_audio_feat, indicator=indicator, train_with_CFG=use_CFG_during_training)
                # compute the basic losses
                if args.training_loss_style == 'diffposetalk' or args.training_loss_style == 'diffposetalk+vae':
                    if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess' or args.dataset_type == "celebv-text-full+ravdess-FLMAE" or args.dataset_type == "celebv-text-medium+ravdess-FLMAE"):
                        loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    elif args.use_vertex_space:
                        # TODO: this needs implementation, compute_loss_espnet is also not implemented
                        torch.cuda.empty_cache()
                        seq_vertices = convert_sequence(target[:, args.n_prev_motions:], flame["neutral"], 
                                            (flame["expression2vert_mean"], flame["expression2vert_std"]), 
                                            flame["expression2vert_model"], device, 16)
                        with torch.no_grad():
                            gt_vertices = convert_sequence(motion_coef_in[:, :], flame["neutral"],
                                                (flame["expression2vert_mean"], flame["expression2vert_std"]),
                                                flame["expression2vert_model"], device, 16)
                        # print the memory size of flame["neutral"]
                        seq_vertices = seq_vertices.to(device)
                        gt_vertices = gt_vertices.to(device)
                        loss_dict = compute_loss_espnet(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, gt_vertices, seq_vertices, end_idx, return_dict=True)
                    else:
                        loss_dict = compute_loss_no_vert(
                            args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)

                    # if args.use_vertex_space:
                    #     loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    # else:
                    #     loss_dict = compute_loss_no_vert(
                    #         args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                elif args.training_loss_style == "model_see_model_do":
                    if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                        loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                        selector_loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, stylized_target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                        for key in selector_loss_dict:
                            loss_dict["stylized_" + key] = selector_loss_dict[key]
                    else:
                        raise ValueError(f"unknown config not recognized during loss computation")
                elif args.training_loss_style == "model_see_model_do_ver2":
                    if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                        loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                        style_guide_loss = torch.nn.functional.mse_loss(training_guide[:, args.n_prev_motions:], motion_coef_in)
                        loss_dict["style_guide_recon"] = style_guide_loss
                    else:
                        raise ValueError(f"unknown config not recognized during loss computation")
                else:
                    raise ValueError(f"Generator Model style {args.generator_model_style} not recognized during loss computation")
                            
                # compute additional losses
                if args.style_enc_model_style[:3] == 'vae':
                    kl_loss = compute_KL_loss(mu_pair[i], logvar_pair[i])
                    loss_dict['kl_div'] = kl_loss
                if args.use_style_adherence_loss:
                    if args.use_cross_style:
                        style_adherence_loss = StyleAdherenceLoss()(target, motion_coef_pair[1-i])
                    else:
                        style_adherence_loss = StyleAdherenceLoss()(target, motion_coef_pair[i])
                    loss_dict['style_adherence'] = style_adherence_loss
                
                if do_compute_style_adherence_loss:
                    loss_weights["style_adherence"] = torch.tensor(1.0, device=device)
                    style_adherence_loss = StyleAdherenceLoss()(target, motion_coef_in)
                    loss_dict['style_adherence'] = style_adherence_loss
                
                if args.use_static_loss:
                    static_loss = torch.nn.functional.mse_loss(static_features[:, args.n_prev_motions:], motion_coef_in)
                    loss_dict['static_loss'] = static_loss

                for key in loss_dict:
                    # note that head_trans could be None in the case of a single frame
                    if loss_weights[key] > 0 and loss_dict[key] is not None:
                        losses[key] += loss_dict[key]
        
            # aggregate losses
            loss = 0
            for key in losses:
                if loss_weights[key] > 0:
                    loss_log[key].append(losses[key].item())
                    loss += losses[key] * loss_weights[key]
            loss_log['loss'].append(loss.item())
    
    description = 'Train loss: ['
    for key in loss_log:
        if key == "loss":
            description += f'{key}: {np.mean(loss_log[key]):.3e}, '
        if key != 'loss' and loss_weights[key] > 0:
            if loss_weights[key] > 0:
                description += f'{key}: {np.mean(loss_log[key]):.3e}, '
    description += ']'
    
    if logger is not None:
        # log to logger (TensorBoard or JSON)
        logger.add_scalar(f'{mode}/loss', np.mean(loss_log['loss']), current_iter)
        for key in loss_log:
            if key != 'loss' and loss_weights[key] > 0:
                logger.add_scalar(f'{mode}/{key}', np.mean(loss_log[key]), current_iter)
        
    if is_training:
        model.train()

    if do_save:
        # save the metrics:
        save_path = do_save_path
        # save as a json file
        for key in loss_log:
            mean_val = np.mean(loss_log[key])
            std_val = np.std(loss_log[key])

            # convert to a single float
            loss_log[key] = {"mean": float(mean_val),
                             "std": float(std_val),
                             "n_samples": len(loss_log[key])}

        with open(save_path, 'w') as f:
            json.dump(loss_log, f)
        
    
    # del some files to clear up memory
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()

    # visualize 10 samples
    # test_loader = val_loader
    if do_render:
        test_dataset = dataset
        file_names = test_dataset.file_names
        # make a directory to save the numpy files for the current iteration
        os.makedirs(os.path.join(out_abc_dir, "iteration_{}".format(current_iter)), exist_ok=True)
        for i in range(0, 10):
            audio = test_dataset.data[file_names[i]]['audio']
            audio = (audio - audio.mean()) / (audio.std() + 1e-5)
            expression_coef = test_dataset.data[file_names[i]]['expression_code']
            expression_coef = (expression_coef - test_dataset.coef_stats['exp_mean'].detach().cpu().numpy()) / (test_dataset.coef_stats['exp_std'].detach().cpu().numpy() + 1e-9)
            head_rot = test_dataset.data[file_names[i]]['head_orientation']
            head_rot = (head_rot - test_dataset.coef_stats['pose_mean'].detach().cpu().numpy()) / (test_dataset.coef_stats['pose_std'].detach().cpu().numpy() + 1e-9)
            # convert to tensor and cast to device
            expression_coef = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
            head_rot = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
            shape_coef = torch.zeros([1, 100], device=device).float()
            motion_coeff = torch.cat([expression_coef, head_rot], dim=2).float().to(device)
            style_coeff = style_enc(motion_coeff[:, :500, :]).to(device)
            audio = torch.from_numpy(audio).to(device).float()    

            ################ test code
            # audio_unit = test_dataset.audio_unit
            # style_feat = style_coeff
            # out_abc_dir = "/code"
            # current_iter = 0
            ################ test code
            
            with torch.no_grad():
                coef = infer_coeffs(
                    model,
                    args,
                    audio, 
                    shape_coef,
                    test_dataset.audio_unit,
                    style_coeff,
                )

            expression_code = coef[0, :, :64]
            head_rot = coef[0, :, 64:]
            # denormalize
            expression_code = expression_code.to(device)
            head_rot = head_rot.to(device)

            expression_code = expression_code * test_dataset.coef_stats['exp_std'].to(device) + test_dataset.coef_stats['exp_mean'].to(device)
            head_rot = head_rot * test_dataset.coef_stats['pose_std'].to(device) + test_dataset.coef_stats['pose_mean'].to(device)
            # print(expression_code.shape, head_rot.shape)
            
            expression_code = expression_code.cpu().numpy()
            head_rot = head_rot.cpu().numpy()

            # save to files for visualization
            if out_abc_dir is not None:
                # save the numpy to the corresponding iteration
                expcode_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "exp_code_example_{}.pkl".format(file_names[i]))
                pkl.dump(expression_code, open(expcode_path, "wb"))
                heawdrot_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "head_rot_example_{}.pkl".format(file_names[i]))
                pkl.dump(head_rot, open(heawdrot_path, "wb"))
                # save the audio as a wav file
                audio_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "audio_example_{}.wav".format(file_names[i]))
                wavfile.write(audio_path, 16000, audio.cpu().numpy())
            express_code_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "exp_code_example_{}.pkl".format(file_names[i]))
            rotation_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "head_rot_example_{}.pkl".format(file_names[i]))
            audio_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "audio_example_{}.wav".format(file_names[i]))
            output_numpy_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "output_example_{}.npy".format(file_names[i]))
            alembic_path = os.path.join(out_abc_dir, "iteration_{}".format(current_iter), "output_example_{}.abc".format(file_names[i]))
            # run the expressionnet code here through subprocess
            
            # generate_vertices_alembic_video(
            #     express_code_path,
            #     rotation_path,
            #     output_numpy_path,
            #     alembic_path,
            #     fps = 25
            # )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def load_loss_weights(args): 
    loss_weights = {
        "noise": torch.tensor(1).float().to(model.device),
        "vert": torch.tensor(args.l_vert).float().to(model.device),
        "vel": torch.tensor(args.l_vel).float().to(model.device),
        "smooth": torch.tensor(args.l_smooth).float().to(model.device),
        "head_angle": torch.tensor(args.l_head_angle).float().to(model.device),
        "head_vel": torch.tensor(args.l_head_vel).float().to(model.device),
        "head_smooth": torch.tensor(args.l_head_smooth).float().to(model.device),
        "head_trans": torch.tensor(args.l_head_trans).float().to(model.device),
    }
    # if uses vertex loss, and on HDTF dataset, we scale it as per the paper
    if not args.use_vertex_space:
        print("Not using vertex space loss")
        loss_weights["vel"] *= 4.5E-8
        loss_weights["smooth"] *= 4E-7
    if not (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess') and args.use_vertex_space:
        print("Using vertex space loss on non-HDTF dataset")
        loss_weights["vert"] *= 1E-7
        loss_weights["vel"] *= 1E-7
        loss_weights["smooth"] *= 2E-8
    # add a loss for the kl divergence
    if args.training_loss_style == 'diffposetalk+vae':
        print("Using VAE loss")
        loss_weights["kl_div"] = torch.tensor(args.l_kl_div).float().to(model.device)
    if args.training_loss_style == 'model_see_model_do':
        loss_weights["stylized_noise"] = loss_weights["noise"]
        loss_weights["stylized_vert"] = loss_weights["vert"]
        loss_weights["stylized_vel"] = loss_weights["vel"]
        loss_weights["stylized_smooth"] = loss_weights["smooth"]
        loss_weights["stylized_head_angle"] = loss_weights["head_angle"]
        loss_weights["stylized_head_vel"] = loss_weights["head_vel"]
        loss_weights["stylized_head_smooth"] = loss_weights["head_smooth"]
        loss_weights["stylized_head_trans"] = loss_weights["head_trans"]
    if args.training_loss_style == 'model_see_model_do_ver2':
        loss_weights["style_guide_recon"] = 1
    if args.use_style_adherence_loss:
        loss_weights["style_adherence"] = 0.1
    if args.use_static_loss:
        loss_weights["static_loss"] = 0.1
    return loss_weights

def debug_encoding_pipeline(val_loader, style_encoder, device, step=10):
    """
    Debug the encoding pipeline by checking values at each step
    Returns statistics at different stages of processing
    """
    stats = {
        'raw_motion_stats': [],
        'encoder_output_stats': [],
        'final_encoding_stats': []
    }
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= step:  # Only check first few batches
                break
                
            audio_pair, coef_pair, _ = data
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            
            # Check motion coefficients
            motion_pair = [coef_pair[0]["motion"], coef_pair[1]["motion"]]
            for motion in motion_pair:
                stats['raw_motion_stats'].append({
                    'min': motion.min().item(),
                    'max': motion.max().item(),
                    'mean': motion.mean().item(),
                    'has_nan': torch.isnan(motion).any().item(),
                    'has_inf': torch.isinf(motion).any().item()
                })
            
            # Check encoder outputs
            encoder_outputs = [style_encoder(motion) for motion in motion_pair]
            for output in encoder_outputs:
                if isinstance(output, tuple):  # VAE case
                    output = output[1]  # Get mu
                stats['encoder_output_stats'].append({
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item()
                })
    
    return stats

def print_debug_stats(stats):
    """Print readable summary of debug statistics"""
    for stage, measurements in stats.items():
        print(f"\n=== {stage} ===")
        if measurements:
            means = np.mean([m['mean'] for m in measurements])
            mins = np.mean([m['min'] for m in measurements])
            maxs = np.mean([m['max'] for m in measurements])
            nans = any([m['has_nan'] for m in measurements])
            infs = any([m['has_inf'] for m in measurements])
            
            print(f"Average mean: {means:.6f}")
            print(f"Average min: {mins:.6f}")
            print(f"Average max: {maxs:.6f}")
            print(f"Contains NaN: {nans}")
            print(f"Contains Inf: {infs}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DiffTalkingHead: Speech-Driven 3D Facial Animation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=100000, help='iteration to test')
    parser.add_argument('--use_vertex_space', action='store_true', help='use vertex space for loss')
    parser.add_argument('--continue_from', type=str, default=None, help='Path to the experiment directory to continue from')
    parser.add_argument('--generator_model_style', type=str, default='diffposetalk', help='style of the generator model')
    parser.add_argument('--style_enc_model_style', type=str, default='diffposetalk', help='style of the style encoder model')
    parser.add_argument('--training_loss_style', type=str, default='diffposetalk', help='style of the training loss')
    parser.add_argument('--use_cross_style', action='store_true', help='use indicator for padded frames') # if it's cross style, we swap the style code for window 1 and 2
    parser.add_argument('--do_save_path', type=Path, default=None, help='Path to save the loss metrics')
    parser.add_argument('--do_ignore_style', action='store_true', help='render the output')
    parser.add_argument('--style_token_count', type=int, default=10, help='number of potential style tokens')
    parser.add_argument('--style_token_attention_heads', type=int, default=8, help='number of attention heads for style tokens')
    parser.add_argument('--feat_to_film', type=str, default="audio", help='indiciate what level to perform film conditioning')
    parser.add_argument('--do_ignore_shape', action='store_true', help='indiciate what level to perform film conditioning')
    parser.add_argument('--batch_overfit_size', type=int, default=-1, help='if > 0, will overfit on a small batch of data for debugging')
    parser.add_argument('--do_ignore_cfg', action='store_true', help='decide whether to use the cfg during training')
    parser.add_argument('--use_style_adherence_loss', action='store_true', help='decide whether to use the style_adherence_loss during training')
    parser.add_argument('--use_static_loss', action='store_true', help='decide whether to use the static_loss during training')
    parser.add_argument('--num_of_basis', type=int, default=1, help='number of basis for the style encoder')
    parser.add_argument('--regularize_alpha', type=str, default="None", help='number of basis for the style encoder')
    parser.add_argument("--prob_cross_style", type=float, default=1.0, help="probability of using cross style")
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='use TensorBoard for logging (falls back to JSON if unavailable)')
    parser.add_argument('--use_normalization', action='store_true',
                        help='Whether to use normalization in the style encoder')
    # Dataset
    options.add_data_options(parser)
    # Model
    options.add_model_options(parser)
    # Training
    options.add_training_options(parser)
    
    ############## test code ##############
    custom_args_list = [
        '--exp_name', 'celebv_text_ravdess_static_dynamic_vae_basis_size_4_no_head_alpha_swap_prob_50',
        '--data_root', 'data/',
        '--use_indicator',
        '--use_cross_style',
        '--batch_size', '16',
        '--num_of_basis', '4',
        '--scheduler', 'Warmup',
        '--audio_model', 'hubert',
        '--style_enc_model_style', 'vae2',
        '--generator_model_style', 'diffposetalk_static_dynamic_k_basis_no_head_alpha',
        '--training_loss_style', 'diffposetalk+vae',
        '--dataset_type', 'celebv-text-medium+ravdess-FLMAE',
        '--num_workers', '2',
        '--d_style', '256',
        '--l_kl_div', '1E-7',
        '--l_smooth', '1E1',
        '--max_iter', '2000000',
        '--prob_cross_style', '0.5',
        '--use_vertex_space',  # defined but not passed in original script
    ]
    ############## test code ##############
    
    custom_args_list = None
    # Additional options depending on previous options
    options.add_additional_options(parser, custom_args_list)
    # copied_model_args = parser.parse_args(custom_args_list)
    args = parser.parse_args(custom_args_list)
    option_text = utils.get_option_text(args, parser)
    
    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load flame model (only necesary if we compute vertex losses)
    if (args.l_vert > 0 or args.l_vel > 0) and args.use_vertex_space:
        if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == "flame_mead_ravdess" or args.dataset_type == "celebv-text-full+ravdess-FLMAE" or args.dataset_type == "celebv-text-medium+ravdess-FLMAE":
            import inspect
            from collections import namedtuple
            import numpy as _np
            # restore removed aliases expected by old libs
            _aliases = [
                ('bool', _np.bool_),  # type alias
                ('int', int),
                ('float', float),
                ('complex', complex),
                ('object', object),
                ('unicode', str),
                ('str', str),
            ]
            for name, target in _aliases:
                if not hasattr(_np, name):
                    setattr(_np, name, target)
            import numpy as np
            # If you're on Python 3.11+, chumpy also uses inspect.getargspec (removed):
            import inspect as _inspect
            from collections import namedtuple as _namedtuple
            if not hasattr(_inspect, "getargspec"):
                def _shim_getargspec(func):
                    fas = _inspect.getfullargspec(func)
                    ArgSpec = _namedtuple("ArgSpec", "args varargs keywords defaults")
                    return ArgSpec(fas.args, fas.varargs, fas.varkw, fas.defaults)
                _inspect.getargspec = _shim_getargspec
            if not hasattr(inspect, "getargspec"):
                def _shim_getargspec(func):
                    fas = inspect.getfullargspec(func)
                    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
                    return ArgSpec(fas.args, fas.varargs, fas.varkw, fas.defaults)
                inspect.getargspec = _shim_getargspec  # monkey-patch for chumpy
            flame = FLAME(FLAMEConfig, dtype=torch.float32, device=device)
    else:
        flame = None
        
    if args.mode == 'test':
        # loaded_args = load_args(Path(args.continue_from))

        loaded_args = load_args_with_defaults(Path(args.continue_from), parser)
        loaded_args.continue_from = args.continue_from
        loaded_args.mode = 'test'
        test_do_save_path = args.do_save_path
        args = loaded_args
    

    # Style Encoder
    if args.style_enc_model_style == 'diffposetalk': # this needs to load a pre-trained model
        print("Building Style Encoder: style = diffposetalk")
        if args.style_enc_ckpt:
            # Build mode
            enc_model_data = torch.load(args.style_enc_ckpt, map_location=device, weights_only=False)
            enc_model_args = utils.NullableArgs(enc_model_data['args'])
            # style_enc = StyleEncoder(enc_model_args).to(device)
            style_enc = get_style_encoder(enc_model_args, args.style_enc_model_style).to(device)
            style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
            style_enc.eval()
            # freeze the style encoder
            for param in style_enc.parameters():
                param.requires_grad = False
        else:
            style_enc = None
        print("Style Encoder built")
    else: # other wise we can just use the factory function
        print(f"Building Style Encoder: style = {args.style_enc_model_style}")
        style_enc = get_style_encoder(args, args.style_enc_model_style).to(device)
        print("Style Encoder built")

    # Build diffusion model
    print("Building DiffTalkingHead: style = diffposetalk")
    model = get_difftalkinghead_model(args, device=device)
    print("DiffTalkingHead built")
    if model is None:
        raise ValueError(f"Model style {args.generator_model_style} not recognized")

    # load any pretrained_models
    if args.continue_from is not None:
        exp_dir = Path(args.continue_from)
        print(f"Starting from pre-trained, continuing from {exp_dir}") 
        args, model, style_enc, start_iter = load_pretrained_model(args, model, style_enc, parser=parser)    
        print("Pretraineed model loaded")
    else:
        # Making folders for Logging
        exp_dir = Path('experiments/dpt') / f'{args.exp_name}-{datetime.now().strftime("%y%m%d_%H%M%S")}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        start_iter = 0
    
    # cast model to device
    model.to(device)
    style_enc.to(device)

    if args.mode == 'train':
        # prep the optimizer
        if args.style_enc_model_style == 'diffposetalk':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            # otherwise the optimizer should work for both the style encoder and the model
            optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, style_enc.parameters()), 'lr': args.lr},
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}
            ])
        # save args (even if they have been save already in the case of retraining)
        save_args(args, exp_dir)
        # prep the logging
        log_dir = exp_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        out_abc_dir = exp_dir / 'out_abc'
        out_abc_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the logger (replaces SummaryWriter)
        logger = TrainingLogger(log_dir, use_tensorboard=getattr(args, 'use_tensorboard', True))
        
        # save args after creating the directories
        if option_text is not None:
            with open(log_dir / 'options.log', 'w') as f:
                f.write(option_text)
            logger.add_text('options', option_text)
        
        # Import colorama for colored output (optional, falls back gracefully)
        try:
            from colorama import Back, Fore, Style
            print(Back.RED + Fore.YELLOW + Style.BRIGHT + exp_dir.name + Style.RESET_ALL)
        except ImportError:
            print(f"=== {exp_dir.name} ===")
        
        print('model parameters: ', count_parameters(model))
        print(f"loading dataset {args.dataset_type}")
        # Dataset
        # args.data_root = "/mnt/f/chrome_downloads/processed_data"

        if args.use_normalization and args.stats_file is not None:
            stats_file = args.stats_file
        else:
            stats_file = None

        train_dataset, val_dataset, train_loader, val_loader = create_datasets(
            dataset_type=args.dataset_type,
            data_root=str(args.data_root),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_se=False,
            no_head_pose=args.no_head_pose,
            fps=args.fps,
            n_motions=args.n_motions,
            rot_repr=args.rot_repr,
            stats_file=stats_file,
            batch_overfit_size=args.batch_overfit_size if hasattr(args, 'batch_overfit_size') else -1,
        )
        # scheduler
        if args.scheduler == 'Warmup':
            from scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
        elif args.scheduler == 'WarmupThenDecay':
            from scheduler import GradualWarmupScheduler
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                                    args.lr * args.min_lr_ratio)
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
        else:
            scheduler = None
            
        train(args, model, style_enc, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, logger,
                flame, out_abc_dir=out_abc_dir, start_iter=start_iter)
        
        # Close the logger when training is done
        logger.close()
    else:
        # load the dataset
        # args.data_root = "/mnt/f/chrome_downloads/processed_data"
        print(f"loading dataset {args.dataset_type} for testing") 
        args.batch_size = 2  # double the batch size for testing, since we don't need to backprop       
        if args.use_normalization and args.stats_file is not None:
            stats_file = args.stats_file
        else:
            stats_file = None
        train_dataset, val_dataset, train_loader, val_loader = create_datasets(
            dataset_type=args.dataset_type,
            data_root=str(args.data_root),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_se=False,
            no_head_pose=args.no_head_pose,
            fps=args.fps,
            n_motions=args.n_motions,
            rot_repr=args.rot_repr,
            stats_file=stats_file,
            batch_overfit_size=args.batch_overfit_size if hasattr(args, 'batch_overfit_size') else -1,
        )
        print("Dataset loaded")

        # Test the model on validation set
        with torch.no_grad():
            string_test_do_save_path = str(test_do_save_path) # convert to string so we can do string operations
            test_test_do_save_path = Path(string_test_do_save_path.split(".")[0] + "_with_style_test.json") # change the name of the file and convert back to Path
            test(args, load_loss_weights(args), model, style_enc, val_loader, args.iter, 5, 'test', None, flame, do_save=True, do_save_path=test_test_do_save_path, do_ignore_style=False, do_compute_style_adherence_loss=True)

            train_do_save_path = Path(string_test_do_save_path.split(".")[0] + "_with_style_train.json") # change the name of the file and convert back to Path
            test(args, load_loss_weights(args), model, style_enc, train_loader, args.iter, 1, 'test', None, flame, do_save=True, do_save_path=train_do_save_path, do_ignore_style=False, do_compute_style_adherence_loss=True)

            test_test_do_save_path = Path(string_test_do_save_path.split(".")[0] + "_no_style_test.json") # change the name of the file and convert back to Path
            test(args, load_loss_weights(args), model, style_enc, val_loader, args.iter, 5, 'test', None, flame, do_save=True, do_save_path=test_test_do_save_path, do_ignore_style=True, do_compute_style_adherence_loss=True)

            train_do_save_path = Path(string_test_do_save_path.split(".")[0] + "_no_style_train.json") # change the name of the file and convert back to Path
            test(args, load_loss_weights(args), model, style_enc, train_loader, args.iter, 1, 'test', None, flame, do_save=True, do_save_path=train_do_save_path, do_ignore_style=True, do_compute_style_adherence_loss=True)