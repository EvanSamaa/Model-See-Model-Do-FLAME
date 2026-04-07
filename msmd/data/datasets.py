import io
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
import os
import lmdb
import numpy as np
import torch
import torchaudio
from torch.utils import data
from scipy.interpolate import interp1d
from tqdm import tqdm
import librosa
import cv2
import pandas as pd



__dir__ = Path(__file__).parent

# https://github.com/pytorch/audio/issues/2950 , https://github.com/pytorch/audio/issues/2356
torchaudio.set_audio_backend('soundfile')

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class ConcatDataset_with_stats(data.ConcatDataset):
    def __init__(self, datasets, coef_stats):
        super(ConcatDataset_with_stats, self).__init__(datasets)
        self.coef_stats = coef_stats

def get_dataset_SE(args, device, only_eval=False, batch_overfit_size=-1):
    print("loading dataset from ", args.data_root)
    try:
        args.data_root = Path(args.data_root)
    except:
        print("dataroot already a path")

    if batch_overfit_size > 0:
        # if we are batch overfitting, we are also not gonna random pad the audio
        do_random_pad = False
    else:
        do_random_pad = True
    if args.dataset_type == "HDTF_TFHP+flame_mead_ravdess":
        HDTF_data_root = Path("/data/HDTF_TFHP/lmdb/")
        mead_ravdess_data_root = Path("/mnt/f/chrome_downloads/mead_ravdess_30fps/")
        if os.path.exists(mead_ravdess_data_root):
            pass
        else:
            mead_ravdess_data_root = Path("/data/mead_ravdess_30fps/")
        
        coef_stats_file: Path = Path(args.stats_file)
        if not coef_stats_file.is_absolute():
            coef_stats_file = HDTF_data_root / coef_stats_file

        train_dataset_hdtf = LmdbDatasetForSE(HDTF_data_root, HDTF_data_root / 'train.txt', coef_stats_file, args.fps, args.n_motions,
                                    rot_repr=args.rot_repr)
        val_dataset_hdtf = LmdbDatasetForSE(HDTF_data_root, HDTF_data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                rot_repr=args.rot_repr)
        
        
        val_data_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "val_mead_ravdess_0.1.pickle")
        val_audio_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "val_mead_ravdess_rawaudio_0.1.pickle")
        train_data_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "train_mead_ravdess_0.1.pickle")
        train_audio_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "train_mead_ravdess_rawaudio_0.1.pickle")
        
        val_data_pkl_mead_ravdess = pickle.load(open(val_data_pkl_file_mead_ravdess, "rb"))
        val_audio_pkl_mead_ravdess = pickle.load(open(val_audio_pkl_file_mead_ravdess, "rb"))
        train_data_pkl_mead_ravdess = pickle.load(open(train_data_pkl_file_mead_ravdess, "rb"))
        train_audio_pkl_mead_ravdess = pickle.load(open(train_audio_pkl_file_mead_ravdess, "rb"))

        train_dataset_mead_ravdess = PickleDataset_Evan_MEAD_RAVDESS(train_data_pkl_mead_ravdess, train_audio_pkl_mead_ravdess, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset_mead_ravdess = PickleDataset_Evan_MEAD_RAVDESS(val_data_pkl_mead_ravdess, val_audio_pkl_mead_ravdess, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        weight_ravdess_train = 1.0 / len(train_dataset_mead_ravdess)
        weight_hdtf_train = 1.0 / len(train_dataset_hdtf)
        weight_ravdess_val = 1.0 / len(val_dataset_mead_ravdess)
        weight_hdtf_val = 1.0 / len(val_dataset_hdtf)

        weights_train = [weight_hdtf_train] * len(train_dataset_hdtf) + [weight_ravdess_train] * len(train_dataset_mead_ravdess)
        weights_val = [weight_hdtf_val] * len(val_dataset_hdtf) + [weight_ravdess_val] * len(val_dataset_mead_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_hdtf, train_dataset_mead_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_hdtf, val_dataset_mead_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=True), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=True), sampler=sampler_val)
    elif args.dataset_type == "ravdess+celebv-text-medium":
        data_root = args.data_root

        ravedess_root = Path("/data/ravdess/processed_data")

        train_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl', 
                                           ravedess_root / 'processed_ravdess_30fps_v3_keys_train.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, device=device, celebv_text=False, full_dataset=True)
        val_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl',
                                            ravedess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, device=device, celebv_text=False, full_dataset=True)

        train_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, 
                                            full_dataset=True, celebv_text=True)
        val_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=True, 
                                            full_dataset=True, celebv_text=True)
        
        weight_ravdess_train = 1.0 / len(train_dataset_ravdess)
        weight_celebv_text_train = 1.0 / len(train_dataset_celebv_text)
        weight_ravdess_val = 1.0 / len(val_dataset_ravdess)
        weight_celebv_text_val = 1.0 / len(val_dataset_celebv_text)

        weights_train = [weight_celebv_text_train] * len(train_dataset_celebv_text) + [weight_ravdess_train] * len(train_dataset_ravdess)
        weights_val = [weight_celebv_text_val] * len(val_dataset_celebv_text) + [weight_ravdess_val] * len(val_dataset_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_celebv_text, train_dataset_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_celebv_text, val_dataset_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=True), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=True), sampler=sampler_val)
    elif args.dataset_type == "celebv-text-medium":
        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_medium_v3.pkl"):
            raw_data.update(chunk)
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, full_dataset=True,
                                            SE=True, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad,
                                            pre_loaded_raw_dataset=raw_data)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, full_dataset=True,
                                            SE=True, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad,
                                            pre_loaded_raw_dataset=raw_data)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=True))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=True))
        
    return train_dataset, val_dataset, train_loader, val_loader

def get_full_ravdess(args, device):
    ravdess_data_root = Path("/mnt/f/chrome_downloads/processed_data/")
    if os.path.exists(ravdess_data_root):
        pass
    else:
        ravdess_data_root = Path("/data/ravdess/processed_data")
    train_dataset = PickleDataset_Evan(ravdess_data_root / 'processed_ravdess_30fps_v3.pkl', 
                                           ravdess_data_root / 'processed_ravdess_30fps_v3_keys.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=False, SE=False, device=device, celebv_text=False, full_dataset=True)
    
    return train_dataset

def get_dataset(args, device, only_eval=False, batch_overfit_size=-1):
    print("loading dataset from ", args.data_root)
    try:
        args.data_root = Path(args.data_root)
    except:
        print("dataroot already a path")

    if batch_overfit_size > 0:
        # if we are batch overfitting, we are also not gonna random pad the audio
        do_random_pad = False
    else:
        do_random_pad = True

    if args.dataset_type == "HDTF_TFHP":
        data_root = args.data_root
        coef_stats_file: Path = Path(args.stats_file)
        if not coef_stats_file.is_absolute():
            coef_stats_file = data_root / coef_stats_file

        train_dataset = LmdbDataset(args.data_root, args.data_root / 'train.txt', coef_stats_file, args.fps, args.n_motions,
                                    rot_repr=args.rot_repr, do_random_pad=do_random_pad, batch_overfit_size=batch_overfit_size)
        val_dataset = LmdbDataset(args.data_root, args.data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                rot_repr=args.rot_repr, do_random_pad=do_random_pad, batch_overfit_size=batch_overfit_size)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    elif args.dataset_type == "iconic_speeches":
        
        data_root = args.data_root
        train_dataset = FileBased_iconic_dataset()
        val_dataset = FileBased_iconic_dataset()

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=FileBased_iconic_dataset.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=FileBased_iconic_dataset.get_collate_fn(SE=False))
                    
    elif args.dataset_type == "HDTF_TFHP+flame_mead_ravdess":
        HDTF_data_root = Path("/data/HDTF_TFHP/lmdb/")
        mead_ravdess_data_root = Path("/mnt/f/chrome_downloads/mead_ravdess_30fps/")
        if os.path.exists(mead_ravdess_data_root):
            pass
        else:
            mead_ravdess_data_root = Path("/data/mead_ravdess_30fps/")
        


        coef_stats_file: Path = Path(args.stats_file)
        if not coef_stats_file.is_absolute():
            coef_stats_file = HDTF_data_root / coef_stats_file

        
        train_dataset_hdtf = LmdbDataset(HDTF_data_root, HDTF_data_root / 'train.txt', coef_stats_file, args.fps, args.n_motions,
                                    rot_repr=args.rot_repr, do_random_pad=do_random_pad, batch_overfit_size=batch_overfit_size)
        val_dataset_hdtf = LmdbDataset(HDTF_data_root, HDTF_data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                rot_repr=args.rot_repr, do_random_pad=do_random_pad, batch_overfit_size=batch_overfit_size)
        
        
        val_data_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "val_mead_ravdess_0.1.pickle")
        val_audio_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "val_mead_ravdess_rawaudio_0.1.pickle")
        train_data_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "train_mead_ravdess_0.1.pickle")
        train_audio_pkl_file_mead_ravdess = os.path.join(mead_ravdess_data_root, "train_mead_ravdess_rawaudio_0.1.pickle")
        
        val_data_pkl_mead_ravdess = pickle.load(open(val_data_pkl_file_mead_ravdess, "rb"))
        val_audio_pkl_mead_ravdess = pickle.load(open(val_audio_pkl_file_mead_ravdess, "rb"))
        train_data_pkl_mead_ravdess = pickle.load(open(train_data_pkl_file_mead_ravdess, "rb"))
        train_audio_pkl_mead_ravdess = pickle.load(open(train_audio_pkl_file_mead_ravdess, "rb"))

        train_dataset_mead_ravdess = PickleDataset_Evan_MEAD_RAVDESS(train_data_pkl_mead_ravdess, train_audio_pkl_mead_ravdess, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset_mead_ravdess = PickleDataset_Evan_MEAD_RAVDESS(val_data_pkl_mead_ravdess, val_audio_pkl_mead_ravdess, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        weight_ravdess_train = 1.0 / len(train_dataset_mead_ravdess)
        weight_hdtf_train = 1.0 / len(train_dataset_hdtf)
        weight_ravdess_val = 1.0 / len(val_dataset_mead_ravdess)
        weight_hdtf_val = 1.0 / len(val_dataset_hdtf)

        weights_train = [weight_hdtf_train] * len(train_dataset_hdtf) + [weight_ravdess_train] * len(train_dataset_mead_ravdess)
        weights_val = [weight_hdtf_val] * len(val_dataset_hdtf) + [weight_ravdess_val] * len(val_dataset_mead_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_hdtf, train_dataset_mead_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_hdtf, val_dataset_mead_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=False), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=False), sampler=sampler_val)
    elif args.dataset_type == "flame_mead_ravdess":
        val_data_pkl_file = os.path.join(args.data_root, "val_mead_ravdess_0.1.pickle")
        val_audio_pkl_file = os.path.join(args.data_root, "val_mead_ravdess_rawaudio_0.1.pickle")
        train_data_pkl_file = os.path.join(args.data_root, "train_mead_ravdess_0.1.pickle")
        train_audio_pkl_file = os.path.join(args.data_root, "train_mead_ravdess_rawaudio_0.1.pickle")
        
        val_data_pkl = pickle.load(open(val_data_pkl_file, "rb"))
        val_audio_pkl = pickle.load(open(val_audio_pkl_file, "rb"))
        train_data_pkl = pickle.load(open(train_data_pkl_file, "rb"))
        train_audio_pkl = pickle.load(open(train_audio_pkl_file, "rb"))

        train_dataset = PickleDataset_Evan_MEAD_RAVDESS(train_data_pkl, train_audio_pkl, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan_MEAD_RAVDESS(val_data_pkl, val_audio_pkl, None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(SE=False))        
    elif args.dataset_type == "celebv-text-toy":
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_toy.pkl",
                                            args.data_root / 'processed_data_30fps_toy_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_toy.pkl",
                                            args.data_root / 'processed_data_30fps_toy_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)   
         
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "celebv-text":
        # TODO: Implement this
        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_v2.pkl"):
            raw_data.update(chunk)
        # print(len(raw_data.keys()))
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_v2.pkl", 
                                                args.data_root / 'processed_data_30fps_v2_keys_train.txt', 
                                                None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                                full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_v2.pkl",
                                            args.data_root / 'processed_data_30fps_v2_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                            full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "celebv-text-v2":
        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_v3.pkl"):
            raw_data.update(chunk)
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_v3.pkl", 
                                                args.data_root / 'processed_data_30fps_v3_keys_train.txt', 
                                                None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                                full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_v3.pkl",
                                            args.data_root / 'processed_data_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                            full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "celebv-text-medium-v2":
        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_medium_v3.pkl"):
            raw_data.update(chunk)
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                            full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                            full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))    
    elif args.dataset_type == "celebv-text-toy-v2":
        train_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_toy_v3.pkl",
                                            args.data_root / 'processed_data_30fps_toy_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / "processed_data_30fps_toy_v3.pkl",
                                            args.data_root / 'processed_data_30fps_toy_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)            


        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "ravdess":
        data_root = args.data_root

        train_dataset = PickleDataset_Evan(args.data_root / 'processed_ravdess_30fps_v3.pkl', 
                                           data_root / 'processed_ravdess_30fps_v3_keys_train.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset = PickleDataset_Evan(args.data_root / 'processed_ravdess_30fps_v3.pkl',
                                            data_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "ravdess+celebv-text-full":
        data_root = args.data_root
        if os.path.exists("/mnt/f/chrome_downloads/processed_data"):
            ravedess_root = Path("/mnt/f/chrome_downloads/processed_data")
        else:    
            ravedess_root = Path("/data/ravdess/processed_data")

        train_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl', 
                                           ravedess_root / 'processed_ravdess_30fps_v3_keys_train.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)
        val_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl',
                                            ravedess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)

        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_v3.pkl"):
            raw_data.update(chunk)
        train_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_v3.pkl", 
                                                args.data_root / 'processed_data_30fps_v3_keys_train.txt', 
                                                None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                                full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        val_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_v3.pkl",
                                            args.data_root / 'processed_data_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, 
                                            full_dataset=True, pre_loaded_raw_dataset=raw_data, batch_overfit_size=batch_overfit_size, random_crop=do_random_pad)
        
        weight_ravdess_train = 1.0 / len(train_dataset_ravdess) / 2
        weight_celebv_text_train = 1.0 / len(train_dataset_celebv_text) * 2
        weight_ravdess_val = 1.0 / len(val_dataset_ravdess) / 2
        weight_celebv_text_val = 1.0 / len(val_dataset_celebv_text) * 2

        weights_train = [weight_celebv_text_train] * len(train_dataset_celebv_text) + [weight_ravdess_train] * len(train_dataset_ravdess)
        weights_val = [weight_celebv_text_val] * len(val_dataset_celebv_text) + [weight_ravdess_val] * len(val_dataset_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_celebv_text, train_dataset_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_celebv_text, val_dataset_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False), sampler=sampler_val)
    elif args.dataset_type == "ravdess+celebv-text-medium":
        data_root = args.data_root
        if os.path.exists("/mnt/f/chrome_downloads/processed_data"):
            ravedess_root = Path("/mnt/f/chrome_downloads/processed_data")
        else:    
            ravedess_root = Path("/data/ravdess/processed_data")

        train_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl', 
                                           ravedess_root / 'processed_ravdess_30fps_v3_keys_train.txt', 
                                           None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)
        val_dataset_ravdess = PickleDataset_Evan(ravedess_root / 'processed_ravdess_30fps_v3.pkl',
                                            ravedess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device, celebv_text=False, full_dataset=True)

        raw_data = {}
        for chunk in PickleDataset_Evan.load_dict_in_chunks_static(args.data_root / "processed_data_30fps_medium_v3.pkl"):
            raw_data.update(chunk)


        train_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_train.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, 
                                            SE=False, pre_loaded_raw_dataset=raw_data,
                                            full_dataset=True)
        val_dataset_celebv_text = PickleDataset_Evan(args.data_root / "processed_data_30fps_medium_v3.pkl",
                                            args.data_root / 'processed_data_30fps_medium_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, 
                                            SE=False, pre_loaded_raw_dataset=raw_data, 
                                            full_dataset=True)
        
        weight_ravdess_train = 1.0 / len(train_dataset_ravdess)
        weight_celebv_text_train = 1.0 / len(train_dataset_celebv_text)
        weight_ravdess_val = 1.0 / len(val_dataset_ravdess)
        weight_celebv_text_val = 1.0 / len(val_dataset_celebv_text)

        weights_train = [weight_celebv_text_train] * len(train_dataset_celebv_text) + [weight_ravdess_train] * len(train_dataset_ravdess)
        weights_val = [weight_celebv_text_val] * len(val_dataset_celebv_text) + [weight_ravdess_val] * len(val_dataset_ravdess)

        train_dataset = data.ConcatDataset([train_dataset_celebv_text, train_dataset_ravdess])
        val_dataset = data.ConcatDataset([val_dataset_celebv_text, val_dataset_ravdess])

        sampler_train = data.WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
        sampler_val = data.WeightedRandomSampler(weights_val, num_samples=len(weights_val), replacement=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False), sampler=sampler_train)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False), sampler=sampler_val)

    return train_dataset, val_dataset, train_loader, val_loader

def get_dataset_lmdb(args, device, only_eval=False):
    try:
        args.data_root = Path(args.data_root)
    except:
        print("dataroot already a path")
    if args.dataset_type == "celebv-text-toy-v2":
        if not only_eval:
            train_dataset = LMDB_Dataset_Evan(args.data_root / "processed_data_30fps_v3.lmdb",
                                                args.data_root / 'processed_data_30fps_toy_v3_keys_train.txt',
                                                None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device)
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_dataset = LMDB_Dataset_Evan(args.data_root / "processed_data_30fps_v3.lmdb",
                                            args.data_root / 'processed_data_30fps_toy_v3_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device)            
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "celebv-text-toy":
        if not only_eval:
            train_dataset = LMDB_Dataset_Evan(args.data_root / "processed_data_30fps_v2.lmdb",
                                                args.data_root / 'processed_data_30fps_toy_keys_train.txt',
                                                None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device)
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
        val_dataset = LMDB_Dataset_Evan(args.data_root / "processed_data_30fps_v2.lmdb",
                                            args.data_root / 'processed_data_30fps_toy_keys_valid.txt',
                                            None, original_fps=30, coef_fps=25, no_head_pose=args.no_head_pose, SE=False, device=device)            
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=PickleDataset_Evan.get_collate_fn(SE=False))
    elif args.dataset_type == "HDTF_TFHP":
        data_root = args.data_root
        coef_stats_file: Path = Path(args.stats_file)
        if not coef_stats_file.is_absolute():
            coef_stats_file = data_root / coef_stats_file
        train_dataset = LmdbDataset(args.data_root, args.data_root / 'train.txt', coef_stats_file, args.fps, args.n_motions,
                                    rot_repr=args.rot_repr)
        val_dataset = LmdbDataset(args.data_root, args.data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                rot_repr=args.rot_repr)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
    elif args.dataset_type == "celebv-text":
        return get_dataset(args, device, only_eval)
    elif args.dataset_type == "celebv-text-v2":
        return get_dataset(args, device, only_eval)
    elif args.dataset_type == "celebv-text-medium-v2":
        return get_dataset(args, device, only_eval)
    else:
        return get_dataset(args, device, only_eval)
    if only_eval:
        return None, val_dataset, None,  val_loader
    else:
        return train_dataset, val_dataset, train_loader, val_loader

def incremental_mean_and_std(train_dataset, SE=False):
    exp_sum = 0
    exp_sum_of_squares = 0
    pose_sum = 0
    pose_sum_of_squares = 0
    num_elements = 0
    for i in tqdm(range(len(train_dataset))):
    # for i in range(len(train_dataset)):
        entry_i = train_dataset[i]
        if not SE:
            # Extract expression and pose tensors for both frames
            exp_0 = entry_i[1][0]['motion'][:, :64]
            exp_1 = entry_i[1][1]['motion'][:, :64]
            pose_0 = entry_i[1][0]['motion'][:, 64:]
            pose_1 = entry_i[1][1]['motion'][:, 64:]
        if SE:
            exp_0 = entry_i[0][:, :64]
            exp_1 = entry_i[1][:, :64]
            pose_0 = entry_i[0][:, 64:]
            pose_1 = entry_i[1][:, 64:]         
        # Update sum and sum of squares for expressions
        exp_sum += exp_0.sum(dim=0)
        exp_sum_of_squares += (exp_0 ** 2).sum(dim=0)
        exp_sum += exp_1.sum(dim=0)
        exp_sum_of_squares += (exp_1 ** 2).sum(dim=0)

        # Update sum and sum of squares for poses
        pose_sum += pose_0.sum(dim=0)
        pose_sum_of_squares += (pose_0 ** 2).sum(dim=0)
        pose_sum += pose_1.sum(dim=0)
        pose_sum_of_squares += (pose_1 ** 2).sum(dim=0)
        
        # Update the total number of elements processed
        num_elements += exp_0.shape[0] + exp_1.shape[0]

    # Compute the mean for expressions and poses
    exp_mean = exp_sum / num_elements
    pose_mean = pose_sum / num_elements

    # Compute the variance for expressions and poses
    exp_var = (exp_sum_of_squares / num_elements) - (exp_mean ** 2)
    pose_var = (pose_sum_of_squares / num_elements) - (pose_mean ** 2)

    # Standard deviation is the square root of variance
    exp_std = torch.sqrt(exp_var)
    pose_std = torch.sqrt(pose_var)

    return exp_mean, exp_std, pose_mean, pose_std

class DynamicObjectDebugger:
    def __init__(self):
        pass

class LmdbDataset(data.Dataset):
    def __init__(self, lmdb_dir, split_file, coef_stats_file=None, coef_fps=25, n_motions=100, crop_strategy='random',
                 rot_repr='aa', batch_overfit_size=-1, do_random_pad=True):
        self.split_file = split_file
        self.lmdb_dir = Path(lmdb_dir)
        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.do_random_pad = do_random_pad
        self.batch_overfit_size = batch_overfit_size
        self.crop_strategy = crop_strategy
        if not self.do_random_pad:
            self.crop_strategy = 'begin'
        self.rot_representation = rot_repr

        # Read split file
        self.entries = []
        with open(self.split_file, 'r') as f:
            for line in f:
                self.entries.append(line.strip())
        if self.batch_overfit_size > 0:
            self.entries = self.entries[:self.batch_overfit_size]
        # Load lmdb
        self.lmdb_env = lmdb.open(str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']
            self.audio_clip_len = round(self.audio_unit * self.clip_len)

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        # Read audio and coef
        with self.lmdb_env.begin(write=False) as txn:
            meta_key = f'{self.entries[index]}/metadata'.encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata['n_frames']

        # Crop the audio and coef
        if self.crop_strategy == 'random':
            start_frame = np.random.randint(0, seq_len - self.coef_total_len + 1)
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'end':
            start_frame = seq_len - self.coef_total_len
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
        coef_keys = ['shape', 'exp', 'pose']
        coef_dict = {k: [] for k in coef_keys}
        audio = []
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip):
                key = f'{self.entries[index]}/{clip_idx:03d}'.encode()
                start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                entry = pickle.loads(txn.get(key))
                for coef_key in coef_keys:
                    coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])

                audio_data = entry['audio']
                audio_clip, sr = torchaudio.load(io.BytesIO(audio_data))
                assert sr == 16000, f'Invalid sampling rate: {sr}'
                audio_clip = audio_clip.squeeze()
                audio.append(audio_clip[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)])

        coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
        assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'
        audio = torch.cat(audio, dim=0)
        assert audio.shape[0] == self.coef_total_len * self.audio_unit, f'Invalid audio length: {audio.shape[0]}'
        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        if self.rot_representation == 'aa':
            keys = ['shape', 'exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_representation}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                         for k in keys}

        # Extract two consecutive audio/coef clips
        audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
        coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                     {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]

        return audio_pair, coef_pair, (audio_mean, audio_std)
    def get_inference_item(self, index):
        # Read audio and coef
        
        with self.lmdb_env.begin(write=False) as txn:
            meta_key = f'{self.entries[index]}/metadata'.encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata['n_frames']
        # Crop the audio and coef
        start_frame = 0

        coef_keys = ['shape', 'exp', 'pose']
        coef_dict = {k: [] for k in coef_keys}
        audio = []
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
        end_clip = seq_len // self.clip_len
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip):
                key = f'{self.entries[index]}/{clip_idx:03d}'.encode()
                # start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                # end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                entry = pickle.loads(txn.get(key))
                for coef_key in coef_keys:
                    # coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])
                    coef_dict[coef_key].append(entry['coef'][coef_key])

                audio_data = entry['audio']
                audio_clip, sr = torchaudio.load(io.BytesIO(audio_data))
                assert sr == 16000, f'Invalid sampling rate: {sr}'
                audio_clip = audio_clip.squeeze()
                # audio.append(audio_clip[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)])
                audio.append(audio_clip)
        coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
        audio = torch.cat(audio, dim=0)
        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        if self.rot_representation == 'aa':
            keys = ['shape', 'exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_representation}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                         for k in keys}

        return audio, coef_dict, (audio_mean, audio_std)
        
class LmdbDatasetForSE(data.Dataset):
    def __init__(self, lmdb_dir, split_file, coef_stats_file=None, coef_fps=25, n_motions=100, crop_strategy='random',
                 rot_repr='aa', no_head_pose=False):
        self.split_file = split_file
        self.lmdb_dir = Path(lmdb_dir)
        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

        self.crop_strategy = crop_strategy
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose

        # Read split file
        self.entries = defaultdict(list)
        with open(self.split_file, 'r') as f:
            for line in f:
                person_id = line.strip().split('/')[0]
                self.entries[person_id].append(line.strip().split()[0])
        self.person_ids = list(self.entries.keys())

        # Load lmdb
        # lmdb_env = lmdb.open((b"/data/HDTF_TFHP/lmdb/"), readonly=True, lock=False, readahead=False, meminit=False)
        self.lmdb_env = lmdb.open(str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        # Read coef
        with self.lmdb_env.begin(write=False) as txn:
            key = random.choice(self.entries[self.person_ids[index]])
            meta_key = f'{key}/metadata'.encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata['n_frames']

        # Crop the audio and coef
        if self.crop_strategy == 'random':
            start_frame = np.random.randint(0, seq_len - self.coef_total_len + 1)
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'end':
            start_frame = seq_len - self.coef_total_len
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        coef_keys = ['exp', 'pose']
        coef_dict = {k: [] for k in coef_keys}
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip):
                clip_key = f'{key}/{clip_idx:03d}'.encode()
                start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                entry = pickle.loads(txn.get(clip_key))
                for coef_key in coef_keys:
                    coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])

        coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
        assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

        if self.rot_representation == 'aa':
            coef_keys = ['exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_representation}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                         for k in coef_keys}

        if self.no_head_pose:
            if self.rot_representation == 'aa':
                mouth_pose_coef = coef_dict['pose'][:, 3:]
            else:
                raise ValueError(f'Unknown rotation representation: {self.rot_representation}')
            motion_coef = torch.cat([coef_dict['exp'], mouth_pose_coef], dim=-1)
        else:
            motion_coef = torch.cat([coef_dict[k] for k in coef_keys], dim=-1)

        if self.rot_representation == 'aa':
            # Remove mouth rotation around y, z axis
            motion_coef = motion_coef[:, :-2]

        # Extract two consecutive coef clips
        coef_pair = [motion_coef[:self.n_motions].clone(), motion_coef[-self.n_motions:].clone()]

        return coef_pair

class FileBased_celebv_text_dataset(data.Dataset):
    def __init__(self):
        self.DATA_ROOT = "/data/celebv-text/"
        self.AUDIO_ROOT = os.path.join(self.DATA_ROOT, "audio", "celebvtext_audio")
        self.VIDEO_ROOT = os.path.join(self.DATA_ROOT, "Videos", "celebvtext_6")
        self.EXPRESSION_CODE_ROOT = os.path.join(self.DATA_ROOT, "expression_code_ver2")
        self.HEAD_ROTATION_ROOT = os.path.join(self.DATA_ROOT, "head_orientations")
        
        self.VIDEO_KEY_PATH = os.path.join(self.DATA_ROOT, "processed_data", "processed_data_30fps_medium_v3_keys_valid.txt")
        # load video keys
        self.keys = pd.read_csv(self.VIDEO_KEY_PATH, header=None).values.squeeze()

        # load coef_stats_dict 
        with open(os.path.join(self.DATA_ROOT, "processed_data", "celebv_text_v3_coeff_stats.pkl"), "rb") as f:
            self.coef_stats = pickle.load(f)
    def __len__(self):
        return len(self.keys)
    
    def query_for_video(self, video_name, device="cuda"):
        expression_coef_path = os.path.join(self.EXPRESSION_CODE_ROOT, video_name + "_code_savgol_boundbox+smooth_expression.pkl")
        head_rot_path = os.path.join(self.HEAD_ROTATION_ROOT, video_name + ".pkl")
        video_path = os.path.join(self.VIDEO_ROOT, f"{video_name}.mp4")
        expression_coef = pickle.load(open(expression_coef_path, "rb")).detach().cpu().numpy()
        head_rot = pickle.load(open(head_rot_path, "rb"))
        # normalize expression_coef and head_rot
        expression_coef = (expression_coef - self.coef_stats['exp_mean'].detach().cpu().numpy()) / (self.coef_stats['exp_std'].detach().cpu().numpy() + 1e-9)
        head_rot = (head_rot - self.coef_stats['pose_mean'].detach().cpu().numpy()) / (self.coef_stats['pose_std'].detach().cpu().numpy() + 1e-9)
        audio = librosa.load(os.path.join(self.AUDIO_ROOT, f"{video_name}.m4a"), sr=16000)[0]
        # normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        # get FPS
        gt_video_data = cv2.VideoCapture(video_path)
        gt_fps = gt_video_data.get(cv2.CAP_PROP_FPS)
        # convert exp_coef and head_rot to the correct FPS
        x = np.linspace(0, 1, num=expression_coef.shape[0])
        xnew = np.linspace(0, 1, num=int(round(expression_coef.shape[0]/gt_fps*25)))
        # resample expression coef
        f_exp = interp1d(x, expression_coef, axis=0)
        expression_coef = f_exp(xnew)
        # also resample head_orientation to 25 fps
        f_head = interp1d(x, head_rot, axis=0)
        head_rot = f_head(xnew)
        # convert to tensor and cast to device
        expression_coef = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
        head_rot = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
        shape_coef = torch.zeros([1, 100], device=device).float()
        motion_coeff = torch.cat([expression_coef, head_rot], dim=2).float().to(device)
        # get FPS
        
        return audio, motion_coeff, shape_coef

class FileBased_ravdess(data.Dataset):
    def __init__(self):
        self.DATA_ROOT = "/data/ravdess/"
        self.AUDIO_ROOT = os.path.join(self.DATA_ROOT, "audio")
        self.VIDEO_ROOT = os.path.join(self.DATA_ROOT, "videos")
        self.EXPRESSION_CODE_ROOT = os.path.join(self.DATA_ROOT, "expression_code_ver2")
        self.HEAD_ROTATION_ROOT = os.path.join(self.DATA_ROOT, "head_orientations")
        
        self.VIDEO_KEY_PATH = os.path.join(self.DATA_ROOT, "processed_data", "processed_ravdess_30fps_v3_keys.txt")
        # load video keys
        self.keys = pd.read_csv(self.VIDEO_KEY_PATH, header=None).values.squeeze()

        # load coef_stats_dict 
        with open(os.path.join("/data/celebv-text/", "processed_data", "celebv_text_v3_coeff_stats.pkl"), "rb") as f:
            self.coef_stats = pickle.load(f)
    def __len__(self):
        return len(self.keys)
    def get_k_indices_for_each_emotion(self, k=1, do_randommize=True):        
        emotion_to_videos_dict = {}
        for key in self.keys:
            attributes_if_mead = key.split("_") # split the file name to see the attribute of the video
            attributes_if_ravdess = key.split("-")
            if len(attributes_if_mead) == 5:
                emotion = attributes_if_mead[1]
                if "mead_"+emotion not in emotion_to_videos_dict:
                    emotion_to_videos_dict["mead_"+emotion] = []
                emotion_to_videos_dict["mead_"+emotion].append(key)
            elif len(attributes_if_ravdess) == 7:
                emotion = attributes_if_ravdess[2]
                if emotion not in emotion_to_videos_dict:
                    emotion_to_videos_dict[emotion] = []
                if (not attributes_if_ravdess[1] == "02") and (attributes_if_ravdess[3] == "02" or attributes_if_ravdess[2] == "01"):
                    emotion_to_videos_dict[emotion].append(key)
        # select indexes from each emotion
        output_indexes = {}
        for emotion in emotion_to_videos_dict:
            if do_randommize:
                indexes = np.random.choice(len(emotion_to_videos_dict[emotion]), k)
            else:
                indexes = list(range(k))
            output_indexes[emotion] = [emotion_to_videos_dict[emotion][kk] for kk in indexes]
        return output_indexes

    def query_for_video(self, video_name, device="cuda"):
        expression_coef_path = os.path.join(self.EXPRESSION_CODE_ROOT, video_name + "_code_savgol_boundbox+smooth_expression.pkl")
        head_rot_path = os.path.join(self.HEAD_ROTATION_ROOT, video_name + ".pkl")
        video_path = os.path.join(self.VIDEO_ROOT, f"{video_name}.mp4")
        expression_coef = pickle.load(open(expression_coef_path, "rb")).detach().cpu().numpy()
        head_rot = pickle.load(open(head_rot_path, "rb"))
        # normalize expression_coef and head_rot
        expression_coef = (expression_coef - self.coef_stats['exp_mean'].detach().cpu().numpy()) / (self.coef_stats['exp_std'].detach().cpu().numpy() + 1e-9)
        head_rot = (head_rot - self.coef_stats['pose_mean'].detach().cpu().numpy()) / (self.coef_stats['pose_std'].detach().cpu().numpy() + 1e-9)
        audio = librosa.load(os.path.join(self.AUDIO_ROOT, f"{video_name}.wav"), sr=16000)[0]
        # normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        # get FPS
        # gt_video_data = cv2.VideoCapture(video_path)
        # gt_fps = gt_video_data.get(cv2.CAP_PROP_FPS)
        gt_fps = 29.97
        # convert exp_coef and head_rot to the correct FPS
        x = np.linspace(0, 1, num=expression_coef.shape[0])
        xnew = np.linspace(0, 1, num=int(round(expression_coef.shape[0]/gt_fps*25)))
        # resample expression coef
        f_exp = interp1d(x, expression_coef, axis=0)
        expression_coef = f_exp(xnew)
        # also resample head_orientation to 25 fps
        f_head = interp1d(x, head_rot, axis=0)
        head_rot = f_head(xnew)
        # convert to tensor and cast to device
        expression_coef = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
        head_rot = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
        shape_coef = torch.zeros([1, 100], device=device).float()
        motion_coeff = torch.cat([expression_coef, head_rot], dim=2).float().to(device)
        # get FPS
        
        return audio, motion_coeff, shape_coef

class FileBased_arbitrary(data.Dataset):
    def __init__(self):
        self.DATA_ROOT = "/data/ravdess/"
        self.AUDIO_ROOT = os.path.join(self.DATA_ROOT, "audio")
        self.VIDEO_ROOT = os.path.join(self.DATA_ROOT, "videos")
        self.EXPRESSION_CODE_ROOT = os.path.join(self.DATA_ROOT, "expression_code_ver2")
        self.HEAD_ROTATION_ROOT = os.path.join(self.DATA_ROOT, "head_orientations")
        
        self.VIDEO_KEY_PATH = os.path.join(self.DATA_ROOT, "processed_data", "processed_ravdess_30fps_v3_keys.txt")
        # load video keys
        self.keys = pd.read_csv(self.VIDEO_KEY_PATH, header=None).values.squeeze()

        # load coef_stats_dict 
        with open(os.path.join("/data/celebv-text/", "processed_data", "celebv_text_v3_coeff_stats.pkl"), "rb") as f:
            self.coef_stats = pickle.load(f)
    def __len__(self):
        return len(self.keys)

    def query_for_video(self, expression_coef_path, head_rot_path, audio_path, device="cuda"):
        # expression_coef_path = os.path.join(self.EXPRESSION_CODE_ROOT, video_name + "_code_savgol_boundbox+smooth_expression.pkl")
        # head_rot_path = os.path.join(self.HEAD_ROTATION_ROOT, video_name + ".pkl")
        expression_coef = pickle.load(open(expression_coef_path, "rb"))
        if type(expression_coef) == torch.Tensor:
            expression_coef = expression_coef.detach().cpu().numpy()
        head_rot = pickle.load(open(head_rot_path, "rb"))
        # normalize expression_coef and head_rot
        expression_coef = (expression_coef - self.coef_stats['exp_mean'].detach().cpu().numpy()) / (self.coef_stats['exp_std'].detach().cpu().numpy() + 1e-9)
        head_rot = (head_rot - self.coef_stats['pose_mean'].detach().cpu().numpy()) / (self.coef_stats['pose_std'].detach().cpu().numpy() + 1e-9)
        audio = librosa.load(audio_path, sr=16000)[0]
        # normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        # get FPS
        # gt_video_data = cv2.VideoCapture(video_path)
        # gt_fps = gt_video_data.get(cv2.CAP_PROP_FPS)
        gt_fps = 29.97
        # convert exp_coef and head_rot to the correct FPS
        x = np.linspace(0, 1, num=expression_coef.shape[0])
        xnew = np.linspace(0, 1, num=int(round(expression_coef.shape[0]/gt_fps*25)))
        # resample expression coef
        f_exp = interp1d(x, expression_coef, axis=0)
        expression_coef = f_exp(xnew)
        # also resample head_orientation to 25 fps
        f_head = interp1d(x, head_rot, axis=0)
        head_rot = f_head(xnew)
        # convert to tensor and cast to device
        expression_coef = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
        head_rot = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
        shape_coef = torch.zeros([1, 100], device=device).float()
        motion_coeff = torch.cat([expression_coef, head_rot], dim=2).float().to(device)
        # get FPS
        
        return audio, motion_coeff, shape_coef

class FileBased_iconic_dataset(data.Dataset):
    def __init__(self):
        self.DATA_ROOT = "/data/evan_iconic_speech/"
        self.AUDIO_ROOT = os.path.join(self.DATA_ROOT, "audio")
        self.VIDEO_ROOT = os.path.join(self.DATA_ROOT, "videos")
        self.EXPRESSION_CODE_ROOT = os.path.join(self.DATA_ROOT, "expression_code_ver2")
        self.HEAD_ROTATION_ROOT = os.path.join(self.DATA_ROOT, "head_orientations")
        
        # load video keys
        self.keys = os.listdir(self.AUDIO_ROOT)
        new_keys = []
        for i in range(len(self.keys)):
            if self.keys[i].endswith(".wav"):
                new_keys.append(self.keys[i].split(".")[0])
        self.keys = new_keys

        # load coef_stats_dict 
        with open(os.path.join("/data/celebv-text", "processed_data", "celebv_text_v3_coeff_stats.pkl"), "rb") as f:
            self.coef_stats = pickle.load(f)
    def __len__(self):
        return len(self.keys)
    
    def query_for_video(self, video_name, device="cuda"):
        expression_coef_path = os.path.join(self.EXPRESSION_CODE_ROOT, video_name + "_code_savgol_boundbox+smooth_expression.pkl")
        head_rot_path = os.path.join(self.HEAD_ROTATION_ROOT, video_name + ".pkl")
        video_path = os.path.join(self.VIDEO_ROOT, f"{video_name}.mp4")
        expression_coef = pickle.load(open(expression_coef_path, "rb")).detach().cpu().numpy()
        head_rot = pickle.load(open(head_rot_path, "rb"))
        # normalize expression_coef and head_rot
        expression_coef = (expression_coef - self.coef_stats['exp_mean'].detach().cpu().numpy()) / (self.coef_stats['exp_std'].detach().cpu().numpy() + 1e-9)
        head_rot = (head_rot - self.coef_stats['pose_mean'].detach().cpu().numpy()) / (self.coef_stats['pose_std'].detach().cpu().numpy() + 1e-9)
        audio = librosa.load(os.path.join(self.AUDIO_ROOT, f"{video_name}.wav"), sr=16000)[0]
        # normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        # get FPS
        gt_video_data = cv2.VideoCapture(video_path)
        gt_fps = gt_video_data.get(cv2.CAP_PROP_FPS)
        # convert exp_coef and head_rot to the correct FPS
        x = np.linspace(0, 1, num=expression_coef.shape[0])
        xnew = np.linspace(0, 1, num=int(round(expression_coef.shape[0]/gt_fps*25)))
        # resample expression coef
        f_exp = interp1d(x, expression_coef, axis=0)
        expression_coef = f_exp(xnew)
        # also resample head_orientation to 25 fps
        f_head = interp1d(x, head_rot, axis=0)
        head_rot = f_head(xnew)
        # convert to tensor and cast to device
        expression_coef = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
        head_rot = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
        shape_coef = torch.zeros([1, 100], device=device).float()
        motion_coeff = torch.cat([expression_coef, head_rot], dim=2).float().to(device)
        audio = torch.from_numpy(audio).to(device).float()
        # get FPS
        
        return audio, motion_coeff, shape_coef

    @staticmethod
    def get_collate_fn_legacy(SE):
        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(0, len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                for i in range(0, len(batch)):
                    audio_0.append(batch[i][0][0])
                    audio_1.append(batch[i][0][1])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])

                # stack them in the first dimension
                audio_0 = torch.stack(audio_0, dim=0)
                audio_1 = torch.stack(audio_1, dim=0)
                motion_0 = torch.stack(motion_0, dim=0)
                motion_1 = torch.stack(motion_1, dim=0)
                shape_0 = torch.stack(shape_0, dim=0)
                shape_1 = torch.stack(shape_1, dim=0)

                # aggregate the audio mean and std
                audio_mean = torch.tensor(audio_mean).to(motion_0.device).float()
                audio_std = torch.tensor(audio_std).to(motion_0.device).float()

                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                # compute mean of mena
                audio_mean = audio_mean.mean()
                # compute mean of std
                audio_std = audio_std.mean()

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
        return collate_fn   
    
    @staticmethod
    def get_collate_fn(SE):
        def pad_or_trim_audio(audio_tensor, target_length=64000):
            """Helper function to ensure all audio tensors have the same length"""
            current_length = audio_tensor.size(0)
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
            elif current_length > target_length:
                # Trim to target length
                return audio_tensor[:target_length]
            return audio_tensor

        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                # First pass: determine max audio length in batch
                target_length = 64000  # Fixed target length for audio

                # Process each item in batch
                for i in range(len(batch)):
                    # Pad or trim audio to target length
                    audio_0_padded = pad_or_trim_audio(batch[i][0][0], target_length)
                    audio_1_padded = pad_or_trim_audio(batch[i][0][1], target_length)
                    
                    # Append processed items
                    audio_0.append(audio_0_padded)
                    audio_1.append(audio_1_padded)
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])

                try:
                    # Stack all tensors
                    audio_0 = torch.stack(audio_0, dim=0)
                    audio_1 = torch.stack(audio_1, dim=0)
                    motion_0 = torch.stack(motion_0, dim=0)
                    motion_1 = torch.stack(motion_1, dim=0)
                    shape_0 = torch.stack(shape_0, dim=0)
                    shape_1 = torch.stack(shape_1, dim=0)
                except RuntimeError as e:
                    shapes_info = {
                        'audio_0': [x.shape for x in audio_0],
                        'audio_1': [x.shape for x in audio_1],
                        'motion_0': [x.shape for x in motion_0],
                        'motion_1': [x.shape for x in motion_1],
                        'shape_0': [x.shape for x in shape_0],
                        'shape_1': [x.shape for x in shape_1]
                    }
                    raise RuntimeError(f"Failed to stack tensors. Shapes: {shapes_info}. Original error: {str(e)}")

                # Process audio statistics
                audio_mean = torch.tensor(audio_mean).float().mean()
                audio_std = torch.tensor(audio_std).float().mean()

                # Create coefficient dictionaries
                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)

        return collate_fn            

# self is an empty class
class PickleDataset_Evan(data.Dataset):

    @staticmethod
    def load_dict_in_chunks_static(file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    yield chunk
                except EOFError:
                    break

    def load_dict_in_chunks(self, file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    yield chunk
                except EOFError:
                    break  # End of file reached
    
    def __init__(self, pkl_file, split_file, coef_stats_file=None, original_fps=30, coef_fps=25, n_motions=100,
                 rot_repr='aa', no_head_pose=False, clip_len=100, device='cpu', SE=True, full_dataset=False, pre_loaded_raw_dataset=None, celebv_text=True, random_crop=True, batch_overfit_size=-1):
        self.split_file = split_file
        self.pkl_file = pkl_file
        self.valid_id = []
        # load the valid id file (only for celebv-text)
        if celebv_text:
            with open("/data/celebv-text/keys.txt", 'r') as f:
                for line in f:
                    self.valid_id.append(line.strip())
        # load the split file
        self.file_names = []
        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip()
                # self.file_names.append(line.strip())
                if celebv_text:
                    if name in self.valid_id:
                        self.file_names.append(name)
                else:
                    self.file_names.append(name)
        # if overfit_mode is not -1, only take the first overfit_mode entries
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]

        # load the data 
        if pre_loaded_raw_dataset is not None:
            raw_data = pre_loaded_raw_dataset
        elif not full_dataset:
            raw_data = pickle.load(open(pkl_file, 'rb'))
        else:
            raw_data = {}
            for chunk in self.load_dict_in_chunks(pkl_file):
                raw_data.update(chunk)
        self.data = {}
        for key in self.file_names:
            self.data[key] = raw_data[key]
        
        # resample the data to 25 fps:
        # resample the head_orientation and expression_code to 25 fps from 30 fps
        if original_fps != coef_fps:
            for key in self.file_names:
                original_dict = self.data[key]
                original_expression_code = original_dict["expression_code"]
                original_head_orientation = original_dict["head_orientation"]
                new_dict = {"audio": original_dict["audio"]}
                # resample expressioncode to 25 fps down from 30 using interp1d
                # original_expression_code.shape
                x = np.linspace(0, 1, num=original_expression_code.shape[0])
                xnew = np.linspace(0, 1, num=int(round(original_expression_code.shape[0]/original_fps*coef_fps)))
                f_exp = interp1d(x, original_expression_code, axis=0)
                new_expression_code = f_exp(xnew)
                # also resample head_orientation to 25 fps
                f_head = interp1d(x, original_head_orientation, axis=0)
                new_head_orientation = f_head(xnew)
                new_dict["expression_code"] = new_expression_code
                new_dict["head_orientation"] = new_head_orientation
                self.data[key] = new_dict
                del original_dict
        print("finished data resampling")

        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file, allow_pickle=True))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')
        self.device = device
        self.coef_fps = coef_fps
        self.clip_len = clip_len
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.random_crop = random_crop
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose
        self.SE = SE


        # Read split file
        self.entries = self.file_names
        if coef_stats_file is None:
            exp_mean, exp_std, pose_mean, pose_std = incremental_mean_and_std(self, self.SE)
            self.coef_stats = {}
            self.coef_stats['exp_mean'] = exp_mean   
            self.coef_stats['exp_std'] = exp_std
            self.coef_stats['pose_mean'] = pose_mean
            self.coef_stats['pose_std'] = pose_std
        self.coef_stats = {x: torch.tensor(self.coef_stats[x]).float() for x in self.coef_stats}

    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["expression_code"]
        head_orientation = clip_dict["head_orientation"]

        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the goal clip
        goal_total_length = self.coef_total_len
        goal_each_clip_length = self.clip_len
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # select a starting frame to ensure that after cropping, the second clip will have at least half of goal_each_clip_length
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
                end_frame1 = start_frame1 + goal_each_clip_length
                start_frame2 = start_frame1 + goal_each_clip_length
                end_frame2 = start_frame2 + goal_each_clip_length
            elif current_clip_length == goal_total_length:
                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            else:
                frames_to_pad = goal_total_length - current_clip_length
                # split this down the middle randomly
                frames_to_pad_front = np.random.randint(0, frames_to_pad)
                frames_to_pad_back = frames_to_pad - frames_to_pad_front
                frames_to_pad_front = int(round(frames_to_pad_front))
                frames_to_pad_back = int(round(frames_to_pad_back))
                expression_code = np.pad(expression_code, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                head_orientation = np.pad(head_orientation, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                
                # audio frames to pad = frames_to_pad * audio_unit
                # note that the audio might be slightly shorter or longer than the video, so we need to pad the audio twice
                audio_frames_to_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_frames_to_pad_back = int(round(frames_to_pad_back * self.audio_unit))
                
                audio = np.pad(audio, ((int(audio_frames_to_pad_front), int(audio_frames_to_pad_back))), 'constant', constant_values=0)
                audio_length = audio.shape[0]

                # if the audio is still shorter than the goal length, pad it with zeros at the end (this is not elegant but what can we do......)
                audio_minimal_length = goal_total_length * self.audio_unit
                audio_minimal_length = int(round(audio_minimal_length))
                if audio_length < audio_minimal_length:
                    audio = np.pad(audio, (0, audio_minimal_length - audio_length), 'constant', constant_values=0)

                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            # Crop the audio and coef
        else:
            start_frame1 = 0
            end_frame1 = goal_each_clip_length
            start_frame2 = goal_each_clip_length
            end_frame2 = goal_each_clip_length * 2

            # pad the audio and coef at the end 
            expression_code = np.pad(expression_code, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            head_orientation = np.pad(head_orientation, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            audio = np.pad(audio, (0, int(round(goal_total_length * self.audio_unit)) - audio.shape[0]), 'constant', constant_values=0)


        expression_code_frame_0 = expression_code[start_frame1:end_frame1]
        expression_code_frame_1 = expression_code[start_frame2:end_frame2]
        head_orientation_frame_0 = head_orientation[start_frame1:end_frame1]
        head_orientation_frame_1 = head_orientation[start_frame2:end_frame2]
        audio_frame_0 = audio[int(start_frame1 * self.audio_unit):int(end_frame1 * self.audio_unit)]
        audio_frame_1 = audio[int(start_frame2 * self.audio_unit):int(end_frame2 * self.audio_unit)]


        # if audio_frame_0.shape[0] != 64000 or audio_frame_1.shape[0] != 64000:
        #     print(self.entries[index])
        #     print(start_frame1, end_frame1, start_frame2, end_frame2)
        #     print(current_clip_length, goal_total_length)
        #     print(audio_frame_0.shape, audio_frame_1.shape)
        #     print(audio.shape, expression_code.shape, head_orientation.shape)

        # concatenate expression and head orientation
        
        expression_code_frame_0 = torch.tensor(expression_code_frame_0).float()
        expression_code_frame_1 = torch.tensor(expression_code_frame_1).float()
        head_orientation_frame_0 = torch.tensor(head_orientation_frame_0).float()
        head_orientation_frame_1 = torch.tensor(head_orientation_frame_1).float()

        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code_frame_0 = (expression_code_frame_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            expression_code_frame_1 = (expression_code_frame_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            head_orientation_frame_0 = (head_orientation_frame_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            head_orientation_frame_1 = (head_orientation_frame_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        
        motion_coef_frame_0 = torch.cat([expression_code_frame_0, head_orientation_frame_0], axis=-1)
        motion_coef_frame_1 = torch.cat([expression_code_frame_1, head_orientation_frame_1], axis=-1)

        shape_frame_0 = torch.zeros((motion_coef_frame_0.shape[0], 100)).float()
        shape_frame_1 = torch.zeros((motion_coef_frame_1.shape[0], 100)).float()

        coef_dict_0 = {"shape": shape_frame_0, "motion": motion_coef_frame_0}
        coef_dict_1 = {"shape": shape_frame_1, "motion": motion_coef_frame_1}

        # turning all the numpy arrays into torch tensors
        audio_frame_0 = torch.tensor(audio_frame_0).float()
        audio_frame_1 = torch.tensor(audio_frame_1).float()
        
        if self.SE:
            return [motion_coef_frame_0, motion_coef_frame_1]
        else:
            return [audio_frame_0, audio_frame_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)
    
    def get_k_indices_for_each_emotion(self, k=2):
        # 1 to 8
        print(self.entries)
        emotions = ["01", "02", "03", "04", "05", "06", "07", "08"]
        emotion_indices = {}
        for emotion in emotions:
            emotion_indices[emotion] = []
            for count in range(0, k):
                # randomly sample from the entries
                index = np.random.randint(0, len(self.entries))
                count = 0
                while self.entries[index].split("-")[2] != emotion:
                    # print(emotion, self.entries[index].split("-")[2])
                    index = np.random.randint(0, len(self.entries))
                    count += 1
                    if count >= 100:
                        break
                if self.entries[index].split("-")[2] == emotion:
                    emotion_indices[emotion].append(index)
                else:
                    continue
        return emotion_indices

    def query_for_video(self, index):
        video_name = self.entries[index]
        if not video_name in self.file_names:
            Exception("Video name not found in the dataset")
        expression_code = self.data[video_name]["expression_code"]
        pose = self.data[video_name]["head_orientation"]
        audio = self.data[video_name]["audio"]
        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # concatenate expression and head orientation
        expression_code = torch.tensor(expression_code).float()
        pose = torch.tensor(pose).float()
        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
        
        motion = torch.cat([expression_code, pose], axis=-1)
        shape = torch.zeros((motion.shape[0], 100)).float()
        coef_dict = {"shape": shape, "motion": motion}
        
        # turning all the numpy arrays into torch tensors
        audio = torch.tensor(audio).float()

        return audio, coef_dict, (audio_mean, audio_std)


    @staticmethod
    def get_collate_fn_legacy(SE):
        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(0, len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                for i in range(0, len(batch)):
                    audio_0.append(batch[i][0][0])
                    audio_1.append(batch[i][0][1])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])

                # stack them in the first dimension
                audio_0 = torch.stack(audio_0, dim=0)
                audio_1 = torch.stack(audio_1, dim=0)
                motion_0 = torch.stack(motion_0, dim=0)
                motion_1 = torch.stack(motion_1, dim=0)
                shape_0 = torch.stack(shape_0, dim=0)
                shape_1 = torch.stack(shape_1, dim=0)

                # aggregate the audio mean and std
                audio_mean = torch.tensor(audio_mean).to(motion_0.device).float()
                audio_std = torch.tensor(audio_std).to(motion_0.device).float()

                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                # compute mean of mena
                audio_mean = audio_mean.mean()
                # compute mean of std
                audio_std = audio_std.mean()

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
        return collate_fn   
    
    @staticmethod
    def get_collate_fn(SE):
        def pad_or_trim_audio(audio_tensor, target_length=64000):
            """Helper function to ensure all audio tensors have the same length"""
            current_length = audio_tensor.size(0)
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
            elif current_length > target_length:
                # Trim to target length
                return audio_tensor[:target_length]
            return audio_tensor

        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                # First pass: determine max audio length in batch
                target_length = 64000  # Fixed target length for audio

                # Process each item in batch
                for i in range(len(batch)):
                    # Pad or trim audio to target length
                    audio_0_padded = pad_or_trim_audio(batch[i][0][0], target_length)
                    audio_1_padded = pad_or_trim_audio(batch[i][0][1], target_length)
                    
                    # Append processed items
                    audio_0.append(audio_0_padded)
                    audio_1.append(audio_1_padded)
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])

                try:
                    # Stack all tensors
                    audio_0 = torch.stack(audio_0, dim=0)
                    audio_1 = torch.stack(audio_1, dim=0)
                    motion_0 = torch.stack(motion_0, dim=0)
                    motion_1 = torch.stack(motion_1, dim=0)
                    shape_0 = torch.stack(shape_0, dim=0)
                    shape_1 = torch.stack(shape_1, dim=0)
                except RuntimeError as e:
                    shapes_info = {
                        'audio_0': [x.shape for x in audio_0],
                        'audio_1': [x.shape for x in audio_1],
                        'motion_0': [x.shape for x in motion_0],
                        'motion_1': [x.shape for x in motion_1],
                        'shape_0': [x.shape for x in shape_0],
                        'shape_1': [x.shape for x in shape_1]
                    }
                    raise RuntimeError(f"Failed to stack tensors. Shapes: {shapes_info}. Original error: {str(e)}")

                # Process audio statistics
                audio_mean = torch.tensor(audio_mean).float().mean()
                audio_std = torch.tensor(audio_std).float().mean()

                # Create coefficient dictionaries
                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)

        return collate_fn            

class LMDB_Dataset_Evan(data.Dataset):    
    # UPDATED __init__ signature and logic
    def __init__(self, lmdb_file, split_file, coef_stats_file=None, original_fps=30, coef_fps=25, n_motions=100,
                 rot_repr='aa', no_head_pose=False, clip_len=100, device='cpu', SE=True, 
                 celebv_text=True, random_crop=True, batch_overfit_size=-1): # <-- ADDED arguments
        self.split_file = split_file
        self.lmdb_file = lmdb_file
        self.valid_id = []
        # load the valid id file
        if celebv_text:
            with open("/data/celebv-text/keys.txt", 'r') as f:
                for line in f:
                    self.valid_id.append(line.strip())
        # load the split file
        self.file_names = []
        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip()
                if celebv_text:
                    if name in self.valid_id:
                        self.file_names.append(name)
                else:
                    self.file_names.append(name)
        
        # --- ADDED: Overfitting functionality ---
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]

        # load the data 
        # Note: pre_loaded_raw_dataset argument was removed as it's less common for LMDB
        raw_data = {}
        env = lmdb.open(str(lmdb_file), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            for key in tqdm(self.file_names):
                raw_data[key] = pickle.loads(txn.get(key.strip().encode()))

        self.data = {}
        for key in self.file_names:
            self.data[key] = raw_data[key]
        
        # resample the head_orientation and expression_code to 25 fps from 30 fps
        if original_fps != coef_fps:
            for key in self.file_names:
                original_dict = self.data[key]
                original_expression_code = original_dict["expression_code"]
                original_head_orientation = original_dict["head_orientation"]
                new_dict = {"audio": original_dict["audio"]}
                
                total_frames_original = original_expression_code.shape[0]
                x = np.linspace(0, 1, num=total_frames_original)
                num_new_frames = int(round(total_frames_original / original_fps * coef_fps))
                xnew = np.linspace(0, 1, num=num_new_frames)
                
                f_exp = interp1d(x, original_expression_code, axis=0)
                new_expression_code = f_exp(xnew)
                
                f_head = interp1d(x, original_head_orientation, axis=0)
                new_head_orientation = f_head(xnew)
                
                new_dict["expression_code"] = new_expression_code
                new_dict["head_orientation"] = new_head_orientation
                self.data[key] = new_dict
                del original_dict
            print("finished data resampling")

        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')
        
        self.device = device
        self.coef_fps = coef_fps
        self.clip_len = clip_len
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        
        # --- ADDED: Store random_crop flag ---
        self.random_crop = random_crop 
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose
        self.SE = SE

        self.entries = self.file_names
        if coef_stats_file is None:
            exp_mean, exp_std, pose_mean, pose_std = incremental_mean_and_std(self, self.SE)
            self.coef_stats = {}
            self.coef_stats['exp_mean'] = exp_mean  
            self.coef_stats['exp_std'] = exp_std
            self.coef_stats['pose_mean'] = pose_mean
            self.coef_stats['pose_std'] = pose_std
        self.coef_stats = {x: torch.tensor(self.coef_stats[x]).float() for x in self.coef_stats}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["expression_code"]
        head_orientation = clip_dict["head_orientation"]

        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        goal_total_length = self.coef_total_len
        goal_each_clip_length = self.clip_len
        current_clip_length = expression_code.shape[0]
        
        # --- UPDATED: Flexible cropping/padding logic ---
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
            else:
                start_frame1 = 0
            
            if current_clip_length < goal_total_length:
                frames_to_pad = goal_total_length - current_clip_length
                frames_to_pad_front = np.random.randint(0, frames_to_pad + 1)
                frames_to_pad_back = frames_to_pad - frames_to_pad_front
                
                expression_code = np.pad(expression_code, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant')
                head_orientation = np.pad(head_orientation, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant')
                
                audio_frames_to_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_frames_to_pad_back = int(round(frames_to_pad_back * self.audio_unit))
                audio = np.pad(audio, (audio_frames_to_pad_front, audio_frames_to_pad_back), 'constant')

                audio_minimal_length = int(round(goal_total_length * self.audio_unit))
                if audio.shape[0] < audio_minimal_length:
                    audio = np.pad(audio, (0, audio_minimal_length - audio.shape[0]), 'constant')
        else: # Deterministic padding (at the end)
             start_frame1 = 0
             if current_clip_length < goal_total_length:
                pad_len = goal_total_length - current_clip_length
                expression_code = np.pad(expression_code, ((0, pad_len), (0, 0)), 'constant')
                head_orientation = np.pad(head_orientation, ((0, pad_len), (0, 0)), 'constant')
                
                audio_pad_len = int(round(goal_total_length * self.audio_unit)) - audio.shape[0]
                if audio_pad_len > 0:
                    audio = np.pad(audio, (0, audio_pad_len), 'constant')
        
        end_frame1 = start_frame1 + goal_each_clip_length
        start_frame2 = start_frame1 + goal_each_clip_length
        end_frame2 = start_frame2 + goal_each_clip_length

        expression_code_frame_0 = expression_code[start_frame1:end_frame1]
        expression_code_frame_1 = expression_code[start_frame2:end_frame2]
        head_orientation_frame_0 = head_orientation[start_frame1:end_frame1]
        head_orientation_frame_1 = head_orientation[start_frame2:end_frame2]
        audio_frame_0 = audio[int(start_frame1 * self.audio_unit):int(end_frame1 * self.audio_unit)]
        audio_frame_1 = audio[int(start_frame2 * self.audio_unit):int(end_frame2 * self.audio_unit)]
        
        expression_code_frame_0 = torch.tensor(expression_code_frame_0).float()
        expression_code_frame_1 = torch.tensor(expression_code_frame_1).float()
        head_orientation_frame_0 = torch.tensor(head_orientation_frame_0).float()
        head_orientation_frame_1 = torch.tensor(head_orientation_frame_1).float()

        if self.coef_stats is not None:
            expression_code_frame_0 = (expression_code_frame_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            expression_code_frame_1 = (expression_code_frame_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            head_orientation_frame_0 = (head_orientation_frame_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            head_orientation_frame_1 = (head_orientation_frame_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        motion_coef_frame_0 = torch.cat([expression_code_frame_0, head_orientation_frame_0], axis=-1)
        motion_coef_frame_1 = torch.cat([expression_code_frame_1, head_orientation_frame_1], axis=-1)

        shape_frame_0 = torch.zeros((motion_coef_frame_0.shape[0], 100)).float()
        shape_frame_1 = torch.zeros((motion_coef_frame_1.shape[0], 100)).float()

        coef_dict_0 = {"shape": shape_frame_0, "motion": motion_coef_frame_0}
        coef_dict_1 = {"shape": shape_frame_1, "motion": motion_coef_frame_1}

        audio_frame_0 = torch.tensor(audio_frame_0).float()
        audio_frame_1 = torch.tensor(audio_frame_1).float()
        
        if self.SE:
            return [motion_coef_frame_0, motion_coef_frame_1]
        else:
            return [audio_frame_0, audio_frame_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)

    # --- ADDED: Missing helper methods ---

    def get_k_indices_for_each_emotion(self, k=2):
        emotions = ["01", "02", "03", "04", "05", "06", "07", "08"]
        emotion_indices = {emotion: [] for emotion in emotions}
        for emotion in emotions:
            count = 0
            while len(emotion_indices[emotion]) < k:
                index = np.random.randint(0, len(self.entries))
                if self.entries[index].split("-")[2] == emotion:
                    if index not in emotion_indices[emotion]:
                         emotion_indices[emotion].append(index)
                count += 1
                if count > len(self.entries): # Avoid infinite loop
                    print(f"Warning: Could not find {k} unique samples for emotion {emotion}")
                    break
        return emotion_indices

    def query_for_video(self, index):
        video_name = self.entries[index]
        if video_name not in self.file_names:
            raise Exception("Video name not found in the dataset")
            
        expression_code = self.data[video_name]["expression_code"]
        pose = self.data[video_name]["head_orientation"]
        audio = self.data[video_name]["audio"]
        
        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        expression_code = torch.tensor(expression_code).float()
        pose = torch.tensor(pose).float()

        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
        
        motion = torch.cat([expression_code, pose], axis=-1)
        shape = torch.zeros((motion.shape[0], 100)).float()
        coef_dict = {"shape": shape, "motion": motion}
        
        audio = torch.tensor(audio).float()
        return audio, coef_dict, (audio_mean, audio_std)

    @staticmethod
    def get_collate_fn(SE):
        def pad_or_trim_audio(audio_tensor, target_length=64000):
            current_length = audio_tensor.size(0)
            if current_length < target_length:
                padding = target_length - current_length
                return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
            elif current_length > target_length:
                return audio_tensor[:target_length]
            return audio_tensor

        def collate_fn(batch):
            if SE:
                coef_0 = [item[0] for item in batch]
                coef_1 = [item[1] for item in batch]
                return [torch.stack(coef_0, dim=0), torch.stack(coef_1, dim=0)]
            else:
                target_length = 64000 
                
                audio_0 = [pad_or_trim_audio(item[0][0], target_length) for item in batch]
                audio_1 = [pad_or_trim_audio(item[0][1], target_length) for item in batch]
                motion_0 = [item[1][0]["motion"] for item in batch]
                motion_1 = [item[1][1]["motion"] for item in batch]
                shape_0 = [item[1][0]["shape"] for item in batch]
                shape_1 = [item[1][1]["shape"] for item in batch]
                audio_means = [item[2][0] for item in batch]
                audio_stds = [item[2][1] for item in batch]

                audio_0 = torch.stack(audio_0, dim=0)
                audio_1 = torch.stack(audio_1, dim=0)
                motion_0 = torch.stack(motion_0, dim=0)
                motion_1 = torch.stack(motion_1, dim=0)
                shape_0 = torch.stack(shape_0, dim=0)
                shape_1 = torch.stack(shape_1, dim=0)

                audio_mean = torch.tensor(audio_means).float().mean()
                audio_std = torch.tensor(audio_stds).float().mean()

                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
        return collate_fn

    @staticmethod
    def get_collate_fn_legacy(SE):
        # This legacy function is kept for completeness, though get_collate_fn is preferred.
        def collate_fn(batch):
            if SE:
                coef_0 = torch.stack([item[0] for item in batch], dim=0)
                coef_1 = torch.stack([item[1] for item in batch], dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = torch.stack([item[0][0] for item in batch], dim=0)
                audio_1 = torch.stack([item[0][1] for item in batch], dim=0)
                motion_0 = torch.stack([item[1][0]["motion"] for item in batch], dim=0)
                motion_1 = torch.stack([item[1][1]["motion"] for item in batch], dim=0)
                shape_0 = torch.stack([item[1][0]["shape"] for item in batch], dim=0)
                shape_1 = torch.stack([item[1][1]["shape"] for item in batch], dim=0)
                
                audio_mean = torch.tensor([item[2][0] for item in batch]).float().mean()
                audio_std = torch.tensor([item[2][1] for item in batch]).float().mean()

                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
        return collate_fn


class PickleDataset_Evan_MEAD_RAVDESS(data.Dataset):
     
    def incremental_mean_and_std(self, SE=False):
        exp_sum = 0
        exp_sum_of_squares = 0
        pose_sum = 0
        pose_sum_of_squares = 0
        shape_sum = 0
        shape_sum_of_squares = 0
        num_elements = 0
        num_elements_shape = 0
        for i in tqdm(range(len(self.file_names))):
        # for i in range(len(train_dataset)):
            entry_i = self.motion_dict[self.file_names[i]]
            # Extract expression and pose tensors for both frames
            if len(entry_i["exp"].shape) == 3:
                exp = entry_i['exp'][0]
                pose = entry_i['global_pose'][0]
                jaw = entry_i['jaw'][0]
                shape = entry_i['shape'][0]
            else:                
                exp = entry_i['exp']
                pose = entry_i['global_pose']
                jaw = entry_i['jaw']
                shape = entry_i['shape']
            pose = np.concatenate([pose, jaw], axis=1)
            # Update sum and sum of squares for expressions
            exp_sum += exp.sum(axis=0)
            exp_sum_of_squares += (exp ** 2).sum(axis=0)
            
            # Update sum and sum of squares for poses
            pose_sum += pose.sum(axis=0)
            pose_sum_of_squares += (pose ** 2).sum(axis=0)
            
            shape_sum += shape.sum(axis=0)
            shape_sum_of_squares += (shape ** 2).sum(axis=0)
            
            # Update the total number of elements processed
            num_elements += pose.shape[0]
            num_elements_shape += shape.shape[0]
        # Compute the mean for expressions and poses
        exp_mean = exp_sum / num_elements
        pose_mean = pose_sum / num_elements
        shape_mean = shape_sum / num_elements_shape

        # Compute the variance for expressions and poses
        exp_var = (exp_sum_of_squares / num_elements) - (exp_mean ** 2)
        pose_var = (pose_sum_of_squares / num_elements) - (pose_mean ** 2)
        shape_var = (shape_sum_of_squares / num_elements_shape) - (shape_mean ** 2)

        # Standard deviation is the square root of variance
        exp_std = np.sqrt(exp_var)
        pose_std = np.sqrt(pose_var)
        shape_std = np.sqrt(shape_var)

        exp_std = np.where(np.isnan(exp_std), 1, exp_std)
        pose_std = np.where(np.isnan(pose_std), 1, pose_std)
        shape_std = np.where(np.isnan(shape_std), 1, shape_std)

        exp_mean = torch.tensor(exp_mean).float()
        exp_std = torch.tensor(exp_std).float()
        pose_mean = torch.tensor(pose_mean).float()
        pose_std = torch.tensor(pose_std).float()
        shape_mean = torch.tensor(shape_mean).float()
        shape_std = torch.tensor(shape_std).float()

        return exp_mean, exp_std, pose_mean, pose_std, shape_mean, shape_std

    def __init__(self, pkl_file_motion, pkl_file_audio, coef_stats_file=None, original_fps=30, coef_fps=25, n_motions=100,
                 rot_repr='aa', no_head_pose=False, clip_len=100, device='cpu', SE=True, full_dataset=False, pre_loaded_raw_dataset=None, celebv_text=True, random_crop=True, batch_overfit_size=-1):
        self.motion_dict = pkl_file_motion
        self.audio_dict = pkl_file_audio
        self.file_names = sorted(list(self.motion_dict.keys()))

        # if overfit_mode is not -1, only take the first overfit_mode entries
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]

        # resample the head_orientation and expression_code to 25 fps from 30 fps
        if original_fps != coef_fps:
            for key in self.file_names:
                original_dict = self.motion_dict[key]
                if len(original_dict["exp"].shape) == 3:
                    original_expression_code = original_dict["exp"][0]
                    original_head_orientation = original_dict["global_pose"][0]
                    original_jaw = original_dict["jaw"][0]
                    original_shape = original_dict["shape"][0]
                else:
                    original_expression_code = original_dict["exp"]
                    original_head_orientation = original_dict["global_pose"]
                    original_jaw = original_dict["jaw"]
                    original_shape = original_dict["shape"]

                # resample expressioncode to 25 fps down from 30 using interp1d
                x = np.linspace(0, 1, num=original_expression_code.shape[0])
                xnew = np.linspace(0, 1, num=int(round(original_expression_code.shape[0]/original_fps*coef_fps)))
                # sample for expression code, head orientation and jaw  
                f_exp = interp1d(x, original_expression_code, axis=0, fill_value="extrapolate", bounds_error=False)
                new_expression_code = f_exp(xnew)
                f_head = interp1d(x, original_head_orientation, axis=0, fill_value="extrapolate", bounds_error=False)
                new_head_orientation = f_head(xnew)
                f_jaw = interp1d(x, original_jaw, axis=0, fill_value="extrapolate", bounds_error=False)
                new_jaw = f_jaw(xnew)

                self.motion_dict[key]["exp"] = new_expression_code[:, :50]
                self.motion_dict[key]["global_pose"] = new_head_orientation
                self.motion_dict[key]["jaw"] = new_jaw
                self.motion_dict[key]["shape"] = original_shape[:, :100]
                del original_dict
        print("finished data resampling")

        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')
        self.device = device
        self.coef_fps = coef_fps
        self.clip_len = clip_len
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.random_crop = random_crop
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose
        self.SE = SE


        # Read split file
        self.entries = self.file_names
        if coef_stats_file is None:
            exp_mean, exp_std, pose_mean, pose_std, shape_mean, shape_std = self.incremental_mean_and_std(self.SE)
            self.coef_stats = {}
            self.coef_stats['exp_mean'] = exp_mean   
            self.coef_stats['exp_std'] = exp_std
            self.coef_stats['pose_mean'] = pose_mean
            self.coef_stats['pose_std'] = pose_std
            self.coef_stats['shape_mean'] = shape_mean
            self.coef_stats['shape_std'] = shape_std
        self.coef_stats = {x: torch.tensor(self.coef_stats[x]).float() for x in self.coef_stats}

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        clip_dict_motion = self.motion_dict[self.entries[index]]
        clip_dict_audio = self.audio_dict[self.entries[index]]
        audio = clip_dict_audio
        # extract motion features
        shape_code = clip_dict_motion["shape"]
        expression_code = clip_dict_motion["exp"]
        head_orientation = clip_dict_motion["global_pose"]
        jaw = clip_dict_motion["jaw"]
        pose = np.concatenate([head_orientation, jaw], axis=1)



        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the goal clip
        goal_total_length = self.coef_total_len
        goal_each_clip_length = self.clip_len
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # select a starting frame to ensure that after cropping, the second clip will have at least half of goal_each_clip_length
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
                end_frame1 = start_frame1 + goal_each_clip_length
                start_frame2 = start_frame1 + goal_each_clip_length
                end_frame2 = start_frame2 + goal_each_clip_length
            elif current_clip_length == goal_total_length:
                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            else:
                frames_to_pad = goal_total_length - current_clip_length
                # split this down the middle randomly
                frames_to_pad_front = np.random.randint(0, frames_to_pad)
                frames_to_pad_back = frames_to_pad - frames_to_pad_front
                frames_to_pad_front = int(round(frames_to_pad_front))
                frames_to_pad_back = int(round(frames_to_pad_back))
                expression_code = np.pad(expression_code, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                pose = np.pad(pose, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant', constant_values=0)
                
                # audio frames to pad = frames_to_pad * audio_unit
                # note that the audio might be slightly shorter or longer than the video, so we need to pad the audio twice
                audio_frames_to_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_frames_to_pad_back = int(round(frames_to_pad_back * self.audio_unit))
                
                audio = np.pad(audio, ((int(audio_frames_to_pad_front), int(audio_frames_to_pad_back))), 'constant', constant_values=0)
                audio_length = audio.shape[0]

                # if the audio is still shorter than the goal length, pad it with zeros at the end (this is not elegant but what can we do......)
                audio_minimal_length = goal_total_length * self.audio_unit
                audio_minimal_length = int(round(audio_minimal_length))
                if audio_length < audio_minimal_length:
                    audio = np.pad(audio, (0, audio_minimal_length - audio_length), 'constant', constant_values=0)

                start_frame1 = 0
                end_frame1 = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2 = goal_each_clip_length * 2
            # Crop the audio and coef
        else:
            start_frame1 = 0
            end_frame1 = goal_each_clip_length
            start_frame2 = goal_each_clip_length
            end_frame2 = goal_each_clip_length * 2

            # pad the audio and coef at the end 
            expression_code = np.pad(expression_code, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            pose = np.pad(pose, ((0, int(round(goal_total_length - current_clip_length))), (0, 0)), 'constant', constant_values=0)
            audio = np.pad(audio, (0, int(round(goal_total_length * self.audio_unit)) - audio.shape[0]), 'constant', constant_values=0)


        expression_code_frame_0 = expression_code[start_frame1:end_frame1]
        expression_code_frame_1 = expression_code[start_frame2:end_frame2]
        pose_frame_0 = pose[start_frame1:end_frame1]
        pose_frame_1 = pose[start_frame2:end_frame2]
        audio_frame_0 = audio[int(start_frame1 * self.audio_unit):int(end_frame1 * self.audio_unit)]
        audio_frame_1 = audio[int(start_frame2 * self.audio_unit):int(end_frame2 * self.audio_unit)]

        # concatenate expression and head orientation
        expression_code_frame_0 = torch.tensor(expression_code_frame_0).float()
        expression_code_frame_1 = torch.tensor(expression_code_frame_1).float()
        pose_frame_0 = torch.tensor(pose_frame_0).float()
        pose_frame_1 = torch.tensor(pose_frame_1).float()
        shape_frame_0 = torch.tensor(shape_code).float().mean(dim=0, keepdim=True).repeat(pose_frame_0.shape[0], 1)
        shape_frame_1 = torch.tensor(shape_code).float().mean(dim=0, keepdim=True).repeat(pose_frame_1.shape[0], 1)

        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code_frame_0 = (expression_code_frame_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            expression_code_frame_1 = (expression_code_frame_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose_frame_0 = (pose_frame_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            pose_frame_1 = (pose_frame_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            shape_frame_0 = (shape_frame_0 - self.coef_stats['shape_mean']) / (self.coef_stats['shape_std'] + 1e-9)
            shape_frame_1 = (shape_frame_1 - self.coef_stats['shape_mean']) / (self.coef_stats['shape_std'] + 1e-9)

        coef_dict_0 = {"shape": shape_frame_0, "exp": expression_code_frame_0, "pose": pose_frame_0}
        coef_dict_1 = {"shape": shape_frame_1, "exp": expression_code_frame_1, "pose": pose_frame_1}

        # turning all the numpy arrays into torch tensors
        audio_frame_0 = torch.tensor(audio_frame_0).float()
        audio_frame_1 = torch.tensor(audio_frame_1).float()
        
        if self.SE:
            motion_coef_0 = torch.cat([coef_dict_0["exp"], coef_dict_0["pose"]], dim=-1)[:, :-2]
            motion_coef_1 = torch.cat([coef_dict_1["exp"], coef_dict_1["pose"]], dim=-1)[:, :-2]
            return [motion_coef_0, motion_coef_1]
        else:
            return [audio_frame_0, audio_frame_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)
    
    @staticmethod
    def get_collate_fn_legacy(SE):
        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(0, len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                motion_0 = []
                motion_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                for i in range(0, len(batch)):
                    audio_0.append(batch[i][0][0])
                    audio_1.append(batch[i][0][1])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])
                    motion_0.append(batch[i][1][0]["motion"])
                    motion_1.append(batch[i][1][1]["motion"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])

                # stack them in the first dimension
                audio_0 = torch.stack(audio_0, dim=0)
                audio_1 = torch.stack(audio_1, dim=0)
                motion_0 = torch.stack(motion_0, dim=0)
                motion_1 = torch.stack(motion_1, dim=0)
                shape_0 = torch.stack(shape_0, dim=0)
                shape_1 = torch.stack(shape_1, dim=0)

                # aggregate the audio mean and std
                audio_mean = torch.tensor(audio_mean).to(motion_0.device).float()
                audio_std = torch.tensor(audio_std).to(motion_0.device).float()

                coef_0 = {"shape": shape_0, "motion": motion_0}
                coef_1 = {"shape": shape_1, "motion": motion_1}

                # compute mean of mena
                audio_mean = audio_mean.mean()
                # compute mean of std
                audio_std = audio_std.mean()

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
        return collate_fn   
    
    @staticmethod
    def get_collate_fn(SE):
        def pad_or_trim_audio(audio_tensor, target_length=64000):
            """Helper function to ensure all audio tensors have the same length"""
            current_length = audio_tensor.size(0)
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
            elif current_length > target_length:
                # Trim to target length
                return audio_tensor[:target_length]
            return audio_tensor

        def collate_fn(batch):
            if SE:
                coef_0 = []
                coef_1 = []
                for i in range(len(batch)):
                    coef_0.append(batch[i][0])
                    coef_1.append(batch[i][1])
                coef_0 = torch.stack(coef_0, dim=0)
                coef_1 = torch.stack(coef_1, dim=0)
                return [coef_0, coef_1]
            else:
                audio_0 = []
                audio_1 = []
                pose_0 = []
                pose_1 = []
                exp_0 = []
                exp_1 = []
                shape_0 = []
                shape_1 = []
                audio_mean = []
                audio_std = []

                # First pass: determine max audio length in batch
                target_length = 64000  # Fixed target length for audio

                # Process each item in batch
                for i in range(len(batch)):
                    # Pad or trim audio to target length
                    audio_0_padded = pad_or_trim_audio(batch[i][0][0], target_length)
                    audio_1_padded = pad_or_trim_audio(batch[i][0][1], target_length)
                    
                    # Append processed items
                    audio_0.append(audio_0_padded)
                    audio_1.append(audio_1_padded)
                    exp_0.append(batch[i][1][0]["exp"])
                    exp_1.append(batch[i][1][1]["exp"])
                    shape_0.append(batch[i][1][0]["shape"])
                    shape_1.append(batch[i][1][1]["shape"])
                    pose_0.append(batch[i][1][0]["pose"])
                    pose_1.append(batch[i][1][1]["pose"])
                    audio_mean.append(batch[i][2][0])
                    audio_std.append(batch[i][2][1])

                try:
                    # Stack all tensors
                    audio_0 = torch.stack(audio_0, dim=0)
                    audio_1 = torch.stack(audio_1, dim=0)
                    exp_0 = torch.stack(exp_0, dim=0)
                    exp_1 = torch.stack(exp_1, dim=0)
                    pose_0 = torch.stack(pose_0, dim=0)
                    pose_1 = torch.stack(pose_1, dim=0)
                    shape_0 = torch.stack(shape_0, dim=0)
                    shape_1 = torch.stack(shape_1, dim=0)
                except RuntimeError as e:
                    shapes_info = {
                        'audio_0': [x.shape for x in audio_0],
                        'audio_1': [x.shape for x in audio_1],
                        'pose_0': [x.shape for x in pose_0],
                        'pose_1': [x.shape for x in pose_1],
                        'exp_0': [x.shape for x in exp_0],
                        'exp_1': [x.shape for x in exp_1],
                        'shape_0': [x.shape for x in shape_0],
                        'shape_1': [x.shape for x in shape_1]
                    }
                    raise RuntimeError(f"Failed to stack tensors. Shapes: {shapes_info}. Original error: {str(e)}")

                # Process audio statistics
                audio_mean = torch.tensor(audio_mean).float().mean()
                audio_std = torch.tensor(audio_std).float().mean()

                # Create coefficient dictionaries
                coef_0 = {"shape": shape_0, "pose": pose_0, "exp": exp_0}
                coef_1 = {"shape": shape_1, "pose": pose_1, "exp": exp_1}

                return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)

        return collate_fn            

    def query_for_video(self, video_name):
        if not video_name in self.file_names:
            Exception("Video name not found in the dataset")
        clip_dict_motion = self.motion_dict[video_name]
        clip_dict_audio = self.audio_dict[video_name]
        audio = clip_dict_audio
        # extract motion features
        shape_code = clip_dict_motion["shape"]
        expression_code = clip_dict_motion["exp"]
        head_orientation = clip_dict_motion["global_pose"]
        jaw = clip_dict_motion["jaw"]
        pose = np.concatenate([head_orientation, jaw], axis=1)



        # normalize the audio
        audio_mean = audio.mean() # note these are calculated before padding to ensure that the mean and std are normalized correctly.
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # length of the current clip
        current_clip_length = expression_code.shape[0]
        
        # select a starting frame to ensure that after cropping, the second clip will have at least half of goal_each_clip_length

        # concatenate expression and head orientation
        expression_code = torch.tensor(expression_code).float()
        pose = torch.tensor(pose).float()
        shape = torch.tensor(shape_code).float().mean(dim=0, keepdim=True).repeat(pose.shape[0], 1)

        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            shape = (shape - self.coef_stats['shape_mean']) / (self.coef_stats['shape_std'] + 1e-9)

        coef_dict = {"shape": shape, "exp": expression_code, "pose": pose}
        
        # turning all the numpy arrays into torch tensors
        audio = torch.tensor(audio).float()
        
    
        return audio, coef_dict, (audio_mean, audio_std)

    def get_k_indices_for_each_emotion(self, k=1, do_randommize=False):        
        emotion_to_videos_dict = {}
        for key in self.file_names:
            attributes_if_mead = key.split("_") # split the file name to see the attribute of the video
            attributes_if_ravdess = key.split("-")
            if len(attributes_if_mead) == 5:
                emotion = attributes_if_mead[1]
                if "mead_"+emotion not in emotion_to_videos_dict:
                    emotion_to_videos_dict["mead_"+emotion] = []
                emotion_to_videos_dict["mead_"+emotion].append(key)
            elif len(attributes_if_ravdess) == 7:
                emotion = attributes_if_ravdess[2]
                if "ravdess_"+emotion not in emotion_to_videos_dict:
                    emotion_to_videos_dict["ravdess_"+emotion] = []
                emotion_to_videos_dict["ravdess_"+emotion].append(key)

        # select indexes from each emotion
        output_indexes = {}
        k=2
        for emotion in emotion_to_videos_dict:
            if do_randommize:
                indexes = np.random.choice(len(emotion_to_videos_dict[emotion]), k)
            else:
                indexes = list(range(k))
            output_indexes[emotion] = [emotion_to_videos_dict[emotion][kk] for kk in indexes]
        return output_indexes

if __name__ == "__main__":

    data_root = "/mnt/f/chrome_downloads/mead_ravdess_30fps"
    data_pkl_file = os.path.join(data_root, "train_mead_ravdess_0.1.pickle")
    audio_pkl_file = os.path.join(data_root, "train_mead_ravdess_rawaudio_0.1.pickle")
    loaded_pickle_motion = pickle.load(open(data_pkl_file, 'rb'))
    loaded_pickle_audio = pickle.load(open(audio_pkl_file, 'rb'))
    loaded_pickle_audio.keys()
    keys = ["W037_surprised_level_1_045", "01-01-07-01-01-02-07"]
    data_0 = loaded_pickle_motion[keys[0]]        
    data_1 = loaded_pickle_motion[keys[1]]
    audio_0 = loaded_pickle_audio[keys[0]]
    audio_1 = loaded_pickle_audio[keys[1]]
    mead_keys = ['shape', 'exp', 'global_pose', 'jaw', 'actor', 'emotion', 'level', 'sentence', ]
    ravdess_keys = ['shape', 'exp', 'global_pose', 'jaw']
    

    test_dataset = PickleDataset_Evan_MEAD_RAVDESS(loaded_pickle_motion, loaded_pickle_audio, coef_stats_file=None, original_fps=30, coef_fps=25, n_motions=100, rot_repr='aa', no_head_pose=False, clip_len=100, device='cpu', SE=True, full_dataset=False, pre_loaded_raw_dataset=None, celebv_text=True, random_crop=True, batch_overfit_size=-1)
    hdtf_coef_stats = dict(np.load("/data/HDTF_TFHP/lmdb/stats_train.npz"))
    hdtf_coef_stats = {x: torch.tensor(hdtf_coef_stats[x]) for x in hdtf_coef_stats}
    


    video_lists = test_dataset.get_k_indices_for_each_emotion(k=1, do_randommize=False)
    for key in video_lists:
        for video in video_lists[key]:
            print(video)
            print(test_dataset.query_for_video(video)[0].shape)
            print(test_dataset.query_for_video(video)[1]["exp"].shape)
            print(test_dataset.query_for_video(video)[1]["pose"].shape)
            print(test_dataset.query_for_video(video)[1]["shape"].shape)
            break


    self = test_dataset






    std_arr = test_dataset.coef_stats["shape_std"]
    std_arr = std_arr.cpu().numpy()
    std_arr = np.where(np.isnan(std_arr) , 1, std_arr)

    audio_pair, coef_pair, (audio_mean, audio_std) = test_dataset[0]
    print(coef_pair[0]["pose"].shape)
    print(coef_pair[0]["exp"].shape)
    print(coef_pair[0]["shape"].shape)
    
    dataloader = data.DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=PickleDataset_Evan_MEAD_RAVDESS.get_collate_fn(False))
    audio_pair, coef_pair, (audio_mean, audio_std) = dataloader.__iter__().__next__()
    print(coef_pair[0]["pose"].shape)
    print(coef_pair[0]["exp"].shape)
    print(coef_pair[0]["shape"].shape)
    
    



    args.dataset_type = "celebv-text-toy-v2"
    args.data_root='/mnt/f/chrome_downloads/processed_data'
    train_dataset, val_dataset, train_loader, val_loader = get_dataset_lmdb(args, "cuda")
    print(args.dataset_type)
        # print(i[0][0].shape)
        # print(i[1][0]["motion"].shape)
        # print(i[1][0]["shape"].shape)
        # print(i[2][0].shape)
        # break

    for batch in train_loader:
        print(batch[0][0].shape)
        print(batch[1][0]["motion"].shape)
        print(batch[1][0]["shape"].shape)
        print(batch[2][0].shape)
        # break

    
    self = DynamicObjectDebugger() 

    pkl_file = '/data/celebv-text/processed_data/processed_data_30fps_toy.pkl'
    split_file = "/data/celebv-text/processed_data/processed_data_30fps_toy_keys_train.txt"

    pkl_file = '/mnt/f/chrome_downloads/processed_data/processed_data_30fps_toy.pkl'
    split_file = "/mnt/f/chrome_downloads/processed_data/processed_data_30fps_toy_keys_train.txt"

    self = PickleDataset_Evan(pkl_file, split_file, coef_stats_file=None, coef_fps=25, n_motions=100, SE=False)
    data_0 = self[0]

    print(data_0[0][0].std())
    print(data_0[0][1].std())
    dataloader = data.DataLoader(self, batch_size=10, shuffle=True, collate_fn=PickleDataset_Evan.get_collate_fn(False))



    # for batch in dataloader:
    #     print(batch[1][0]["motion"].mean(), batch[1][0]["motion"].std())
    #     print(batch[1][1]["motion"].mean(), batch[1][1]["motion"].std())
    #     break



    import pickle as pkl
    import lmdb
    import os
    import numpy as np
    lmdb_path = "/data/HDTF_TFHP/lmdb/"
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin(write=False)
    # load any key
    for val in txn.cursor():
        print(len(val))
        val = val
        break
    value = pkl.loads(val[1])
    print(value["coef"].keys())
    shape = value["coef"]["shape"]
    pose = value["coef"]["pose"]
    exp = value["coef"]["exp"]

    print(shape.shape)
    print(pose.shape)
    print(exp.shape)
