"""
Dataset classes for Style Encoder training.
Contains all dataset implementations for various data formats and sources.
"""

import io
import pickle
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
import os
import cv2
import lmdb
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.interpolate import interp1d
from torch.utils import data
from tqdm import tqdm

# Suppress audio backend warnings
torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


# =============================================================================
# Utility Functions
# =============================================================================

def pad_or_trim_audio(audio_tensor: torch.Tensor, target_length: int = 64000) -> torch.Tensor:
    """Ensure audio tensor has exact target length by padding or trimming."""
    current_length = audio_tensor.size(0)
    if current_length < target_length:
        padding = target_length - current_length
        return torch.nn.functional.pad(audio_tensor, (0, padding), 'constant', 0)
    elif current_length > target_length:
        return audio_tensor[:target_length]
    return audio_tensor


def compute_incremental_stats(dataset, is_se: bool = False) -> Tuple[torch.Tensor, ...]:
    """Compute mean and std statistics incrementally over a dataset."""
    exp_sum = 0
    exp_sum_of_squares = 0
    pose_sum = 0
    pose_sum_of_squares = 0
    num_elements = 0
    
    for i in tqdm(range(len(dataset)), desc="Computing statistics"):
        entry = dataset[i]
        if not is_se:
            exp_0 = entry[1][0]['motion'][:, :64]
            exp_1 = entry[1][1]['motion'][:, :64]
            pose_0 = entry[1][0]['motion'][:, 64:]
            pose_1 = entry[1][1]['motion'][:, 64:]
        else:
            exp_0 = entry[0][:, :64]
            exp_1 = entry[1][:, :64]
            pose_0 = entry[0][:, 64:]
            pose_1 = entry[1][:, 64:]
        
        exp_sum += exp_0.sum(dim=0) + exp_1.sum(dim=0)
        exp_sum_of_squares += (exp_0 ** 2).sum(dim=0) + (exp_1 ** 2).sum(dim=0)
        pose_sum += pose_0.sum(dim=0) + pose_1.sum(dim=0)
        pose_sum_of_squares += (pose_0 ** 2).sum(dim=0) + (pose_1 ** 2).sum(dim=0)
        num_elements += exp_0.shape[0] + exp_1.shape[0]

    exp_mean = exp_sum / num_elements
    pose_mean = pose_sum / num_elements
    exp_var = (exp_sum_of_squares / num_elements) - (exp_mean ** 2)
    pose_var = (pose_sum_of_squares / num_elements) - (pose_mean ** 2)
    
    return exp_mean, torch.sqrt(exp_var), pose_mean, torch.sqrt(pose_var)


def resample_coefficients(data: np.ndarray, original_fps: int, target_fps: int) -> np.ndarray:
    """Resample coefficient data from original FPS to target FPS."""
    if original_fps == target_fps:
        return data
    x = np.linspace(0, 1, num=data.shape[0])
    xnew = np.linspace(0, 1, num=int(round(data.shape[0] / original_fps * target_fps)))
    f = interp1d(x, data, axis=0, fill_value="extrapolate", bounds_error=False)
    return f(xnew)


def load_pickle_in_chunks(file_path: Path) -> Generator[Dict, None, None]:
    """Load a pickle file in chunks to handle large files."""
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_coef_stats(stats_file: Optional[Path]) -> Optional[Dict[str, torch.Tensor]]:
    """Load coefficient statistics from file."""
    if stats_file is None:
        return None
    coef_stats = dict(np.load(stats_file, allow_pickle=True))
    return {k: torch.tensor(v).float() for k, v in coef_stats.items()}


# =============================================================================
# Collate Functions
# =============================================================================

def get_se_collate_fn(is_se: bool):
    """Get appropriate collate function for Style Encoder datasets."""
    def collate_fn(batch):
        if is_se:
            coef_0 = torch.stack([item[0] for item in batch], dim=0)
            coef_1 = torch.stack([item[1] for item in batch], dim=0)
            return [coef_0, coef_1]
        else:
            target_length = 64000
            audio_0 = [pad_or_trim_audio(item[0][0], target_length) for item in batch]
            audio_1 = [pad_or_trim_audio(item[0][1], target_length) for item in batch]
            
            try:
                audio_0 = torch.stack(audio_0, dim=0)
                audio_1 = torch.stack(audio_1, dim=0)
                motion_0 = torch.stack([item[1][0]["motion"] for item in batch], dim=0)
                motion_1 = torch.stack([item[1][1]["motion"] for item in batch], dim=0)
                shape_0 = torch.stack([item[1][0]["shape"] for item in batch], dim=0)
                shape_1 = torch.stack([item[1][1]["shape"] for item in batch], dim=0)
            except RuntimeError as e:
                shapes_info = {
                    'audio_0': [x.shape for x in audio_0],
                    'motion_0': [item[1][0]["motion"].shape for item in batch],
                }
                raise RuntimeError(f"Failed to stack tensors. Shapes: {shapes_info}. Error: {e}")

            audio_mean = torch.tensor([item[2][0] for item in batch]).float().mean()
            audio_std = torch.tensor([item[2][1] for item in batch]).float().mean()
            
            coef_0 = {"shape": shape_0, "motion": motion_0}
            coef_1 = {"shape": shape_1, "motion": motion_1}
            return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
    
    return collate_fn


def get_mead_ravdess_collate_fn(is_se: bool):
    """Get collate function for MEAD/RAVDESS datasets."""
    def collate_fn(batch):
        if is_se:
            coef_0 = torch.stack([item[0] for item in batch], dim=0)
            coef_1 = torch.stack([item[1] for item in batch], dim=0)
            return [coef_0, coef_1]
        else:
            target_length = 64000
            audio_0 = [pad_or_trim_audio(item[0][0], target_length) for item in batch]
            audio_1 = [pad_or_trim_audio(item[0][1], target_length) for item in batch]
            
            audio_0 = torch.stack(audio_0, dim=0)
            audio_1 = torch.stack(audio_1, dim=0)
            exp_0 = torch.stack([item[1][0]["exp"] for item in batch], dim=0)
            exp_1 = torch.stack([item[1][1]["exp"] for item in batch], dim=0)
            pose_0 = torch.stack([item[1][0]["pose"] for item in batch], dim=0)
            pose_1 = torch.stack([item[1][1]["pose"] for item in batch], dim=0)
            shape_0 = torch.stack([item[1][0]["shape"] for item in batch], dim=0)
            shape_1 = torch.stack([item[1][1]["shape"] for item in batch], dim=0)
            
            audio_mean = torch.tensor([item[2][0] for item in batch]).float().mean()
            audio_std = torch.tensor([item[2][1] for item in batch]).float().mean()
            
            coef_0 = {"shape": shape_0, "pose": pose_0, "exp": exp_0}
            coef_1 = {"shape": shape_1, "pose": pose_1, "exp": exp_1}
            return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
    
    return collate_fn


# =============================================================================
# Base Dataset Classes
# =============================================================================

class BaseMotionDataset(data.Dataset):
    """Base class for motion datasets with common functionality."""
    
    def __init__(
        self,
        coef_fps: int = 25,
        n_motions: int = 100,
        clip_len: int = 100,
        is_se: bool = True,
        no_head_pose: bool = False,
        random_crop: bool = True,
    ):
        self.coef_fps = coef_fps
        self.clip_len = clip_len
        self.n_motions = n_motions
        self.is_se = is_se
        self.no_head_pose = no_head_pose
        self.random_crop = random_crop
        
        self.audio_unit = 16000.0 / coef_fps
        self.n_audio_samples = round(self.audio_unit * n_motions)
        self.coef_total_len = int(n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        
        self.coef_stats: Optional[Dict[str, torch.Tensor]] = None
        self.entries: List[str] = []
    
    def _normalize_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Normalize audio and return mean/std for denormalization."""
        audio_mean = audio.mean()
        audio_std = audio.std()
        normalized = (audio - audio_mean) / (audio_std + 1e-5)
        return normalized, audio_mean, audio_std
    
    def _crop_and_pad_data(
        self,
        expression_code: np.ndarray,
        head_orientation: np.ndarray,
        audio: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """Crop or pad data to required length and return frame indices."""
        goal_total_length = self.coef_total_len
        goal_each_clip_length = self.clip_len
        current_clip_length = expression_code.shape[0]
        
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
            elif current_clip_length == goal_total_length:
                start_frame1 = 0
            else:
                # Need to pad
                frames_to_pad = goal_total_length - current_clip_length
                frames_to_pad_front = np.random.randint(0, frames_to_pad + 1)
                frames_to_pad_back = frames_to_pad - frames_to_pad_front
                
                expression_code = np.pad(
                    expression_code,
                    ((frames_to_pad_front, frames_to_pad_back), (0, 0)),
                    'constant'
                )
                head_orientation = np.pad(
                    head_orientation,
                    ((frames_to_pad_front, frames_to_pad_back), (0, 0)),
                    'constant'
                )
                
                audio_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_pad_back = int(round(frames_to_pad_back * self.audio_unit))
                audio = np.pad(audio, (audio_pad_front, audio_pad_back), 'constant')
                
                # Ensure minimum audio length
                audio_min_len = int(round(goal_total_length * self.audio_unit))
                if audio.shape[0] < audio_min_len:
                    audio = np.pad(audio, (0, audio_min_len - audio.shape[0]), 'constant')
                
                start_frame1 = 0
        else:
            start_frame1 = 0
            if current_clip_length < goal_total_length:
                pad_len = goal_total_length - current_clip_length
                expression_code = np.pad(expression_code, ((0, pad_len), (0, 0)), 'constant')
                head_orientation = np.pad(head_orientation, ((0, pad_len), (0, 0)), 'constant')
                
                audio_target_len = int(round(goal_total_length * self.audio_unit))
                if audio.shape[0] < audio_target_len:
                    audio = np.pad(audio, (0, audio_target_len - audio.shape[0]), 'constant')
        
        end_frame1 = start_frame1 + goal_each_clip_length
        start_frame2 = start_frame1 + goal_each_clip_length
        end_frame2 = start_frame2 + goal_each_clip_length
        
        return expression_code, head_orientation, audio, start_frame1, end_frame1, start_frame2, end_frame2
    
    def _normalize_coefficients(
        self,
        exp: torch.Tensor,
        pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize expression and pose coefficients using stored stats."""
        if self.coef_stats is not None:
            exp = (exp - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
        return exp, pose


# =============================================================================
# LMDB Dataset Classes
# =============================================================================
class LmdbDataset(BaseMotionDataset):
    """Dataset for loading motion data from LMDB format (HDTF_TFHP style)."""
    
    def __init__(
        self,
        lmdb_dir: Path,
        split_file: Path,
        coef_stats_file: Optional[Path] = None,
        original_fps: int = 30,
        coef_fps: int = 25,
        n_motions: int = 100,
        seq_len: int = 100,
        crop_strategy: str = 'random',
        rot_repr: str = 'aa',
        batch_overfit_size: int = -1,
        do_random_pad: bool = True,
        no_head_pose = False,
        SE=False
    ):
        super().__init__(
            coef_fps=coef_fps,
            n_motions=n_motions,
            is_se=False,
            random_crop=do_random_pad,
        )
        
        self.lmdb_dir = Path(lmdb_dir)
        self.split_file = split_file
        self.crop_strategy = crop_strategy if do_random_pad else 'begin'
        self.rot_representation = rot_repr
        
        # Read split file
        with open(split_file, 'r') as f:
            self.entries = [line.strip() for line in f]
        
        if batch_overfit_size > 0:
            self.entries = self.entries[:batch_overfit_size]
        
        # Load LMDB
        self.lmdb_env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )
        self.clip_len = 100
        self.audio_clip_len = round(self.audio_unit * self.clip_len)
        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.random_crop = do_random_pad
        self.no_head_pose = no_head_pose
        self.SE = SE
        # load all data to memory
        self.data = {}
        entries_to_remove = []
        with self.lmdb_env.begin(write=False) as txn:
            print(len(self.entries))
            for key in self.entries:
                try:
                    self.data[key] = (pickle.loads(txn.get(key.encode())))
                except:
                    print(f"key {key} not found in lmdb")
                    entries_to_remove.append(key)

        # remove entries that are not found in lmdb 
        for key in entries_to_remove:
            self.entries.remove(key)

        # resample data to target fps
        if original_fps != coef_fps:
            print("start data resampling")
            for key in self.entries:
                original_dict = self.data[key]
                original_expression_code = original_dict["exp_params"]
                original_jaw = original_dict["jaw_params"]
                original_head_orientation = original_dict["head_rotation"]
                new_dict = {"audio": original_dict["audio"]}
                # resample expressioncode to 25 fps down from 30 using interp1d
                # original_expression_code.shape
                x = np.linspace(0, 1, num=original_expression_code.shape[0])
                xnew = np.linspace(0, 1, num=int(round(original_expression_code.shape[0]/original_fps*coef_fps)))
                f_exp = interp1d(x, original_expression_code, axis=0)
                new_expression_code = f_exp(xnew)
                f_exp = interp1d(x, original_jaw, axis=0)
                new_jaw = f_exp(xnew)
                # also resample head_rotation to 25 fps
                f_head = interp1d(x, original_head_orientation, axis=0)
                new_head_orientation = f_head(xnew)
                new_dict["exp_params"] = new_expression_code
                new_dict["jaw_params"] = new_jaw
                new_dict["head_rotation"] = new_head_orientation
                self.data[key] = new_dict
                del original_dict
            print("finished data resampling")     
        else:
            print("no resampling needed")

        import os
        # Load or compute stats
        stats_file_path = Path(lmdb_dir) / "coef_stats.npz"

        if coef_stats_file is not None:
            self.coef_stats = load_coef_stats(coef_stats_file)
        elif stats_file_path.exists():
            print(f"Loading stats from {stats_file_path.absolute()}")
            self.coef_stats = load_coef_stats(stats_file_path)
        else:
            print(f"No stats file found. Computing statistics from data and saving to {stats_file_path.absolute()}")
            exp_sum  = np.zeros(100)
            exp_sq   = np.zeros(100)
            pose_sum = np.zeros(6)
            pose_sq  = np.zeros(6)
            n_frames = 0

            for key in tqdm(self.entries, desc="Computing stats"):
                exp  = self.data[key]["exp_params"]    # (T, 100)
                jaw  = self.data[key]["jaw_params"]    # (T, 3)
                head = self.data[key]["head_rotation"] # (T, 3)
                pose = np.concatenate([jaw, head], axis=1)  # (T, 6)

                exp_sum  += exp.sum(axis=0)
                exp_sq   += (exp ** 2).sum(axis=0)
                pose_sum += pose.sum(axis=0)
                pose_sq  += (pose ** 2).sum(axis=0)
                n_frames += exp.shape[0]

            exp_mean  = exp_sum / n_frames
            pose_mean = pose_sum / n_frames
            exp_std   = np.sqrt(np.maximum((exp_sq / n_frames) - (exp_mean ** 2), 0))
            pose_std  = np.sqrt(np.maximum((pose_sq / n_frames) - (pose_mean ** 2), 0))

            exp_std  = np.where(exp_std  < 1e-9, 1.0, exp_std)
            pose_std = np.where(pose_std < 1e-9, 1.0, pose_std)

            np.savez(
                stats_file_path,
                exp_mean=exp_mean,
                exp_std=exp_std,
                pose_mean=pose_mean,
                pose_std=pose_std,
            )
            print(f"Stats saved to {stats_file_path.absolute()}")

            self.coef_stats = {
                'exp_mean':  torch.tensor(exp_mean).float(),
                'exp_std':   torch.tensor(exp_std).float(),
                'pose_mean': torch.tensor(pose_mean).float(),
                'pose_std':  torch.tensor(pose_std).float(),
            }
    
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, index: int):        
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["exp_params"]
        jaw_params = clip_dict["jaw_params"]
        # expression_code = np.concatenate([expression_code, jaw_params], axis=1)
        head_orientation = clip_dict["head_rotation"]
        pose = np.concatenate([jaw_params, head_orientation], axis=1)

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
        
        expression_code_frame_0 = torch.tensor(expression_code_frame_0).float()
        expression_code_frame_1 = torch.tensor(expression_code_frame_1).float()
        pose_frame_0 = torch.tensor(pose_frame_0).float()
        pose_frame_1 = torch.tensor(pose_frame_1).float()

        # normalize coef if applicable
        if self.coef_stats is not None:
            expression_code_frame_0 = (expression_code_frame_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            expression_code_frame_1 = (expression_code_frame_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose_frame_0 = (pose_frame_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            pose_frame_1 = (pose_frame_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        
        motion_coef_frame_0 = torch.cat([expression_code_frame_0, pose_frame_0], axis=-1)
        motion_coef_frame_1 = torch.cat([expression_code_frame_1, pose_frame_1], axis=-1)

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

    def get_item_full(self, index: int):
        """
        Retrieves the entire audio and motion sequence for a given index 
        without cropping, padding, or splitting into windows.
        """
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["exp_params"]
        jaw_params = clip_dict["jaw_params"]
        head_orientation = clip_dict["head_rotation"]
        
        # Combine jaw and head into pose, matching __getitem__ exactly
        pose = np.concatenate([jaw_params, head_orientation], axis=1)  # (T, 6)

        # 1. Normalize the audio
        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)
        
        # 2. Convert to Torch Tensors
        expression_code = torch.tensor(expression_code).float()
        pose = torch.tensor(pose).float()
        audio = torch.tensor(audio).float()

        # 3. Normalize coefficients if stats are provided
        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        # 4. Concatenate final motion vector: exp(100) + jaw(3) + head(3) = (T, 106)
        motion_coef = torch.cat([expression_code, pose], axis=-1)

        # Create dummy shape tensor (matching __getitem__ behavior)
        shape = torch.zeros((motion_coef.shape[0], 100)).float()

        coef_dict = {"shape": shape, "motion": motion_coef}

        return audio, coef_dict, (audio_mean, audio_std)

# class LmdbDatasetForSE(data.Dataset):
#     """
#     Cached LMDB Dataset for Style Encoder.
#     Performs resampling and normalization once at startup to maximize training throughput.
#     """
    
#     def __init__(
#         self,
#         lmdb_dir: Path,
#         split_file: Path,
#         coef_stats_file: Optional[Path] = None,
#         input_fps: int = 30,
#         coef_fps: int = 25,
#         n_motions: int = 100,
#         no_head_pose: bool = False,
#     ):
#         self.lmdb_dir = Path(lmdb_dir)
#         self.input_fps = input_fps
#         self.coef_fps = coef_fps
#         self.n_motions = n_motions
#         self.no_head_pose = no_head_pose
#         self.coef_stats = load_coef_stats(coef_stats_file)
        
#         # 1. Parse split file and group by person
#         self.person_to_keys = defaultdict(list)
#         all_keys = []
#         with open(split_file, 'r') as f:
#             for line in f:
#                 full_key = line.strip().split()[0]
#                 person_id = full_key.split('/')[0]
#                 self.person_to_keys[person_id].append(full_key)
#                 all_keys.append(full_key)
        
#         self.person_ids = list(self.person_to_keys.keys())

#         # 2. Open LMDB Environment
#         self.lmdb_env = lmdb.open(
#             str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False
#         )
        
#         with self.lmdb_env.begin(write=False) as txn:
#             self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']

#         # 3. Cache, Resample, and Normalize everything
#         self.cached_data = {}
#         print(f"Caching and resampling {len(all_keys)} sequences in memory...")
        
#         for key in tqdm(all_keys):
#             self.cached_data[key] = self._load_and_process_sequence(key)
            
#         # Close environment as we now have everything in RAM
#         self.lmdb_env.close()

#     def _load_and_process_sequence(self, key: str) -> torch.Tensor:
#         """Internal helper to pull all chunks for a key and process them."""
#         with self.lmdb_env.begin(write=False) as txn:
#             # Get metadata for this specific sequence
#             meta_key = f'{key}/metadata'.encode()
#             metadata = pickle.loads(txn.get(meta_key))
#             seq_len = metadata['n_frames']
            
#             coef_keys = ['exp', 'pose']
#             raw_coefs = {k: [] for k in coef_keys}
            
#             # Stitch chunks together
#             num_clips = (seq_len - 1) // self.clip_len + 1
#             for i in range(num_clips):
#                 clip_key = f'{key}/{i:03d}'.encode()
#                 entry = pickle.loads(txn.get(clip_key))
#                 for k in coef_keys:
#                     raw_coefs[k].append(entry['coef'][k])
            
#             # Concatenate
#             processed = {k: np.concatenate(raw_coefs[k], axis=0) for k in coef_keys}
            
#             # Resample to target FPS
#             if self.input_fps != self.coef_fps:
#                 for k in coef_keys:
#                     processed[k] = resample_coefficients(processed[k], self.input_fps, self.coef_fps)
            
#             # Convert to Tensor
#             exp = torch.from_numpy(processed['exp']).float()
#             pose = torch.from_numpy(processed['pose']).float()

#             # Normalize using stored stats
#             if self.coef_stats is not None:
#                 exp = (exp - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
#                 pose = (pose - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

#             # Refactor into motion coefficient
#             if self.no_head_pose:
#                 mouth_pose = pose[:, 3:]
#                 motion_coef = torch.cat([exp, mouth_pose], dim=-1)
#             else:
#                 motion_coef = torch.cat([exp, pose], dim=-1)
            
#             # Strip mouth rotation around y, z (last 2 dims)
#             return motion_coef[:, :-2]

#     def __len__(self) -> int:
#         return len(self.person_ids)

#     def __getitem__(self, index: int):
#         # Pick a random video for the selected person
#         person_id = self.person_ids[index]
#         key = random.choice(self.person_to_keys[person_id])
        
#         full_motion = self.cached_data[key]
#         seq_len = full_motion.shape[0]
        
#         # We need a window of roughly 2.1 * n_motions for the SE pair logic
#         required_len = int(self.n_motions * 2.1)
        
#         if seq_len <= required_len:
#             # If sequence is too short, pad it
#             padding = required_len - seq_len + 1
#             full_motion = torch.nn.functional.pad(full_motion, (0, 0, 0, padding), mode='constant', value=0)
#             seq_len = full_motion.shape[0]

#         # Random crop within the processed sequence
#         start = random.randint(0, seq_len - required_len)
#         motion_window = full_motion[start : start + required_len]
        
#         return [
#             motion_window[:self.n_motions].clone(),
#             motion_window[-self.n_motions:].clone()
#         ]

# =============================================================================
# Pickle Dataset Classes
# =============================================================================

class PickleDataset(BaseMotionDataset):
    """Dataset for loading motion data from pickle files (CelebV-Text style)."""
    
    def __init__(
        self,
        pkl_file: Path,
        split_file: Path,
        coef_stats_file: Optional[Path] = None,
        original_fps: int = 30,
        coef_fps: int = 25,
        n_motions: int = 100,
        rot_repr: str = 'aa',
        no_head_pose: bool = False,
        clip_len: int = 100,
        is_se: bool = True,
        full_dataset: bool = False,
        pre_loaded_raw_dataset: Optional[Dict] = None,
        celebv_text: bool = True,
        random_crop: bool = True,
        batch_overfit_size: int = -1,
    ):
        super().__init__(
            coef_fps=coef_fps,
            n_motions=n_motions,
            clip_len=clip_len,
            is_se=is_se,
            no_head_pose=no_head_pose,
            random_crop=random_crop,
        )
        
        self.pkl_file = pkl_file
        self.split_file = split_file
        self.original_fps = original_fps
        self.rot_representation = rot_repr
        
        # Load valid IDs for celebv-text
        self.valid_ids = set()
        if celebv_text:
            valid_id_file = Path("/data/celebv-text/keys.txt")
            if valid_id_file.exists():
                with open(valid_id_file, 'r') as f:
                    self.valid_ids = {line.strip() for line in f}
        
        # Load split file
        self.file_names = []
        with open(split_file, 'r') as f:
            for line in f:
                name = line.strip()
                if celebv_text and self.valid_ids and name not in self.valid_ids:
                    continue
                self.file_names.append(name)
        
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]
        
        # Load data
        if pre_loaded_raw_dataset is not None:
            raw_data = pre_loaded_raw_dataset
        elif full_dataset:
            raw_data = {}
            for chunk in load_pickle_in_chunks(pkl_file):
                raw_data.update(chunk)
        else:
            raw_data = pickle.load(open(pkl_file, 'rb'))
        
        self.data = {key: raw_data[key] for key in self.file_names}
        
        # Resample to target FPS
        if original_fps != coef_fps:
            self._resample_data()
        
        self.entries = self.file_names
        
        # Load or compute stats
        if coef_stats_file is not None:
            self.coef_stats = load_coef_stats(coef_stats_file)
        else:
            print('Warning: No stats file found. Computing statistics...')
            exp_mean, exp_std, pose_mean, pose_std = compute_incremental_stats(self, is_se)
            self.coef_stats = {
                'exp_mean': exp_mean,
                'exp_std': exp_std,
                'pose_mean': pose_mean,
                'pose_std': pose_std,
            }
        
        self.coef_stats = {k: torch.tensor(v).float() for k, v in self.coef_stats.items()}
    
    def _resample_data(self):
        """Resample all data to target FPS."""
        for key in self.file_names:
            original = self.data[key]
            exp = resample_coefficients(original["expression_code"], self.original_fps, self.coef_fps)
            head = resample_coefficients(original["head_orientation"], self.original_fps, self.coef_fps)
            self.data[key] = {
                "audio": original["audio"],
                "expression_code": exp,
                "head_orientation": head,
            }
        print("Finished data resampling")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, index: int):
        clip_dict = self.data[self.entries[index]]
        audio = clip_dict["audio"]
        expression_code = clip_dict["expression_code"]
        head_orientation = clip_dict["head_orientation"]
        
        audio, audio_mean, audio_std = self._normalize_audio(audio)
        
        expression_code, head_orientation, audio, sf1, ef1, sf2, ef2 = self._crop_and_pad_data(
            expression_code, head_orientation, audio
        )
        
        # Extract frames
        exp_0 = torch.tensor(expression_code[sf1:ef1]).float()
        exp_1 = torch.tensor(expression_code[sf2:ef2]).float()
        pose_0 = torch.tensor(head_orientation[sf1:ef1]).float()
        pose_1 = torch.tensor(head_orientation[sf2:ef2]).float()
        audio_0 = torch.tensor(audio[int(sf1 * self.audio_unit):int(ef1 * self.audio_unit)]).float()
        audio_1 = torch.tensor(audio[int(sf2 * self.audio_unit):int(ef2 * self.audio_unit)]).float()
        
        # Normalize coefficients
        exp_0, pose_0 = self._normalize_coefficients(exp_0, pose_0)
        exp_1, pose_1 = self._normalize_coefficients(exp_1, pose_1)
        
        motion_0 = torch.cat([exp_0, pose_0], dim=-1)
        motion_1 = torch.cat([exp_1, pose_1], dim=-1)
        
        if self.is_se:
            return [motion_0, motion_1]
        
        shape_0 = torch.zeros((motion_0.shape[0], 100)).float()
        shape_1 = torch.zeros((motion_1.shape[0], 100)).float()
        
        coef_dict_0 = {"shape": shape_0, "motion": motion_0}
        coef_dict_1 = {"shape": shape_1, "motion": motion_1}
        
        return [audio_0, audio_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)
    
    def query_for_video(self, index: int):
        """Get full video data for inference."""
        video_name = self.entries[index]
        if video_name not in self.file_names:
            raise ValueError("Video name not found in dataset")
        
        exp = self.data[video_name]["expression_code"]
        pose = self.data[video_name]["head_orientation"]
        audio = self.data[video_name]["audio"]
        
        audio, audio_mean, audio_std = self._normalize_audio(audio)
        
        exp = torch.tensor(exp).float()
        pose = torch.tensor(pose).float()
        exp, pose = self._normalize_coefficients(exp, pose)
        
        motion = torch.cat([exp, pose], dim=-1)
        shape = torch.zeros((motion.shape[0], 100)).float()
        
        coef_dict = {"shape": shape, "motion": motion}
        audio = torch.tensor(audio).float()
        
        return audio, coef_dict, (audio_mean, audio_std)


class MeadRavdessDataset(BaseMotionDataset):
    """Dataset for MEAD and RAVDESS data with FLAME parameters."""
    
    def __init__(
        self,
        motion_dict: Dict,
        audio_dict: Dict,
        coef_stats_file: Optional[Path] = None,
        original_fps: int = 30,
        coef_fps: int = 25,
        n_motions: int = 100,
        no_head_pose: bool = False,
        clip_len: int = 100,
        is_se: bool = True,
        random_crop: bool = True,
        batch_overfit_size: int = -1,
    ):
        super().__init__(
            coef_fps=coef_fps,
            n_motions=n_motions,
            clip_len=clip_len,
            is_se=is_se,
            no_head_pose=no_head_pose,
            random_crop=random_crop,
        )
        
        self.motion_dict = motion_dict
        self.audio_dict = audio_dict
        self.file_names = sorted(list(motion_dict.keys()))
        
        if batch_overfit_size > 0:
            self.file_names = self.file_names[:batch_overfit_size]
        
        # Resample to target FPS
        if original_fps != coef_fps:
            self._resample_data(original_fps)
        
        self.entries = self.file_names
        
        # Load or compute stats
        if coef_stats_file is not None:
            self.coef_stats = load_coef_stats(coef_stats_file)
        else:
            print('Computing statistics...')
            self.coef_stats = self._compute_stats()
        
        self.coef_stats = {k: torch.tensor(v).float() for k, v in self.coef_stats.items()}
    
    def _resample_data(self, original_fps: int):
        """Resample FLAME parameters to target FPS."""
        for key in self.file_names:
            original = self.motion_dict[key]
            
            # Handle different data shapes
            exp = original["exp"][0] if len(original["exp"].shape) == 3 else original["exp"]
            pose = original["global_pose"][0] if len(original["global_pose"].shape) == 3 else original["global_pose"]
            jaw = original["jaw"][0] if len(original["jaw"].shape) == 3 else original["jaw"]
            shape = original["shape"][0] if len(original["shape"].shape) == 3 else original["shape"]
            
            exp = resample_coefficients(exp, original_fps, self.coef_fps)
            pose = resample_coefficients(pose, original_fps, self.coef_fps)
            jaw = resample_coefficients(jaw, original_fps, self.coef_fps)
            
            self.motion_dict[key] = {
                "exp": exp[:, :50],
                "global_pose": pose,
                "jaw": jaw,
                "shape": shape[:, :100],
            }
    
    def _compute_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std for all coefficient types."""
        exp_sum = pose_sum = shape_sum = 0
        exp_sq = pose_sq = shape_sq = 0
        n_frames = n_shape = 0
        
        for key in tqdm(self.file_names, desc="Computing stats"):
            entry = self.motion_dict[key]
            exp = entry['exp']
            pose = np.concatenate([entry['global_pose'], entry['jaw']], axis=1)
            shape = entry['shape']
            
            exp_sum += exp.sum(axis=0)
            exp_sq += (exp ** 2).sum(axis=0)
            pose_sum += pose.sum(axis=0)
            pose_sq += (pose ** 2).sum(axis=0)
            shape_sum += shape.sum(axis=0)
            shape_sq += (shape ** 2).sum(axis=0)
            
            n_frames += pose.shape[0]
            n_shape += shape.shape[0]
        
        exp_mean = exp_sum / n_frames
        pose_mean = pose_sum / n_frames
        shape_mean = shape_sum / n_shape
        
        exp_std = np.sqrt((exp_sq / n_frames) - (exp_mean ** 2))
        pose_std = np.sqrt((pose_sq / n_frames) - (pose_mean ** 2))
        shape_std = np.sqrt((shape_sq / n_shape) - (shape_mean ** 2))
        
        # Handle NaN
        exp_std = np.where(np.isnan(exp_std), 1, exp_std)
        pose_std = np.where(np.isnan(pose_std), 1, pose_std)
        shape_std = np.where(np.isnan(shape_std), 1, shape_std)
        
        return {
            'exp_mean': exp_mean, 'exp_std': exp_std,
            'pose_mean': pose_mean, 'pose_std': pose_std,
            'shape_mean': shape_mean, 'shape_std': shape_std,
        }
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, index: int):
        motion = self.motion_dict[self.entries[index]]
        audio = self.audio_dict[self.entries[index]]
        
        exp = motion["exp"]
        pose = np.concatenate([motion["global_pose"], motion["jaw"]], axis=1)
        shape = motion["shape"]
        
        audio, audio_mean, audio_std = self._normalize_audio(audio)
        
        # Reuse crop/pad logic
        exp, pose, audio, sf1, ef1, sf2, ef2 = self._crop_and_pad_data(exp, pose, audio)
        
        # Extract frames
        exp_0 = torch.tensor(exp[sf1:ef1]).float()
        exp_1 = torch.tensor(exp[sf2:ef2]).float()
        pose_0 = torch.tensor(pose[sf1:ef1]).float()
        pose_1 = torch.tensor(pose[sf2:ef2]).float()
        audio_0 = torch.tensor(audio[int(sf1 * self.audio_unit):int(ef1 * self.audio_unit)]).float()
        audio_1 = torch.tensor(audio[int(sf2 * self.audio_unit):int(ef2 * self.audio_unit)]).float()
        
        # Normalize
        if self.coef_stats is not None:
            exp_0 = (exp_0 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            exp_1 = (exp_1 - self.coef_stats['exp_mean']) / (self.coef_stats['exp_std'] + 1e-9)
            pose_0 = (pose_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            pose_1 = (pose_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
        
        shape_0 = torch.tensor(shape).float().mean(dim=0, keepdim=True).expand(pose_0.shape[0], -1)
        shape_1 = torch.tensor(shape).float().mean(dim=0, keepdim=True).expand(pose_1.shape[0], -1)
        
        if self.coef_stats is not None:
            shape_0 = (shape_0 - self.coef_stats['shape_mean']) / (self.coef_stats['shape_std'] + 1e-9)
            shape_1 = (shape_1 - self.coef_stats['shape_mean']) / (self.coef_stats['shape_std'] + 1e-9)
        
        if self.is_se:
            motion_0 = torch.cat([exp_0, pose_0], dim=-1)[:, :-2]
            motion_1 = torch.cat([exp_1, pose_1], dim=-1)[:, :-2]
            return [motion_0, motion_1]
        
        coef_0 = {"shape": shape_0, "exp": exp_0, "pose": pose_0}
        coef_1 = {"shape": shape_1, "exp": exp_1, "pose": pose_1}
        
        return [audio_0, audio_1], [coef_0, coef_1], (audio_mean, audio_std)
    
    def get_emotion_samples(self, k: int = 1, randomize: bool = False) -> Dict[str, List[str]]:
        """Get k samples for each emotion category."""
        emotion_to_videos = defaultdict(list)
        
        for key in self.file_names:
            mead_parts = key.split("_")
            ravdess_parts = key.split("-")
            
            if len(mead_parts) == 5:
                emotion = f"mead_{mead_parts[1]}"
                emotion_to_videos[emotion].append(key)
            elif len(ravdess_parts) == 7:
                emotion = f"ravdess_{ravdess_parts[2]}"
                emotion_to_videos[emotion].append(key)
        
        result = {}
        for emotion, videos in emotion_to_videos.items():
            if randomize:
                indices = np.random.choice(len(videos), min(k, len(videos)), replace=False)
            else:
                indices = list(range(min(k, len(videos))))
            result[emotion] = [videos[i] for i in indices]
        
        return result


# =============================================================================
# File-Based Dataset Classes (for inference)
# =============================================================================

class FileBasedDataset(data.Dataset):
    """Base class for file-based datasets used in inference."""
    
    def __init__(
        self,
        data_root: str,
        audio_subdir: str,
        video_subdir: str,
        exp_subdir: str,
        head_subdir: str,
        keys_file: Optional[str] = None,
        coef_stats_path: str = "/data/celebv-text/processed_data/celebv_text_v3_coeff_stats.pkl",
    ):
        self.data_root = Path(data_root)
        self.audio_root = self.data_root / audio_subdir
        self.video_root = self.data_root / video_subdir
        self.exp_root = self.data_root / exp_subdir
        self.head_root = self.data_root / head_subdir
        
        # Load keys
        if keys_file:
            self.keys = pd.read_csv(keys_file, header=None).values.squeeze().tolist()
        else:
            self.keys = [f.stem for f in self.audio_root.iterdir() if f.suffix in ['.wav', '.m4a']]
        
        # Load stats
        with open(coef_stats_path, "rb") as f:
            self.coef_stats = pickle.load(f)
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def query_for_video(self, video_name: str, device: str = "cuda"):
        """Load and preprocess data for a single video."""
        exp_path = self.exp_root / f"{video_name}_code_savgol_boundbox+smooth_expression.pkl"
        head_path = self.head_root / f"{video_name}.pkl"
        
        exp = pickle.load(open(exp_path, "rb"))
        if isinstance(exp, torch.Tensor):
            exp = exp.detach().cpu().numpy()
        head = pickle.load(open(head_path, "rb"))
        
        # Normalize
        exp = (exp - self.coef_stats['exp_mean'].cpu().numpy()) / (self.coef_stats['exp_std'].cpu().numpy() + 1e-9)
        head = (head - self.coef_stats['pose_mean'].cpu().numpy()) / (self.coef_stats['pose_std'].cpu().numpy() + 1e-9)
        
        # Load audio
        audio_ext = '.wav' if (self.audio_root / f"{video_name}.wav").exists() else '.m4a'
        audio, _ = librosa.load(self.audio_root / f"{video_name}{audio_ext}", sr=16000)
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        
        # Resample to 25 FPS if needed
        video_path = self.video_root / f"{video_name}.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            gt_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            gt_fps = 29.97
        
        exp = resample_coefficients(exp, gt_fps, 25)
        head = resample_coefficients(head, gt_fps, 25)
        
        # Convert to tensors
        exp = torch.from_numpy(exp).to(device).unsqueeze(0).float()
        head = torch.from_numpy(head).to(device).unsqueeze(0).float()
        shape = torch.zeros([1, 100], device=device).float()
        motion = torch.cat([exp, head], dim=2)
        
        return audio, motion, shape


class CelebvTextFileDataset(FileBasedDataset):
    """File-based dataset for CelebV-Text."""
    
    def __init__(self):
        super().__init__(
            data_root="/data/celebv-text/",
            audio_subdir="audio/celebvtext_audio",
            video_subdir="Videos/celebvtext_6",
            exp_subdir="expression_code_ver2",
            head_subdir="head_orientations",
            keys_file="/data/celebv-text/processed_data/processed_data_30fps_medium_v3_keys_valid.txt",
        )


class RavdessFileDataset(FileBasedDataset):
    """File-based dataset for RAVDESS."""
    
    def __init__(self):
        super().__init__(
            data_root="/data/ravdess/",
            audio_subdir="audio",
            video_subdir="videos",
            exp_subdir="expression_code_ver2",
            head_subdir="head_orientations",
            keys_file="/data/ravdess/processed_data/processed_ravdess_30fps_v3_keys.txt",
        )

class MeadDataset(BaseMotionDataset):
    """Dataset for loading MEAD data from pickle + wav files."""
    
    def __init__(
        self,
        entries_path: str = "data/overlapped_mead_entries_sample100.csv",
        audio_data_root: str = "data/m2f_MEAD",
        pkl_path: str = "data/FLAME_mead_ravdess/val_mead_ravdess_0.1.pickle",
        coef_stats_file: Optional[Path] = None,
        coef_fps: int = 25,
        n_motions: int = 100,
        batch_overfit_size: int = -1,
        no_head_pose: bool = False,
        SE: bool = False,
        random_crop: bool = True,
    ):
        super().__init__(
            coef_fps=coef_fps,
            n_motions=n_motions,
            is_se=SE,
            random_crop=random_crop,
        )
        
        self.audio_data_root = audio_data_root
        self.no_head_pose = no_head_pose
        self.SE = SE
        self.coef_fps = coef_fps
        self.n_motions = n_motions
        self.audio_unit = 16000. / coef_fps
        self.n_audio_samples = round(self.audio_unit * n_motions)
        self.coef_total_len = int(n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.clip_len = n_motions
        self.random_crop = random_crop

        # Load entries
        import pandas as pd
        all_entries = pd.read_csv(entries_path, header=None).values.squeeze().tolist()[1:]
        
        # Load pickle
        print("Loading MEAD pickle...")
        raw_pkl = {}
        for chunk in load_pickle_in_chunks(pkl_path):
            for key, value in chunk.items():
                raw_pkl[key] = value
        
        # Filter to entries that exist in both the pkl and on disk
        self.entries = []
        self.data = {}
        for key in all_entries:
            audio_path = os.path.join(audio_data_root, key, "audio.wav")
            if key in raw_pkl and os.path.exists(audio_path):
                self.entries.append(key)
                self.data[key] = raw_pkl[key]
            else:
                print(f"Skipping {key}: missing from pkl or audio not found")
        
        if batch_overfit_size > 0:
            self.entries = self.entries[:batch_overfit_size]
            self.data = {k: self.data[k] for k in self.entries}
        
        print(f"Loaded {len(self.entries)} MEAD entries")

        # Load or compute stats
        stats_save_path = Path(pkl_path).parent / "mead_coef_stats.npz"
        if coef_stats_file is not None:
            self.coef_stats = load_coef_stats(coef_stats_file)
        elif stats_save_path.exists():
            print(f"Loading stats from {stats_save_path}")
            self.coef_stats = load_coef_stats(stats_save_path)
        else:
            print("No stats file provided, computing from data...")
            exp_sum  = np.zeros(100)
            exp_sq   = np.zeros(100)
            pose_sum = np.zeros(6)
            pose_sq  = np.zeros(6)
            n_frames = 0
            for key in tqdm(self.entries, desc="Computing stats"):
                d = self.data[key]
                exp  = d["exp"].squeeze(0)       # (T, 100)
                jaw  = d["jaw"].squeeze(0)       # (T, 3)
                head = d["global_pose"].squeeze(0) # (T, 3)
                pose = np.concatenate([jaw, head], axis=1)  # (T, 6)
                exp_sum  += exp.sum(axis=0)
                exp_sq   += (exp ** 2).sum(axis=0)
                pose_sum += pose.sum(axis=0)
                pose_sq  += (pose ** 2).sum(axis=0)
                n_frames += exp.shape[0]
            exp_mean  = exp_sum / n_frames
            pose_mean = pose_sum / n_frames
            exp_std   = np.sqrt(np.maximum((exp_sq / n_frames) - exp_mean ** 2, 0))
            pose_std  = np.sqrt(np.maximum((pose_sq / n_frames) - pose_mean ** 2, 0))
            exp_std   = np.where(exp_std  < 1e-9, 1.0, exp_std)
            pose_std  = np.where(pose_std < 1e-9, 1.0, pose_std)
            self.coef_stats = {
                'exp_mean':  torch.tensor(exp_mean).float(),
                'exp_std':   torch.tensor(exp_std).float(),
                'pose_mean': torch.tensor(pose_mean).float(),
                'pose_std':  torch.tensor(pose_std).float(),
            }
            np.savez(
                stats_save_path,
                exp_mean=exp_mean,
                exp_std=exp_std,
                pose_mean=pose_mean,
                pose_std=pose_std,
            )
            print(f"Stats saved to {stats_save_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        key = self.entries[index]
        d = self.data[key]

        # Load audio from disk
        audio_path = os.path.join(self.audio_data_root, key, "audio.wav")
        audio, _ = librosa.load(audio_path, sr=16000)

        # Extract motion — squeeze out batch dim: (1, T, C) -> (T, C)
        expression_code = d["exp"].squeeze(0)        # (T, 100)
        jaw_params      = d["jaw"].squeeze(0)        # (T, 3)
        head_orientation = d["global_pose"].squeeze(0)  # (T, 3)
        pose = np.concatenate([jaw_params, head_orientation], axis=1)  # (T, 6)

        # Normalize audio
        audio_mean = audio.mean()
        audio_std  = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        goal_total_length    = self.coef_total_len
        goal_each_clip_length = self.clip_len
        current_clip_length  = expression_code.shape[0]

        # Crop/pad to get two consecutive windows, matching LmdbDataset logic
        if self.random_crop:
            if current_clip_length > goal_total_length:
                start_frame1 = np.random.randint(0, current_clip_length - goal_total_length + 1)
                end_frame1   = start_frame1 + goal_each_clip_length
                start_frame2 = end_frame1
                end_frame2   = start_frame2 + goal_each_clip_length
            elif current_clip_length == goal_total_length:
                start_frame1 = 0
                end_frame1   = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2   = goal_each_clip_length * 2
            else:
                frames_to_pad       = goal_total_length - current_clip_length
                frames_to_pad_front = int(round(np.random.randint(0, frames_to_pad + 1)))
                frames_to_pad_back  = frames_to_pad - frames_to_pad_front
                expression_code = np.pad(expression_code, ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant')
                pose            = np.pad(pose,            ((frames_to_pad_front, frames_to_pad_back), (0, 0)), 'constant')
                audio_pad_front = int(round(frames_to_pad_front * self.audio_unit))
                audio_pad_back  = int(round(frames_to_pad_back  * self.audio_unit))
                audio = np.pad(audio, (audio_pad_front, audio_pad_back), 'constant')
                audio_minimal_length = int(round(goal_total_length * self.audio_unit))
                if audio.shape[0] < audio_minimal_length:
                    audio = np.pad(audio, (0, audio_minimal_length - audio.shape[0]), 'constant')
                start_frame1 = 0
                end_frame1   = goal_each_clip_length
                start_frame2 = goal_each_clip_length
                end_frame2   = goal_each_clip_length * 2
        else:
            start_frame1 = 0
            end_frame1   = goal_each_clip_length
            start_frame2 = goal_each_clip_length
            end_frame2   = goal_each_clip_length * 2
            expression_code = np.pad(expression_code, ((0, max(0, goal_total_length - current_clip_length)), (0, 0)), 'constant')
            pose            = np.pad(pose,            ((0, max(0, goal_total_length - current_clip_length)), (0, 0)), 'constant')
            audio = np.pad(audio, (0, max(0, int(round(goal_total_length * self.audio_unit)) - audio.shape[0])), 'constant')

        # Slice windows
        exp_0  = torch.tensor(expression_code[start_frame1:end_frame1]).float()
        exp_1  = torch.tensor(expression_code[start_frame2:end_frame2]).float()
        pose_0 = torch.tensor(pose[start_frame1:end_frame1]).float()
        pose_1 = torch.tensor(pose[start_frame2:end_frame2]).float()
        audio_0 = torch.tensor(audio[int(start_frame1 * self.audio_unit):int(end_frame1 * self.audio_unit)]).float()
        audio_1 = torch.tensor(audio[int(start_frame2 * self.audio_unit):int(end_frame2 * self.audio_unit)]).float()

        # Normalize
        if self.coef_stats is not None:
            exp_0  = (exp_0  - self.coef_stats['exp_mean'])  / (self.coef_stats['exp_std']  + 1e-9)
            exp_1  = (exp_1  - self.coef_stats['exp_mean'])  / (self.coef_stats['exp_std']  + 1e-9)
            pose_0 = (pose_0 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)
            pose_1 = (pose_1 - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        motion_0 = torch.cat([exp_0, pose_0], dim=-1)   # (T, 106)
        motion_1 = torch.cat([exp_1, pose_1], dim=-1)

        shape_0 = torch.zeros((motion_0.shape[0], 100)).float()
        shape_1 = torch.zeros((motion_1.shape[0], 100)).float()

        coef_dict_0 = {"shape": shape_0, "motion": motion_0}
        coef_dict_1 = {"shape": shape_1, "motion": motion_1}

        if self.SE:
            return [motion_0, motion_1]
        else:
            return [audio_0, audio_1], [coef_dict_0, coef_dict_1], (audio_mean, audio_std)

    def get_item_full(self, index: int):
        """Returns full unsplit sequence, matching LmdbDataset.get_item_full."""
        key = self.entries[index]
        d = self.data[key]

        audio_path = os.path.join(self.audio_data_root, key, "audio.wav")
        audio, _ = librosa.load(audio_path, sr=16000)

        expression_code  = d["exp"].squeeze(0)          # (T, 100)
        jaw_params       = d["jaw"].squeeze(0)           # (T, 3)
        head_orientation = d["global_pose"].squeeze(0)   # (T, 3)
        pose = np.concatenate([jaw_params, head_orientation], axis=1)  # (T, 6)

        audio_mean = audio.mean()
        audio_std  = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        expression_code = torch.tensor(expression_code).float()
        pose            = torch.tensor(pose).float()
        audio           = torch.tensor(audio).float()

        if self.coef_stats is not None:
            expression_code = (expression_code - self.coef_stats['exp_mean'])  / (self.coef_stats['exp_std']  + 1e-9)
            pose            = (pose            - self.coef_stats['pose_mean']) / (self.coef_stats['pose_std'] + 1e-9)

        motion_coef = torch.cat([expression_code, pose], dim=-1)  # (T, 106)
        shape = torch.zeros((motion_coef.shape[0], 100)).float()
        coef_dict = {"shape": shape, "motion": motion_coef}

        return audio, coef_dict, (audio_mean, audio_std)


if __name__ == "__main__":

    import os

    # Update these paths to point to your local data directory
    entries_path = "data/overlapped_mead_entries_sample100.csv"
    all_entris = pd.read_csv(entries_path, header=None).values.squeeze().tolist()[1:]

    # save audio root
    audio_data_root = "data/m2f_MEAD"
    # load all data
    BS_pkl_path = "data/FLAME_mead_ravdess/val_mead_ravdess_0.1.pickle"
    BS_pkl = {}
    for chunk in load_pickle_in_chunks(BS_pkl_path):
        for key, value in chunk.items():
            BS_pkl[key] = value
    BS_pkl.keys()

    # get one example 
    sample_name = all_entris[0]
    
    data = BS_pkl[sample_name]
    # get audio
    sample_audio_path = os.path.join(audio_data_root, sample_name)
    sample_audio = os.path.join(sample_audio_path, "audio.wav")  
    sample_audio, _ = librosa.load(sample_audio, sr=16000)
    print(f"Audio shape: {sample_audio.shape}, mean: {sample_audio.mean()}, std: {sample_audio.std()}")
    # Audio shape: (51883,), mean: 4.8825921112438664e-05, std: 0.01993979699909687

    exp_data = data["exp"]
    head_data = data["global_pose"]
    jaw_data = data["jaw"]
    print(f"exp shape: {exp_data.shape}, head shape: {head_data.shape}, jaw shape: {jaw_data.shape}")
    # exp shape: (1, 190, 100), head shape: (1, 190, 3), jaw shape: (1, 190, 3)




    # Test LmdbDataset
    train_dataset = LmdbDataset(
        lmdb_dir="data/FLAME_celebv-text/processed_flame_param/flame_param_merged.lmdb",
        split_file="data/FLAME_celebv-text/processed_data_30fps_toy_v3_keys_valid.txt",
        coef_stats_file=None,
        n_motions=100,
        batch_overfit_size=10,
        SE=True
    )

        
    # Add this debug script to test exactly where it hangs
    import torch
    from torch.utils.data import DataLoader

    # First, test the dataset directly
    print("Testing dataset directly...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"  Sample {i}: {[s.shape for s in sample]}")

    # Test collate function directly
    print("\nTesting collate function...")
    samples = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
    collate_fn = get_se_collate_fn(True)
    batch = collate_fn(samples)
    print(f"  Batch shapes: {[b.shape for b in batch]}")

    # Test sampler
    print("\nTesting sampler...")
    if hasattr(train_loader, 'sampler'):
        sampler = train_loader.sampler
        indices = list(sampler)[:10]
        print(f"  First 10 indices: {indices}")

    # Test dataloader with timeout
    print("\nTesting dataloader (with timeout)...")
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("DataLoader iteration timed out!")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

    try:
        batch = next(iter(train_loader))
        print(f"  Got batch: {[b.shape for b in batch]}")
        signal.alarm(0)  # Cancel alarm
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")