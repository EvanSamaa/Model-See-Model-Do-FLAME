"""
Dataset Factory Module.

Provides a single factory function to create datasets and dataloaders
for all supported dataset types.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
from torch.utils import data
import sys
from .msmd_datasets import (
    LmdbDataset,
    PickleDataset,
    MeadRavdessDataset,
    get_se_collate_fn,
    get_mead_ravdess_collate_fn,
    load_pickle_in_chunks,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    dataset_type: str
    data_root: Path
    batch_size: int = 32
    num_workers: int = 4
    is_se: bool = False
    no_head_pose: bool = False
    fps: int = 25
    n_motions: int = 100
    rot_repr: str = 'aa'
    stats_file: Optional[Path] = None
    batch_overfit_size: int = -1
    validation_only: bool = False
    use_test_set: bool = False


class DatasetFactory:
    """Factory class for creating datasets and dataloaders."""
    
    # Dataset type configurations
    DATASET_CONFIGS = {
        # LMDB-based datasets
        'HDTF_TFHP': {
            'loader': '_load_hdtf',
            'collate': 'standard',
        },
        # Pickle-based CelebV-Text datasets
        'celebv-text-toy': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_toy.pkl',
            'keys_prefix': 'processed_data_30fps_toy_keys',
            'full_dataset': False,
        },
        'celebv-text': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_v2.pkl',
            'keys_prefix': 'processed_data_30fps_v2_keys',
            'full_dataset': True,
        },
        'celebv-text-v2': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_v3.pkl',
            'keys_prefix': 'processed_data_30fps_v3_keys',
            'full_dataset': True,
        },
        'celebv-text-toy-v2': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_toy_v3.pkl',
            'keys_prefix': 'processed_data_30fps_toy_v3_keys',
            'full_dataset': False,
        },
        'celebv-text-medium': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_medium_v3.pkl',
            'keys_prefix': 'processed_data_30fps_medium_v3_keys',
            'full_dataset': True,
        },
        'celebv-text-medium-v2': {
            'loader': '_load_celebv_text',
            'pkl_file': 'processed_data_30fps_medium_v3.pkl',
            'keys_prefix': 'processed_data_30fps_medium_v3_keys',
            'full_dataset': True,
        },
        # RAVDESS dataset
        'ravdess': {
            'loader': '_load_ravdess',
        },
        # Combined datasets
        'ravdess+celebv-text-medium': {
            'loader': '_load_combined_ravdess_celebv',
            'celebv_pkl': 'processed_data_30fps_medium_v3.pkl',
            'celebv_keys': 'processed_data_30fps_medium_v3_keys',
        },
        'ravdess+celebv-text-full': {
            'loader': '_load_combined_ravdess_celebv',
            'celebv_pkl': 'processed_data_30fps_v3.pkl',
            'celebv_keys': 'processed_data_30fps_v3_keys',
        },
        # MEAD/RAVDESS FLAME datasets
        'flame-mead-100': {
            'loader': '_load_mead',
        },
        'flame_mead_ravdess': {
            'loader': '_load_mead_ravdess',
        },
        'HDTF_TFHP+flame_mead_ravdess': {
            'loader': '_load_combined_hdtf_mead_ravdess',
        },
        # Special combined for SE
        'celebv-text-medium+ravdess-FLMAE': {
            'loader': '_load_celebv_ravdess_flmae',
        },
        'celebv-text-full+ravdess-FLMAE': {
            'loader': '_load_celebv_ravdess_flmae_full',
        },
        'celebv-text-flmae-full': {
            'loader': '_load_celebv_flmae_full',
        },
        'ravdess-flmae': {
            'loader': '_load_ravdess_flmae',
        },



    }
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_root = Path(config.data_root)
    
    def create_datasets(self) -> Tuple[Optional[data.Dataset], data.Dataset, Optional[data.DataLoader], data.DataLoader]:
        """
        Create train and validation datasets with their dataloaders.
        
        Returns:
            Tuple of (train_dataset, val_dataset, train_loader, val_loader)
            If validation_only is True, train_dataset and train_loader will be None.
        """
        dataset_type = self.config.dataset_type
        
        if dataset_type not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Available types: {list(self.DATASET_CONFIGS.keys())}")
        
        cfg = self.DATASET_CONFIGS[dataset_type]
        loader_method = getattr(self, cfg['loader'])
        
        return loader_method(cfg)
    
    def _get_stats_file(self) -> Optional[Path]:
        """Get the coefficient statistics file path."""
        if self.config.stats_file is None:
            return None
        
        stats_file = Path(self.config.stats_file)
        if not stats_file.is_absolute():
            stats_file = self.data_root / stats_file
        return stats_file
    
    def _create_dataloader(
        self,
        dataset: data.Dataset,
        shuffle: bool = True,
        drop_last: bool = True,
        collate_fn=None,
        sampler=None,
        persistent_workers: bool = True,
    ) -> data.DataLoader:
        """Create a DataLoader with standard settings."""
        
        dataset_size = len(dataset)
        
        # Auto-adjust batch size for small datasets
        effective_batch_size = min(self.config.batch_size, dataset_size)
        
        # Don't drop last if dataset is too small
        effective_drop_last = drop_last and (dataset_size > self.config.batch_size)
        
        # Disable persistent_workers for small datasets
        use_persistent = (
            persistent_workers 
            and self.config.num_workers > 0 
            and self.config.batch_overfit_size <= 0
        )
        
        kwargs = {
            'batch_size': effective_batch_size,
            'num_workers': self.config.num_workers,
            'drop_last': effective_drop_last,
            'pin_memory': True,
        }
        
        if sampler is not None:
            kwargs['sampler'] = sampler
        else:
            kwargs['shuffle'] = shuffle
        
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        
        if use_persistent:
            kwargs['persistent_workers'] = True
        
        return data.DataLoader(dataset, **kwargs)

    # ==========================================================================
    # Dataset Loaders
    # ==========================================================================
    def _load_mead(self, cfg: Dict) -> Tuple:
        """Load MEAD dataset from pickle + wav files."""
        from refactored_datasets import MeadDataset  # adjust import as needed
        
        common_kwargs = {
            'coef_stats_file': self._get_stats_file(),
            'coef_fps': self.config.fps,
            'n_motions': self.config.n_motions,
            'no_head_pose': self.config.no_head_pose,
            'SE': self.config.is_se,
            'random_crop': self.config.batch_overfit_size <= 0,
            'batch_overfit_size': self.config.batch_overfit_size,
        }
        
        train_dataset = None
        train_loader = None

        val_dataset = MeadDataset(
            # validation_only=True,
            # use_test_set=self.config.use_test_set,
            **common_kwargs,
        )

        if not self.config.validation_only:
            train_dataset = MeadDataset(
                validation_only=False,
                **common_kwargs,
            )

        collate_fn = get_se_collate_fn(self.config.is_se)

        if not self.config.validation_only:
            train_loader = self._create_dataloader(train_dataset, collate_fn=collate_fn, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, collate_fn=collate_fn, persistent_workers=False, shuffle=False)

        return train_dataset, val_dataset, train_loader, val_loader
    def _load_hdtf(self, cfg: Dict) -> Tuple:
        """Load HDTF_TFHP LMDB dataset."""
        stats_file = self._get_stats_file()
        do_random_pad = self.config.batch_overfit_size <= 0
        
        train_dataset = None
        train_loader = None
        
        if self.config.is_se:
            if not self.config.validation_only:
                train_dataset = LmdbDatasetForSE(
                    self.data_root, self.data_root / 'train.txt', stats_file,
                    self.config.fps, self.config.n_motions,
                    rot_repr=self.config.rot_repr, no_head_pose=self.config.no_head_pose
                )
            val_dataset = LmdbDatasetForSE(
                self.data_root, self.data_root / 'val.txt', stats_file,
                self.config.fps, self.config.n_motions,
                rot_repr=self.config.rot_repr, no_head_pose=self.config.no_head_pose
            )
        else:
            if not self.config.validation_only:
                train_dataset = LmdbDataset(
                    self.data_root, self.data_root / 'train.txt', stats_file,
                    self.config.fps, self.config.n_motions,
                    rot_repr=self.config.rot_repr,
                    do_random_pad=do_random_pad,
                    batch_overfit_size=self.config.batch_overfit_size
                )
            val_dataset = LmdbDataset(
                self.data_root, self.data_root / 'val.txt', stats_file,
                self.config.fps, self.config.n_motions,
                rot_repr=self.config.rot_repr,
                do_random_pad=do_random_pad,
                batch_overfit_size=self.config.batch_overfit_size
            )
        
        if not self.config.validation_only:
            train_loader = self._create_dataloader(train_dataset)
        val_loader = self._create_dataloader(val_dataset, persistent_workers=False)
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_celebv_text(self, cfg: Dict) -> Tuple:
        """Load CelebV-Text pickle dataset."""
        pkl_file = self.data_root / cfg['pkl_file']
        keys_prefix = cfg['keys_prefix']
        full_dataset = cfg.get('full_dataset', False)
        do_random_pad = self.config.batch_overfit_size <= 0
        
        # Pre-load raw data for full datasets
        raw_data = None
        if full_dataset:
            raw_data = {}
            for chunk in load_pickle_in_chunks(pkl_file):
                raw_data.update(chunk)
        
        common_kwargs = {
            'coef_stats_file': self._get_stats_file(),
            'original_fps': 30,
            'coef_fps': self.config.fps,
            'n_motions': self.config.n_motions,
            'no_head_pose': self.config.no_head_pose,
            'is_se': self.config.is_se,
            'full_dataset': full_dataset,
            'pre_loaded_raw_dataset': raw_data,
            'random_crop': do_random_pad,
            'batch_overfit_size': self.config.batch_overfit_size,
        }
        
        train_dataset = None
        train_loader = None
        
        if not self.config.validation_only:
            train_dataset = PickleDataset(
                pkl_file, self.data_root / f'{keys_prefix}_train.txt', **common_kwargs
            )
        val_dataset = PickleDataset(
            pkl_file, self.data_root / f'{keys_prefix}_valid.txt', **common_kwargs
        )
        
        collate_fn = get_se_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(train_dataset, collate_fn=collate_fn)
        val_loader = self._create_dataloader(val_dataset, collate_fn=collate_fn, persistent_workers=False)
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_ravdess(self, cfg: Dict) -> Tuple:
        """Load RAVDESS pickle dataset."""
        ravdess_root = self._get_ravdess_root()
        do_random_pad = self.config.batch_overfit_size <= 0
        
        common_kwargs = {
            'coef_stats_file': None,
            'original_fps': 30,
            'coef_fps': self.config.fps,
            'no_head_pose': self.config.no_head_pose,
            'is_se': self.config.is_se,
            'full_dataset': True,
            'celebv_text': False,
            'random_crop': do_random_pad,
            'batch_overfit_size': self.config.batch_overfit_size,
        }
        
        train_dataset = None
        train_loader = None
        
        if not self.config.validation_only:
            train_dataset = PickleDataset(
                ravdess_root / 'processed_ravdess_30fps_v3.pkl',
                ravdess_root / 'processed_ravdess_30fps_v3_keys_train.txt',
                **common_kwargs
            )
        val_dataset = PickleDataset(
            ravdess_root / 'processed_ravdess_30fps_v3.pkl',
            ravdess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
            **common_kwargs
        )
        
        collate_fn = get_se_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(train_dataset, collate_fn=collate_fn)
        val_loader = self._create_dataloader(val_dataset, collate_fn=collate_fn, persistent_workers=False)
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_mead_ravdess(self, cfg: Dict) -> Tuple:
        """Load MEAD/RAVDESS FLAME dataset."""
        mead_ravdess_root = self._get_mead_ravdess_root()
        do_random_pad = self.config.batch_overfit_size <= 0
        
        # Load pickle files - only load what's needed
        val_motion = pickle.load(open(mead_ravdess_root / "val_mead_ravdess_0.1.pickle", "rb"))
        val_audio = pickle.load(open(mead_ravdess_root / "val_mead_ravdess_rawaudio_0.1.pickle", "rb"))
        
        common_kwargs = {
            'coef_stats_file': None,
            'original_fps': 30,
            'coef_fps': self.config.fps,
            'no_head_pose': self.config.no_head_pose,
            'is_se': self.config.is_se,
            'random_crop': do_random_pad,
            'batch_overfit_size': self.config.batch_overfit_size,
        }
        
        train_dataset = None
        train_loader = None
        
        if not self.config.validation_only:
            train_motion = pickle.load(open(mead_ravdess_root / "train_mead_ravdess_0.1.pickle", "rb"))
            train_audio = pickle.load(open(mead_ravdess_root / "train_mead_ravdess_rawaudio_0.1.pickle", "rb"))
            train_dataset = MeadRavdessDataset(train_motion, train_audio, **common_kwargs)
        
        val_dataset = MeadRavdessDataset(val_motion, val_audio, **common_kwargs)
        
        collate_fn = get_mead_ravdess_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(train_dataset, collate_fn=collate_fn)
        val_loader = self._create_dataloader(val_dataset, collate_fn=collate_fn, persistent_workers=False)
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_combined_ravdess_celebv(self, cfg: Dict) -> Tuple:
        """Load combined RAVDESS + CelebV-Text dataset with weighted sampling."""
        ravdess_root = self._get_ravdess_root()
        
        # Load CelebV-Text
        celebv_pkl = self.data_root / cfg['celebv_pkl']
        celebv_keys = cfg['celebv_keys']
        
        raw_data = {}
        for chunk in load_pickle_in_chunks(celebv_pkl):
            raw_data.update(chunk)
        
        common_kwargs = {
            'coef_stats_file': None,
            'original_fps': 30,
            'coef_fps': self.config.fps,
            'no_head_pose': self.config.no_head_pose,
            'is_se': self.config.is_se,
            'full_dataset': True,
        }
        
        train_dataset = None
        train_loader = None
        train_sampler = None
        
        # Validation datasets (always loaded)
        val_ravdess = PickleDataset(
            ravdess_root / 'processed_ravdess_30fps_v3.pkl',
            ravdess_root / 'processed_ravdess_30fps_v3_keys_valid.txt',
            celebv_text=False, **common_kwargs
        )
        val_celebv = PickleDataset(
            celebv_pkl, self.data_root / f'{celebv_keys}_valid.txt',
            pre_loaded_raw_dataset=raw_data, **common_kwargs
        )
        val_dataset, val_sampler = self._create_weighted_concat([val_celebv, val_ravdess])
        
        if not self.config.validation_only:
            # Training datasets
            train_ravdess = PickleDataset(
                ravdess_root / 'processed_ravdess_30fps_v3.pkl',
                ravdess_root / 'processed_ravdess_30fps_v3_keys_train.txt',
                celebv_text=False, **common_kwargs
            )
            train_celebv = PickleDataset(
                celebv_pkl, self.data_root / f'{celebv_keys}_train.txt',
                pre_loaded_raw_dataset=raw_data, **common_kwargs
            )
            train_dataset, train_sampler = self._create_weighted_concat([train_celebv, train_ravdess])
        
        collate_fn = get_se_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, sampler=train_sampler
            )
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, sampler=val_sampler, persistent_workers=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_combined_hdtf_mead_ravdess(self, cfg: Dict) -> Tuple:
        """Load combined HDTF + MEAD/RAVDESS dataset."""
        hdtf_root = Path("/data/HDTF_TFHP/lmdb/")
        mead_ravdess_root = self._get_mead_ravdess_root()
        stats_file = hdtf_root / "stats_train.npz" if self._get_stats_file() is None else self._get_stats_file()
        do_random_pad = self.config.batch_overfit_size <= 0
        
        train_dataset = None
        train_loader = None
        
        # Load validation data (always needed)
        if self.config.is_se:
            val_hdtf = LmdbDatasetForSE(
                hdtf_root, hdtf_root / 'val.txt', stats_file,
                self.config.fps, self.config.n_motions, rot_repr=self.config.rot_repr
            )
        else:
            val_hdtf = LmdbDataset(
                hdtf_root, hdtf_root / 'val.txt', stats_file,
                self.config.fps, self.config.n_motions, rot_repr=self.config.rot_repr,
                do_random_pad=do_random_pad, batch_overfit_size=self.config.batch_overfit_size
            )
        
        val_motion = pickle.load(open(mead_ravdess_root / "val_mead_ravdess_0.1.pickle", "rb"))
        val_audio = pickle.load(open(mead_ravdess_root / "val_mead_ravdess_rawaudio_0.1.pickle", "rb"))
        val_mead = MeadRavdessDataset(
            val_motion, val_audio,
            no_head_pose=self.config.no_head_pose, is_se=self.config.is_se,
            random_crop=do_random_pad, batch_overfit_size=self.config.batch_overfit_size
        )
        val_dataset, val_sampler = self._create_weighted_concat([val_hdtf, val_mead])
        
        if not self.config.validation_only:
            # Load HDTF training
            if self.config.is_se:
                train_hdtf = LmdbDatasetForSE(
                    hdtf_root, hdtf_root / 'train.txt', stats_file,
                    self.config.fps, self.config.n_motions, rot_repr=self.config.rot_repr
                )
            else:
                train_hdtf = LmdbDataset(
                    hdtf_root, hdtf_root / 'train.txt', stats_file,
                    self.config.fps, self.config.n_motions, rot_repr=self.config.rot_repr,
                    do_random_pad=do_random_pad, batch_overfit_size=self.config.batch_overfit_size
                )
            
            # Load MEAD/RAVDESS training
            train_motion = pickle.load(open(mead_ravdess_root / "train_mead_ravdess_0.1.pickle", "rb"))
            train_audio = pickle.load(open(mead_ravdess_root / "train_mead_ravdess_rawaudio_0.1.pickle", "rb"))
            train_mead = MeadRavdessDataset(
                train_motion, train_audio,
                no_head_pose=self.config.no_head_pose, is_se=self.config.is_se,
                random_crop=do_random_pad, batch_overfit_size=self.config.batch_overfit_size
            )
            train_dataset, train_sampler = self._create_weighted_concat([train_hdtf, train_mead])
        
        collate_fn = get_mead_ravdess_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, sampler=train_sampler
            )
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, sampler=val_sampler, persistent_workers=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_celebv_ravdess_flmae_full(self, cfg: Dict) -> Tuple:
        """Load full CelebV-Text + RAVDESS for FLMAE training."""
        celebv_lmdb_path = self.config.data_root / "FLAME_celebv-text/processed_flame_param/flame_param_merged.lmdb"
        celebv_split_root = self.config.data_root / "FLAME_celebv-text"
        stats_file = self._get_stats_file()
        
        train_dataset = None
        train_loader = None
        
        # Validation CelebV-Text
        val_dataset_celebv = LmdbDataset(
            celebv_lmdb_path, celebv_split_root / 'processed_data_30fps_v3_keys_valid.txt', stats_file,
            30, self.config.fps, self.config.n_motions,
            no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        # Handle RAVDESS splits
        ravdess_lmdb = self.config.data_root / "FLAME_ravdess/processed_flame_param_ravdess_25fps.lmdb"
        ravdess_root = self.config.data_root / 'FLAME_ravdess'
        ravdess_split = ravdess_root / 'processed_ravdess_30fps_v3_keys_train.txt' 
        if not os.path.exists(ravdess_split):
            ravdess_keys = ravdess_root / "processed_flame_param_ravdess_25fps_keys.txt"
            with open(ravdess_keys, 'r') as f:
                all_keys = f.readlines()
            split_idx = int(0.9 * len(all_keys))
            import random
            random.seed(42)
            random.shuffle(all_keys)
            train_keys = all_keys[:split_idx]
            valid_keys = all_keys[split_idx:]
            with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', 'w') as f:
                f.writelines(train_keys)
            with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt', 'w') as f:
                f.writelines(valid_keys)
        
        # Validation RAVDESS
        val_dataset_ravdess = LmdbDataset(
            ravdess_lmdb, ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt', stats_file,
            25, self.config.fps, self.config.n_motions,
            no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        val_dataset, val_sampler = self._create_weighted_concat([val_dataset_celebv, val_dataset_ravdess])
        
        if not self.config.validation_only:
            # Training CelebV-Text
            train_dataset_celebv = LmdbDataset(
                celebv_lmdb_path, celebv_split_root / 'processed_data_30fps_v3_keys_train.txt', stats_file,
                30, self.config.fps, self.config.n_motions,
                no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )
            
            # Training RAVDESS
            train_dataset_ravdess = LmdbDataset(
                ravdess_lmdb, ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', stats_file,
                25, self.config.fps, self.config.n_motions,
                no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )
            
            train_dataset, train_sampler = self._create_weighted_concat([train_dataset_celebv, train_dataset_ravdess])
        
        collate_fn = get_se_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, sampler=train_sampler
            )
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, sampler=val_sampler, persistent_workers=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_celebv_ravdess_flmae(self, cfg: Dict) -> Tuple:
        """Load CelebV-Text + RAVDESS for FLMAE training."""
        celebv_lmdb_path = self.config.data_root / "FLAME_celebv-text/processed_flame_param/flame_param_merged.lmdb"
        celebv_split_root = self.config.data_root / "FLAME_celebv-text"
        stats_file = self._get_stats_file()
        
        train_dataset = None
        train_loader = None
        
        # Validation CelebV-Text
        val_dataset_celebv = LmdbDataset(
            celebv_lmdb_path, celebv_split_root / 'processed_data_30fps_medium_v3_keys_valid.txt', stats_file,
            30, self.config.fps, self.config.n_motions,
            no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        # Handle RAVDESS splits
        ravdess_lmdb = self.config.data_root / "FLAME_ravdess/processed_flame_param_ravdess_25fps.lmdb"
        ravdess_root = self.config.data_root / 'FLAME_ravdess'
        ravdess_split = ravdess_root / 'processed_ravdess_30fps_v3_keys_train.txt' 
        if not os.path.exists(ravdess_split):
            ravdess_keys = ravdess_root / "processed_flame_param_ravdess_25fps_keys.txt"
            with open(ravdess_keys, 'r') as f:
                all_keys = f.readlines()
            split_idx = int(0.9 * len(all_keys))
            import random
            random.seed(42)
            random.shuffle(all_keys)
            train_keys = all_keys[:split_idx]
            valid_keys = all_keys[split_idx:]
            with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', 'w') as f:
                f.writelines(train_keys)
            with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt', 'w') as f:
                f.writelines(valid_keys)
        
        # Validation RAVDESS
        val_dataset_ravdess = LmdbDataset(
            ravdess_lmdb, ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt', stats_file,
            25, self.config.fps, self.config.n_motions,
            no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        val_dataset, val_sampler = self._create_weighted_concat([val_dataset_celebv, val_dataset_ravdess])
        
        if not self.config.validation_only:
            # Training CelebV-Text
            train_dataset_celebv = LmdbDataset(
                celebv_lmdb_path, celebv_split_root / 'processed_data_30fps_medium_v3_keys_train.txt', stats_file,
                30, self.config.fps, self.config.n_motions,
                no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )
            
            # Training RAVDESS
            train_dataset_ravdess = LmdbDataset(
                ravdess_lmdb, ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', stats_file,
                25, self.config.fps, self.config.n_motions,
                no_head_pose=self.config.no_head_pose, SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )
            
            train_dataset, train_sampler = self._create_weighted_concat([train_dataset_celebv, train_dataset_ravdess])
        
        collate_fn = get_se_collate_fn(self.config.is_se)
        if not self.config.validation_only:
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, sampler=train_sampler
            )
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, sampler=val_sampler, persistent_workers=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader
    
    def _load_celebv_flmae_full(self, cfg: Dict) -> Tuple:
        """Load only the full CelebV-Text FLAME dataset (returns LmdbDataset)."""
        celebv_lmdb_path = self.config.data_root / "FLAME_celebv-text/processed_flame_param/flame_param_merged.lmdb"
        celebv_split_root = self.config.data_root / "FLAME_celebv-text"
        stats_file = self._get_stats_file()
        
        train_dataset = None
        train_loader = None
        
        if self.config.use_test_set:
            key_set = celebv_split_root / 'processed_data_30fps_v3_keys_test.txt'
        else:
            key_set = celebv_split_root / 'processed_data_30fps_v3_keys_valid.txt'
        # Validation Dataset
        val_dataset = LmdbDataset(
            celebv_lmdb_path, 
            key_set,
            stats_file,
            30, # Original FPS for CelebV
            self.config.fps, 
            self.config.n_motions,
            no_head_pose=self.config.no_head_pose, 
            SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        if not self.config.validation_only:
            # Training Dataset
            train_dataset = LmdbDataset(
                celebv_lmdb_path, 
                celebv_split_root / 'processed_data_30fps_v3_keys_train.txt', 
                stats_file,
                30, 
                self.config.fps, 
                self.config.n_motions,
                no_head_pose=self.config.no_head_pose, 
                SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )
            
        collate_fn = get_se_collate_fn(self.config.is_se)
        
        if not self.config.validation_only:
            # Standard shuffle=True is used since we aren't using a weighted sampler
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, shuffle=True
            )
            
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, persistent_workers=False, shuffle=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader

    def _load_ravdess_flmae(self, cfg: Dict) -> Tuple:
        """Load only the RAVDESS FLAME dataset (returns LmdbDataset)."""
        ravdess_lmdb = self.config.data_root / "FLAME_ravdess/processed_flame_param_ravdess_25fps.lmdb"
        ravdess_root = self.config.data_root / 'FLAME_ravdess'
        stats_file = self._get_stats_file()
        
        # --- Split Logic (Preserved from original) ---
        # Checks if split files exist, otherwise creates them based on a 90/10 random split
        ravdess_split = ravdess_root / 'processed_ravdess_30fps_v3_keys_train.txt' 
        if not os.path.exists(ravdess_split):
            ravdess_keys = ravdess_root / "processed_flame_param_ravdess_25fps_keys.txt"
            if ravdess_keys.exists():
                with open(ravdess_keys, 'r') as f:
                    all_keys = f.readlines()
                split_idx = int(0.9 * len(all_keys))
                import random
                # Ensure deterministic split creation
                random.seed(42)
                random.shuffle(all_keys)
                train_keys = all_keys[:split_idx]
                valid_keys = all_keys[split_idx:]
                
                # Write splits to disk
                with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', 'w') as f:
                    f.writelines(train_keys)
                if self.config.use_test_set:
                    with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_test.txt', 'w') as f:
                        f.writelines(valid_keys)
                else:
                    with open(ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt', 'w') as f:
                        f.writelines(valid_keys)
        # ---------------------------------------------

        train_dataset = None
        train_loader = None
        
        # Validation Dataset
        if self.config.use_test_set:
            key_set = ravdess_root / 'processed_flame_param_ravdess_25fps_keys_test.txt'
        else:
            key_set = ravdess_root / 'processed_flame_param_ravdess_25fps_keys_valid.txt'
        

        val_dataset = LmdbDataset(
            ravdess_lmdb, 
            key_set, 
            stats_file,
            25, # Original FPS for RAVDESS FLAME
            self.config.fps, 
            self.config.n_motions,
            no_head_pose=self.config.no_head_pose, 
            SE=self.config.is_se,
            batch_overfit_size=self.config.batch_overfit_size
        )
        
        if not self.config.validation_only:
            # Training Dataset
            train_dataset = LmdbDataset(
                ravdess_lmdb, 
                ravdess_root / 'processed_flame_param_ravdess_25fps_keys_train.txt', 
                stats_file,
                25, 
                self.config.fps, 
                self.config.n_motions,
                no_head_pose=self.config.no_head_pose, 
                SE=self.config.is_se,
                batch_overfit_size=self.config.batch_overfit_size
            )

        collate_fn = get_se_collate_fn(self.config.is_se)
        
        if not self.config.validation_only:
            # Standard shuffle=True
            train_loader = self._create_dataloader(
                train_dataset, collate_fn=collate_fn, shuffle=True
            )
            
        val_loader = self._create_dataloader(
            val_dataset, collate_fn=collate_fn, persistent_workers=False, shuffle=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader

    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _create_weighted_concat(
        self,
        datasets: list,
    ) -> Tuple[data.ConcatDataset, data.WeightedRandomSampler]:
        """Create a ConcatDataset with weighted sampling to balance datasets."""
        weights = []
        for ds in datasets:
            w = 1.0 / len(ds)
            weights.extend([w] * len(ds))
        
        concat_dataset = data.ConcatDataset(datasets)
        sampler = data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        
        return concat_dataset, sampler
    
    @staticmethod
    def _get_ravdess_root() -> Path:
        """Get RAVDESS data root, checking multiple possible locations."""
        candidates = [
            Path("/mnt/f/chrome_downloads/processed_data"),
            Path("/data/ravdess/processed_data"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[-1]  # Default to last option
    
    @staticmethod
    def _get_mead_ravdess_root() -> Path:
        """Get MEAD/RAVDESS data root, checking multiple possible locations."""
        candidates = [
            Path("/mnt/f/chrome_downloads/mead_ravdess_30fps"),
            Path("/data/mead_ravdess_30fps"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[-1]

    

# =============================================================================
# Convenience Function
# =============================================================================

def create_datasets(
    dataset_type: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    is_se: bool = False,
    no_head_pose: bool = False,
    fps: int = 25,
    n_motions: int = 100,
    rot_repr: str = 'aa',
    stats_file: Optional[str] = None,
    batch_overfit_size: int = -1,
    validation_only: bool = False,
    use_test_set: bool = False,
) -> Tuple[Optional[data.Dataset], data.Dataset, Optional[data.DataLoader], data.DataLoader]:
    """
    Create train and validation datasets with their dataloaders.
    
    This is the main entry point for dataset creation.
    
    Args:
        dataset_type: Type of dataset to create (e.g., 'celebv-text', 'HDTF_TFHP')
        data_root: Root directory containing the data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        is_se: Whether this is for Style Encoder training
        no_head_pose: Whether to exclude head pose from coefficients
        fps: Target frames per second
        n_motions: Number of motion frames per sample
        rot_repr: Rotation representation ('aa' for axis-angle)
        stats_file: Path to coefficient statistics file
        batch_overfit_size: If > 0, limit dataset to this many samples (for debugging)
        validation_only: If True, only load validation dataset (returns None for train)
    
    Returns:
        Tuple of (train_dataset, val_dataset, train_loader, val_loader)
        If validation_only is True, train_dataset and train_loader will be None.
    
    Example:
        >>> # Full training setup
        >>> train_ds, val_ds, train_loader, val_loader = create_datasets(
        ...     dataset_type='celebv-text-medium',
        ...     data_root='/data/celebv-text/processed_data',
        ...     batch_size=32,
        ...     is_se=True,
        ... )
        
        >>> # Validation only (faster loading for evaluation)
        >>> _, val_ds, _, val_loader = create_datasets(
        ...     dataset_type='celebv-text-medium',
        ...     data_root='/data/celebv-text/processed_data',
        ...     batch_size=32,
        ...     validation_only=True,
        ... )
    """
    config = DatasetConfig(
        dataset_type=dataset_type,
        data_root=Path(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        is_se=is_se,
        no_head_pose=no_head_pose,
        fps=fps,
        n_motions=n_motions,
        rot_repr=rot_repr,
        stats_file=Path(stats_file) if stats_file else None,
        batch_overfit_size=batch_overfit_size,
        validation_only=validation_only,
        use_test_set=use_test_set,
    )
    
    factory = DatasetFactory(config)
    return factory.create_datasets()


def get_available_dataset_types() -> list:
    """Return list of available dataset types."""
    return list(DatasetFactory.DATASET_CONFIGS.keys())


if __name__ == "__main__":
    # Example usage - full training
    print("Testing full dataset loading...")
    train_ds, val_ds, train_loader, val_loader = create_datasets(
        dataset_type='celebv-text-medium+ravdess-FLMAE',
        data_root='data/',
        batch_size=32,
        is_se=False,
        batch_overfit_size=10,
    )
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    # Example usage - validation only
    print("\nTesting validation-only loading...")
    train_ds_val, val_ds_val, train_loader_val, val_loader_val = create_datasets(
        dataset_type='celebv-text-medium+ravdess-FLMAE',
        data_root='data/',
        batch_size=32,
        is_se=False,
        batch_overfit_size=10,
        validation_only=True,
    )
    print(f"Train dataset: {train_ds_val}")  # Should be None
    print(f"Train loader: {train_loader_val}")  # Should be None
    print(f"Validation dataset size: {len(val_ds_val)}")