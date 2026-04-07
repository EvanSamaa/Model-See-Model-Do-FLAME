# Data Directory

Place your processed datasets here.

## Expected Structure

```
data/
├── coef_stats.npz                          # Coefficient statistics file
│                                           # (mean/std per coefficient dimension)
├── FLAME_celebv-text/                      # CelebV-Text FLAME parameters
│   └── processed_flame_param/
│       └── flame_param_merged.lmdb/
├── FLAME_mead_ravdess/                     # MEAD + RAVDESS FLAME parameters
│   ├── train_mead_ravdess_*.pickle
│   └── val_mead_ravdess_*.pickle
└── m2f_MEAD/                               # Raw MEAD audio (optional)
```

## Generating `coef_stats.npz`

Run the stats computation script on your LMDB dataset:

```bash
python -c "
from msmd.data.msmd_datasets import compute_incremental_stats
# See compute_stats.py for a full example
"
```
