"""
Style Encoder Training Script.

Train a style encoder model using contrastive learning on motion coefficients.
"""

import argparse
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
import sys

from msmd.data.dataset_factory import create_datasets, get_available_dataset_types
from msmd.data.msmd_datasets import get_se_collate_fn
from msmd.models.style_encoder import StyleEncoder_celebv, get_style_encoder, StyleEncoder_VAE
import msmd.options.se as options


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with console and optional file output.
    
    Args:
        log_dir: Directory for log files. If None, only console logging.
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("style_encoder")
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpoints, keeping only the best and latest.
    
    Attributes:
        save_dir: Directory to save checkpoints
        best_metric: Best validation metric seen so far
        best_path: Path to the best checkpoint
        latest_path: Path to the latest checkpoint
    """
    
    def __init__(
        self,
        save_dir: Path,
        metric_name: str = "val_loss",
        mode: str = "min",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            metric_name: Name of metric to track for best model
            mode: "min" if lower is better, "max" if higher is better
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        
        self.best_path = self.save_dir / "best.pt"
        self.latest_path = self.save_dir / "latest.pt"
        
        self.logger = logger or logging.getLogger("style_encoder")
    
    def is_better(self, metric: float) -> bool:
        """Check if the given metric is better than the current best."""
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric
    
    def save(
        self,
        state: dict,
        metric: Optional[float] = None,
        iteration: int = 0
    ) -> dict:
        """
        Save checkpoint, updating best if metric improved.
        
        Args:w
            state: Dictionary containing model state, optimizer state, etc.
            metric: Current validation metric (for best model tracking)
            iteration: Current training iteration
        
        Returns:
            Dictionary with save info: {"saved_latest": bool, "saved_best": bool}
        """
        save_info = {"saved_latest": False, "saved_best": False}
        
        # Add metadata
        state["iteration"] = iteration
        state["timestamp"] = datetime.now().isoformat()
        
        # Always save latest
        torch.save(state, self.latest_path)
        save_info["saved_latest"] = True
        self.logger.debug(f"Saved latest checkpoint at iteration {iteration}")
        
        # Check if this is the best model
        if metric is not None and self.is_better(metric):
            old_best = self.best_metric
            self.best_metric = metric
            state["best_metric"] = metric
            torch.save(state, self.best_path)
            save_info["saved_best"] = True
            self.logger.info(
                f"New best model! {self.metric_name}: {old_best:.4e} -> {metric:.4e}"
            )
        
        return save_info
    
    def load_latest(self, device: str = "cpu") -> Optional[dict]:
        """Load the latest checkpoint if it exists."""
        if self.latest_path.exists():
            self.logger.info(f"Loading latest checkpoint from {self.latest_path}")
            return torch.load(self.latest_path, map_location=device, weights_only=False)
        return None
    
    def load_best(self, device: str = "cpu") -> Optional[dict]:
        """Load the best checkpoint if it exists."""
        if self.best_path.exists():
            self.logger.info(f"Loading best checkpoint from {self.best_path}")
            checkpoint = torch.load(self.best_path, map_location=device, weights_only=False)
            if "best_metric" in checkpoint:
                self.best_metric = checkpoint["best_metric"]
            return checkpoint
        return None


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """Handles saving and loading of experiment configurations."""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_path = exp_dir / "config.json"
    
    def save(self, args: argparse.Namespace) -> None:
        """Save configuration to JSON file."""
        config = vars(args).copy()
        
        # Convert non-serializable types
        for key, value in config.items():
            if isinstance(value, Path):
                config[key] = str(value)
            elif hasattr(value, '__dict__'):
                config[key] = str(value)
        
        # Add metadata
        config["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def load(self) -> argparse.Namespace:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Remove metadata before creating namespace
        config.pop("_metadata", None)
        
        return argparse.Namespace(**config)
    
    def exists(self) -> bool:
        """Check if config file exists."""
        return self.config_path.exists()


# =============================================================================
# Utility Functions
# =============================================================================

def nt_xent_loss(feat_a: torch.Tensor, feat_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Args:
        feat_a: Features from first view [batch_size, feature_dim]
        feat_b: Features from second view [batch_size, feature_dim]
        temperature: Temperature parameter for scaling
    
    Returns:
        Scalar loss tensor
    """
    batch_size = feat_a.shape[0]
    
    # Normalize features
    feat_a = torch.nn.functional.normalize(feat_a, dim=1)
    feat_b = torch.nn.functional.normalize(feat_b, dim=1)
    
    # Concatenate features
    features = torch.cat([feat_a, feat_b], dim=0)
    
    # Compute similarity matrix
    similarity = torch.matmul(features, features.T) / temperature
    
    # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=feat_a.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=feat_a.device).bool()
    similarity = similarity.masked_fill(mask, float('-inf'))
    
    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(similarity, labels)
    
    return loss


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infinite_data_loader(data_loader):
    """Create an infinite iterator over a data loader."""
    while True:
        yield from data_loader


def get_model_path(exp_name: str, iteration: int, model_type: str = 'SE') -> Tuple[Path, dict]:
    """
    Get the path to a saved model checkpoint.
    
    Args:
        exp_name: Experiment name
        iteration: Training iteration
        model_type: Type of model ('SE' for style encoder)
    
    Returns:
        Tuple of (checkpoint_path, metadata)
    """
    exp_root = Path('experiments') / model_type
    
    # Find matching experiment directory
    for exp_dir in exp_root.iterdir():
        if exp_dir.name.startswith(exp_name):
            # Check for new-style checkpoints first
            for ckpt_name in ["best.pt", "latest.pt"]:
                checkpoint_path = exp_dir / 'checkpoints' / ckpt_name
                if checkpoint_path.exists():
                    return checkpoint_path, {}
            
            # Fall back to old-style iteration checkpoints
            checkpoint_path = exp_dir / 'checkpoints' / f'iter_{iteration:07}.pt'
            if checkpoint_path.exists():
                return checkpoint_path, {}
    
    raise ValueError(f"Checkpoint not found for {exp_name}")


# =============================================================================
# Training Functions
# =============================================================================

def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    checkpoint_manager: CheckpointManager,
    writer: Optional[SummaryWriter] = None,
    start_iter: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Main training loop for the Style Encoder.
    
    Args:
        args: Training arguments
        model: Style encoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        checkpoint_manager: CheckpointManager instance
        writer: TensorBoard writer
        start_iter: Starting iteration (for resuming training)
        logger: Logger instance
    """
    logger = logger or logging.getLogger("style_encoder")
    device = next(model.parameters()).device
    
    model.encoder.train()
    data_iter = iter(infinite_data_loader(train_loader))
    
    loss_log = deque(maxlen=args.log_smooth_win)
    pbar = tqdm(
        range(start_iter, args.max_iter + 1),
        initial=start_iter,
        total=args.max_iter + 1,
        dynamic_ncols=True
    )
    
    optimizer.zero_grad()
    
    for it in pbar:
        # Load data
        coef_pair = next(data_iter)
        coef_pair = [coef.to(device) for coef in coef_pair]
        # Forward pass
        feat_a = model(coef_pair[0])
        feat_b = model(coef_pair[1])
        
        # Compute loss
        loss = nt_xent_loss(feat_a, feat_b, args.temperature)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update loss tracking
        loss_val = loss.item()
        loss_log.append(loss_val)
        avg_loss = np.mean(loss_log)
        pbar.set_description(f'Loss: {avg_loss:.3e}')
        
        # Periodic logging
        if it % args.log_iter == 0:
            if writer is not None:
                writer.add_scalar('train/loss', loss_val, it)
                writer.add_scalar('train/loss_smooth', avg_loss, it)
            
            if it > 0:  # Skip initial log to avoid cluttering
                logger.debug(f"Iter {it}: loss={loss_val:.4e}, avg_loss={avg_loss:.4e}")
        
        # Validation
        val_loss = None
        if (it % args.val_iter == 0 and it > 0) or it == args.max_iter:
            val_loss = validate(
                args, model, val_loader, it, 
                n_rounds=5, mode='val', writer=writer, logger=logger
            )
        
        # Save checkpoint (only at validation intervals or final iteration)
        if (it % args.val_iter == 0 and it > 0) or it == args.max_iter:
            checkpoint_state = {
                'args': args,
                'encoder': model.encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_manager.save(checkpoint_state, metric=val_loss, iteration=it)
        
        # Clean up
        del loss, feat_a, feat_b
        torch.cuda.empty_cache()
    
    logger.info(f"Training complete. Best val_loss: {checkpoint_manager.best_metric:.4e}")


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    test_loader,
    current_iter: int,
    n_rounds: int = 10,
    mode: str = 'val',
    writer: Optional[SummaryWriter] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Validate the model.
    
    Args:
        args: Training arguments
        model: Style encoder model
        test_loader: Test/validation data loader
        current_iter: Current training iteration
        n_rounds: Number of validation rounds
        mode: 'val' or 'test'
        writer: TensorBoard writer
        logger: Logger instance
    
    Returns:
        Average validation loss
    """
    logger = logger or logging.getLogger("style_encoder")
    was_training = model.encoder.training
    device = next(model.parameters()).device
    model.encoder.eval()
    
    loss_log = []
    
    for _ in range(n_rounds):
        for coef_pair in test_loader:
            coef_pair = [coef.to(device) for coef in coef_pair]
            
            feat_a = model(coef_pair[0])
            feat_b = model(coef_pair[1])
            loss = nt_xent_loss(feat_a, feat_b, args.temperature)
            
            loss_log.append(loss.item())
    
    avg_loss = np.mean(loss_log)
    std_loss = np.std(loss_log)
    
    logger.info(f"[{mode.upper()}] Iter {current_iter}: loss={avg_loss:.4e} ± {std_loss:.4e}")
    
    if writer is not None:
        writer.add_scalar(f'{mode}/loss', avg_loss, current_iter)
        writer.add_scalar(f'{mode}/loss_std', std_loss, current_iter)
    
    if was_training:
        model.encoder.train()
    
    return avg_loss


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args(custom_args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Style Encoder')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=100000, help='Iteration to test')
    parser.add_argument('--continue_from', type=str, default=None,
                       help='Path to experiment directory to continue from')
    parser.add_argument("--batch_overfit_size", type=int, default=-1,
                        help="If > 0, overfit to a small batch of this size for debugging")
    parser.add_argument('--use_normalization', action='store_true',
                        help='Whether to use normalization in the style encoder')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level')
    
    options.add_model_options(parser)
    options.add_data_options(parser)
    options.add_training_options(parser)
    
    args = parser.parse_args(custom_args)
    
    # Convert paths
    args.data_root = Path(args.data_root)
    if args.stats_file:
        args.stats_file = Path(args.stats_file)
    
    return args


# =============================================================================
# Main Entry Points
# =============================================================================

def run_training(args: argparse.Namespace) -> None:
    """Main training entry point."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup experiment directory
    if args.continue_from is not None:
        exp_dir = Path(args.continue_from)
        config_manager = ConfigManager(exp_dir)
        if config_manager.exists():
            # Preserve args we want to override from command line
            new_max_iter = args.max_iter
            
            saved_args = config_manager.load()
            saved_args.continue_from = str(exp_dir)
            args = saved_args
            
            # Override with new values
            args.max_iter = new_max_iter
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        exp_dir = Path('experiments/SE') / f'{args.exp_name}-{timestamp}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_manager = ConfigManager(exp_dir)
    
    # Setup logging
    log_dir = exp_dir / 'logs'
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_dir, level=log_level)
    
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=exp_dir / 'checkpoints',
        metric_name="val_loss",
        mode="min",
        logger=logger
    )
    
    # Build or load model
    start_iter = 0
    model = get_style_encoder(args).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Try to resume from checkpoint
    if args.continue_from is not None:
        checkpoint = checkpoint_manager.load_latest(device)
        if checkpoint is not None:
            model.encoder.load_state_dict(checkpoint['encoder'])
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    logger.info("Loaded optimizer state")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            start_iter = checkpoint.get('iteration', 0)
            logger.info(f"Resuming from iteration {start_iter}")
    
    # Create datasets
    logger.info(f"Loading dataset: {args.dataset_type}")
    stats_file = args.stats_file if args.use_normalization else None
    
    train_dataset, val_dataset, train_loader, val_loader = create_datasets(
        dataset_type=args.dataset_type,
        data_root=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_se=True,
        no_head_pose=args.no_head_pose,
        fps=args.fps,
        n_motions=args.n_motions,
        rot_repr=args.rot_repr,
        stats_file=str(stats_file) if stats_file else None,
        batch_overfit_size=args.batch_overfit_size if hasattr(args, 'batch_overfit_size') else -1,
    )
    

    # ============================================================================
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
    # ============================================================================


    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Save config
    config_manager.save(args)
    
    # Setup TensorBoard
    writer = SummaryWriter(str(log_dir))
    
    # Log experiment info
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    writer.add_text('config', json.dumps(vars(args), indent=2, default=str))
    
    # Train
    train(
        args, model, train_loader, val_loader, optimizer,
        checkpoint_manager, writer, start_iter=start_iter, logger=logger
    )
    
    writer.close()
    logger.info("Training finished!")


def run_testing(args: argparse.Namespace) -> None:
    """Main testing entry point."""
    # Setup logging (console only for testing)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(level=log_level)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load model
    checkpoint_path, _ = get_model_path(args.exp_name, args.iter, 'SE')
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = get_style_encoder(args).to(device)
    model.encoder.load_state_dict(model_data['encoder'], strict=False)
    model.eval()
    
    # Create test dataset
    _, _, _, test_loader = create_datasets(
        dataset_type=args.dataset_type,
        data_root=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_se=True,
        no_head_pose=args.no_head_pose,
        fps=args.fps,
        n_motions=args.n_motions,
        rot_repr=args.rot_repr,
        batch_overfit_size=args.batch_overfit_size if hasattr(args, 'overfit_size') else -1,
    )
    
    # Run test
    validate(args, model, test_loader, args.iter, n_rounds=100, mode='test', logger=logger)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        run_training(args)
    else:
        run_testing(args)


if __name__ == '__main__':
    main()