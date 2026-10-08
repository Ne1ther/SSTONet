# -*- coding: utf-8 -*-
"""Training callbacks for model training.

Provides callbacks for early stopping and model checkpointing.
"""

import os
import torch
import torch.nn as nn
from typing import Optional


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        restore_best: Whether to restore best model weights
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to save state from

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        filepath: Path template for checkpoint files (e.g., 'model_epoch{epoch}.pt')
        save_best_only: Only save when validation loss improves
        monitor: Metric to monitor ('val_loss' or 'train_loss')
    """

    def __init__(
        self,
        filepath: str,
        save_best_only: bool = True,
        monitor: str = 'val_loss'
    ):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_loss = float('inf')

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss: float,
        val_loss: Optional[float] = None,
        **kwargs
    ) -> None:
        """Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            train_loss: Training loss
            val_loss: Validation loss (if available)
            **kwargs: Additional items to save
        """
        loss = val_loss if self.monitor == 'val_loss' and val_loss is not None else train_loss

        if self.save_best_only:
            if loss < self.best_loss:
                self.best_loss = loss
                self._save(epoch, model, optimizer, train_loss, val_loss, **kwargs)
        else:
            self._save(epoch, model, optimizer, train_loss, val_loss, **kwargs)

    def _save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss: float,
        val_loss: Optional[float],
        **kwargs
    ) -> None:
        """Save checkpoint to file."""
        filepath = self.filepath.format(epoch=epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            **kwargs
        }
        torch.save(checkpoint, filepath)
