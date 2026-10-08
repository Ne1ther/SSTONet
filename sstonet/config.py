# -*- coding: utf-8 -*-
"""Global configuration for SSTONet"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DataConfig:
    """Data configuration"""
    root: Path = Path("data/nx_exports")
    result_subdir: str = "."
    n_samples: int = 2000

    @property
    def result_dir(self) -> Path:
        return self.root / self.result_subdir


@dataclass
class ModelConfig:
    """Model configuration"""
    hidden_dim: int = 128
    output_dim: int = 64
    trunk_input_dim: int = 3  # x, y, z coordinates


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 500
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return cls.from_dict(cfg_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        config = cls()
        if 'data' in d:
            for k, v in d['data'].items():
                if k == 'root':
                    v = Path(v)
                setattr(config.data, k, v)
        if 'model' in d:
            for k, v in d['model'].items():
                setattr(config.model, k, v)
        if 'training' in d:
            for k, v in d['training'].items():
                setattr(config.training, k, v)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'data': {
                'root': str(self.data.root),
                'result_subdir': self.data.result_subdir,
                'n_samples': self.data.n_samples,
            },
            'model': {
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'trunk_input_dim': self.model.trunk_input_dim,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'epochs': self.training.epochs,
                'val_split': self.training.val_split,
                'seed': self.training.seed,
                'device': self.training.device,
            }
        }


# Global default config
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration"""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set global configuration"""
    global _default_config
    _default_config = config


def load_config(path: str) -> Config:
    """Load and set global configuration from YAML file"""
    config = Config.from_yaml(path)
    set_config(config)
    return config
