# -*- coding: utf-8 -*-
"""Tests for trainer basic functionality."""

import pytest
import torch
import numpy as np
from sstonet.models import DeepONet, MIONet
from sstonet.training import DeepONetTrainer, MIONetTrainer
from sstonet.callbacks import EarlyStopping, ModelCheckpoint


def test_deeponet_trainer_creation():
    """Test DeepONetTrainer creation."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[32],
        trunk_hidden=[32],
        output_dim=16
    )
    trainer = DeepONetTrainer(model, lr=1e-3, device="cpu")
    assert trainer.model is not None
    assert trainer.optimizer is not None


def test_deeponet_trainer_fit():
    """Test DeepONetTrainer fit method."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )
    trainer = DeepONetTrainer(model, lr=1e-3, device="cpu")

    # Small synthetic dataset
    X_train = np.random.randn(20, 10)
    coords = np.random.randn(30, 3)
    T_train = np.random.randn(20, 30)

    history = trainer.fit(
        X_train, coords, T_train,
        epochs=5,
        batch_size=4,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 5


def test_deeponet_trainer_predict():
    """Test DeepONetTrainer predict method."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )
    trainer = DeepONetTrainer(model, lr=1e-3, device="cpu")

    # Train on small dataset
    X_train = np.random.randn(20, 10)
    coords = np.random.randn(30, 3)
    T_train = np.random.randn(20, 30)

    trainer.fit(X_train, coords, T_train, epochs=2, verbose=False)

    # Predict
    X_test = np.random.randn(5, 10)
    T_pred = trainer.predict(X_test, coords)

    assert T_pred.shape == (5, 30)


def test_mionet_trainer_creation():
    """Test MIONetTrainer creation."""
    input_dims = {'input1': 10, 'input2': 20}
    model = MIONet(
        input_dims=input_dims,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )
    trainer = MIONetTrainer(model, lr=1e-3, input_dims=input_dims)
    assert trainer.model is not None
    assert trainer.input_dims == input_dims


def test_mionet_trainer_fit():
    """Test MIONetTrainer fit method."""
    input_dims = {'input1': 10, 'input2': 20}
    model = MIONet(
        input_dims=input_dims,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )
    trainer = MIONetTrainer(model, lr=1e-3, input_dims=input_dims)

    # Small synthetic dataset
    X_train = np.random.randn(20, 30)  # 10 + 20
    coords = np.random.randn(30, 3)
    T_train = np.random.randn(20, 30)

    history = trainer.fit(
        X_train, coords, T_train,
        epochs=5,
        batch_size=4,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 5


def test_early_stopping():
    """Test EarlyStopping callback."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )

    early_stop = EarlyStopping(patience=3, min_delta=0.01)

    # Simulate improving loss
    assert not early_stop(1.0, model)
    assert not early_stop(0.9, model)
    assert not early_stop(0.8, model)

    # Simulate stagnating loss
    assert not early_stop(0.81, model)
    assert not early_stop(0.82, model)
    assert early_stop(0.83, model)  # Should trigger


def test_model_checkpoint(tmp_path):
    """Test ModelCheckpoint callback."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8
    )
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_path = tmp_path / "model_epoch{epoch}.pt"
    checkpoint = ModelCheckpoint(str(checkpoint_path), save_best_only=True)

    # Simulate training
    checkpoint(epoch=1, model=model, optimizer=optimizer, train_loss=1.0, val_loss=1.0)
    checkpoint(epoch=2, model=model, optimizer=optimizer, train_loss=0.8, val_loss=0.8)

    # Check that best model was saved
    assert (tmp_path / "model_epoch2.pt").exists()
