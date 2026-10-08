# -*- coding: utf-8 -*-
"""Tests for MLP and MLPTrainer."""

import pytest
import torch
import numpy as np
from sstonet.models import MLP, MLPTrainer


def test_mlp_creation():
    """Test MLP model creation."""
    model = MLP(
        input_dim=10,
        output_dim=100,
        hidden_dims=[64, 32],
    )
    assert model.input_dim == 10
    assert model.output_dim == 100
    assert model.count_parameters() > 0


def test_mlp_creation_with_bottleneck():
    """Test MLP model creation with bottleneck."""
    model = MLP(
        input_dim=10,
        output_dim=100,
        hidden_dims=[64, 32],
        bottleneck_dim=16,
    )
    assert model.bottleneck_dim == 16


def test_mlp_forward():
    """Test MLP forward pass."""
    model = MLP(
        input_dim=10,
        output_dim=50,
        hidden_dims=[32, 16],
    )
    x = torch.randn(4, 10)
    output = model(x)
    assert output.shape == (4, 50)


def test_mlp_trainer_creation():
    """Test MLPTrainer creation."""
    model = MLP(input_dim=10, output_dim=50, hidden_dims=[32])
    trainer = MLPTrainer(model, lr=1e-3, device="cpu")
    assert trainer.model is not None
    assert trainer.optimizer is not None


def test_mlp_trainer_fit():
    """Test MLPTrainer fit method."""
    model = MLP(input_dim=10, output_dim=30, hidden_dims=[16])
    trainer = MLPTrainer(model, lr=1e-3, device='cpu')

    X_train = np.random.randn(20, 10)
    T_train = np.random.randn(20, 30)

    history = trainer.fit(
        X_train, T_train,
        epochs=3,
        batch_size=4,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 3


def test_mlp_trainer_predict():
    """Test MLPTrainer predict method."""
    model = MLP(input_dim=10, output_dim=30, hidden_dims=[16])
    trainer = MLPTrainer(model, lr=1e-3, device='cpu')

    X_train = np.random.randn(20, 10)
    T_train = np.random.randn(20, 30)

    trainer.fit(X_train, T_train, epochs=2, verbose=False)

    X_test = np.random.randn(5, 10)
    T_pred = trainer.predict(X_test)

    assert T_pred.shape == (5, 30)


def test_mlp_trainer_evaluate():
    """Test MLPTrainer evaluate method."""
    model = MLP(input_dim=10, output_dim=30, hidden_dims=[16])
    trainer = MLPTrainer(model, lr=1e-3, device='cpu')

    X_train = np.random.randn(20, 10)
    T_train = np.random.randn(20, 30)

    trainer.fit(X_train, T_train, epochs=2, verbose=False)

    X_test = np.random.randn(5, 10)
    T_test = np.random.randn(5, 30)
    metrics = trainer.evaluate(X_test, T_test)

    assert 'RMSE' in metrics
    assert 'MAE' in metrics
