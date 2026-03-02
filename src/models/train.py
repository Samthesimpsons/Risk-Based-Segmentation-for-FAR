"""Shared training utilities for PyTorch-based recommendation models."""

import random
from collections.abc import Callable

import numpy as np
import torch


def set_random_seeds(seed: int) -> None:
    """Set seeds for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pytorch_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loss_function: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    number_of_epochs: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Generic PyTorch training loop. Returns per-epoch average losses."""
    model.to(device)
    model.train()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    epoch_losses: list[float] = []

    for _ in range(number_of_epochs):
        total_loss = 0.0
        batch_count = 0

        for batch in data_loader:
            optimizer.zero_grad()
            loss = loss_function(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count if batch_count > 0 else 0.0
        epoch_losses.append(average_loss)

    return epoch_losses
