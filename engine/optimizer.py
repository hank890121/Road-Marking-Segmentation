from abc import ABC, abstractclassmethod
from typing import Any, Optional, Protocol, Tuple

import torch
from torch.cuda.amp import GradScaler


class Optimizer(ABC):
    """An object to optimize/update models' parameters."""

    @abstractclassmethod
    def get_lr(self) -> list[tuple[str, float]]:
        """Get current learning rate."""

    @abstractclassmethod
    def zero_grad(self) -> None:
        """Reset current backpropagated gradient."""

    @abstractclassmethod
    def step(self) -> None:
        """Optimize the registered target parameters."""

    @abstractclassmethod
    def torch(self) -> torch.optim.Optimizer:
        """Convert to pytorch format."""


def get_torch_lr(optimizer: torch.optim.Optimizer) -> list[tuple[str, float]]:
    return [(param["name"], param["lr"]) for param in optimizer.param_groups]


class AdamW(Optimizer):
    """AdamW optimizer provided by Pytorch."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.optimizer = torch.optim.AdamW(params=params)

    def get_lr(self) -> list[tuple[str, float]]:
        return get_torch_lr(self.optimizer)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, scaler: Optional[GradScaler] = None) -> None:
        if scaler:
            scaler.step(self.torch())
            scaler.update()
        else:
            self.optimizer.step()

    def torch(self) -> torch.optim.Optimizer:
        return self.optimizer


class SGD(Optimizer):
    """AdamW optimizer provided by Pytorch."""

    def __init__(self, params: dict[str, Any], lr: float, weight_decay: float) -> None:
        self.optimizer = torch.optim.SGD(
            params=params, lr=lr, weight_decay=weight_decay
        )

    def get_lr(self) -> list[tuple[str, float]]:
        return get_torch_lr(self.optimizer)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, scaler=None) -> None:
        if scaler:
            scaler.step(self.torch())
            scaler.update()
        else:
            self.optimizer.step()

    def torch(self) -> torch.optim.Optimizer:
        return self.optimizer
