from abc import ABC, abstractclassmethod
from typing import Any, Optional, Protocol, Tuple

import torch

from engine.optimizer import Optimizer


class Scheduler(ABC):
    """An object to schedule optimizers."""

    begin: int

    @abstractclassmethod
    def step(self) -> None:
        """Schedule the registered optimizers."""


class SchedulerGroup:
    def __init__(self, schedulers: list[Scheduler] = None) -> None:
        self.schedulers = schedulers or []

    def add_scheduler(self, scheduler: Scheduler) -> None:
        self.schedulers.append(scheduler)

    def remove_scheduler(self, scheduler: Scheduler) -> None:
        self.schedulers.remove(scheduler)

    def step(self, epochs: int) -> None:
        for sched in self.schedulers:
            if epochs >= sched.begin:
                sched.step()


class WarmUp(Scheduler):
    def __init__(
        self, optimizer: Optimizer, start_lr: float, max_epochs: int, begin: int = 0
    ) -> None:
        super().__init__()
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer.torch(),
            start_lr,
            total_iters=max_epochs,
        )
        self.begin = begin

    def step(self) -> None:
        self.scheduler.step()


class Polynomial(Scheduler):

    def __init__(
        self, optimizer: Optimizer, max_epochs: int, power: float, begin: int = 0
    ) -> None:
        super().__init__()
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer.torch(),
            max_epochs,
            power,
        )
        self.begin = begin

    def step(self) -> None:
        self.scheduler.step()


class ReduceOnPlateau(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str,
        factor: float,
        patience: int,
        metrics: dict[str, Any],
        key: str,
    ) -> None:
        self.metrics = metrics
        self.key = key
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.torch(), mode, factor, patience
        )

    def step(self) -> None:
        self.scheduler.step(self.metrics[self.key])
