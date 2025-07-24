import builtins
import torch
from dataclasses import dataclass
from typing import Protocol, Optional
from abc import ABC, abstractmethod

from torch.nn.modules import Module
from rich import print as rprint


class EntropyLossComputer(Protocol):

    def compute(self, x: torch.Tensor) -> torch.Tensor: ...


class BasicEntropy:
    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        probs = x.softmax(dim=1)
        log_probs = torch.log(probs + self.eps)
        return -(probs * log_probs).sum(dim=1)


class MaximumSquaredLoss:
    def __init__(self, ignored_index: int = 255) -> None:
        self.ignored_index = ignored_index

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        probs = x.softmax(dim=1)
        return -probs[probs != self.ignored_index].sqrt().mean() / 2


class SoftLossComputer(Protocol):

    @abstractmethod
    def compute(
        self,
        pred: torch.Tensor,
        soft_ann: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> torch.Tensor: ...


class NoThreshold:
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, pred: torch.Tensor, soft_ann: torch.Tensor, criterion: Module
    ) -> torch.Tensor:
        return criterion(pred, soft_ann.argmax(1)).mean()


class GlobalThreshold:

    def __init__(self, threshold: float | list[float] = 0.968) -> None:
        assert isinstance(threshold, float) or isinstance(threshold, list)
        self.threshold = threshold

    def compute(
        self,
        pred: torch.Tensor,
        soft_ann: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        prob, ann = soft_ann.max(1)

        match type(self.threshold):
            case builtins.float:
                ge = prob >= self.threshold
            case builtins.list:
                threshold = torch.zeros(prob.shape).cuda()
                for i, th in enumerate(self.threshold):
                    threshold[ann == i] = th
                ge = prob >= threshold

        weight = len(torch.where(ge)[0]) / len(prob.flatten())
        return criterion(pred, ann).mean() * weight


class PixelThreshold:
    def __init__(self, threshold: float | list[float] = 0.968) -> None:
        assert isinstance(threshold, float) or isinstance(threshold, list)
        self.threshold = threshold

    def compute(
        self,
        pred: torch.Tensor,
        soft_ann: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        prob, ann = soft_ann.max(1)
        loss = criterion(pred, ann)

        match type(self.threshold):
            case builtins.float:
                ge = prob >= self.threshold
            case builtins.list:
                threshold = torch.zeros(prob.shape).cuda()
                for i, th in enumerate(self.threshold):
                    threshold[ann == i] = th
                ge = prob >= threshold

        if ge.any():
            return loss[ge].mean()
        return torch.zeros(1, requires_grad=True).cuda()


class EMAModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, alpha: float = 0.999) -> None:
        super().__init__()
        self.model = model
        self.alpha = alpha

    @torch.no_grad()
    def update(self, iteration: int, model: torch.nn.Module) -> None:
        alpha = min(1 - 1 / (iteration - 1), self.alpha) if iteration > 1 else 0
        for eparam, param in zip(self.model.parameters(), model.parameters()):
            if not eparam.shape:
                eparam.data = alpha * eparam.data + (1 - alpha) * param.data
            else:
                eparam.data[:] = (
                    alpha * eparam[:].data[:] + (1 - alpha) * param[:].data[:]
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# if __name__ == "__main__":
#     from engine.dataloader import ImgAnnDataset
#     from engine.transform import LoadImg, Normalize
#     from engine.builder import build_model, build_loss_function
#     from engine.category import Category
#     from rich import print

#     dataset_name = "vpgnet"

#     path_dict = {
#         "vpgnet": {
#             "train_data_root": "data/vpgnet/clear/train",
#             "target_train_data_root": "data/vpgnet/night/train",
#             "val_data_root": "data/vpgnet/clear/val",
#             "target_val_data_root": "data/vpgnet/night/val",
#             "img_prefix": "images",
#             "ann_prefix": "labels",
#             "img_suffix": ".png",
#             "ann_suffix": ".png",
#         }
#     }

#     dataloader = ImgAnnDataset(
#         root=path_dict[dataset_name]["train_data_root"],
#         img_prefix=path_dict[dataset_name]["img_prefix"],
#         ann_prefix=path_dict[dataset_name]["ann_prefix"],
#         img_suffix=path_dict[dataset_name]["img_suffix"],
#         ann_suffix=path_dict[dataset_name]["ann_suffix"],
#         transforms=[
#             LoadImg(),
#             Normalize(),
#         ],
#         max_len=10,
#     ).get_loader(
#         batch_size=4,
#         shuffle=True,
#         num_workers=4,
#         drop_last=True,
#         pin_memory=False,
#         infinite=False,
#     )

#     categories = Category.load("data/csv/vpgnet.csv")

#     loss_function = build_loss_function(
#         {"name": "focal_loss", "gamma": 2, "normalized": False}
#     )
#     model = build_model(
#         {
#             "name": "segformer",
#             "pretrained": "nvidia/mit-b0",
#             "num_classes": len(categories),
#         }
#     ).cuda()
#     ema = build_model(
#         {
#             "name": "segformer",
#             "pretrained": "nvidia/mit-b0",
#             "num_classes": len(categories),
#         }
#     ).cuda()
#     copy_parameters(model, ema)

#     soft_loss_computing = GlobalThresholdLoss()
#     # soft_loss_computing = LocalThresholdLoss()

#     for data in dataloader:
#         img = data["img"].cuda()

#         pred = model(img)
#         loss = compute_pseudo_loss(
#             pred,
#             ema,
#             img,
#             loss_function,
#             # soft_loss_computing,
#         )
#         print(loss)
#         loss.backward()
