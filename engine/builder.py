from typing import Any
from engine.ema import (
    BasicEntropy,
    MaximumSquaredLoss,
    EMAModel,
    EntropyLossComputer,
    GlobalThreshold,
    NoThreshold,
    PixelThreshold,
    SoftLossComputer,
)

from engine.inferencer import BasicInferencer, Inferencer, SlideInferencer
from engine.losses.diceloss import DiceLoss
from rich import print as rprint
from rich import print_json
from segmentation_models_pytorch.losses import FocalLoss
from torch.nn import CrossEntropyLoss, Module, BCEWithLogitsLoss
from engine.models.segformer import Segformer


def build_criterion(cfg: dict[str, Any], show: bool = True) -> Module:
    if show:
        rprint(cfg)

    assert cfg.get("name") in [
        "cross_entropy_loss",
        "focal_loss",
        "dice_loss",
        "bce_with_logits_loss",
    ]

    match cfg.get("name"):
        case "cross_entropy_loss":
            loss_function = CrossEntropyLoss(
                weight=cfg.get("weight"),
                ignore_index=cfg.get("ignore_index"),
                reduction=cfg.get("reduction"),
                label_smoothing=cfg.get("label_smoothing"),
            )

        case "focal_loss":
            loss_function = FocalLoss(
                mode="multiclass",
                alpha=cfg.get("alpha"),
                gamma=cfg.get("gamma"),
                ignore_index=cfg.get("ignore_index"),
                normalized=cfg.get("normalized"),
                reduction="none",
            )
        case "dice_loss":
            loss_function = DiceLoss()
        case "bce_with_logits_loss":
            loss_function = BCEWithLogitsLoss(
                reduction=cfg.get("reduction"),
            )

    return loss_function.to(cfg.get("device"))


def build_inferencer(cfg: dict[str, Any], show: bool = True) -> Inferencer:
    if show:
        rprint(cfg)

    assert cfg.get("mode") in ["basic", "slide"]

    match cfg.get("mode"):
        case "basic":
            inferencer = BasicInferencer()
        case "slide":
            inferencer = SlideInferencer(
                crop_size=cfg["crop_size"],
                stride=cfg["stride"],
                num_categories=cfg["num_categories"],
            )

    return inferencer


def build_model(cfg: dict[str, Any], show: bool = True) -> Module:
    if show:
        rprint(cfg)

    assert cfg.get("name") in ["segformer"]

    match cfg.get("name"):
        case "segformer":
            model = Segformer.from_pretrained(
                pretrained_model_name_or_path=cfg.get("pretrained"),
                num_labels=cfg.get("num_classes"),
            )

    return model


def build_soft_loss_computer(
    cfg: dict[str, Any], show: bool = True
) -> SoftLossComputer:
    if show:
        rprint(cfg)

    match cfg.get("name"):
        case "NoThreshold":
            soft_loss_computer = NoThreshold()
        case "GlobalThreshold":
            soft_loss_computer = GlobalThreshold(threshold=cfg.get("threshold"))
        case "PixelThreshold":
            soft_loss_computer = PixelThreshold(threshold=cfg.get("threshold"))
        case _:
            raise NotImplementedError()

    return soft_loss_computer


def build_entropy_loss_computer(
    cfg: dict[str, Any], show: bool = True
) -> EntropyLossComputer:
    if show:
        rprint(cfg)

    match cfg.get("name"):
        case "BasicEntropy":
            entropy_computer = BasicEntropy()
        case "MaximumSquaredLoss":
            entropy_computer = MaximumSquaredLoss()
        case _:
            raise NotImplementedError()

    return entropy_computer
