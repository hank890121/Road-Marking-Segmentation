import torch
from rich import print
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader
from simple_parsing import Serializable
from dataclasses import dataclass
import os

from engine import builder, transform
from engine.category import Category
from engine.dataloader import ImgAnnDataset
from engine.inferencer import Inferencer
from engine.metric import Metrics
from engine.misc import set_seed
import numpy as np

from engine.validator import Validator


@dataclass
class TestingConfig(Serializable):
    category_csv: str
    datasets: list[tuple[str, str]]

    logdir: str

    batch_size: int
    image_scale: tuple[int, int]
    crop_size: tuple[int, int] | None
    stride: tuple[int, int] | None

    seed: int
    num_workers: int
    pin_memory: bool


def main(cfg: TestingConfig, exp_name: str, checkpoint: str):
    set_seed(cfg.seed)

    categories = Category.load(cfg.category_csv)

    model = builder.build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            "num_classes": len(categories),
        }
    ).cuda()

    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]

    dataloaders = []
    for dataset_name, dataset_root in cfg.datasets:
        dataloader = ImgAnnDataset(
            root=dataset_root,
            transforms=val_transforms,
            img_prefix="images",
            ann_prefix="labels",
            img_suffix=".jpg",
            ann_suffix=".png",
        ).get_loader(
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=cfg.pin_memory,
        )
        dataloaders.append((dataset_name, dataloader))

    logdir = os.path.join(cfg.logdir, exp_name)
    metrics = Metrics(num_categories=len(categories), nan_to_num=0)

    if cfg.crop_size is not None and cfg.stride is not None:
        assert cfg.image_scale[0] > cfg.crop_size[0]
        assert cfg.image_scale[1] > cfg.crop_size[1]
        inferencer = builder.build_inferencer(
            {
                "mode": "slide",
                "crop_size": cfg.crop_size,
                "stride": cfg.stride,
                "num_categories": len(categories),
            }
        )
    else:
        inferencer = builder.build_inferencer({"mode": "basic"})

    model.load_state_dict(torch.load(os.path.join(logdir, checkpoint))["model"])

    validator = Validator(
        dataloaders,
        model,
        torch.nn.CrossEntropyLoss(),
        inferencer,
        metrics,
        categories,
    )

    torch.compile(model)

    with Progress() as progress:
        val_outputs = validator.validate(progress)


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 4
    cfg = TestingConfig.load(sys.argv[1])
    exp_name = sys.argv[2]
    checkpoint = sys.argv[3]
    main(cfg, exp_name, checkpoint)
