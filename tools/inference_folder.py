from rich.progress import Progress

from engine.builder import build_model, build_inferencer
from engine.category import Category
import torch
import os

from engine import transform
from engine.dataloader import ImgAnnDataset
from engine.inferencer import Inferencer
from engine.visualizer import IdMapVisualizer, ImgSaver
from pathlib import Path

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference a folder of images.")
    parser.add_argument("csv", type=str, help="The csv file for categories")
    parser.add_argument("image_directory", type=str, help="Directory containing the input images.")
    parser.add_argument("suffix", type=str, help="Suffix of the input images.")
    parser.add_argument("save_dir", type=str, help="The directory to save the predictions.")
    parser.add_argument("checkpoint", type=str, help="The path of checkpoint for the model.")
    parser.add_argument("batch_size", type=int, help="The batch size for inference.")
    parser.add_argument("height", type=int, help="The height of the prediction.")
    parser.add_argument("width", type=int, help="The height of the prediction.")

    parser.add_argument("--sliding-window", action="store_true", help="Whether to use sliding window inference from orignal SegFormer")

    args = parser.parse_args()
    return args


def inference_folder(
    image_scale: tuple[int, int],
    categories: list[Category],
    inferencer: Inferencer,
    model: torch.nn.Module,
    folder_dir: str,
    save_dir: str,
    batch_size: int,
    suffix: str = ".jpg",
    device: str = "cuda",
    num_workers: int = 16,
):
    dataloader = ImgAnnDataset(
        root=folder_dir,
        transforms=[
            transform.LoadImg(),
            transform.Resize(image_scale),
            transform.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        img_prefix="",
        ann_prefix="",
        img_suffix=suffix,
        ann_suffix="",
        check_exist=False,
    ).get_loader(batch_size, num_workers=num_workers)

    img_saver = ImgSaver(save_dir, IdMapVisualizer(categories))

    with Progress() as progress:
        task = progress.add_task(f"Inference", total=len(dataloader))

        for data in dataloader:
            img = data["img"].to(device)
            pred = inferencer.inference(model, img)
            for p, path in zip(pred, data["img_path"]):
                img_saver.save_pred(p[None, :], f"{Path(path).stem}.png")
            progress.advance(task, 1)

        progress.remove_task(task)


def main(args):
    categories = Category.load(args.csv)
    device = "cuda"

    # assert not os.path.exists(args.save_dir), "Ouput directory already exists!"

    model = build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            "num_classes": len(categories),
        }
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.eval()

    if args.sliding_window:
        assert args.height > 512
        assert args.width > 512
        inferencer = build_inferencer(
            {
                "mode": "slide",
                "crop_size": (512, 512),
                "stride": (384, 384),
                "num_categories": len(categories),
            }
        )
    else:
        inferencer = build_inferencer({"mode": "basic"})

    with torch.inference_mode(), torch.cuda.amp.autocast():
        inference_folder(
            (args.height, args.width),
            categories,
            inferencer,
            model,
            args.image_directory,
            args.save_dir,
            args.batch_size,
            args.suffix,
            device,
            num_workers=16,  # Adjust according to your system's resources.
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
