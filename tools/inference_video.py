from rich import print
from rich.progress import Progress

from engine.builder import build_model, build_inferencer
from engine.category import Category
import torch

import cv2

from engine import transform
from engine.dataloader import ImgAnnDataset
from engine.inferencer import Inferencer
from engine.visualizer import IdMapVisualizer, ImgSaver
from pathlib import Path

import argparse


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
    parser.add_argument(
        "video_path", type=str, help="Directory of the input video."
    )
    parser.add_argument(
        "save_dir", type=str, help="The directory to save the predictions. The format must be mp4."
    )
    parser.add_argument(
        "checkpoint", type=str, help="The path of checkpoint for the model."
    )
    parser.add_argument("height", type=int, help="The height of the prediction.")
    parser.add_argument("width", type=int, help="The height of the prediction.")
    parser.add_argument("framerate", type=int, help="The framerate of the output video.")

    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Whether to use sliding window inference from orignal SegFormer",
    )

    args = parser.parse_args()
    return args


def inference_video(
    image_scale: tuple[int, int],
    categories: list[Category],
    inferencer: Inferencer,
    model: torch.nn.Module,
    video_dir: str,
    save_dir: str,
    frame_rate: float = 30,
    device: str = "cuda",
):
    transforms = transform.Composition(
        [
            transform.NDArrayImgToTensor(),
            transform.Resize(image_scale),
            transform.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    capture = cv2.VideoCapture(video_dir)
    writer = cv2.VideoWriter(
        save_dir,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        tuple(reversed(image_scale)),
    )

    visualizer = IdMapVisualizer(categories)

    print("Inferencing, please wait...")
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        data = transforms.transform({"img": frame})
        img = data["img"].to(device)[None, :]
        pred = inferencer.inference(model, img)
        visualization = visualizer.visualize(pred.argmax(1).cpu())
        writer.write(visualization)

    capture.release()
    writer.release()


def main(args):
    categories = Category.load(args.csv)
    device = "cuda"

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
        inference_video(
            (args.height, args.width),
            categories,
            inferencer,
            model,
            args.video_path,
            args.save_dir,
            args.framerate,
            device,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
