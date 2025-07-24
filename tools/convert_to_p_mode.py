import argparse
import os
import os.path as osp

import cv2
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from PIL import Image

from engine.category import Category


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the RGB-mode png labels to P-mode.")
    parser.add_argument(
        "csv", help="The csv file for categories"
    )
    parser.add_argument("source_dir", help="The directory containing the RGB-mode png labels.")
    parser.add_argument("target_dir", help="The directory to save the P-mode png labels.")
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Number of processes to use for parallel conversion (default: 8), enter 0 to disable.",
    )

    return parser.parse_args()


def convert(
    categories: list[Category],
    source_dir: str,
    target_dir: str,
    filename: str,
) -> None:
    image = cv2.imread(osp.join(source_dir, filename))
    output = np.zeros(image.shape[:2], dtype=np.uint8)
    for cat in categories:
        class_mask = np.equal(image, (cat.b, cat.g, cat.r))
        output[np.all(class_mask, axis=-1)] = cat.id
    output = Image.fromarray(output).convert("P")
    output.putpalette(
        np.array([[cat.r, cat.g, cat.b] for cat in categories], dtype=np.uint8)
    )
    output.save(osp.join(target_dir, filename), bitmap_format=".png")


def main():
    args = parse_args()

    assert not osp.exists(args.target_dir), "Target folder already exist!"

    categories = Category.load(args.csv)
    filenames = os.listdir(args.source_dir)
    assert len(filenames) > 0, "No label detected."

    os.mkdir(args.target_dir)

    with joblib_progress("[Converting]", total=len(filenames)):
        Parallel(args.nproc)(
            delayed(convert)(categories, args.source_dir, args.target_dir, fn)
            for fn in filenames
        )


main()
