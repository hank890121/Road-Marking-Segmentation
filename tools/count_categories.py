import argparse
import json

import torch
from rich import print
from rich.progress import track
from rich.table import Table

from engine.category import Category, count_categories
from engine.dataloader import ImgAnnDataset
from engine.transform import LoadAnn, Resize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count pixels per category in a dataset."
    )
    parser.add_argument("csv", type=str, help="The csv file for categories")
    parser.add_argument("label_directory", type=str, help="Directory containing the P-mode png labels.")
    parser.add_argument(
        "--rcs-file-savepath",
        type=str,
        default=None,
        help="If specified, a category statistics file used for class-balanced sampling will be saved.",
    )
    args = parser.parse_args()
    return args


def count_dataset_categories(dataloader, categories, rcs: list | None = None):
    counts = torch.zeros(len(categories)).int()
    for data in track(dataloader, description="Counting categories..."):
        label = data["ann"]
        count = count_categories(label, categories)
        counts += count

        if rcs is not None:
            rcs.append({"filename": data["ann_path"][0], "count": count.tolist()})
    return counts.tolist()


def print_counts_table(categories, counts):
    table = Table()
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Abbr.")
    table.add_column("count", style="blue", justify="right")
    for cat, count in zip(categories, counts):
        table.add_row(str(cat.id), cat.name, cat.abbr, str(count))
    print(table)


def main(args):
    categories = Category.load(args.csv, False)

    dataloader = ImgAnnDataset(
        args.label_directory,
        img_prefix="",
        ann_prefix="",
        img_suffix="",
        ann_suffix=".png",
        transforms=[
            LoadAnn(),
            Resize(),
        ],
        check_exist=False,
    ).get_loader(1, False, 0, False, False)

    if args.rcs_file_savepath:
        rcs = []
        counts = count_dataset_categories(dataloader, categories, rcs)
        with open(args.rcs_file_savepath, "w") as f:
            json.dump(rcs, f)
    else:
        counts = count_dataset_categories(dataloader, categories)

    print_counts_table(categories, counts)


if __name__ == "__main__":
    args = parse_args()
    main(args)
