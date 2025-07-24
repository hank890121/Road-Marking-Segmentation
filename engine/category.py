from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Union

import torch
from rich import print as rprint
from rich.table import Table


@dataclass
class Category:
    """A class represent a semantic category."""

    id: int
    name: str
    abbr: str
    r: int
    g: int
    b: int

    def count(self, id_map: torch.Tensor) -> int:
        """Counts how many pixels in a id map belong to this category by its id.

        Args:
            id_map (torch.Tensor): A HxW or NxHxW map that consists of category ids.

        Returns:
            int: Number of pixels belongs to this category.
        """

        assert (
            len(id_map.shape) == 2 or len(id_map.shape) == 3
        ), "Shape of a category map should be HxW or NxHxW."
        return torch.where(id_map == self.id)[0].shape[0]

    @staticmethod
    def load(csv_path: str, show: bool = True) -> list[Category]:
        """Load a category definition csv.

        Args:
            csv_path (str): A path to a category definition csv.
            show (bool, optional): Print the category table after loaded. Defaults to True.

        Returns:
            list[Category]: A list of categories, sorted by category id.
        """
        with open(csv_path, "r", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            _ = next(reader)  # headers are not needed.
            cats = [
                Category(id, name, abbr, int(r), int(g), int(b))
                for id, (name, abbr, r, g, b) in enumerate(csv.reader(file))
            ]
            cats = sorted(cats, key=lambda x: x.id)
        if show:
            Category.print(cats)
        return cats

    @staticmethod
    def print(categories: Union[Category, list[Category]]) -> None:
        """Print out the category table given a category of a list of categories.

        Args:
            categories (Union[Category, list[Category]]): One category or a list of categories.
        """

        if not isinstance(categories, Category) and not isinstance(categories, list):
            print("Input argument should be a category or a list of categories!")
            return

        if isinstance(categories, Category):
            categories = [categories]

        table = Table()
        table.add_column("ID", justify="right")
        table.add_column("Name")
        table.add_column("Abbr.")
        table.add_column("R", justify="right", style="red")
        table.add_column("G", justify="right", style="green")
        table.add_column("B", justify="right", style="cyan")
        for cat in categories:
            table.add_row(
                str(cat.id), cat.name, cat.abbr, str(cat.r), str(cat.g), str(cat.b)
            )
        rprint(table)


def count_categories(id_map: torch.Tensor, categories: list[Category]) -> torch.Tensor:
    """Counts every category in one id map given a list of categories.

    Args:
        id_map (torch.Tensor): A HxW or NxHxW map that consists of category ids
        categories (list[Category]): A list of categories.

    Returns:
        torch.Tensor: A list of numbers of pixels belong to corresponding categories.
    """
    return torch.Tensor([cat.count(id_map) for cat in categories]).int()


if __name__ == "__main__":
    cats = Category.load("./data/csv/ceymo.csv")
