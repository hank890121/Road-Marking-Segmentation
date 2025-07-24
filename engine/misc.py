import random
from collections import Counter

import numpy as np
import torch
from typing import TypeVar, Any
from rich.table import Table
from rich import print as rprint


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dict_add(dict1, dict2):
    return dict(Counter(dict1) + Counter(dict2))


T = TypeVar("T")


def to_tuple(value: T, number: int) -> tuple[T]:
    return tuple([value for _ in range(number)])


def print_table(headers: list[str], rows: list[Any]):
    table = Table()
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    rprint(table)


class NanException(Exception):
    def __repr__(self) -> str:
        return "Got unexpected NaN."
