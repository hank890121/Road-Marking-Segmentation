import math
from typing import Union

import numpy as np
import torch
from engine.category import Category
from numpy.typing import NDArray
from pathlib import Path
from torchvision.utils import save_image as torch_save_img
import cv2
from matplotlib import pyplot as plt


class IdMapVisualizer:
    def __init__(self, categories: list[Category], nrow: int = 4) -> None:
        self.categories = categories
        self.nrow = nrow

    def _visualize_one(self, id_map: Union[torch.Tensor, NDArray]) -> NDArray[np.uint8]:
        """Visualize an id map (BGR) with given categories.

        Args:
            id_map (Union[torch.Tensor, NDArray]): A HxW map that consists of category ids.
            nrow (int, optional): Number of visualizations in a row. Defaults to 4.

        Returns:
            NDArray[np.uint8]: A visualized 8-bit BGR image (HxWx3) given categories.
        """

        w, h = id_map.shape
        vis = np.zeros((w, h, 3), dtype=np.uint8)
        for cat in self.categories:
            vis[id_map == cat.id] = (cat.b, cat.g, cat.r)
        return vis

    def _make_grid(
        self, vis_list: list[np.ndarray], nrow: int = 4
    ) -> NDArray[np.uint8]:
        """Re-arrange a list of visualizations into grid.

        Args:
            vis_list (list[np.ndarray]): A list of visualizations
            nrow (int, optional): Number of visualizations in a row. Defaults to 8.

        Returns:
            NDArray[np.uint8]: A grid of visualizations.
        """
        n = len(vis_list)  # batch size
        m = n % nrow  # number of rows
        ncol = math.ceil(n / nrow)  # number of columns
        if n > nrow and m:
            for _ in range(m):
                vis_list.append(np.zeros(vis_list[0].shape))
        return np.concatenate(
            [
                np.concatenate(vis_list[nc * nrow : (nc + 1) * nrow], 1)
                for nc in range(ncol)
            ],
            0,
        )

    def visualize(self, id_map: Union[torch.Tensor, NDArray]) -> NDArray[np.uint8]:
        """Visualize a batch of id map (NxHxW) or one index map (HxW) to a grid of BGR image (HxWx3) with given categories.

        Args:
            id_map (Union[torch.Tensor, NDArray]): A HxW or NxHxW map that consists of category ids.
            nrow (int, optional): Number of visualizations in a row. Defaults to 8.

        Returns:
            NDArray[np.uint8]: A grid visualization of id map with given categories.
        """
        assert (
            len(id_map.shape) == 2 or len(id_map.shape) == 3
        ), "The shape of index map should be NxWxH or WxH."

        if len(id_map.shape) == 2:
            id_map = np.expand_dims(id_map, 0)

        return self._make_grid(
            [self._visualize_one(im) for im in id_map],
            id_map.shape[0] if id_map.shape[0] < self.nrow else self.nrow,
        )


class ImgSaver:
    def __init__(self, root: str, visualizer: IdMapVisualizer, nrow: int = 4) -> None:
        self.visualizer = visualizer
        self.root = Path(root)
        if not self.root.exists():
            self.root.mkdir()
        self.nrow = nrow

    def save_img(
        self, img: torch.Tensor, filename: str, normalize: bool = True
    ) -> None:
        torch_save_img(
            img.cpu(), self.root / filename, normalize=normalize, nrow=self.nrow
        )

    def save_ann(self, ann: torch.Tensor, filename: str) -> None:
        cv2.imwrite(
            str(self.root / filename),
            self.visualizer.visualize(ann.cpu()),
        )

    def save_pred(self, pred: torch.Tensor, filename: str) -> None:
        self.save_ann(pred.argmax(1), filename)

    def save_heatmap(self, heatmap: torch.Tensor, filename: str) -> None:
        h = self.visualizer._make_grid(heatmap.cpu().numpy(), self.nrow)
        plt.imsave(str(self.root / filename), h)
