import torch
from rich import print
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader

from engine.category import Category
from engine.inferencer import Inferencer
from engine.metric import Metrics
import numpy as np


class Validator:
    """
    This validator can process unlimited number of dataloaders, and present the iou table automatively.\\
    The param 'dataloaders' should be a sequence of tuple[name_of_dataloder, dataloder].\\
    Ex. [("Loader1", dataloader1), ("Loader2", dataloader2), ("Loader3", dataloader3), ...]\\

    """

    def __init__(
        self,
        dataloaders: list[tuple[str, DataLoader]],
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        inferencer: Inferencer,
        metrics: Metrics,
        categories: list[Category],
    ) -> None:
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.inferencer = inferencer
        self.metrics = metrics
        self.categories = categories

    @torch.inference_mode()
    def _validate_one_domain(
        self, name: str, dataloader: DataLoader, prog: Progress
    ) -> tuple[float, np.ndarray]:

        task = prog.add_task(f"{name} Validating", total=len(dataloader))

        avg_loss = 0
        for data in dataloader:
            img = data["img"].cuda()
            ann = data["ann"].cuda()

            pred = self.inferencer.inference(self.model, img)
            loss = self.criterion(pred, ann)

            self.metrics.compute_and_accum(pred.argmax(1), ann)
            avg_loss += loss.mean().item()

            prog.update(task, advance=1)

        avg_loss /= len(dataloader)
        iou = self.metrics.get_and_reset()["IoU"]

        prog.remove_task(task)

        return avg_loss, iou

    def _make_table(self, ious: list[np.ndarray]) -> Table:

        table = Table()
        table.add_column("Id", justify="right")
        table.add_column("Name")

        for name, _ in self.dataloaders:
            table.add_column(f"{name} IoU")

        for cat, *iou in zip(self.categories, *ious):
            table.add_row(
                str(cat.id),
                cat.name,
                *[f"{_iou:.5f}" for _iou in iou],
            )

        table.add_row(
            "",
            "",
            *[f"{iou.mean():.5f}" for iou in ious],
        )
        return table

    def validate(self, prog: Progress):

        # Validation
        self.model.eval()

        losses = []  # [loss from dataloader1, ...]
        ious = []  # [[iou1, ...] from dataloader1, ...]

        for name, dataloader in self.dataloaders:
            loss, iou = self._validate_one_domain(name, dataloader, prog)
            losses.append(loss)
            ious.append(iou)

        # IoU table.
        table = self._make_table(ious)
        print(table)

        return [
            (name, loss, iou)
            for (name, dataloader), loss, iou in zip(self.dataloaders, losses, ious)
        ]  # TODO: better way to return validation result for visulization & logging.
