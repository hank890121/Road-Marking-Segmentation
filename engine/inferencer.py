from typing import Protocol
import torch
from engine.slide_inference import slide_inference


class Inferencer(Protocol):
    def inference(self, model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor: ...


class BasicInferencer:
    def inference(self, model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        return model(img)


class SlideInferencer:
    def __init__(
        self, crop_size: tuple[int, int], stride: tuple[int, int], num_categories: int
    ) -> None:
        self.crop_size = crop_size
        self.stride = stride
        self.num_categories = num_categories

    def inference(self, model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        return slide_inference(
            model, img, self.crop_size, self.stride, self.num_categories
        )
