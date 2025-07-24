from typing import Any
from transformers import SegformerForSemanticSegmentation, SegformerDecodeHead
import torch
from torch.nn import functional as F


import math

import torch
import torch.utils.checkpoint


class Segformer(SegformerForSemanticSegmentation):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pred = super().forward(img)["logits"]
        pred = F.interpolate(pred, img.shape[-2:], mode="bilinear")
        return pred


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_variables
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None


def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x, lam)


class SegformerWithDomainClassifier(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 4, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 1, 4, 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, img: torch.Tensor, train: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not train:
            pred = super().forward(img)["logits"]
            pred = F.interpolate(pred, img.shape[-2:], mode="bilinear")
            return pred
        else:
            hidden_states = self.segformer(img, output_hidden_states=True).hidden_states
            pred = self.decode_head(hidden_states)["logits"]
            pred = F.interpolate(pred, img.shape[-2:], mode="bilinear")

            domain_pred = self.domain_classifier(grad_reverse(hidden_states[-1]))
            return {"pred": pred, "domain_pred": domain_pred}


class SegformerDiscriminator(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states):
        reversed_hidden_states = [grad_reverse(state) for state in hidden_states]
        pred = super().forward(reversed_hidden_states)
        return pred


class SegformerLogitsDiscriminator(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        reversed_hidden_states = [grad_reverse(state) for state in hidden_states]
        pred = super().forward(reversed_hidden_states)
        return pred
