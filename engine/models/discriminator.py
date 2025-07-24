import torch
from transformers import SegformerDecodeHead
from engine.models.segformer import grad_reverse


class LogitsDiscriminator(torch.nn.Module):
    def __init__(self, num_categories: int, num_domains: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            # Input is 1 x 64 x 64
            torch.nn.Conv2d(num_categories + 3, 64, (4, 4), (2, 2), (1, 1), bias=True),
            torch.nn.LeakyReLU(0.2, True),
            # State size. 64 x 32 x 32
            torch.nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            # State size. 128 x 16 x 16
            torch.nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            # State size. 256 x 8 x 8
            torch.nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            # State size. 512 x 4 x 4
            torch.nn.Conv2d(512, num_domains, (4, 4), (1, 1), (0, 0), bias=True),
            # torch.nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.layers(grad_reverse(torch.cat([image, x], dim=1)))


# if __name__ == "__main__":
#     discrimiator = Discriminator().cuda()

#     features = torch.rand((1, 256, 16, 16)).cuda()
#     out = discrimiator(features)
#     print(out.shape)
