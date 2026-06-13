import torch
import torch.nn as nn
from typing import Tuple

class CheckboardMaskedConv3d(nn.Conv3d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, **kwargs
        )

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if isinstance(kernel_size, int):
            kd = kh = kw = kernel_size
        else:
            kd, kh, kw = kernel_size

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    if (d + h + w) % 2 == 1:
                        self.mask[:, :, d, h, w] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)

class Quantizer3D:

    @staticmethod
    def quantize(inputs: torch.Tensor, quantize_type: str = "noise") -> torch.Tensor:
        if quantize_type == "noise":
            half = 0.5
            noise = torch.empty_like(inputs).uniform_(-half, half)
            return inputs + noise
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

def create_3d_checkerboard_mask(
    D: int,
    H: int,
    W: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_idx = torch.arange(D, device=device).view(-1, 1, 1)
    h_idx = torch.arange(H, device=device).view(1, -1, 1)
    w_idx = torch.arange(W, device=device).view(1, 1, -1)

    anchor_mask = ((d_idx + h_idx + w_idx) % 2 == 0).float()
    non_anchor_mask = 1.0 - anchor_mask

    return anchor_mask, non_anchor_mask

def conv1x1_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
