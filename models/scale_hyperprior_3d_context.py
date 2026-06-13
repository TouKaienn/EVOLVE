import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste

from .context_layers_3d import (
    CheckboardMaskedConv3d,
    Quantizer3D,
    create_3d_checkerboard_mask,
    conv1x1_3d,
)

MODEL_CONFIGS = {
    "small": {
        "depths": [2, 2, 4, 0],
        "channels": [96, 192, 384, 320],
        "hyper_depths": [1, 3],
        "hyper_channels": [256, 256],
        "context_depths": 2,
        "groups": [0, 16, 16, 32, 64, 192],
    },
}

def get_model_config(model_size: str = "small") -> dict:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_size: {model_size}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size]

def conv2x2_down_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

def deconv2x2_up_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

def conv4x4_down_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

def deconv4x4_up_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

def dwconv3x3_3d(ch: int) -> nn.Module:
    return nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch)

class PConv3x3_3D(nn.Module):

    def __init__(self, N: int, N1: int):
        super().__init__()
        self.N = N
        self.N1 = N1
        self.pconv = nn.Conv3d(N1, N1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1, x2 = x.split([self.N1, self.N - self.N1], dim=1)
        x1 = self.pconv(x1)
        return torch.cat((x1, x2), dim=1)

class PConvRB3D(nn.Module):

    def __init__(self, N: int, partial_ratio: int = 4, mlp_ratio: int = 4):
        super().__init__()
        N1 = N // partial_ratio
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            PConv3x3_3D(N, N1),
            conv1x1_3d(N, middle_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1_3d(middle_ch, N),
        )

    def forward(self, x):
        return x + self.branch(x)

class DWConvRB3D(nn.Module):

    def __init__(self, N: int, mlp_ratio: int = 2):
        super().__init__()
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            dwconv3x3_3d(N),
            conv1x1_3d(N, middle_ch),
            nn.LeakyReLU(),
            conv1x1_3d(middle_ch, N),
        )

    def forward(self, x):
        return x + self.branch(x)

class AnalysisTransform3D(nn.Module):

    def __init__(
        self,
        in_ch: int = 1,
        channels: List[int] = None,
        depths: List[int] = None,
        mlp_ratio: int = 4,
        partial_ratio: int = 4
    ):
        super().__init__()

        if channels is None:
            channels = [96, 192, 384, 320]
        if depths is None:
            depths = [2, 2, 4, 0]

        C1, C2, C3, C4 = channels
        L1, L2, L3, L4 = depths

        layers = []

        layers.append(conv4x4_down_3d(in_ch, C1))
        for _ in range(L1):
            layers.append(PConvRB3D(C1, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(conv2x2_down_3d(C1, C2))
        for _ in range(L2):
            layers.append(PConvRB3D(C2, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(conv2x2_down_3d(C2, C3))
        for _ in range(L3):
            layers.append(PConvRB3D(C3, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(conv2x2_down_3d(C3, C4))
        for _ in range(L4):
            layers.append(PConvRB3D(C4, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)

class SynthesisTransform3D(nn.Module):

    def __init__(
        self,
        out_ch: int = 1,
        channels: List[int] = None,
        depths: List[int] = None,
        mlp_ratio: int = 4,
        partial_ratio: int = 4
    ):
        super().__init__()

        if channels is None:
            channels = [96, 192, 384, 320]
        if depths is None:
            depths = [2, 2, 4, 0]

        C1, C2, C3, C4 = channels
        L1, L2, L3, L4 = depths

        layers = []

        layers.append(deconv2x2_up_3d(C4, C3))
        for _ in range(L3):
            layers.append(PConvRB3D(C3, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(deconv2x2_up_3d(C3, C2))
        for _ in range(L2):
            layers.append(PConvRB3D(C2, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(deconv2x2_up_3d(C2, C1))
        for _ in range(L1):
            layers.append(PConvRB3D(C1, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(deconv4x4_up_3d(C1, out_ch))

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)

class HyperAnalysisTransform3D(nn.Module):

    def __init__(
        self,
        M: int = 320,
        hyper_channels: List[int] = None,
        hyper_depths: List[int] = None,
        mlp_ratio: int = 4,
        partial_ratio: int = 4
    ):
        super().__init__()

        if hyper_channels is None:
            hyper_channels = [256, 256]
        if hyper_depths is None:
            hyper_depths = [1, 3]

        C5, C6 = hyper_channels
        L5, L6 = hyper_depths

        layers = []

        for _ in range(L5):
            layers.append(PConvRB3D(M, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(conv2x2_down_3d(M, C5))

        for _ in range(L6):
            layers.append(PConvRB3D(C5, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(conv2x2_down_3d(C5, C6))

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)

class HyperSynthesisTransform3D(nn.Module):

    def __init__(
        self,
        M: int = 320,
        hyper_channels: List[int] = None,
        hyper_depths: List[int] = None,
        mlp_ratio: int = 4,
        partial_ratio: int = 4
    ):
        super().__init__()

        if hyper_channels is None:
            hyper_channels = [256, 256]
        if hyper_depths is None:
            hyper_depths = [1, 3]

        C5, C6 = hyper_channels
        L5, L6 = hyper_depths

        layers = []

        layers.append(deconv2x2_up_3d(C6, C5))

        for _ in range(L6):
            layers.append(PConvRB3D(C5, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        layers.append(deconv2x2_up_3d(C5, 2 * M))

        for _ in range(L5):
            layers.append(PConvRB3D(2 * M, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio))

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
    min_val: float = SCALES_MIN,
    max_val: float = SCALES_MAX,
    levels: int = SCALES_LEVELS
) -> torch.Tensor:
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

def build_cc_transforms_3d(
    groups: List[int],
    num_slices: int,
    context_depths: int = 2
) -> nn.ModuleList:
    cc_transforms = nn.ModuleList()

    for i in range(1, num_slices):
        in_ch = sum(groups[1:i+1])

        out_ch = groups[i + 1] * 2

        layers = []
        for _ in range(context_depths):
            layers.append(DWConvRB3D(in_ch, mlp_ratio=2))
        layers.append(conv1x1_3d(in_ch, out_ch))

        cc_transforms.append(nn.Sequential(*layers))

    return cc_transforms

def build_context_prediction_3d(groups: List[int], num_slices: int) -> nn.ModuleList:
    context_prediction = nn.ModuleList()

    for i in range(num_slices):
        in_ch = groups[i + 1]
        out_ch = 2 * groups[i + 1]

        context_prediction.append(
            CheckboardMaskedConv3d(
                in_ch, out_ch,
                kernel_size=5, padding=2, stride=1
            )
        )

    return context_prediction

def build_param_aggregation_3d(
    groups: List[int],
    num_slices: int,
    M: int,
    context_depths: int = 2
) -> nn.ModuleList:
    param_aggregation = nn.ModuleList()

    for i in range(num_slices):
        hyper_ch = 2 * M

        if i == 0:
            cc_ch = 0
        else:
            cc_ch = 2 * groups[i + 1]

        spatial_ch = 2 * groups[i + 1]

        in_ch = hyper_ch + cc_ch + spatial_ch
        out_ch = 2 * groups[i + 1]

        layers = []
        for _ in range(context_depths):
            layers.append(DWConvRB3D(in_ch, mlp_ratio=2))
        layers.append(conv1x1_3d(in_ch, out_ch))

        param_aggregation.append(nn.Sequential(*layers))

    return param_aggregation

class ScaleHyperprior3DContext(nn.Module):

    def __init__(
        self,
        model_size: str = "small",
        num_slices: int = 5,
        groups: Optional[List[int]] = None,
        mlp_ratio: int = 4,
        partial_ratio: int = 4,
    ):
        super().__init__()

        config = get_model_config(model_size)
        self.model_size = model_size

        channels = config["channels"]
        depths = config["depths"]
        hyper_channels = config["hyper_channels"]
        hyper_depths = config["hyper_depths"]

        self.M = channels[3]
        self.N_hyper = hyper_channels[0]

        self.num_slices = num_slices

        if groups is None:
            self.groups = config["groups"]
        else:
            self.groups = groups

        assert sum(self.groups[1:]) == self.M, \
            f"Groups must sum to M={self.M}, got {sum(self.groups[1:])}"

        self.g_a = AnalysisTransform3D(
            in_ch=1,
            channels=channels,
            depths=depths,
            mlp_ratio=mlp_ratio,
            partial_ratio=partial_ratio
        )

        self.g_s = SynthesisTransform3D(
            out_ch=1,
            channels=channels,
            depths=depths,
            mlp_ratio=mlp_ratio,
            partial_ratio=partial_ratio
        )

        self.h_a = HyperAnalysisTransform3D(
            M=self.M,
            hyper_channels=hyper_channels,
            hyper_depths=hyper_depths,
            mlp_ratio=mlp_ratio,
            partial_ratio=partial_ratio
        )

        self.h_s = HyperSynthesisTransform3D(
            M=self.M,
            hyper_channels=hyper_channels,
            hyper_depths=hyper_depths,
            mlp_ratio=mlp_ratio,
            partial_ratio=partial_ratio
        )

        context_depths = config.get("context_depths", 2)
        self.cc_transforms = build_cc_transforms_3d(self.groups, num_slices, context_depths)
        self.context_prediction = build_context_prediction_3d(self.groups, num_slices)
        self.param_aggregation = build_param_aggregation_3d(self.groups, num_slices, self.M, context_depths)

        self.entropy_bottleneck = EntropyBottleneck(self.N_hyper)
        self.gaussian_conditional = GaussianConditional(None)

        self.quantizer = Quantizer3D()

        self.lmbda = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
        self.levels = len(self.lmbda)

        self.Gain = nn.Parameter(torch.tensor(
            [1.0, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0]
        ), requires_grad=True)

    def aux_loss(self) -> torch.Tensor:
        return self.entropy_bottleneck.loss()

    def update(self, scale_table: Optional[torch.Tensor] = None, force: bool = False) -> bool:
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)

        return updated

    def forward(
        self,
        x: torch.Tensor,
        noisequant: bool = False,
        stage: int = 3,
        s: int = 7
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-6

        if stage > 1:
            if s != 0:
                QuantizationRegulator = torch.abs(self.Gain[s]) + eps
            else:
                QuantizationRegulator = self.Gain[0].detach()
        else:
            QuantizationRegulator = self.Gain[-1].detach()

        ReQuantizationRegulator = 1.0 / QuantizationRegulator.clone().detach()

        y = self.g_a(x)
        B, C, D, H, W = y.size()

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_offset = z_offset.unsqueeze(-1)
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        params = self.h_s(z_hat)
        if params.shape[2:] != y.shape[2:]:
            params = params[:, :, :D, :H, :W]
        latent_means, latent_scales = params.chunk(2, dim=1)

        anchor_mask, non_anchor_mask = create_3d_checkerboard_mask(D, H, W, x.device)
        anchor_mask = anchor_mask.unsqueeze(0).unsqueeze(0)
        non_anchor_mask = non_anchor_mask.unsqueeze(0).unsqueeze(0)

        y_slices = torch.split(y, self.groups[1:], dim=1)

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, D, H, W, device=x.device),
            [2 * g for g in self.groups[1:]], dim=1
        )

        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []

        for slice_idx, y_slice in enumerate(y_slices):
            slice_ch = self.groups[slice_idx + 1]

            if slice_idx == 0:
                support_slices_ch = None
            else:
                support = torch.cat(y_hat_slices[:slice_idx], dim=1)
                support_slices_ch = self.cc_transforms[slice_idx - 1](support)

            hyper_support = torch.cat([latent_means, latent_scales], dim=1)

            if support_slices_ch is not None:
                support = torch.cat([support_slices_ch, hyper_support], dim=1)
            else:
                support = hyper_support

            anchor_input = torch.cat([ctx_params_anchor_split[slice_idx], support], dim=1)
            params_anchor = self.param_aggregation[slice_idx](anchor_input)
            means_anchor, scales_anchor = params_anchor.chunk(2, dim=1)

            scales_hat_split = torch.zeros_like(y_slice)
            means_hat_split = torch.zeros_like(y_slice)

            scales_hat_split = scales_hat_split + scales_anchor * anchor_mask
            means_hat_split = means_hat_split + means_anchor * anchor_mask

            y_anchor = y_slice * anchor_mask
            if noisequant:
                y_anchor_hat = self.quantizer.quantize(
                    y_anchor * QuantizationRegulator, "noise"
                ) * ReQuantizationRegulator
                y_anchor_hat_gs = self.quantizer.quantize(
                    y_anchor * QuantizationRegulator, "ste"
                ) * ReQuantizationRegulator
            else:
                y_anchor_hat = self.quantizer.quantize(
                    (y_anchor - means_anchor) * QuantizationRegulator, "ste"
                ) * ReQuantizationRegulator + means_anchor
                y_anchor_hat_gs = y_anchor_hat

            y_anchor_hat = y_anchor_hat * anchor_mask
            y_anchor_hat_gs = y_anchor_hat_gs * anchor_mask

            masked_context = self.context_prediction[slice_idx](y_anchor_hat)

            non_anchor_input = torch.cat([masked_context, support], dim=1)
            params_non_anchor = self.param_aggregation[slice_idx](non_anchor_input)
            means_non_anchor, scales_non_anchor = params_non_anchor.chunk(2, dim=1)

            scales_hat_split = scales_hat_split + scales_non_anchor * non_anchor_mask
            means_hat_split = means_hat_split + means_non_anchor * non_anchor_mask

            y_non_anchor = y_slice * non_anchor_mask
            if noisequant:
                y_non_anchor_hat = self.quantizer.quantize(
                    y_non_anchor * QuantizationRegulator, "noise"
                ) * ReQuantizationRegulator
                y_non_anchor_hat_gs = self.quantizer.quantize(
                    y_non_anchor * QuantizationRegulator, "ste"
                ) * ReQuantizationRegulator
            else:
                y_non_anchor_hat = self.quantizer.quantize(
                    (y_non_anchor - means_non_anchor) * QuantizationRegulator, "ste"
                ) * ReQuantizationRegulator + means_non_anchor
                y_non_anchor_hat_gs = y_non_anchor_hat

            y_non_anchor_hat = y_non_anchor_hat * non_anchor_mask
            y_non_anchor_hat_gs = y_non_anchor_hat_gs * non_anchor_mask

            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice * QuantizationRegulator,
                scales_hat_split * QuantizationRegulator,
                means=means_hat_split * QuantizationRegulator
            )

            y_hat_slice = y_anchor_hat + y_non_anchor_hat
            y_hat_slice_gs = y_anchor_hat_gs + y_non_anchor_hat_gs

            y_hat_slices.append(y_hat_slice)
            y_hat_slices_for_gs.append(y_hat_slice_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: torch.Tensor, s: int = 7, inputscale: float = 0) -> Dict:
        if inputscale != 0:
            QuantizationRegulator = torch.tensor(inputscale, device=x.device, dtype=x.dtype)
        else:
            assert s in range(0, self.levels), f"s should be in range(0, {self.levels})"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = 1.0 / QuantizationRegulator

        y = self.g_a(x)
        B, C, D, H, W = y.size()

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:])

        params = self.h_s(z_hat)
        if params.shape[2:] != y.shape[2:]:
            params = params[:, :, :D, :H, :W]
        latent_means, latent_scales = params.chunk(2, dim=1)

        anchor_mask, non_anchor_mask = create_3d_checkerboard_mask(D, H, W, x.device)
        anchor_mask = anchor_mask.unsqueeze(0).unsqueeze(0)
        non_anchor_mask = non_anchor_mask.unsqueeze(0).unsqueeze(0)

        y_slices = torch.split(y, self.groups[1:], dim=1)

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, D, H, W, device=x.device),
            [2 * g for g in self.groups[1:]], dim=1
        )

        y_strings = []
        y_hat_slices = []

        for slice_idx, y_slice in enumerate(y_slices):
            if slice_idx == 0:
                support_slices_ch = None
            else:
                support = torch.cat(y_hat_slices[:slice_idx], dim=1)
                support_slices_ch = self.cc_transforms[slice_idx - 1](support)

            hyper_support = torch.cat([latent_means, latent_scales], dim=1)
            support = torch.cat([support_slices_ch, hyper_support], dim=1) \
                if support_slices_ch is not None else hyper_support

            anchor_input = torch.cat([ctx_params_anchor_split[slice_idx], support], dim=1)
            params_anchor = self.param_aggregation[slice_idx](anchor_input)
            means_anchor, scales_anchor = params_anchor.chunk(2, dim=1)

            y_anchor = y_slice * anchor_mask
            y_anchor_scaled = y_anchor * QuantizationRegulator
            means_anchor_scaled = means_anchor * anchor_mask * QuantizationRegulator
            scales_anchor_scaled = scales_anchor * anchor_mask * QuantizationRegulator + 1e-6 * (1 - anchor_mask)

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_scaled)
            anchor_strings = self.gaussian_conditional.compress(
                y_anchor_scaled, indexes_anchor, means=means_anchor_scaled
            )

            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_scaled
            )
            y_anchor_decode = anchor_quantized * ReQuantizationRegulator * anchor_mask

            masked_context = self.context_prediction[slice_idx](y_anchor_decode)
            non_anchor_input = torch.cat([masked_context, support], dim=1)
            params_non_anchor = self.param_aggregation[slice_idx](non_anchor_input)
            means_non_anchor, scales_non_anchor = params_non_anchor.chunk(2, dim=1)

            y_non_anchor = y_slice * non_anchor_mask
            y_non_anchor_scaled = y_non_anchor * QuantizationRegulator
            means_non_anchor_scaled = means_non_anchor * non_anchor_mask * QuantizationRegulator
            scales_non_anchor_scaled = scales_non_anchor * non_anchor_mask * QuantizationRegulator + 1e-6 * (1 - non_anchor_mask)

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_scaled)
            non_anchor_strings = self.gaussian_conditional.compress(
                y_non_anchor_scaled, indexes_non_anchor, means=means_non_anchor_scaled
            )

            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_scaled
            )
            y_non_anchor_decode = non_anchor_quantized * ReQuantizationRegulator * non_anchor_mask

            y_hat_slice = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_hat_slice)
            y_strings.append([anchor_strings, non_anchor_strings])

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-3:],
            "y_shape": y.size()[-3:],
        }

    def decompress(
        self,
        strings: List,
        shape: Tuple[int, int, int],
        y_shape: Optional[Tuple[int, int, int]] = None,
        s: int = 7,
        inputscale: float = 0
    ) -> torch.Tensor:
        y_strings, z_strings = strings

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        B = z_hat.shape[0]

        if inputscale != 0:
            QuantizationRegulator = torch.tensor(inputscale, device=z_hat.device, dtype=z_hat.dtype)
        else:
            assert s in range(0, self.levels), f"s should be in range(0, {self.levels})"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = 1.0 / QuantizationRegulator

        params = self.h_s(z_hat)
        if y_shape is not None:
            D, H, W = y_shape
            params = params[:, :, :D, :H, :W]
        else:
            D, H, W = params.shape[2:]

        latent_means, latent_scales = params.chunk(2, dim=1)

        anchor_mask, non_anchor_mask = create_3d_checkerboard_mask(D, H, W, z_hat.device)
        anchor_mask = anchor_mask.unsqueeze(0).unsqueeze(0)
        non_anchor_mask = non_anchor_mask.unsqueeze(0).unsqueeze(0)

        C = self.M
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, D, H, W, device=z_hat.device),
            [2 * g for g in self.groups[1:]], dim=1
        )

        y_hat_slices = []

        for slice_idx in range(self.num_slices):
            slice_ch = self.groups[slice_idx + 1]

            if slice_idx == 0:
                support_slices_ch = None
            else:
                support = torch.cat(y_hat_slices[:slice_idx], dim=1)
                support_slices_ch = self.cc_transforms[slice_idx - 1](support)

            hyper_support = torch.cat([latent_means, latent_scales], dim=1)
            support = torch.cat([support_slices_ch, hyper_support], dim=1) \
                if support_slices_ch is not None else hyper_support

            anchor_input = torch.cat([ctx_params_anchor_split[slice_idx], support], dim=1)
            params_anchor = self.param_aggregation[slice_idx](anchor_input)
            means_anchor, scales_anchor = params_anchor.chunk(2, dim=1)

            means_anchor_scaled = means_anchor * anchor_mask * QuantizationRegulator
            scales_anchor_scaled = scales_anchor * anchor_mask * QuantizationRegulator + 1e-6 * (1 - anchor_mask)
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_scaled)

            anchor_strings = y_strings[slice_idx][0]
            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_scaled
            )
            y_anchor_decode = anchor_quantized * ReQuantizationRegulator * anchor_mask

            masked_context = self.context_prediction[slice_idx](y_anchor_decode)
            non_anchor_input = torch.cat([masked_context, support], dim=1)
            params_non_anchor = self.param_aggregation[slice_idx](non_anchor_input)
            means_non_anchor, scales_non_anchor = params_non_anchor.chunk(2, dim=1)

            means_non_anchor_scaled = means_non_anchor * non_anchor_mask * QuantizationRegulator
            scales_non_anchor_scaled = scales_non_anchor * non_anchor_mask * QuantizationRegulator + 1e-6 * (1 - non_anchor_mask)
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_scaled)

            non_anchor_strings = y_strings[slice_idx][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_scaled
            )
            y_non_anchor_decode = non_anchor_quantized * ReQuantizationRegulator * non_anchor_mask

            y_hat_slice = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)

        return x_hat

class RateDistortionLossContext(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        lmbda: float
    ) -> Dict[str, torch.Tensor]:
        B = target.size(0)
        num_voxels = target[0].numel()

        bpp_y = torch.log(output["likelihoods"]["y"].clamp(min=1e-9)).sum() / (-math.log(2) * num_voxels * B)
        bpp_z = torch.log(output["likelihoods"]["z"].clamp(min=1e-9)).sum() / (-math.log(2) * num_voxels * B)
        bpp_loss = bpp_y + bpp_z

        mse_loss = self.mse(output["x_hat"], target)

        loss = lmbda * mse_loss + bpp_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "bpp_loss": bpp_loss,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }
