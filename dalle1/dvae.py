from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import DvaeConfig


def _init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Match the initialization used by OpenAI's released dVAE."""
    fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    nn.init.normal_(conv.weight, std=fan_in**-0.5)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, total_layers: int, decoder: bool = False):
        super().__init__()
        hidden = max(1, out_channels // 4)
        self.id_path = _init_conv(nn.Conv2d(in_channels, out_channels, 1)) if in_channels != out_channels else nn.Identity()
        if decoder:
            layers = [
                nn.ReLU(),
                nn.Conv2d(in_channels, hidden, 1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, out_channels, 3, padding=1),
            ]
        else:
            layers = [
                nn.ReLU(),
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, out_channels, 1),
            ]
        self.res_path = nn.Sequential(*layers)
        for layer in self.res_path:
            if isinstance(layer, nn.Conv2d):
                _init_conv(layer)
        self.res_gain = 1 / (total_layers**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.res_gain * self.res_path(x)


class Encoder(nn.Module):
    def __init__(self, cfg: DvaeConfig):
        super().__init__()
        layers: list[nn.Module] = [_init_conv(nn.Conv2d(cfg.in_channels, cfg.hidden_channels, 7, padding=3))]
        channels = cfg.hidden_channels
        total_layers = len(cfg.channel_multipliers) * cfg.num_res_blocks
        for index, mult in enumerate(cfg.channel_multipliers):
            out_channels = cfg.hidden_channels * mult
            for block_index in range(cfg.num_res_blocks):
                block_in = channels if block_index == 0 else out_channels
                layers.append(ResBlock(block_in, out_channels, total_layers=total_layers))
            channels = out_channels
            if index < len(cfg.channel_multipliers) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output = _init_conv(nn.Conv2d(channels, cfg.codebook_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # The official implementation keeps the categorical-logit projection
        # in float32 even when the residual trunk uses mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.output(x.float())


class Decoder(nn.Module):
    def __init__(self, cfg: DvaeConfig):
        super().__init__()
        channels = cfg.hidden_channels // 2
        layers: list[nn.Module] = [_init_conv(nn.Conv2d(cfg.codebook_size, channels, 1))]
        total_layers = len(cfg.channel_multipliers) * cfg.num_res_blocks
        for index, mult in enumerate(reversed(cfg.channel_multipliers)):
            out_channels = cfg.hidden_channels * mult
            for block_index in range(cfg.num_res_blocks):
                block_in = channels if block_index == 0 else out_channels
                layers.append(ResBlock(block_in, out_channels, total_layers=total_layers, decoder=True))
            channels = out_channels
            if index < len(cfg.channel_multipliers) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.extend([nn.ReLU(), _init_conv(nn.Conv2d(channels, 2 * cfg.in_channels, 1))])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        projection = self.net[0]
        # OpenAI's decoder also keeps this large code projection in float32.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = F.linear(z.float(), projection.weight[:, :, 0, 0].float(), projection.bias.float())
        z = z.permute(0, 3, 1, 2)
        for index in range(1, len(self.net)):
            z = self.net[index](z)
        return z


class GumbelRelaxedQuantizer(nn.Module):
    def __init__(self, cfg: DvaeConfig):
        super().__init__()
        self.codebook_size = cfg.codebook_size
        self.kl_weight = cfg.kl_weight
        self.temperature = cfg.temperature
        downsample_factor = 2 ** max(0, len(cfg.channel_multipliers) - 1)
        # Reconstruction NLL is averaged over image channels and pixels, while
        # KL is averaged over latent tokens. Match the paper's normalization by
        # dividing KL by (image values / latent tokens). For 256x256 RGB images
        # with a 32x32 latent grid, this factor is 3 * 8**2 = 192.
        self.kl_normalization = cfg.in_channels * downsample_factor**2

    def forward(
        self,
        logits: torch.Tensor,
        schedule: torch.Tensor | None = None,
        *,
        return_ids: bool = True,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
        logits = logits.permute(0, 2, 3, 1).float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        temperature = self.temperature if schedule is None else schedule[0]
        kl_weight = self.kl_weight if schedule is None else schedule[1]
        if self.training:
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        else:
            ids = logits.argmax(dim=-1)
            weights = F.one_hot(ids, self.codebook_size).type_as(logits)

        ids = probs.argmax(dim=-1) if return_ids or return_diagnostics else None
        kl_per_token = (probs * (log_probs + torch.log(logits.new_tensor(self.codebook_size)))).sum(dim=-1)
        kl_loss = (kl_weight / self.kl_normalization) * kl_per_token.mean()
        diagnostics: dict[str, torch.Tensor] = {}
        if return_diagnostics:
            assert ids is not None
            diagnostics = {
                "unweighted_kl": kl_per_token.mean().detach(),
                "posterior_entropy": (-probs * log_probs).sum(dim=-1).mean().detach(),
                "logit_rms": logits.square().mean().sqrt().detach(),
                "logit_abs_max": logits.abs().max().detach(),
                "token_counts": torch.bincount(ids.flatten(), minlength=self.codebook_size).detach(),
            }
        return weights, ids if return_ids else None, kl_loss, diagnostics

    def embed_code(self, ids: torch.Tensor) -> torch.Tensor:
        return F.one_hot(ids, self.codebook_size).float()


class DiscreteVAE(nn.Module):
    def __init__(self, cfg: DvaeConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.quantizer = GumbelRelaxedQuantizer(cfg)
        self.decoder = Decoder(cfg)

    @property
    def downsample_factor(self) -> int:
        return 2 ** max(0, len(self.cfg.channel_multipliers) - 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(map_pixels(images, eps=self.cfg.logit_laplace_eps))
        ids = logits.argmax(dim=1)
        return ids

    @torch.no_grad()
    def decode_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return decode_logit_laplace_mean(
            self.decoder(self.quantizer.embed_code(ids)),
            eps=self.cfg.logit_laplace_eps,
        )

    def forward(
        self,
        images: torch.Tensor,
        schedule: torch.Tensor | None = None,
        return_details: bool = True,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        mapped_images = map_pixels(images, eps=self.cfg.logit_laplace_eps)
        logits = self.encoder(mapped_images)
        quant, ids, kl_loss, diagnostics = self.quantizer(
            logits,
            schedule,
            return_ids=return_details,
            return_diagnostics=return_diagnostics,
        )
        recon_params = self.decoder(quant)
        recon_loss = logit_laplace_nll(recon_params, mapped_images)
        output = {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
        if return_details:
            output.update(
                {
                    "recon": decode_logit_laplace_mean(recon_params, eps=self.cfg.logit_laplace_eps),
                    "ids": ids,
                    "logits": logits,
                }
            )
        output.update(diagnostics)
        return output


def map_pixels(images: torch.Tensor, *, eps: float) -> torch.Tensor:
    pixels = images.add(1).mul(0.5)
    return pixels.mul(1 - 2 * eps).add(eps)


def unmap_pixels(pixels: torch.Tensor, *, eps: float) -> torch.Tensor:
    return pixels.sub(eps).div(1 - 2 * eps).clamp(0, 1).mul(2).sub(1)


def logit_laplace_nll(params: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    mean, log_scale = params.chunk(2, dim=1)
    target = torch.logit(images)
    log_det = torch.log(images) + torch.log1p(-images)
    nll = (target - mean).abs() * torch.exp(-log_scale) + log_scale + torch.log(params.new_tensor(2.0)) + log_det
    return nll.mean()


def decode_logit_laplace_mean(params: torch.Tensor, *, eps: float = 0.1) -> torch.Tensor:
    mean, _log_scale = params.chunk(2, dim=1)
    return unmap_pixels(mean.sigmoid(), eps=eps)
