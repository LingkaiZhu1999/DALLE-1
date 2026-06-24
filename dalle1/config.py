from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    data_dir: str = "/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data"
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 8
    shuffle_buffer: int = 10_000
    max_shards: int | None = None
    caption_key: str = "txt"
    augmentation: str = "center"


@dataclass
class TrainConfig:
    output_dir: str = "runs/dalle1"
    steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 5_000
    mixed_precision: str = "bf16"
    compile: bool = False
    seed: int = 1337
    ema_decay: float = 0.999
    ema_every: int = 25


@dataclass
class DvaeConfig:
    image_size: int = 256
    in_channels: int = 3
    hidden_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    codebook_size: int = 8192
    code_dim: int = 8192
    kl_weight: float = 1e-4
    kl_weight_start: float = 0.0
    kl_weight_end: float = 6.6
    kl_anneal_steps: int = 5_000
    temperature: float = 0.9
    temperature_start: float = 1.0
    temperature_end: float = 1.0 / 16.0
    temperature_anneal_steps: int = 150_000
    logit_laplace_eps: float = 0.1


@dataclass
class DalleTransformerConfig:
    text_vocab_size: int = 49_408
    image_vocab_size: int = 8192
    text_seq_len: int = 256
    text_lowercase: bool = True
    bpe_dropout: float = 0.0
    image_tokens_per_side: int = 32
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    image_loss_weight: float = 1.0
    text_loss_weight: float = 0.125

    @property
    def image_seq_len(self) -> int:
        return self.image_tokens_per_side * self.image_tokens_per_side

    @property
    def seq_len(self) -> int:
        return self.text_seq_len + self.image_seq_len


@dataclass
class DalleConfig:
    data: DataConfig
    train: TrainConfig
    dvae: DvaeConfig
    transformer: DalleTransformerConfig


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], value: dict[str, Any] | None) -> T:
    value = value or {}
    kwargs = {}
    for field in fields(cls):
        raw = value.get(field.name, field.default)
        field_type = field.type
        if is_dataclass(field_type):
            raw = _coerce_dataclass(field_type, raw)
        if field.name == "channel_multipliers" and isinstance(raw, list):
            raw = tuple(raw)
        kwargs[field.name] = raw
    return cls(**kwargs)


def load_config(path: str | Path) -> DalleConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> DalleConfig:
    return DalleConfig(
        data=_coerce_dataclass(DataConfig, raw.get("data")),
        train=_coerce_dataclass(TrainConfig, raw.get("train")),
        dvae=_coerce_dataclass(DvaeConfig, raw.get("dvae")),
        transformer=_coerce_dataclass(DalleTransformerConfig, raw.get("transformer")),
    )
