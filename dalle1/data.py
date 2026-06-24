from __future__ import annotations

import io
import random
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig


class DalleImageTransform:
    def __init__(self, image_size: int, augmentation: str):
        self.image_size = image_size
        self.augmentation = augmentation

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.augmentation == "dvae":
            image = self._random_square_crop(image)
            max_size = max(self.image_size, int(round(9 / 8 * self.image_size)))
            size = random.randint(self.image_size, max_size)
            image = image.resize((size, size), Image.Resampling.BOX)
            image = self._random_crop(image, self.image_size)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif self.augmentation == "transformer":
            image = self._random_square_crop(image, low=3 / 8, high=5 / 8)
            max_size = max(self.image_size, int(round(9 / 8 * self.image_size)))
            size = random.randint(self.image_size, max_size)
            image = image.resize((size, size), Image.Resampling.BOX)
            image = self._random_crop(image, self.image_size)
        else:
            image = transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC)(image)
            image = transforms.CenterCrop(self.image_size)(image)
        tensor = transforms.ToTensor()(image)
        return tensor.mul(2).sub(1)

    def _random_square_crop(self, image: Image.Image, *, low: float = 0.0, high: float = 1.0) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        max_left = width - side
        max_top = height - side
        left = self._offset(max_left, low, high)
        top = self._offset(max_top, low, high)
        return image.crop((left, top, left + side, top + side))

    @staticmethod
    def _offset(max_offset: int, low: float, high: float) -> int:
        if max_offset <= 0:
            return 0
        start = int(low * max_offset)
        stop = max(start, int(high * max_offset))
        return random.randint(start, stop)

    @staticmethod
    def _random_crop(image: Image.Image, size: int) -> Image.Image:
        width, height = image.size
        if width == size and height == size:
            return image
        left = random.randint(0, max(0, width - size))
        top = random.randint(0, max(0, height - size))
        return image.crop((left, top, left + size, top + size))


def image_transform(image_size: int, augmentation: str = "center") -> DalleImageTransform:
    return DalleImageTransform(image_size, augmentation)


def _decode_sample(sample: dict, transform) -> tuple[torch.Tensor, str]:
    image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    caption = sample.get("txt", "")
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8", errors="replace")
    return transform(image.convert("RGB")), str(caption)


def build_webdataset_loader(cfg: DataConfig, *, rank: int = 0, world_size: int = 1) -> DataLoader:
    shards = sorted(str(path) for path in Path(cfg.data_dir).glob("*.tar"))
    if cfg.max_shards is not None:
        shards = shards[: cfg.max_shards]
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {cfg.data_dir}")
    if world_size > 1:
        shards = shards[rank::world_size]
        if not shards:
            raise ValueError(f"Rank {rank} received no shards from {cfg.data_dir}; reduce world size or add shards")
    try:
        import webdataset as wds
    except ModuleNotFoundError:
        dataset = TarShardDataset(shards, cfg.image_size, cfg.augmentation)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    transform = image_transform(cfg.image_size, cfg.augmentation)
    dataset = (
        wds.WebDataset(shards, shardshuffle=False, resampled=True)
        .shuffle(cfg.shuffle_buffer)
        .decode("pil")
        .map(lambda sample: _decode_sample(sample, transform))
        .batched(cfg.batch_size, partial=False)
    )
    return DataLoader(dataset, batch_size=None, num_workers=cfg.num_workers, pin_memory=True)


class TarShardDataset(IterableDataset):
    """Small dependency-free reader for img2dataset tar shards."""

    def __init__(self, shards: list[str], image_size: int, augmentation: str):
        super().__init__()
        self.shards = shards
        self.transform = image_transform(image_size, augmentation)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shards = self.shards
        else:
            shards = self.shards[worker.id :: worker.num_workers]
        for shard in shards:
            captions: dict[str, str] = {}
            pending_images: dict[str, bytes] = {}
            with tarfile.open(shard, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    key = str(Path(member.name).with_suffix(""))
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    if suffix == ".txt":
                        captions[key] = data.decode("utf-8", errors="replace")
                        if key in pending_images:
                            yield self._make_example(pending_images.pop(key), captions[key])
                    elif suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                        if key in captions:
                            yield self._make_example(data, captions[key])
                        else:
                            pending_images[key] = data

    def _make_example(self, image_bytes: bytes, caption: str) -> tuple[torch.Tensor, str]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image), caption
