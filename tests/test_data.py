from __future__ import annotations

import io
import tarfile

import pytest
from PIL import Image

from dalle1.config import DataConfig
from dalle1.data import build_webdataset_loader


def test_webdataset_accepts_prepartitioned_distributed_shards(tmp_path, monkeypatch) -> None:
    pytest.importorskip("webdataset")
    for index in range(4):
        shard = tmp_path / f"{index:05d}.tar"
        image_buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color="red").save(image_buffer, format="PNG")
        with tarfile.open(shard, "w") as archive:
            for name, payload in ((f"{index}.png", image_buffer.getvalue()), (f"{index}.txt", b"caption")):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    loader = build_webdataset_loader(
        DataConfig(data_dir=str(tmp_path), image_size=8, batch_size=1, num_workers=0),
        rank=0,
        world_size=4,
    )

    images, captions = next(iter(loader))
    assert images.shape == (1, 3, 8, 8)
    assert captions == ["caption"]
