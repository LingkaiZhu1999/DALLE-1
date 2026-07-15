from __future__ import annotations

import json

import pytest
import torch

import dalle1.utils as utils
from dalle1.train_dvae import cosine_anneal


def test_cosine_lr_supports_independent_decay_and_nonzero_floor() -> None:
    kwargs = {
        "base_lr": 1e-4,
        "min_lr": 1.25e-6,
        "total_steps": 3_000_000,
        "decay_steps": 1_200_000,
        "warmup_steps": 0,
    }

    assert utils.cosine_lr(0, **kwargs) == pytest.approx(1e-4)
    assert utils.cosine_lr(1_200_000, **kwargs) == pytest.approx(1.25e-6)
    assert utils.cosine_lr(3_000_000, **kwargs) == pytest.approx(1.25e-6)


def test_cosine_anneal_reaches_end_and_stays_there() -> None:
    assert cosine_anneal(0, start=1.0, end=1 / 16, steps=150_000) == pytest.approx(1.0)
    assert cosine_anneal(150_000, start=1.0, end=1 / 16, steps=150_000) == pytest.approx(1 / 16)
    assert cosine_anneal(3_000_000, start=1.0, end=1 / 16, steps=150_000) == pytest.approx(1 / 16)


def test_save_checkpoint_makes_channels_last_tensors_contiguous(tmp_path) -> None:
    weight = torch.randn(8, 4, 3, 3).contiguous(memory_format=torch.channels_last)
    path = tmp_path / "checkpoint.safetensors"

    utils.save_checkpoint(path, model={"weight": weight}, step=12)
    checkpoint = utils.load_checkpoint(path)

    assert checkpoint["step"] == 12
    assert checkpoint["model"]["weight"].is_contiguous()
    torch.testing.assert_close(checkpoint["model"]["weight"], weight)


def test_failed_checkpoint_save_preserves_existing_files(tmp_path, monkeypatch) -> None:
    path = tmp_path / "checkpoint.safetensors"
    meta_path = path.with_suffix(".json")
    path.write_bytes(b"existing tensor checkpoint")
    meta_path.write_text(json.dumps({"step": 11}), encoding="utf-8")

    def fail_save(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(utils, "save_safetensors", fail_save)
    with pytest.raises(RuntimeError, match="simulated save failure"):
        utils.save_checkpoint(path, model={"weight": torch.ones(1)}, step=12)

    assert path.read_bytes() == b"existing tensor checkpoint"
    assert json.loads(meta_path.read_text(encoding="utf-8")) == {"step": 11}
    assert not list(tmp_path.glob(".*.tmp"))
