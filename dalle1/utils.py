from __future__ import annotations

import math
import os
import random
import json
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .config import config_from_dict


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = device()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, dev


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def autocast_context(device_: torch.device, precision: str):
    if device_.type != "cuda" or precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cosine_lr(step: int, *, base_lr: float, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path, meta_path = _checkpoint_paths(path)
    payload = _move_to_cpu(payload)
    model = payload.pop("model", None)
    if not isinstance(model, dict) or not all(isinstance(v, torch.Tensor) for v in model.values()):
        raise TypeError("save_checkpoint expects a tensor state_dict under the 'model' key")
    save_safetensors(model, tensor_path)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    tensor_path, meta_path = _checkpoint_paths(path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing checkpoint tensor file: {tensor_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    cfg = meta.get("cfg")
    if isinstance(cfg, dict) and {"data", "train", "dvae", "transformer"} <= set(cfg):
        meta["cfg"] = config_from_dict(cfg)
    meta["model"] = load_safetensors(tensor_path, device=str(map_location))
    return meta


def _checkpoint_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix != ".safetensors":
        raise ValueError(f"Checkpoints must use the .safetensors extension, got: {path}")
    return path, path.with_suffix(".json")


def _move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {"config": repr(value)}


def init_wandb(cfg: Any, *, job_type: str):
    try:
        wandb_dir = Path(os.environ.get("WANDB_DIR", Path(cfg.train.output_dir).parent / "wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))
        import wandb

        if not hasattr(wandb, "init"):
            raise AttributeError(
                f"imported wandb from {getattr(wandb, '__file__', getattr(wandb, '__path__', None))}, "
                "but it has no init attribute"
            )
        return wandb.init(
            project="dalle1-reproduction",
            job_type=job_type,
            name=Path(cfg.train.output_dir).name,
            config=config_to_dict(cfg),
        )
    except Exception as exc:
        print(f"wandb unavailable; continuing without remote logging: {exc}", flush=True)
        return None


def wandb_log(run: Any, payload: dict[str, Any], *, step: int) -> None:
    if run is not None:
        run.log(payload, step=step)
