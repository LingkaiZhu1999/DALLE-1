from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange

from .config import load_config
from .data import build_webdataset_loader
from .dvae import DiscreteVAE
from .utils import (
    autocast_context,
    cleanup_distributed,
    cosine_lr,
    init_wandb,
    init_distributed,
    is_rank_zero,
    save_checkpoint,
    seed_everything,
    set_lr,
    reduce_mean,
    unwrap_model,
    wandb_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dvae_small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    rank, world_size, _local_rank, dev = init_distributed()
    loader = iter(build_webdataset_loader(cfg.data, rank=rank, world_size=world_size))
    model = DiscreteVAE(cfg.dvae).to(dev)
    if cfg.train.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[dev.index])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    out_dir = Path(cfg.train.output_dir)
    wandb_run = init_wandb(cfg, job_type="train_dvae") if is_rank_zero(rank) else None
    ema_state: dict[str, torch.Tensor] | None = None
    progress = trange(cfg.train.steps, dynamic_ncols=True, disable=not is_rank_zero(rank))
    for step in progress:
        lr = cosine_lr(step, base_lr=cfg.train.lr, total_steps=cfg.train.steps, warmup_steps=cfg.train.warmup_steps)
        set_lr(opt, lr)
        active = unwrap_model(model)
        active.cfg.temperature = linear_anneal(
            step,
            start=cfg.dvae.temperature_start,
            end=cfg.dvae.temperature_end,
            steps=cfg.dvae.temperature_anneal_steps,
        )
        active.cfg.kl_weight = linear_anneal(
            step,
            start=cfg.dvae.kl_weight_start,
            end=cfg.dvae.kl_weight_end,
            steps=cfg.dvae.kl_anneal_steps,
        )
        active.quantizer.temperature = active.cfg.temperature
        active.quantizer.kl_weight = active.cfg.kl_weight
        opt.zero_grad(set_to_none=True)
        accum = cfg.train.grad_accum_steps
        metric_sums = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        for micro_step in range(accum):
            images, _captions = next(loader)
            images = images.to(dev, non_blocking=True)
            sync_context = model.no_sync() if world_size > 1 and micro_step < accum - 1 else nullcontext()
            with sync_context:
                with autocast_context(dev, cfg.train.mixed_precision):
                    out = model(images)
                    loss = out["loss"] / accum
                loss.backward()
            for key in metric_sums:
                metric_sums[key] += reduce_mean(out[key]).item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if is_rank_zero(rank) and cfg.train.ema_decay < 1.0 and (step + 1) % cfg.train.ema_every == 0:
            ema_state = update_ema_state(ema_state, unwrap_model(model).state_dict(), decay=cfg.train.ema_decay)
        if is_rank_zero(rank) and step % cfg.train.log_every == 0:
            progress.set_description(
                f"loss={metric_sums['loss']:.4f} recon={metric_sums['recon_loss']:.4f} kl={metric_sums['kl_loss']:.4f}"
            )
            wandb_log(
                wandb_run,
                {
                    "train/loss": metric_sums["loss"],
                    "train/recon_loss": metric_sums["recon_loss"],
                    "train/kl_loss": metric_sums["kl_loss"],
                    "train/lr": lr,
                    "train/dvae_temperature": active.cfg.temperature,
                    "train/dvae_kl_weight": active.cfg.kl_weight,
                    "train/global_batch_size": cfg.data.batch_size * world_size * accum,
                },
                step=step,
            )
        if is_rank_zero(rank) and (step + 1) % cfg.train.save_every == 0:
            state = ema_state if ema_state is not None else unwrap_model(model).state_dict()
            save_checkpoint(
                out_dir / f"checkpoint-{step + 1}.safetensors",
                model=state,
                cfg=cfg,
                step=step + 1,
            )
            save_checkpoint(
                out_dir / "checkpoint-last.safetensors",
                model=state,
                cfg=cfg,
                step=step + 1,
            )
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


def linear_anneal(step: int, *, start: float, end: float, steps: int) -> float:
    if steps <= 0:
        return end
    alpha = min(1.0, step / steps)
    return start + alpha * (end - start)


def update_ema_state(
    ema_state: dict[str, torch.Tensor] | None,
    state: dict[str, Any],
    *,
    decay: float,
) -> dict[str, torch.Tensor]:
    state_cpu = {key: value.detach().cpu() for key, value in state.items() if torch.is_tensor(value)}
    if ema_state is None:
        return {key: value.clone() for key, value in state_cpu.items()}
    for key, value in state_cpu.items():
        ema_state[key].mul_(decay).add_(value, alpha=1 - decay)
    return ema_state


if __name__ == "__main__":
    main()
