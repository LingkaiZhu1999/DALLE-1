from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
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


@record
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dvae_small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    rank, world_size, _local_rank, dev = init_distributed()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    loader = iter(build_webdataset_loader(cfg.data, rank=rank, world_size=world_size))
    model = DiscreteVAE(cfg.dvae).to(dev, memory_format=torch.channels_last)
    if cfg.train.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[dev.index])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    out_dir = Path(cfg.train.output_dir)
    wandb_run = init_wandb(cfg, job_type="train_dvae") if is_rank_zero(rank) else None
    ema_state: dict[str, torch.Tensor] | None = None
    progress = trange(cfg.train.steps, dynamic_ncols=True, disable=not is_rank_zero(rank))
    last_log_time = time.perf_counter()
    last_log_step = 0
    for step in progress:
        lr = cosine_lr(
            step,
            base_lr=cfg.train.lr,
            min_lr=cfg.train.min_lr,
            total_steps=cfg.train.steps,
            decay_steps=cfg.train.lr_decay_steps,
            warmup_steps=cfg.train.warmup_steps,
        )
        set_lr(opt, lr)
        active = unwrap_model(model)
        active.cfg.temperature = cosine_anneal(
            step,
            start=cfg.dvae.temperature_start,
            end=cfg.dvae.temperature_end,
            steps=cfg.dvae.temperature_anneal_steps,
        )
        active.cfg.kl_weight = cosine_anneal(
            step,
            start=cfg.dvae.kl_weight_start,
            end=cfg.dvae.kl_weight_end,
            steps=cfg.dvae.kl_anneal_steps,
        )
        active.quantizer.temperature = active.cfg.temperature
        active.quantizer.kl_weight = active.cfg.kl_weight
        schedule = torch.tensor(
            [active.cfg.temperature, active.cfg.kl_weight],
            device=dev,
            dtype=torch.float32,
        )
        opt.zero_grad(set_to_none=True)
        accum = cfg.train.grad_accum_steps
        should_log = step % cfg.train.log_every == 0
        metric_sums = torch.zeros(7, device=dev) if should_log else None
        token_counts = torch.zeros(cfg.dvae.codebook_size, device=dev) if should_log else None
        for micro_step in range(accum):
            images, _captions = next(loader)
            images = images.to(dev, non_blocking=True, memory_format=torch.channels_last)
            sync_context = model.no_sync() if world_size > 1 and micro_step < accum - 1 else nullcontext()
            with sync_context:
                with autocast_context(dev, cfg.train.mixed_precision):
                    out = model(images, schedule, False, should_log)
                    loss = out["loss"] / accum
                loss.backward()
            if metric_sums is not None:
                metric_sums += torch.stack([
                    out["loss"].detach(),
                    out["recon_loss"],
                    out["kl_loss"],
                    out["unweighted_kl"],
                    out["posterior_entropy"],
                    out["logit_rms"],
                    out["logit_abs_max"],
                ]) / accum
                token_counts += out["token_counts"]
        if cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if is_rank_zero(rank) and cfg.train.ema_decay < 1.0 and (step + 1) % cfg.train.ema_every == 0:
            ema_state = update_ema_state(ema_state, unwrap_model(model).state_dict(), decay=cfg.train.ema_decay)
        if should_log:
            (
                loss_value,
                recon_value,
                kl_value,
                raw_kl,
                posterior_entropy,
                logit_rms,
                logit_abs_max,
            ) = reduce_mean(metric_sums).tolist()
            if world_size > 1:
                dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)
            token_probs = token_counts / token_counts.sum().clamp_min(1)
            aggregate_entropy = -(token_probs * token_probs.clamp_min(1e-12).log()).sum()
            code_usage = (token_counts > 0).float().mean().item()
            code_perplexity = aggregate_entropy.exp().item()
            top_code_share = token_probs.max().item()
            now = time.perf_counter()
            steps_since_log = step - last_log_step + 1
            images_per_second = cfg.data.batch_size * world_size * accum * steps_since_log / (now - last_log_time)
            last_log_time = now
            last_log_step = step + 1
        if is_rank_zero(rank) and should_log:
            progress.set_description(
                f"loss={loss_value:.4f} recon={recon_value:.4f} kl={kl_value:.4f} img/s={images_per_second:.1f}"
            )
            wandb_log(
                wandb_run,
                {
                    "train/loss": loss_value,
                    "train/recon_loss": recon_value,
                    "train/kl_loss": kl_value,
                    "train/unweighted_kl": raw_kl,
                    "train/posterior_entropy": posterior_entropy,
                    "train/posterior_perplexity": math.exp(posterior_entropy),
                    "train/code_usage_fraction": code_usage,
                    "train/code_perplexity": code_perplexity,
                    "train/top_code_share": top_code_share,
                    "train/encoder_logit_rms": logit_rms,
                    "train/encoder_logit_abs_max": logit_abs_max,
                    "train/lr": lr,
                    "train/dvae_temperature": active.cfg.temperature,
                    "train/dvae_kl_weight": active.cfg.kl_weight,
                    "train/global_batch_size": cfg.data.batch_size * world_size * accum,
                    "train/images_per_second": images_per_second,
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


def cosine_anneal(step: int, *, start: float, end: float, steps: int) -> float:
    if steps <= 0:
        return end
    progress = min(1.0, step / steps)
    alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
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
