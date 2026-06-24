from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange

from .config import load_config
from .data import build_webdataset_loader
from .dvae import DiscreteVAE
from .tokenizer import TextTokenizer
from .transformer import DalleTransformer
from .utils import (
    autocast_context,
    cleanup_distributed,
    cosine_lr,
    init_wandb,
    init_distributed,
    is_rank_zero,
    load_checkpoint,
    reduce_mean,
    save_checkpoint,
    seed_everything,
    set_lr,
    unwrap_model,
    wandb_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer_small.yaml")
    parser.add_argument("--dvae-checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    rank, world_size, _local_rank, dev = init_distributed()
    loader = iter(build_webdataset_loader(cfg.data, rank=rank, world_size=world_size))
    dvae = DiscreteVAE(cfg.dvae).to(dev).eval()
    dvae.load_state_dict(load_checkpoint(args.dvae_checkpoint)["model"], strict=True)
    for param in dvae.parameters():
        param.requires_grad_(False)
    tokenizer = TextTokenizer(
        max_length=cfg.transformer.text_seq_len,
        vocab_size=cfg.transformer.text_vocab_size,
        lowercase=cfg.transformer.text_lowercase,
        bpe_dropout=cfg.transformer.bpe_dropout,
    )
    model = DalleTransformer(cfg.transformer).to(dev)
    if cfg.train.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[dev.index])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    out_dir = Path(cfg.train.output_dir)
    wandb_run = init_wandb(cfg, job_type="train_transformer") if is_rank_zero(rank) else None
    progress = trange(cfg.train.steps, dynamic_ncols=True, disable=not is_rank_zero(rank))
    for step in progress:
        lr = cosine_lr(step, base_lr=cfg.train.lr, total_steps=cfg.train.steps, warmup_steps=cfg.train.warmup_steps)
        set_lr(opt, lr)
        opt.zero_grad(set_to_none=True)
        accum = cfg.train.grad_accum_steps
        metric_sums = {"loss": 0.0, "image_loss": 0.0, "text_loss": 0.0}
        for micro_step in range(accum):
            images, captions = next(loader)
            images = images.to(dev, non_blocking=True)
            text = tokenizer.encode(list(captions)).to(dev, non_blocking=True).clamp_max(cfg.transformer.text_vocab_size - 1)
            with torch.no_grad():
                image_tokens = dvae.encode(images).flatten(1)
            sync_context = model.no_sync() if world_size > 1 and micro_step < accum - 1 else nullcontext()
            with sync_context:
                with autocast_context(dev, cfg.train.mixed_precision):
                    out = model(text, image_tokens)
                    loss = out["loss"] / accum
                loss.backward()
            for key in metric_sums:
                metric_sums[key] += reduce_mean(out[key]).item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if is_rank_zero(rank) and step % cfg.train.log_every == 0:
            progress.set_description(
                f"loss={metric_sums['loss']:.4f} image={metric_sums['image_loss']:.4f} text={metric_sums['text_loss']:.4f}"
            )
            wandb_log(
                wandb_run,
                {
                    "train/loss": metric_sums["loss"],
                    "train/image_loss": metric_sums["image_loss"],
                    "train/text_loss": metric_sums["text_loss"],
                    "train/lr": lr,
                    "train/global_batch_size": cfg.data.batch_size * world_size * accum,
                },
                step=step,
            )
        if is_rank_zero(rank) and (step + 1) % cfg.train.save_every == 0:
            payload = {"model": unwrap_model(model).state_dict(), "cfg": cfg, "step": step + 1}
            save_checkpoint(out_dir / f"checkpoint-{step + 1}.safetensors", **payload)
            save_checkpoint(out_dir / "checkpoint-last.safetensors", **payload)
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
