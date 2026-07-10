#!/usr/bin/env python3
"""Small CIFAR-10 experiment for validating the stage-1 discrete VAE.

The experiment deliberately uses a compact model and a short training run.  It is
intended as a functional/overfitting check, not as a competitive CIFAR-10 model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import trange

from dalle1.config import DvaeConfig
from dalle1.dvae import DiscreteVAE
from dalle1.train_dvae import linear_anneal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cifar10_dvae"))
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--eval-examples", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    # The dVAE expects pixels in [-1, 1]. Horizontal flips are the only useful
    # DALL-E-style augmentation here because CIFAR-10 images are already square.
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(2).sub(1))]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(2).sub(1))]
    )
    train_set = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=args.download)
    eval_set = datasets.CIFAR10(args.data_dir, train=False, transform=eval_transform, download=args.download)
    generator = torch.Generator().manual_seed(args.seed)
    train_ids = torch.randperm(len(train_set), generator=generator)[: args.train_examples].tolist()
    eval_ids = torch.randperm(len(eval_set), generator=generator)[: args.eval_examples].tolist()
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(train_set, train_ids), shuffle=True, drop_last=True, **common)
    eval_loader = DataLoader(Subset(eval_set, eval_ids), shuffle=False, **common)
    return train_loader, eval_loader


@torch.no_grad()
def evaluate(model: DiscreteVAE, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    squared_error = 0.0
    pixels = 0
    token_counts = torch.zeros(model.cfg.codebook_size, dtype=torch.long, device=device)
    preview = None
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        ids = model.encode(images)
        recon = model.decode_tokens(ids)
        squared_error += (recon - images).square().sum().item()
        pixels += images.numel()
        token_counts += torch.bincount(ids.flatten(), minlength=model.cfg.codebook_size)
        if preview is None:
            count = min(16, images.shape[0])
            preview = torch.cat((images[:count], recon[:count])).add(1).div(2).clamp(0, 1).cpu()
    mse = squared_error / pixels
    used = int((token_counts > 0).sum().item())
    probs = token_counts.float() / token_counts.sum().clamp_min(1)
    perplexity = torch.exp(-(probs[probs > 0] * probs[probs > 0].log()).sum()).item()
    metrics = {
        "mse_0_1": mse / 4.0,
        "psnr_db": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
        "codes_used": used,
        "codebook_size": model.cfg.codebook_size,
        "codebook_usage_fraction": used / model.cfg.codebook_size,
        "code_perplexity": perplexity,
    }
    assert preview is not None
    return metrics, preview


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.train_examples < args.batch_size or args.eval_examples < 1:
        raise ValueError("steps and eval-examples must be positive; train-examples must be at least batch-size")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, eval_loader = make_loaders(args)

    cfg = DvaeConfig(
        image_size=32,
        hidden_channels=64,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        codebook_size=256,
        code_dim=256,
        kl_weight_start=0.0,
        kl_weight_end=0.1,
        kl_anneal_steps=max(1, args.steps // 2),
        temperature_start=1.0,
        temperature_end=1.0 / 16.0,
        temperature_anneal_steps=args.steps,
    )
    model = DiscreteVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    iterator = iter(train_loader)
    progress = trange(args.steps, dynamic_ncols=True)
    model.train()
    for step in progress:
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        temperature = linear_anneal(
            step, start=cfg.temperature_start, end=cfg.temperature_end, steps=cfg.temperature_anneal_steps
        )
        kl_weight = linear_anneal(step, start=cfg.kl_weight_start, end=cfg.kl_weight_end, steps=cfg.kl_anneal_steps)
        schedule = torch.tensor([temperature, kl_weight], device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images, schedule, False)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step + 1 == args.steps:
            progress.set_description(
                f"loss={output['loss'].item():.3f} recon={output['recon_loss'].item():.3f} "
                f"kl={output['kl_loss'].item():.3f}"
            )

    metrics, preview = evaluate(model, eval_loader, device)
    torch.save({"model": model.state_dict(), "config": vars(cfg), "step": args.steps}, args.output_dir / "model.pt")
    utils.save_image(preview, args.output_dir / "reconstructions.png", nrow=min(16, preview.shape[0] // 2))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
