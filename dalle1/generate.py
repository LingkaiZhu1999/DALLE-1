from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .dvae import DiscreteVAE
from .tokenizer import TextTokenizer
from .transformer import DalleTransformer
from .utils import device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvae-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="sample.png")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()
    dev = device()
    t_ckpt = load_checkpoint(args.transformer_checkpoint)
    cfg = t_ckpt["cfg"]
    dvae = DiscreteVAE(cfg.dvae).to(dev).eval()
    dvae.load_state_dict(load_checkpoint(args.dvae_checkpoint)["model"], strict=True)
    transformer = DalleTransformer(cfg.transformer).to(dev).eval()
    transformer.load_state_dict(t_ckpt["model"], strict=True)
    tokenizer = TextTokenizer(
        max_length=cfg.transformer.text_seq_len,
        vocab_size=cfg.transformer.text_vocab_size,
        lowercase=cfg.transformer.text_lowercase,
        bpe_dropout=0.0,
    )
    text = tokenizer.encode([args.prompt]).to(dev)
    ids = transformer.sample(text, temperature=args.temperature, top_k=args.top_k)
    image = dvae.decode_tokens(ids).add(1).div(2).clamp(0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(image, args.out)


if __name__ == "__main__":
    main()
