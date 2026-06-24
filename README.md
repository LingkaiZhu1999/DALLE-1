# DALL-E 1 Reproduction

This repository reproduces the complete DALL-E 1 training stack in modern PyTorch:

1. Train a discrete image tokenizer (`dVAE`) with a Gumbel-softmax relaxed ELB, logit-Laplace reconstruction likelihood, and DALL-E-style image preprocessing that maps images to a grid of codebook ids.
2. Train an autoregressive transformer over a single stream of text tokens followed by image tokens.
3. Generate images by sampling image tokens conditioned on text and decoding them with the dVAE.

The implementation follows the DALL-E paper, "Zero-Shot Text-to-Image Generation" by Ramesh et al., which describes autoregressively modeling text and image tokens in one stream. The official OpenAI repository only released the discrete VAE package and explicitly did not include the text-to-image transformer, so this project implements that missing stage as well.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,logging]'
```

## Dataset

The default config points at the local LAION-400M mirror:

```text
/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data
```

The loader expects `img2dataset`/WebDataset shards containing `jpg`, `txt`, and optional `json` files.

## Smoke Test

```bash
pytest
```

## Train The Image Tokenizer

```bash
dalle1-train-dvae --config configs/dvae_small.yaml
```

The dVAE checkpoint contains the encoder, quantizer, and decoder. The large 256px config follows the Appendix A setup: four encoder/decoder groups, bottleneck residual blocks with small residual gain, 8192 categorical codes, 32x32 image-token grids, KL and temperature schedules, AdamW, gradient clipping, and exponentially weighted iterate averaging.

`configs/dvae_h200_256.yaml` uses per-GPU batch size 96 on H200s, matching the large-experiment cluster note in `guideline.md`.

For distributed 256px dVAE training on 8 H200 GPUs, use:

```bash
sbatch scripts/slurm_dvae_h200_256.sh
```

That distributed dVAE job trains with global batch size `96 * 8 = 768`.

## Train The Transformer

```bash
dalle1-train-transformer \
  --config configs/transformer_small.yaml \
  --dvae-checkpoint runs/dvae/checkpoint-last.safetensors
```

For distributed 256px training on 8 H200 GPUs, use:

```bash
sbatch scripts/slurm_transformer_h200_256.sh
```

`configs/transformer_h200_256.yaml` uses per-GPU batch size 4 and 32 gradient accumulation steps, so a 2-node x 4-GPU job trains with global batch size `4 * 8 * 32 = 1024`.
For Appendix B data handling, transformer training uses argmax dVAE image tokens, transformer-specific image augmentation without horizontal flips, lowercased captions, optional BPE dropout, and separate text/image loss weights.

## Generate

```bash
dalle1-generate \
  --dvae-checkpoint runs/dvae/checkpoint-last.safetensors \
  --transformer-checkpoint runs/transformer/checkpoint-last.safetensors \
  --prompt "a stained glass window of a blue robot reading a book" \
  --out samples/robot.png
```

The small configs are intentionally modest so the pipeline can be verified quickly. For a closer DALL-E 1 scale, increase dVAE codebook size toward 8192, image token grid toward `32 x 32`, text length toward 256, and transformer depth/width substantially.

`configs/dalle1_256.yaml` is a full-resolution reference config: 256px images, 8192 image codes, 256 text tokens, and a 32x32 image-token grid. It is still smaller than OpenAI's unreleased production transformer, but it preserves the core architecture and training objective.
