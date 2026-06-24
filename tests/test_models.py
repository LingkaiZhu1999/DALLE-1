import torch

from dalle1.config import DalleTransformerConfig, DvaeConfig
from dalle1.dvae import DiscreteVAE
from dalle1.transformer import DalleTransformer


def test_dvae_roundtrip_shapes():
    cfg = DvaeConfig(
        image_size=32,
        hidden_channels=16,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        codebook_size=32,
        code_dim=32,
    )
    model = DiscreteVAE(cfg)
    images = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    out = model(images)
    assert out["recon"].shape == images.shape
    assert out["ids"].shape == (2, 8, 8)
    assert out["kl_loss"].ndim == 0
    decoded = model.decode_tokens(out["ids"])
    assert decoded.shape == images.shape


def test_transformer_forward_and_sample():
    cfg = DalleTransformerConfig(
        text_vocab_size=128,
        image_vocab_size=32,
        text_seq_len=8,
        image_tokens_per_side=4,
        dim=64,
        depth=2,
        heads=4,
        dropout=0.0,
    )
    model = DalleTransformer(cfg)
    text = torch.randint(0, cfg.text_vocab_size, (2, cfg.text_seq_len))
    image = torch.randint(0, cfg.image_vocab_size, (2, cfg.image_seq_len))
    out = model(text, image)
    assert out["loss"].ndim == 0
    sampled = model.sample(text[:1], steps=cfg.image_seq_len, top_k=8)
    assert sampled.shape == (1, cfg.image_tokens_per_side, cfg.image_tokens_per_side)
