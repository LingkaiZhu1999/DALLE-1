import torch
import torch.nn.functional as F

from dalle1.config import DalleTransformerConfig, DvaeConfig
from dalle1.dvae import DiscreteVAE, map_pixels, unmap_pixels
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


def test_decoder_relaxed_projection_matches_dense_conv():
    cfg = DvaeConfig(
        image_size=16,
        hidden_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        codebook_size=16,
        code_dim=16,
    )
    model = DiscreteVAE(cfg).eval()
    logits = torch.randn(2, 4, 4, cfg.codebook_size)
    weights = F.softmax(logits, dim=-1)

    projected = model.decoder(weights)
    dense = model.decoder.net(weights.permute(0, 3, 1, 2).contiguous())

    torch.testing.assert_close(projected, dense)


def test_dvae_kl_matches_image_space_normalization():
    cfg = DvaeConfig(
        image_size=32,
        in_channels=3,
        hidden_channels=8,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        codebook_size=32,
        code_dim=32,
    )
    model = DiscreteVAE(cfg).eval()
    logits = torch.full((2, cfg.codebook_size, 8, 8), -100.0)
    logits[:, 0] = 100.0
    beta = 6.6

    _, _, kl_loss, _ = model.quantizer(
        logits,
        schedule=torch.tensor([1.0, beta]),
        return_ids=False,
    )

    # A deterministic categorical posterior has KL(q || uniform) = ln(K).
    # The 4x spatial downsampling gives 3 * 4**2 = 48 image values per token.
    expected = beta * torch.log(torch.tensor(float(cfg.codebook_size))) / 48
    torch.testing.assert_close(kl_loss, expected)


def test_pixel_mapping_is_affine_and_invertible():
    images = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    mapped = map_pixels(images, eps=0.1)
    torch.testing.assert_close(mapped, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
    torch.testing.assert_close(unmap_pixels(mapped, eps=0.1), images)


def test_dvae_diagnostics_measure_code_usage():
    cfg = DvaeConfig(
        image_size=8,
        in_channels=3,
        hidden_channels=8,
        channel_multipliers=(1,),
        num_res_blocks=1,
        codebook_size=4,
        code_dim=4,
    )
    model = DiscreteVAE(cfg).eval()
    logits = torch.full((1, 4, 2, 2), -100.0)
    logits[:, 0, 0, :] = 100.0
    logits[:, 1, 1, :] = 100.0

    _, _, _, diagnostics = model.quantizer(logits, return_ids=False, return_diagnostics=True)

    assert diagnostics["token_counts"].tolist() == [2, 2, 0, 0]
    torch.testing.assert_close(diagnostics["posterior_entropy"], torch.tensor(0.0))


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
