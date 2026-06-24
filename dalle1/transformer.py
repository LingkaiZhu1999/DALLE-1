from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import DalleTransformerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(b, n, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DalleTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg.dim, cfg.heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DalleTransformer(nn.Module):
    def __init__(self, cfg: DalleTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.total_vocab = cfg.text_vocab_size + cfg.image_vocab_size
        self.text_embed = nn.Embedding(cfg.text_vocab_size, cfg.dim)
        self.image_embed = nn.Embedding(cfg.image_vocab_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.dim) * 0.02)
        self.mod_embed = nn.Parameter(torch.randn(2, cfg.dim) * 0.02)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.depth)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.to_logits = nn.Linear(cfg.dim, self.total_vocab, bias=False)

    def _embed(self, text: torch.Tensor, image: torch.Tensor | None) -> torch.Tensor:
        text_emb = self.text_embed(text) + self.mod_embed[0]
        if image is None:
            return text_emb
        image_emb = self.image_embed(image) + self.mod_embed[1]
        return torch.cat([text_emb, image_emb], dim=1)

    def forward(
        self, text: torch.Tensor, image: torch.Tensor | None = None, labels: bool = True
    ) -> dict[str, torch.Tensor]:
        x = self._embed(text, image)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.to_logits(self.ln(x))
        out = {"logits": logits}
        if image is not None and labels:
            targets = torch.cat([text, image + self.cfg.text_vocab_size], dim=1)
            pred = logits[:, :-1].reshape(-1, self.total_vocab)
            gold = targets[:, 1:].reshape(-1)
            loss_all = F.cross_entropy(pred, gold, reduction="none").view(targets.shape[0], -1)
            text_positions = max(0, self.cfg.text_seq_len - 1)
            text_loss = loss_all[:, :text_positions].mean() if text_positions else loss_all.new_tensor(0)
            image_loss = loss_all[:, text_positions:].mean()
            out["loss"] = self.cfg.image_loss_weight * image_loss + self.cfg.text_loss_weight * text_loss
            out["text_loss"] = text_loss.detach()
            out["image_loss"] = image_loss.detach()
        return out

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        *,
        steps: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = 256,
    ) -> torch.Tensor:
        self.eval()
        steps = steps or self.cfg.image_seq_len
        image = torch.empty(text.shape[0], 0, dtype=torch.long, device=text.device)
        for _ in range(steps):
            logits = self(text, image, labels=False)["logits"][:, -1, self.cfg.text_vocab_size :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                kth = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, kth, dim=-1)
                logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            image = torch.cat([image, next_token], dim=1)
        side = int(steps**0.5)
        return image.view(text.shape[0], side, side)
