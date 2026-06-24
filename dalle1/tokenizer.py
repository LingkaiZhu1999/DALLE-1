from __future__ import annotations

import hashlib

import torch


class TextTokenizer:
    """Thin wrapper around Hugging Face tokenizers with a deterministic fallback."""

    def __init__(
        self,
        name: str = "gpt2",
        max_length: int = 256,
        vocab_size: int = 49_408,
        *,
        lowercase: bool = True,
        bpe_dropout: float = 0.0,
    ):
        self.max_length = max_length
        self.fallback_vocab_size = vocab_size
        self.lowercase = lowercase
        self.bpe_dropout = bpe_dropout
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if bpe_dropout > 0 and hasattr(tokenizer, "backend_tokenizer"):
                model = tokenizer.backend_tokenizer.model
                if hasattr(model, "dropout"):
                    model.dropout = bpe_dropout
            self.tokenizer = tokenizer
            self.pad_id = int(tokenizer.pad_token_id)
            self.vocab_size = int(len(tokenizer))
        except Exception:
            self.pad_id = 0
            self.vocab_size = vocab_size

    def encode(self, captions: list[str]) -> torch.Tensor:
        if self.lowercase:
            captions = [caption.lower() for caption in captions]
        if self.tokenizer is not None:
            out = self.tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return out.input_ids.long()
        return torch.stack([self._fallback_encode(caption) for caption in captions], dim=0)

    def _fallback_encode(self, caption: str) -> torch.Tensor:
        ids = [1]
        for word in caption.split():
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
            ids.append(2 + int.from_bytes(digest, "little") % (self.fallback_vocab_size - 2))
        ids = ids[: self.max_length]
        ids += [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)
