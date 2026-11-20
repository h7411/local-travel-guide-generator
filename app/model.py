"""LoRA (Low-Rank Adaptation) and Language Model architecture for travel guide generation.

Implements:
- LoRALinear: Low-rank adapter linear layer wrapper
- inject_lora: Injects LoRA adapters into targeted linear layers
- merge_lora_weights: Folds adapter weights (alpha/r * B @ A) into base weights
- TravelTokenizer: Vocabulary and tokenization for travel prompts and guides
- MiniGPT: Compact, efficient GPT-2 style causal language model for fast training and inference
- train_lora_model: Training loop that optimizes only adapter parameters
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.dataset import TRAVEL_DATASET_RAW, TravelRecord, load_dataset


# ============================================================================
# LoRA Adapter Module
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA wrapper around an nn.Linear layer.

    Computes: h = W_0 x + (alpha / r) * B(A(x))
    where:
      W_0 is the frozen pre-trained base weight matrix
      A is initialized from N(0, 1/r)
      B is initialized to 0 (so Delta W = 0 initially)
      alpha / r is the scaling constant
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze base layer weights and bias
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank adapter matrices
        # A: (rank, in_features)
        # B: (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        # Initialize A with Kaiming uniform / Gaussian and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. If merged, uses the base layer directly."""
        base_out = self.base_layer(x)
        if self.merged or self.rank == 0:
            return base_out

        # Compute adapter contribution: (x @ A^T) @ B^T * scaling
        # x: [..., in_features]
        # x_dropout: [..., in_features]
        # intermediate: [..., rank]
        # adapter_out: [..., out_features]
        x_dropout = self.dropout(x)
        adapter_out = F.linear(F.linear(x_dropout, self.lora_A), self.lora_B) * self.scaling
        return base_out + adapter_out

    def merge(self) -> None:
        """Merge adapter weights directly into base weight matrix."""
        if self.merged:
            return
        # Delta W = scaling * (B @ A) of shape (out_features, in_features)
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        self.base_layer.weight.data.add_(delta_w)
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge adapter weights from base weight matrix."""
        if not self.merged:
            return
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        self.base_layer.weight.data.sub_(delta_w)
        self.merged = False

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"


# ============================================================================
# LoRA Injection & Merging Helpers
# ============================================================================

def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> dict[str, Any]:
    """Recursively replaces targeted linear layers with LoRALinear adapter layers.

    All original model weights are frozen. Only LoRA parameters (lora_A, lora_B)
    remain trainable.
    """
    # 1. Freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    targets = [m.lower() for m in target_modules] if target_modules else ["c_attn", "q_proj", "v_proj", "out_proj", "c_proj", "fc", "linear"]

    injected_count = 0

    def _replace_layers(module: nn.Module, parent_name: str = ""):
        nonlocal injected_count
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            # Check if this child matches target linear layer
            is_target = any(t in name.lower() or t in full_name.lower() for t in targets)
            if isinstance(child, nn.Linear) and is_target and not isinstance(child, LoRALinear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                injected_count += 1
            else:
                _replace_layers(child, full_name)

    _replace_layers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "status": "injected",
        "injected_layers_count": injected_count,
        "rank": rank,
        "alpha": alpha,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": round(trainable_pct, 4),
    }


def merge_lora_weights(model: nn.Module) -> int:
    """Merge all LoRALinear layers in model into their base weights."""
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged_count += 1
    return merged_count


def unmerge_lora_weights(model: nn.Module) -> int:
    """Unmerge all LoRALinear layers in model."""
    unmerged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            unmerged_count += 1
    return unmerged_count


# ============================================================================
# Tokenizer
# ============================================================================

class TravelTokenizer:
    """Vocabulary and tokenizer built specifically for travel guide prompts and completions."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, records: list[TravelRecord] | None = None) -> None:
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.token2id: dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id2token: dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}

        corpus_texts = []
        data = records if records is not None else load_dataset()
        for r in data:
            corpus_texts.append(r.to_prompt())
            corpus_texts.append(r.to_completion())

        # Build vocabulary from corpus
        tokens_set: set[str] = set()
        for text in corpus_texts:
            for token in self._tokenize_words(text):
                tokens_set.add(token.lower())

        for token in sorted(tokens_set):
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.bos_id = self.token2id[self.BOS_TOKEN]
        self.eos_id = self.token2id[self.EOS_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]

    def _tokenize_words(self, text: str) -> list[str]:
        # Keep punctuation as separate tokens
        pattern = r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b|[.,;:!?\->|]"
        return re.findall(pattern, text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = self._tokenize_words(text)
        ids = [self.token2id.get(t.lower(), self.unk_id) for t in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            token = self.id2token.get(tid, self.UNK_TOKEN)
            tokens.append(token)

        # Reconstruct natural text with correct spacing around punctuation
        text = ""
        for token in tokens:
            if token in {".", ",", "!", "?", ";", ":"}:
                text = text.rstrip() + token + " "
            elif token in {"-", "|", "->"}:
                text = text.rstrip() + f" {token} "
            else:
                text += token + " "
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)


# ============================================================================
# Compact GPT-Style Causal Language Model
# ============================================================================

class MultiHeadCausalAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 256) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Causal mask: lower triangular matrix
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard Transformer Decoder Block."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 256) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadCausalAttention(hidden_dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniCausalLM(nn.Module):
    """Lightweight Causal Language Model for LoRA fine-tuning and inference."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, max_seq_len) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if targets is not None:
                targets = targets[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shift so tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / max(temperature, 1e-4), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break

        return input_ids


# ============================================================================
# LoRA Training Loop
# ============================================================================

def train_lora_model(
    model: nn.Module,
    training_texts: list[str],
    tokenizer: TravelTokenizer,
    epochs: int = 3,
    lr: float = 1e-3,
    max_len: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fine-tune LoRA adapters on travel recommendation dataset.

    Ensures only adapter weights are trained, base weights stay frozen.
    """
    model.train()
    model.to(device)

    # Filter trainable parameters: strictly LoRA weights
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Did you inject LoRA adapters?")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Prepare batches
    encoded_samples = [
        tokenizer.encode(t, add_special_tokens=True)[:max_len] for t in training_texts
    ]
    loss_history: list[float] = []

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        random.shuffle(encoded_samples)

        for seq in encoded_samples:
            if len(seq) < 2:
                continue
            tensor_seq = torch.tensor([seq], dtype=torch.long, device=device)
            optimizer.zero_grad()
            _, loss = model(tensor_seq, targets=tensor_seq)
            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(round(avg_loss, 4))

    return {
        "status": "completed",
        "epochs": epochs,
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else 0.0,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "loss_reduced": loss_history[-1] < loss_history[0] if len(loss_history) > 1 else True,
        "trainable_params_count": sum(p.numel() for p in trainable_params),
    }
