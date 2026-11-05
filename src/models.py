"""Language models that differ only in the attention layer.

The skeleton (pre-LN block, ReLU FFN, LayerNorm, untied lm_head) is the paper's
WaveLLM. Attention variants:

  softmax  standard causal multi-head attention with RoPE (the baseline)
  cvqiaa   MultiHeadUnifiedQuantumAttention from the paper, as published
           (src/wavellm_fixed.py, only K.mtheH -> K.mH), with the paper's
           positions linspace(0, 0.1 * (T - 1), T)
  born     the baseline with softmax replaced by the paper's Born-rule
           normalisation |s|^2 / sum |s|^2; isolates that one idea
  none     attention output is zero: the model sees only the current token,
           a floor that shows how much a variant gets from context
  cvqiaa2  AnalyticalCVQIAALayer, the grid-free version the authors sent after
           the review (src/paper_v2_replacement.py, as published); it takes
           positions from token indices itself
  cvqiaa2-noenv  the same layer with the wave-packet envelope switched off
           (sigma0 fixed at e^20, so the envelope is sqrt(2) for every pair):
           shows what the envelope adds
  cvqiaa2-wide  the same layer with sigma0 initialised at 40 instead of 2
  softmax-env  softmax attention without RoPE that takes the v2 envelope as
           its positional part (log envelope^2 added to the scores, learnable
           sigma0 per head, init 2): differs from cvqiaa2 only in softmax vs
           the Born rule (and real vs complex q, k)
  softmax-env-wide  the same with sigma0 initialised at 40, the control for
           cvqiaa2-wide
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from paper_v2_replacement import AnalyticalCVQIAALayer  # noqa: E402
from wavellm_fixed import MultiHeadUnifiedQuantumAttention  # noqa: E402


def rope(x, base=10000.0):
    # x: [B, H, T, D], D even
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.arange(T, device=x.device).float()[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, normalisation="softmax", epsilon=1e-9):
        super().__init__()
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.normalisation = normalisation
        self.epsilon = epsilon

    def forward(self, x, positions=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        if self.normalisation == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
            w = s ** 2 * torch.tril(torch.ones(T, T, device=x.device))
            w = w / (w.sum(-1, keepdim=True) + self.epsilon)
            y = w @ v
        return self.out(y.transpose(1, 2).reshape(B, T, C))


def log_envelope_sq(T, log_sigma0):
    # log of envelope^2 from AnalyticalCVQIAALayer (without its 1e-9 guards), shape [H, T, T]
    pos = torch.arange(T, device=log_sigma0.device, dtype=torch.float32)
    dt = (pos[:, None] - pos[None, :]).clamp(min=0)
    s0_sq = (2 * log_sigma0).exp().view(-1, 1, 1)
    st_sq = s0_sq + dt ** 2 / s0_sq
    return torch.log(2 * s0_sq / st_sq) - dt ** 2 / st_sq


class EnvelopeSoftmax(StandardAttention):
    def __init__(self, d_model, n_heads, sigma0=2.0):
        super().__init__(d_model, n_heads)
        self.log_sigma0 = nn.Parameter(torch.log(torch.ones(n_heads) * sigma0))

    def forward(self, x, positions=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        future = ~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        bias = log_envelope_sq(T, self.log_sigma0).masked_fill(future, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class Analytical(nn.Module):
    def __init__(self, d_model, n_heads, envelope=True, sigma0=2.0):
        super().__init__()
        self.layer = AnalyticalCVQIAALayer(d_model, n_heads)
        if sigma0 != 2.0:
            self.layer.log_sigma0.data.fill_(math.log(sigma0))
        if not envelope:
            self.layer.log_sigma0.data.fill_(20.0)
            self.layer.log_sigma0.requires_grad_(False)

    def forward(self, x, positions=None):
        return self.layer(x)


class NoAttention(nn.Module):
    def forward(self, x, positions=None):
        return torch.zeros_like(x)


class Block(nn.Module):
    def __init__(self, kind, d_model, n_heads, d_ff):
        super().__init__()
        if kind == "cvqiaa":
            self.attn = MultiHeadUnifiedQuantumAttention(d_model, n_heads)
        elif kind in ("cvqiaa2", "cvqiaa2-noenv", "cvqiaa2-wide"):
            self.attn = Analytical(d_model, n_heads, envelope=kind != "cvqiaa2-noenv",
                                   sigma0=40.0 if kind == "cvqiaa2-wide" else 2.0)
        elif kind in ("softmax-env", "softmax-env-wide"):
            self.attn = EnvelopeSoftmax(d_model, n_heads, sigma0=40.0 if kind == "softmax-env-wide" else 2.0)
        elif kind == "none":
            self.attn = NoAttention()
        else:
            self.attn = StandardAttention(d_model, n_heads, normalisation=kind)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, positions):
        x = x + self.attn(self.ln1(x), positions)
        x = x + self.ffn(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self, kind, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=4):
        super().__init__()
        self.kind = kind
        self.tok = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Block(kind, d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        # The paper's WaveLLM positions; the standard variants ignore them and use RoPE.
        positions = torch.linspace(0.0, float(T - 1) * 0.1, T, device=idx.device).expand(B, T)
        x = self.tok(idx)
        for layer in self.layers:
            x = layer(x, positions)
        return self.lm_head(self.ln_final(x))

    def attention_params(self):
        return sum(p.numel() for l in self.layers for p in l.attn.parameters())
