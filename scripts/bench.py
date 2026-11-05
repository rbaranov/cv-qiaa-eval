"""Memory and time of one attention layer by context length.

Memory: bytes of tensors autograd saves for backward, counted with
saved_tensors_hooks on CPU (exact and independent of the device's allocator).
Each storage is counted once: autograd can save one buffer under several
shapes (naive softmax saves its output as [B, H, T, T] and as [B*H, T, T]);
counting by shape, as the first version of this script did, counted it twice.
Peak: the largest temporary tensor CV-QIAA builds in the forward pass,
B x T x T x grid_size float32, computed from shapes.
Time: forward + backward on MPS (Apple GPU), mean of 3 runs after a warm-up.

Variants: sdpa (PyTorch scaled_dot_product_attention, the training baseline),
naive softmax (explicit T x T matrix, masked_fill + softmax), born (softmax
replaced by |s|^2 / sum |s|^2), cvqiaa (the paper's layer), cvqiaa2 (the
authors' grid-free version 2, AnalyticalCVQIAALayer).
"""
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import StandardAttention, rope  # noqa: E402
from paper_v2_replacement import AnalyticalCVQIAALayer  # noqa: E402
from wavellm_fixed import MultiHeadUnifiedQuantumAttention  # noqa: E402


class NaiveSoftmax(StandardAttention):
    def forward(self, x, positions=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)), float("-inf"))
        return self.out((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, C))


class V2(AnalyticalCVQIAALayer):
    def forward(self, x, positions=None):
        return super().forward(x)


def make(kind):
    if kind == "cvqiaa2":
        return V2(D, H)
    if kind == "cvqiaa":
        return MultiHeadUnifiedQuantumAttention(D, H)
    if kind == "naive softmax":
        return NaiveSoftmax(D, H)
    return StandardAttention(D, H, normalisation="softmax" if kind == "sdpa" else kind)


def saved_mb(layer, x, pos):
    seen = {}

    def pack(t):
        storage = t.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        layer(x, pos)
    return sum(seen.values()) / 2 ** 20


B, D, H, GRID = 4, 128, 4, 64
KINDS = ["sdpa", "naive softmax", "born", "cvqiaa", "cvqiaa2"]
rows = []
for T in [256, 512, 1024]:
    for kind in KINDS:
        torch.manual_seed(0)
        layer = make(kind)
        x = torch.randn(B, T, D, requires_grad=True)
        pos = torch.linspace(0.0, (T - 1) * 0.1, T).expand(B, T)
        row = {"T": T, "kind": kind, "saved_mb": saved_mb(layer, x, pos)}
        if kind == "cvqiaa":
            row["largest_temporary_mb"] = B * T * T * GRID * 4 / 2 ** 20
        layer, x, pos = layer.to("mps"), x.detach().to("mps").requires_grad_(), pos.to("mps")
        times = []
        for i in range(4):
            torch.mps.synchronize()
            t = time.time()
            layer(x, pos).sum().backward()
            torch.mps.synchronize()
            if i:
                times.append(time.time() - t)
        row["ms_mps"] = 1000 * sum(times) / len(times)
        rows.append(row)
        print(row, flush=True)
        torch.mps.empty_cache()

with open(os.path.join(ROOT, "results", "bench.json"), "w") as f:
    json.dump({"batch": B, "d_model": D, "heads": H, "grid_size": GRID, "rows": rows}, f, indent=1)
