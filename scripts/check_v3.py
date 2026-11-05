"""Checks of version 3 (the authors' final document of 5 November).

1. The v3 layer computes the same as v2: same weights, same input, outputs
   compared exactly.
2. Narrow and wide heads. The text of section 1.1 says a narrow packet spreads
   into an almost uniform profile and a wide one stays local. The weights are
   taken from the layer itself with equal scores for all pairs, so only the
   distance factor E(d)^2 is left; the last row of a 128-token context is
   reported. For comparison, the same with the v1 factor exp(-d^2 / 4 sigma_t^2).
3. Exact zeros. Section 2 claims sparsity from exact phase cancellation.
   Counts causal pairs whose weight is exactly 0 for a freshly initialised
   layer on random input.
4. Scalar phase rotation e^{i w s} (section 1.2, absent in the code). With
   one w for all dimensions the Born rule erases it; with a frequency per
   dimension, as in RoPE, it does not.

The attention weights are captured from the authors' forward pass: its last
torch.matmul is A_normalized @ V.
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paper_v2_replacement import AnalyticalCVQIAALayer as V2Layer  # noqa: E402
from paper_v3_layer import AnalyticalCVQIAALayer as V3Layer  # noqa: E402

D_MODEL, N_HEADS = 128, 4


def attention_weights(layer, x):
    """Run the authors' forward and return A_normalized [B, H, T, T]."""
    calls = []
    matmul = torch.matmul

    def recording(a, b):
        calls.append(a)
        return matmul(a, b)

    torch.matmul = recording
    try:
        layer(x)
    finally:
        torch.matmul = matmul
    return calls[-1]


def same_as_v2():
    torch.manual_seed(0)
    v2 = V2Layer(D_MODEL, N_HEADS)
    v3 = V3Layer(D_MODEL, N_HEADS)
    v3.load_state_dict(v2.state_dict())
    x = torch.randn(4, 128, D_MODEL)
    with torch.no_grad():
        return (v2(x) - v3(x)).abs().max().item()


def positional_weights(sigma0, T=128):
    """Weights of the last token when every pair has the same score."""
    layer = V3Layer(D_MODEL, N_HEADS)
    with torch.no_grad():
        for proj, bias in ((layer.q_real, 1.0), (layer.k_real, 1.0), (layer.q_imag, 0.0), (layer.k_imag, 0.0)):
            proj.weight.zero_()
            proj.bias.fill_(bias)
        layer.log_sigma0.fill_(torch.log(torch.tensor(sigma0)).item())
        w = attention_weights(layer, torch.zeros(1, T, D_MODEL))[0, 0, -1]
    d = torch.arange(T - 1, -1, -1)
    return {"own_token": w[-1].item(), "d_ge_32": w[d >= 32].sum().item()}


def v1_positional_weights(sigma0, T=128):
    d = torch.arange(T, dtype=torch.float64)
    st_sq = sigma0 ** 2 + d ** 2 / sigma0 ** 2
    w = torch.exp(-d ** 2 / (4 * st_sq))
    w = w / w.sum()
    return {"own_token": w[0].item(), "d_ge_32": w[32:].sum().item()}


def exact_zeros(B=20, T=256):
    torch.manual_seed(0)
    layer = V3Layer(D_MODEL, N_HEADS)
    with torch.no_grad():
        A = attention_weights(layer, torch.randn(B, T, D_MODEL))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).expand_as(A)
    return {"causal_pairs": int(causal.sum()), "exact_zeros": int((A[causal] == 0).sum()),
            "min_weight": A[causal].min().item()}


def phase_rotation(T=64, d_head=32, w=0.3):
    torch.manual_seed(0)
    q = torch.randn(T, d_head, dtype=torch.cfloat)
    k = torch.randn(T, d_head, dtype=torch.cfloat)
    pos = torch.arange(T, dtype=torch.float32)
    born = lambda q, k: (q @ k.conj().T).abs() ** 2  # noqa: E731
    base = born(q, k)
    one_w = torch.exp(1j * w * pos)[:, None]
    per_dim = torch.exp(1j * pos[:, None] * w ** torch.arange(1, d_head + 1, dtype=torch.float32) ** 0.5)
    rel = lambda a: ((a - base).abs().max() / base.abs().max()).item()  # noqa: E731
    return {"one_frequency": rel(born(q * one_w, k * one_w)),
            "frequency_per_dimension": rel(born(q * per_dim, k * per_dim))}


def main():
    res = {
        "max_abs_diff_v3_vs_v2": same_as_v2(),
        "positional_weights_v3": {s: positional_weights(s) for s in (0.3, 2.0, 50.0)},
        "positional_weights_v1_formula": {s: v1_positional_weights(s) for s in (0.3, 2.0, 50.0)},
        "exact_zeros": exact_zeros(),
        "born_change_after_phase_rotation": phase_rotation(),
    }
    print(json.dumps(res, indent=2))
    with open(os.path.join(ROOT, "results", "v3-checks.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
