"""Checks of the paper's code that need no training. Writes results/diagnose.json.

1. Paper demo control: the paper's 5-step demo on random tokens, run with
   CV-QIAA, with standard attention and with attention switched off, over 30
   seeds; within a seed all variants get the same data. Without attention the
   model sees only the current token, so if loss drops for it too, the demo
   does not test attention.
2. Spatial overlap at context 128 with the paper's positions 0.1 * index and
   canvas x in [-5, 5]: where each head's weight goes. Plus the paper's own
   formula exp(-dt^2 / 4 sigma(t)^2) without the grid: its range per head.
3. NaN gradients on CPU: count of parameters with NaN gradient after one
   backward pass of the TinyStories-sized model.
4. The same model with |Psi|^2 written as Re^2 + Im^2: NaN over 20 CPU steps.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import LM  # noqa: E402

out = {}

# 1. Paper demo with and without attention (same sizes as the paper's WaveLLM).
SEEDS = 30
demo = {}
for kind in ["cvqiaa", "softmax", "none"]:
    curves = []
    for seed in range(SEEDS):
        data = torch.randint(0, 10, (4, 12), generator=torch.Generator().manual_seed(seed))
        torch.manual_seed(seed)
        model = LM(kind, 10, d_model=32, n_heads=4, d_ff=64, n_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = F.cross_entropy(model(data[:, :-1]).reshape(-1, 10), data[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves.append(losses)
    demo[kind] = [round(sum(c[i] for c in curves) / SEEDS, 3) for i in range(5)]
out["paper_demo_mean_losses_30_seeds"] = demo
print(f"paper demo, 5 steps on random tokens, mean over {SEEDS} seeds:", demo)

# 2. Spatial overlap, copied from FreeEvolutionQuantumAttentionHead.forward.
T, G = 128, 64
pos = torch.linspace(0, (T - 1) * 0.1, T)
grid = torch.linspace(-5, 5, G).view(1, 1, -1)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
overlap = {}
for s0 in [4.0, 0.5]:
    sQ, sK = pos[:, None], pos[None, :]
    dt = torch.clamp(sQ - sK, min=0)
    sig = s0 * torch.sqrt(1 + (dt / s0 ** 2) ** 2)
    eQ = torch.exp(-((grid - sQ[..., None]) ** 2) / (2 * sig[..., None] ** 2))
    eK = torch.exp(-((grid - sK[..., None]) ** 2) / (2 * sig[..., None] ** 2))
    ov = (eQ * eK).sum(-1) / (G * sig)
    strongest = (torch.arange(T) - ov.masked_fill(~causal, -1).argmax(1))
    self_zero = (ov.diagonal() == 0).nonzero()
    overlap[f"sigma0={s0}"] = {
        "causal_pairs": int(causal.sum()),
        "pairs_with_zero_weight": int(((ov ** 2 == 0) & causal).sum()),
        "first_token_that_cannot_see_itself": int(self_zero[0]) if len(self_zero) else None,
        "distance_to_strongest_key_for_token": {i: int(strongest[i]) for i in [10, 30, 60, 127]},
        "token127_overlap_by_distance": {d: float(ov[127, 127 - d]) for d in [0, 1, 5, 20, 60, 127]},
        # |Psi|^2 carries overlap squared; share of that positional weight on the 5 nearest tokens
        "share_of_weight_on_5_nearest_for_token": {
            i: float((ov[i, i - 4:i + 1] ** 2).sum() / (ov[i, :i + 1] ** 2).sum()) for i in [50, 70, 90, 127]},
        "uniform_share_of_5_nearest_for_token": {i: 5 / (i + 1) for i in [50, 70, 90, 127]},
    }
    # The paper's formula (section 2.2) without the grid, for causal pairs where dt = s_Q - s_K.
    t = torch.linspace(0, 1000, 100001)
    formula = torch.exp(-t ** 2 / (4 * s0 ** 2 * (1 + (t / s0 ** 2) ** 2)))
    overlap[f"sigma0={s0}"]["paper_formula_min_max_over_distance"] = [float(formula.min()), float(formula.max())]
out["overlap_ctx128"] = overlap
print(json.dumps(overlap, indent=1))

# 3. NaN gradients on CPU after one step.
torch.manual_seed(0)
model = LM("cvqiaa", 100)
x = torch.randint(0, 100, (32, 128))
F.cross_entropy(model(x).reshape(-1, 100), x.reshape(-1)).backward()
bad = [n for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
total = sum(1 for p in model.parameters() if p.grad is not None)
out["cpu_nan_grad_params"] = {"with_nan": len(bad), "total": total}
print(f"CPU: {len(bad)} of {total} parameter tensors have NaN gradient after one step")

# 4. The NaN fix from the paper's own formula |Psi|^2 = Re^2 + Im^2: 20 CPU steps.
import importlib.util  # noqa: E402
import models  # noqa: E402

src = open(os.path.join(ROOT, "src", "wavellm_fixed.py")).read()
patched = src.replace("wave_intensities = torch.abs(unified_complex_amplitudes) ** 2",
                      "wave_intensities = unified_complex_amplitudes.real ** 2 + unified_complex_amplitudes.imag ** 2")
assert patched != src
spec = importlib.util.spec_from_loader("wavellm_nanfix", loader=None)
nanfix = importlib.util.module_from_spec(spec)
exec(compile(patched, "wavellm_nanfix", "exec"), nanfix.__dict__)
models.MultiHeadUnifiedQuantumAttention = nanfix.MultiHeadUnifiedQuantumAttention
torch.manual_seed(0)
model = models.LM("cvqiaa", 100)
opt = torch.optim.AdamW(model.parameters(), 1e-3)
nan_steps = 0
for _ in range(20):
    loss = F.cross_entropy(model(x).reshape(-1, 100), x.reshape(-1))
    opt.zero_grad()
    loss.backward()
    nan_steps += any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
    opt.step()
out["cpu_nan_after_fix"] = {"steps": 20, "steps_with_nan": nan_steps}
print(f"CPU with Re^2 + Im^2: {nan_steps} of 20 steps with NaN gradient")

with open(os.path.join(ROOT, "results", "diagnose.json"), "w") as f:
    json.dump(out, f, indent=1)
