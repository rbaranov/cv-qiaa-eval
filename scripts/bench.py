"""Time and memory of one attention layer, forward + backward, by context length.

Memory is what MPS holds after the forward pass (activations saved for
backward) minus what it held before.
"""
import gc
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import StandardAttention  # noqa: E402
from wavellm_fixed import MultiHeadUnifiedQuantumAttention  # noqa: E402

dev = "mps"
B, D, H = 4, 128, 4
rows = []
for T in [64, 128, 256, 512, 1024]:
    for kind in ["softmax", "born", "cvqiaa"]:
        torch.manual_seed(0)
        layer = (MultiHeadUnifiedQuantumAttention(D, H) if kind == "cvqiaa"
                 else StandardAttention(D, H, normalisation=kind)).to(dev)
        x = torch.randn(B, T, D, device=dev, requires_grad=True)
        pos = torch.linspace(0.0, (T - 1) * 0.1, T, device=dev).expand(B, T)
        try:
            times = []
            for i in range(4):
                gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
                before = torch.mps.current_allocated_memory()
                t = time.time()
                out = layer(x, pos)
                torch.mps.synchronize()
                held = torch.mps.current_allocated_memory() - before
                out.sum().backward()
                torch.mps.synchronize()
                if i:
                    times.append(time.time() - t)
                del out
            row = {"T": T, "kind": kind, "ms": 1000 * sum(times) / len(times), "activation_mb": held / 2**20}
        except RuntimeError as e:
            row = {"T": T, "kind": kind, "error": str(e).splitlines()[0][:160]}
        rows.append(row)
        print(row, flush=True)
        del layer, x
with open(os.path.join(ROOT, "results", "bench.json"), "w") as f:
    json.dump({"batch": B, "d_model": D, "heads": H, "rows": rows}, f, indent=1)
