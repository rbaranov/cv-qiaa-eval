"""Train one variant on character-level TinyStories and log to results/.

Usage: .venv/bin/python scripts/train.py --kind cvqiaa --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import LM  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True, choices=["softmax", "cvqiaa", "born", "none", "cvqiaa2", "cvqiaa2-noenv", "cvqiaa2-wide", "softmax-env", "softmax-env-wide"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--eval-batches", type=int, default=50)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = p.parse_args()

text = open(os.path.join(ROOT, "data", "TinyStories-valid.txt"), encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def batch(src, gen):
    ix = torch.randint(len(src) - args.ctx - 1, (args.batch,), generator=gen)
    x = torch.stack([src[i:i + args.ctx] for i in ix])
    y = torch.stack([src[i + 1:i + args.ctx + 1] for i in ix])
    return x.to(args.device), y.to(args.device)


@torch.no_grad()
def evaluate(model):
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # same validation batches for every run
    losses = []
    for _ in range(args.eval_batches):
        x, y = batch(val_data, gen)
        losses.append(F.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, prompt="Once upon a time", n=300):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.tensor([[stoi[c] for c in prompt]], device=args.device)
    for _ in range(n):
        logits = model(idx[:, -args.ctx:])[:, -1, :] / 0.8
        probs = F.softmax(logits.float().cpu(), -1)
        nxt = torch.multinomial(probs, 1, generator=gen).to(args.device)
        idx = torch.cat([idx, nxt], 1)
    return "".join(chars[i] for i in idx[0].tolist())


torch.manual_seed(args.seed)
model = LM(args.kind, len(chars)).to(args.device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
warmup = 100


def lr_at(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))


log = {"args": vars(args), "vocab": len(chars), "params": sum(p.numel() for p in model.parameters()),
       "attention_params": model.attention_params(), "curve": [], "nan_step": None}
gen = torch.Generator().manual_seed(args.seed)
t0 = time.time()
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(train_data, gen)
    loss = F.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
    if not torch.isfinite(loss):
        log["nan_step"] = step
        print(f"step {step}: loss is {loss.item()}, stopping")
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % args.eval_every == 0 or step == 0:
        val = evaluate(model)
        elapsed = time.time() - t0
        log["curve"].append({"step": step + 1, "train": loss.item(), "val": val, "seconds": elapsed})
        print(f"{args.kind} seed {args.seed} step {step + 1}: train {loss.item():.4f} val {val:.4f} "
              f"({val / math.log(2):.3f} bpc) {elapsed:.0f}s", flush=True)

log["seconds"] = time.time() - t0
log["final_val"] = log["curve"][-1]["val"] if log["curve"] else None
log["sample"] = sample(model) if log["nan_step"] is None else None
sigma = [[p.exp().tolist() for n, p in l.attn.named_parameters() if n.endswith("log_sigma0") and p.requires_grad]
         for l in model.layers]
if any(sigma):  # learned wave-packet widths, per layer and head
    log["sigma0"] = [s[0] for s in sigma]
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
suffix = "" if args.lr == 1e-3 else f"-lr{args.lr:g}"  # runs with the default lr keep their old names
with open(os.path.join(ROOT, "results", f"train-{args.kind}-seed{args.seed}{suffix}.json"), "w") as f:
    json.dump(log, f, ensure_ascii=False, indent=1)
print(log["sample"])
