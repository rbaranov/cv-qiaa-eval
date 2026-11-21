"""Collect results/train-*.json into a Markdown table (results/summary.md)."""
import glob
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = {}
for path in sorted(glob.glob(os.path.join(ROOT, "results", "train-*.json"))):
    r = json.load(open(path))
    if r["args"]["lr"] != 1e-3:
        continue  # other learning rates are reported separately in README
    runs.setdefault(r["args"]["kind"], []).append(r)

lines = ["| Variant | Runs | Val loss (nats) | Bits per char | Seconds per 3000 steps | Attention params | NaN |",
         "|---|---|---|---|---|---|---|"]
for kind in ["softmax", "born", "cvqiaa", "none"]:
    rs = runs.get(kind, [])
    if not rs:
        continue
    vals = [r["final_val"] for r in rs if r["final_val"] is not None]
    mean = statistics.mean(vals)
    spread = f" ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else ""
    secs = statistics.mean(r["seconds"] for r in rs)
    nans = sum(r["nan_step"] is not None for r in rs)
    lines.append(f"| {kind} | {len(rs)} | {mean:.3f}{spread} | {mean / math.log(2):.3f} | {secs:.0f} | "
                 f"{rs[0]['attention_params']:,} | {nans} |")

lines += ["", "Val loss curves (seed 0):", "", "| Step | " + " | ".join(k for k in runs) + " |",
          "|---|" + "---|" * len(runs)]
seed0 = {k: next((r for r in rs if r["args"]["seed"] == 0), None) for k, rs in runs.items()}
steps = [p["step"] for p in next(iter(seed0.values()))["curve"]]
for i, step in enumerate(steps):
    cells = [f"{seed0[k]['curve'][i]['val']:.3f}" if seed0[k] and i < len(seed0[k]["curve"]) else "" for k in runs]
    lines.append(f"| {step} | " + " | ".join(cells) + " |")

lines += ["", "Text samples (seed 0, prompt "Once upon a time", temperature 0.8):", ""]
for k, r in seed0.items():
    if r and r["sample"]:
        lines += [f"**{k}**", "", "```", r["sample"], "```", ""]

open(os.path.join(ROOT, "results", "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
