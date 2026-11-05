"""Table of final val loss by variant and lr over all seeds; writes results/summary-v2.md."""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = {}
for path in glob.glob(os.path.join(ROOT, "results", "train-*.json")):
    log = json.load(open(path))
    a = log["args"]
    runs.setdefault((a["kind"], a["lr"]), []).append(log)

lines = ["| Variant | lr | Seeds | Final val loss | Training time, min | Learned sigma0 by layer |",
         "|---|---|---|---|---|---|"]
order = ["softmax", "softmax-env", "softmax-env-wide", "born", "cvqiaa", "cvqiaa2", "cvqiaa2-wide", "cvqiaa2-noenv", "none"]
for (kind, lr), logs in sorted(runs.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
    vals = [l["final_val"] for l in logs if l["final_val"] is not None]
    loss = f"{statistics.mean(vals):.3f}" + (f" ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else "")
    minutes = statistics.mean(l["seconds"] for l in logs) / 60
    sig = ""
    if logs[0].get("sigma0"):
        s = sorted(logs, key=lambda l: l["args"]["seed"])[0]
        sig = f"seed {s['args']['seed']}: " + "; ".join(
            "/".join(f"{x:.2g}" for x in layer) for layer in s["sigma0"])
    lines.append(f"| {kind} | {lr:g} | {len(vals)} | {loss} | {minutes:.1f} | {sig} |")

out = "\n".join(lines) + "\n"
open(os.path.join(ROOT, "results", "summary-v2.md"), "w").write(out)
print(out)
