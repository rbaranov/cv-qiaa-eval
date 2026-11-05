#!/bin/sh
# The authors' grid-free layer (cvqiaa2) against softmax: lr 1e-3, 3e-3 and 6e-3,
# three seeds at the first two; the envelope ablation and born at 3e-3.
set -e
cd "$(dirname "$0")/.."
run() { .venv/bin/python scripts/train.py "$@"; }
run --kind cvqiaa2 --seed 0 --lr 3e-3
run --kind cvqiaa2 --seed 0
run --kind cvqiaa2-noenv --seed 0 --lr 3e-3
run --kind born --seed 0 --lr 3e-3
for seed in 1 2; do
  run --kind cvqiaa2 --seed $seed --lr 3e-3
  run --kind softmax --seed $seed --lr 3e-3
  run --kind cvqiaa2 --seed $seed
done
run --kind cvqiaa2 --seed 0 --lr 6e-3
run --kind softmax --seed 0 --lr 6e-3
