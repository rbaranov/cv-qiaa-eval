#!/bin/sh
# What the v2 gap is made of: softmax with the v2 envelope instead of RoPE
# (Born rule vs softmax on the same positional part), and v2 with sigma0 init 40.
set -e
cd "$(dirname "$0")/.."
run() { .venv/bin/python scripts/train.py "$@"; }
run --kind softmax-env --seed 0 --lr 3e-3
run --kind cvqiaa2-wide --seed 0 --lr 3e-3
run --kind softmax-env --seed 1 --lr 3e-3
run --kind softmax-env --seed 0
# sigma0 init 40: more seeds for cvqiaa2-wide and the same init for softmax-env
run --kind cvqiaa2-wide --seed 1 --lr 3e-3
run --kind cvqiaa2-wide --seed 2 --lr 3e-3
run --kind softmax-env-wide --seed 0 --lr 3e-3
run --kind softmax-env-wide --seed 1 --lr 3e-3
