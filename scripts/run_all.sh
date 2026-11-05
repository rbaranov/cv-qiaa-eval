#!/bin/sh
# Seed 0 for every variant, then extra seeds: 1-2 for the fast ones, 1 for cvqiaa.
set -e
cd "$(dirname "$0")/.."
for kind in none softmax born cvqiaa; do .venv/bin/python scripts/train.py --kind $kind --seed 0; done
for seed in 1 2; do for kind in none softmax born; do .venv/bin/python scripts/train.py --kind $kind --seed $seed; done; done
.venv/bin/python scripts/train.py --kind cvqiaa --seed 1
