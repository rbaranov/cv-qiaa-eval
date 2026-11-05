#!/bin/sh
# Learning-rate check: seed 0 at lr 3e-3 for softmax and cvqiaa.
set -e
cd "$(dirname "$0")/.."
for kind in softmax cvqiaa; do .venv/bin/python scripts/train.py --kind $kind --seed 0 --lr 3e-3; done
