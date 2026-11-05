#!/bin/sh
# Two fixes to the paper's code, nothing else:
#   1. K.mtheH -> K.mH (the attribute does not exist; mH is the conjugate transpose);
#   2. targets.view(-1) -> targets.reshape(-1) in the demo (targets is a
#      non-contiguous slice, view() raises RuntimeError).
set -e
cd "$(dirname "$0")/.."
sed -e 's/K\.mtheH/K.mH/' \
    -e 's/targets\.view(-1)/targets.reshape(-1)/' \
    -e 's|^# Nothing is changed here.*|# Copy of src/paper_wavellm.py with two fixes, see scripts/make_fixed.sh.|' \
    -e '/^# src\/wavellm_fixed.py (generated/d' \
    src/paper_wavellm.py > src/wavellm_fixed.py
