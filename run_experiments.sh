#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python train.py
python register.py
python compare.py
