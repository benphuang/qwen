#!/usr/bin/env bash
# Usage: ./02_download.sh [variant ...]    (default: int4 bf16)
# Approx sizes: int4 ~17 GB, bf16 ~54 GB.
set -euo pipefail
source "$(dirname "$0")/config.env"
source "$VENV_DIR/bin/activate"

VARIANTS=("$@")
(( ${#VARIANTS[@]} )) || VARIANTS=(int4 bf16)

for v in "${VARIANTS[@]}"; do
  id="${MODEL_ID[$v]:?unknown variant $v}"
  echo "== $v -> $id =="
  hf download "$id"
done
df -h "$HF_HOME" | tail -n 1
