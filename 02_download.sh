#!/usr/bin/env bash
# Usage: ./02_download.sh [variant ...]    (default: int4 fp8 bf16)
# Approx sizes: int4 ~17 GB, fp8 ~29 GB, bf16 ~54 GB.
set -euo pipefail
source "$(dirname "$0")/config.env"
eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME"

VARIANTS=("$@")
(( ${#VARIANTS[@]} )) || VARIANTS=(int4 fp8 bf16)

for v in "${VARIANTS[@]}"; do
  id="${MODEL_ID[$v]:?unknown variant $v}"
  echo "== $v -> $id =="
  hf download "$id"
done
df -h "$HF_HOME" | tail -n 1
