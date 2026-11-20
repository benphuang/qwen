#!/usr/bin/env bash
# Creates a conda env with vLLM, generates a local API key, and prints hardware info.
set -euo pipefail
source "$(dirname "$0")/config.env"

echo "== Driver / GPUs =="
nvidia-smi | head -n 4
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "== Topology (expect PHB/SYS links, no NV#, since P2P is off) =="
nvidia-smi topo -m || true
echo "== Host RAM / disk =="
free -g | head -n 2
mkdir -p "$QWEN_ROOT" "$HF_HOME" "$(dirname "$KEY_FILE")"
df -h "$HF_HOME" | tail -n 1

echo "== Conda env: $ENV_NAME =="
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda activate "$ENV_NAME"
pip install -U uv

# vLLM >= 0.17 and transformers >= 5.8 are required by the Qwen3.8 recipe.
# Resolving them together lets uv report a conflict instead of silently breaking one.
uv pip install -U vllm "transformers>=5.8.0" --torch-backend=auto
uv pip install -U huggingface_hub requests openai

python - <<'PY'
import torch, vllm, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
print("visible GPUs:", torch.cuda.device_count())
PY

# API key (vLLM reads it from the VLLM_API_KEY env var, so it never shows up in `ps`)
if [[ ! -s "$KEY_FILE" ]]; then
  (umask 077; python -c 'import secrets; print("sk-local-" + secrets.token_urlsafe(32))' > "$KEY_FILE")
fi
chmod 600 "$KEY_FILE"
echo "API key stored in $KEY_FILE"
echo "Done. Next: ./02_download.sh"
