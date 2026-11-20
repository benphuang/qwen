#!/usr/bin/env bash
# Launch vLLM for Qwen3.8-27B. Everything is controlled by env vars, e.g.:
#   GPUS=0,1,2,3 VARIANT=fp8 ./serve.sh
#   GPUS=0,1,2,3,4,5,6,7 VARIANT=bf16 PP=2 SPEC=1 ./serve.sh     # TP=4 x PP=2, MTP speculative decoding
# Run it inside tmux so it survives SSH disconnects.
set -euo pipefail
source "$(dirname "$0")/config.env"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"   # which physical GPUs to use; TP = (#GPUs / PP)
VARIANT="${VARIANT:-int4}"        # bf16 | fp8 | int4
PP="${PP:-1}"                     # pipeline-parallel stages
MAXLEN="${MAXLEN:-262144}"        # native context; see README for the 1M extension
GMU="${GMU:-0.90}"                # gpu-memory-utilization; lower it if other users share the GPUs
MAX_SEQS="${MAX_SEQS:-4}"         # single user: a small value saves activation/graph memory
KV_DTYPE="${KV_DTYPE:-auto}"      # try fp8; may be rejected on Ampere depending on the attention backend
SPEC="${SPEC:-0}"                 # 1 = MTP speculative decoding (3 draft tokens)
TEXT_ONLY="${TEXT_ONLY:-0}"       # 1 = skip the vision tower's memory profiling (no image input)
THINKING="${THINKING:-}"          # "0" = default to non-thinking replies, "1" = default to thinking
EXTRA="${EXTRA:-}"                # anything else for `vllm serve`

eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
(( NGPU % PP == 0 )) || { echo "#GPUs ($NGPU) must be divisible by PP ($PP)"; exit 1; }
TP=$(( NGPU / PP ))
MODEL="${MODEL_ID[$VARIANT]:?unknown VARIANT '$VARIANT' (bf16|fp8|int4)}"

# Refuse to start on GPUs someone else is using (override with FORCE=1).
for g in "${GPU_ARR[@]}"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used > 1000 )) && [[ "${FORCE:-0}" != 1 ]]; then
    echo "GPU $g already has ${used} MiB in use; aborting (FORCE=1 to override)"; exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1                         # P2P is disabled on this box; be explicit
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # try this if startup OOMs
export VLLM_API_KEY="$(cat "$KEY_FILE")"

ARGS=(
  "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$GMU"
  --max-num-seqs "$MAX_SEQS"
  --enable-prefix-caching
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --mm-encoder-tp-mode data
  --disable-custom-all-reduce                     # custom all-reduce needs P2P
)
[[ "$KV_DTYPE" != auto ]] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[[ "$TEXT_ONLY" == 1 ]] && ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0}')
[[ "$SPEC" == 1 ]] && ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
[[ "$THINKING" == 0 ]] && ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
[[ "$THINKING" == 1 ]] && ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')

echo "Serving $MODEL on GPUs $GPUS (TP=$TP, PP=$PP), 127.0.0.1:$PORT, max-model-len=$MAXLEN"
# shellcheck disable=SC2086
exec vllm serve "${ARGS[@]}" $EXTRA
