#!/usr/bin/env bash
# Launch each config, wait until healthy, benchmark it, tear it down, then print a summary.
# Edit MATRIX below. Entries are "GPU-list|variant|PP".
#   ONLY=g4_fp8 ./bench_sweep.sh        run only configs whose name contains this string
#   SPEC=1 ./bench_sweep.sh             same sweep with MTP speculative decoding
#   LENGTHS="8000 64000" ./bench_sweep.sh
# Run inside tmux. A full sweep takes a while (model load + long prefills).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.env"
eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME"

OUT="${OUT:-$HERE/results}"; mkdir -p "$OUT"
LENGTHS="${LENGTHS:-8000 32000 100000 200000}"
START_TIMEOUT="${START_TIMEOUT:-1800}"

# Feasibility notes (weights only, ~14.4 GB usable per card at 0.90 utilization):
#   int4 ~17 GB  -> fits on 2 GPUs      fp8 ~29 GB -> needs 4+      bf16 ~54 GB -> realistically 8 only
MATRIX=(
  "0,1|int4|1"
  "0,1,2,3|int4|1"
  "0,1,2,3|fp8|1"
  "0,1,2,3,4,5,6,7|int4|1"
  "0,1,2,3,4,5,6,7|fp8|1"
  "0,1,2,3,4,5,6,7|bf16|1"
  "0,1,2,3,4,5,6,7|fp8|2"
  "0,1,2,3,4,5,6,7|bf16|2"
)

for cfg in "${MATRIX[@]}"; do
  IFS='|' read -r GPUS VARIANT PP <<< "$cfg"
  n=$(awk -F, '{print NF}' <<< "$GPUS")
  name="g${n}_${VARIANT}_pp${PP}"
  [[ "${SPEC:-0}" == 1 ]] && name+="_spec"
  [[ -n "${ONLY:-}" && "$name" != *"$ONLY"* ]] && continue

  log="$OUT/$name.server.log"
  echo "=== $name ($(date +%H:%M:%S)) ==="
  GPUS="$GPUS" VARIANT="$VARIANT" PP="$PP" setsid "$HERE/serve.sh" > "$log" 2>&1 &
  pid=$!

  ok=0; t0=$SECONDS
  while (( SECONDS - t0 < START_TIMEOUT )); do
    kill -0 "$pid" 2>/dev/null || break                     # server died during startup
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ok=1; break; fi
    sleep 5
  done

  if (( ok )); then
    echo "up after $((SECONDS - t0))s"
    grep -E "GPU KV cache size|Maximum concurrency" "$log" | tee "$OUT/$name.kv.txt"
    python "$HERE/bench_client.py" --port "$PORT" --key-file "$KEY_FILE" --model "$SERVED_NAME" \
      --lengths $LENGTHS --out "$OUT/$name.json" --label "$name" || echo "bench client failed"
  else
    echo "FAILED to start; see $log"
    grep -iE "error|estimated maximum model length|out of memory|cannot|not divisible" "$log" \
      | tail -n 20 | tee "$OUT/$name.fail.txt"
  fi

  # teardown: kill the whole process group, then wait for VRAM to be released
  kill -TERM -- "-$pid" 2>/dev/null; sleep 10; kill -KILL -- "-$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  for _ in $(seq 30); do
    used=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
    (( used < 1000 )) && break
    sleep 2
  done
done

python "$HERE/bench_client.py" --summarize "$OUT"
