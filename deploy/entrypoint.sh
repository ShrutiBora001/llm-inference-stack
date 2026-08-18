#!/usr/bin/env bash
# One entrypoint, several verbs. Every path validates the hardware before it
# measures anything, because a wrong interconnect silently invalidates results.
set -euo pipefail

WORLD="${LIS_WORLD_SIZE:-4}"
SEQ="${LIS_SEQ:-32768}"
LAYOUT="${LIS_LAYOUT:-striped}"
BLOCK="${LIS_BLOCK:-512}"
OUT="${LIS_RESULTS:-/workspace/lis/results}"
mkdir -p "$OUT"

PY_BIN="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)/.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python)"
shopt -s expand_aliases
python() { "$PY_BIN" "$@"; }
torchrun() { "$PY_BIN" -m torch.distributed.run "$@"; }

log() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }

preflight() {
  log "preflight"
  command -v nvidia-smi >/dev/null || { echo "FATAL: no NVIDIA driver"; exit 1; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  [ "$n" -ge "$WORLD" ] || { echo "FATAL: need $WORLD GPUs, found $n"; exit 1; }

  # An all-PCIe topology is valid but makes every overlap number meaningless.
  # Warn loudly rather than silently producing a dull result.
  if nvidia-smi topo -m | grep -q 'NV[0-9]'; then
    echo "  interconnect: NVLink detected"
  else
    echo "  WARNING: no NVLink -- PCIe only. Comm-bound; say so in any writeup."
  fi
  python -c "import torch;print(f'  torch {torch.__version__} cuda {torch.version.cuda} '
             f'nccl {\".\".join(map(str,torch.cuda.nccl.version()))}')"
}

case "${1:-validate}" in
  validate)
    preflight
    log "correctness (backend-agnostic: gloo on CPU, NCCL on GPU)"
    python -m pytest tests/ -q
    log "anti-decoration contract"
    python -m pytest tests/test_contract.py -q -rs
    ;;
  smoke)
    preflight
    log "NCCL correctness + measured bandwidth"
    torchrun --nproc_per_node="$WORLD" vendor/dattn/scripts/smoke_gpu.py 2>&1 | tee "$OUT/smoke.log"
    ;;
  bench)
    preflight
    log "layout A/B: contiguous vs striped, S=$SEQ"
    for lay in contiguous striped; do
      torchrun --nproc_per_node="$WORLD" vendor/dattn/bench/bench_attention.py \
        --seq "$SEQ" --d-model 4096 --heads 32 --dtype bf16 --layout "$lay" \
        --block-size "$BLOCK" --iters 20 --warmup 5 --strategies ring \
        --json "$OUT/layout.jsonl" 2>&1 | tee -a "$OUT/bench.log"
    done
    ;;
  sweep)
    preflight
    log "sequence sweep (BLOCK=$BLOCK)"
    for s in 131072 262144 524288 1048576; do
      torchrun --nproc_per_node="$WORLD" vendor/dattn/bench/bench_attention.py \
        --seq "$s" --d-model 4096 --heads 32 --dtype bf16 --layout "$LAYOUT" \
        --block-size "$BLOCK" --iters 3 --warmup 1 --strategies ring allgather \
        --json "$OUT/sweep.jsonl" 2>&1 | tee -a "$OUT/sweep.log" || \
        echo "  seq $s did not complete (OOM is a result, not a failure)"
    done
    ;;
  profile)
    preflight
    log "nsys timeline"
    nsys profile -o "$OUT/ring_$LAYOUT" --force-overwrite true --trace=cuda,nvtx,osrt \
      torchrun --nproc_per_node="$WORLD" vendor/dattn/bench/bench_attention.py \
        --seq "$SEQ" --d-model 4096 --heads 32 --dtype bf16 --layout "$LAYOUT" \
        --iters 5 --warmup 2 --strategies ring
    ;;
  all)
    "$0" validate && "$0" smoke && "$0" bench && "$0" sweep
    log "done -- copy $OUT off the box BEFORE destroying it"
    ;;
  shell) exec /bin/bash ;;
  *) exec "$@" ;;
esac
