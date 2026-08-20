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

# Takes the number of GPUs this verb actually needs. Kernel work needs one;
# the ring benchmarks need four. Demanding four for a single-kernel run would
# force a 4x more expensive rental for no reason.
preflight() {
  need="${1:-$WORLD}"
  log "preflight (needs $need GPU(s))"
  command -v nvidia-smi >/dev/null || { echo "FATAL: no NVIDIA driver"; exit 1; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  [ "$n" -ge "$need" ] || { echo "FATAL: need $need GPUs, found $n"; exit 1; }

  # An all-PCIe topology is valid but makes every overlap number meaningless.
  # Warn loudly rather than silently producing a dull result.
  if [ "$need" -lt 2 ]; then
    echo "  single GPU: no interconnect to validate"
  elif nvidia-smi topo -m | grep -q 'NV[0-9]'; then
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
  kernel)
    # Phase 1. One GPU is enough - four cannot make a single kernel converge
    # faster, and the rental costs 4x.
    preflight 1
    log "triton fused kernel validation (correctness before speed)"
    python scripts/validate_triton.py --seqs 2048 4096 8192 16384 \
      --out "$OUT/triton_validation.json" 2>&1 | tee "$OUT/triton_validation.log"
    log "full MFU sweep, every backend"
    python bench/kernel.py --seqs 2048 4096 8192 16384 32768 \
      --out "$OUT/kernel.jsonl" 2>&1 | tee -a "$OUT/kernel.log"
    log "contract suite (Triton checks are live now, not skipped)"
    python -m pytest tests/test_contract.py -q -rs 2>&1 | tee "$OUT/contract.log"
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
