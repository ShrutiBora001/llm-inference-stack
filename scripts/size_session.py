#!/usr/bin/env python3
"""Size a GPU session before renting it.

    python scripts/size_session.py
    python scripts/size_session.py --requests 128 --input-len 16384

Answers three questions that decide what to rent: how much KV memory the sweep
needs, how many GPUs that implies, and how long the whole thing takes.

Every number is derived from the benchmark's actual defaults rather than
asserted, so changing `--requests` or `--input-len` changes the estimate. Model
geometry is the one input that must be verified against the model card -- the
KV-per-token arithmetic drives the memory argument, and a wrong `n_kv_heads`
(GQA vs MHA is a 4x error here) would size the session wrong in the expensive
direction.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class ModelGeometry:
    """VERIFY against the model card before spending money on this estimate."""

    name: str
    params_b: float
    layers: int
    n_kv_heads: int
    head_dim: int
    verified: bool = False

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        # K and V, per layer, per KV head. GQA means n_kv_heads < n_heads, which
        # is exactly where an unverified guess goes wrong by a factor of 4.
        return 2 * self.layers * self.n_kv_heads * self.head_dim * dtype_bytes

    def weight_bytes(self, dtype_bytes: int = 2) -> float:
        return self.params_b * 1e9 * dtype_bytes


MODELS = {
    # Verified against the published config.json: 28 layers, 12 attention heads,
    # 2 KV heads (GQA), hidden 1536 -> head_dim 128, max_position_embeddings
    # 131072. Apache 2.0 and ungated, which matters: the Llama configs are
    # gated, so using one would require an HF token on a rented box.
    "qwen2.5-1.5b": ModelGeometry(
        "Qwen/Qwen2.5-1.5B", 1.54, 28, 2, 128, verified=True
    ),
    # NOT verified -- config.json returns 401 without accepting the licence.
    # Kept for comparison only; do not size a rental from these.
    "llama-3.2-1b": ModelGeometry("meta-llama/Llama-3.2-1B", 1.24, 16, 8, 64),
    "llama-3.1-8b": ModelGeometry("meta-llama/Llama-3.1-8B", 8.03, 32, 8, 128),
}

A100_40GB = 40 * 2**30
USABLE = 0.85  # allocator overhead + fragmentation, same figure as upstream


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5-1.5b", choices=sorted(MODELS))
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--prefix-shares", type=int, default=6)
    p.add_argument("--frameworks", type=int, default=4)
    p.add_argument("--gpu-hourly", type=float, default=0.70,
                   help="per-GPU hourly rate")
    args = p.parse_args()

    m = MODELS[args.model]
    kv_per_tok = m.kv_bytes_per_token()
    peak_tokens = args.requests * (args.input_len + args.output_len)
    kv_bytes = peak_tokens * kv_per_tok
    total_bytes = kv_bytes + m.weight_bytes()
    gpus_needed = max(1, -(-int(total_bytes) // int(A100_40GB * USABLE)))

    print(f"MODEL  {m.name}")
    if m.verified:
        print("       geometry verified against the published config.json")
    else:
        print("       !! geometry NOT verified against the model card. "
              "n_kv_heads is the one that matters (GQA vs MHA = 4x).")
    print(f"       {m.layers} layers, {m.n_kv_heads} kv heads, head_dim {m.head_dim}")
    print(f"       KV per token: {kv_per_tok / 1024:.1f} KiB")

    print(f"\nMEMORY at {args.requests} concurrent x "
          f"{args.input_len}+{args.output_len} tokens")
    print(f"       weights      {m.weight_bytes() / 2**30:>7.1f} GiB")
    print(f"       KV at peak   {kv_bytes / 2**30:>7.1f} GiB   "
          f"({peak_tokens:,} tokens)")
    print(f"       total        {total_bytes / 2**30:>7.1f} GiB")
    print(f"       -> {gpus_needed} x A100 40GB "
          f"(at {USABLE:.0%} usable = {A100_40GB * USABLE / 2**30:.1f} GiB each)")

    # Time. Arrival pacing dominates: the sweep is deliberately not saturating,
    # because a saturated queue measures the queue rather than the engine.
    pacing_s = args.requests / args.qps
    runs = args.frameworks * args.prefix_shares
    measure_s = runs * (pacing_s * 1.4)   # 40% tail for the last requests to drain
    startup_s = args.frameworks * 120     # engine init + CUDA graph capture
    install_s = 30 * 60                   # vLLM + SGLang from pip on NGC
    download_s = 5 * 60

    print(f"\nTIME   {runs} runs ({args.frameworks} frameworks x "
          f"{args.prefix_shares} prefix shares)")
    for label, secs in (
        ("install vLLM + SGLang", install_s),
        ("model download", download_s),
        (f"engine startup x{args.frameworks}", startup_s),
        (f"measurement ({pacing_s:.0f}s pacing + drain)", measure_s),
    ):
        print(f"       {label:<34}{secs / 60:>6.0f} min")
    subtotal = install_s + download_s + startup_s + measure_s
    print(f"       {'subtotal':<34}{subtotal / 60:>6.0f} min")
    print(f"       {'x1.8 for first-contact fixes':<34}"
          f"{subtotal * 1.8 / 60:>6.0f} min")

    hours = subtotal * 1.8 / 3600
    cost = hours * args.gpu_hourly * gpus_needed
    print(f"\nCOST   {gpus_needed} GPU x {hours:.1f} hr x "
          f"${args.gpu_hourly:.2f}/hr = ${cost:.2f}")
    print(f"       (measurement alone: "
          f"${measure_s / 3600 * args.gpu_hourly * gpus_needed:.2f} — "
          "setup dominates, which is why one long session beats three short ones)")


if __name__ == "__main__":
    main()
