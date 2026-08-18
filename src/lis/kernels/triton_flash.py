"""Hand-written Triton FlashAttention forward, returning (out, lse).

NEVER EXECUTED. There is no NVIDIA device on the development machine, so this
has not been compiled or run. Expect to fix it on first contact with a GPU.
It is written out in full so the GPU session starts from something concrete --
the same approach taken with nvshmem/ring_bw.cu upstream, which was the right
call there.

The `torch_flash` backend is the working fused path and is preferred by the
dispatcher when both are present. This kernel has to *earn* its place by
beating a real baseline, which is the point: a hand-written kernel that only
beats an unfused Python loop has proven nothing.

Structure is the standard FlashAttention-2 forward:

    for each block of BLOCK_N keys:
        S      = QK^T * scale                     (+ causal mask)
        m_new  = max(m, rowmax(S))
        alpha  = exp(m - m_new)                   rescale the running state
        P      = exp(S - m_new)
        acc    = acc*alpha + P@V
        l      = l*alpha  + rowsum(P)
    out = acc / l
    lse = m + log(l)                              <- what the ring needs

The score tile never leaves SRAM, which is the whole point: the unfused path
writes ~90 GiB to HBM at S=32768 (report section 6.7) and this writes none.
"""

from __future__ import annotations

import math

import torch

from ..lse import PartialAttention

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - the laptop path
    _HAVE_TRITON = False

# Populated after each run so the contract suite can assert the autotuner
# explored the space rather than always taking the first candidate.
LAST_CONFIG: dict = {}


def available() -> bool:
    return _HAVE_TRITON and torch.cuda.is_available()


if _HAVE_TRITON:

    def _configs():
        """Autotune space. Deliberately spans warp and stage counts as well as
        tile sizes: on A100 the best config for a long-sequence causal block is
        usually not the best for a short one, which is the reason to autotune at
        all rather than hardcode."""
        out = []
        for bm in (64, 128):
            for bn in (32, 64, 128):
                for w in (4, 8):
                    for s in (2, 3, 4):
                        out.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s
                        ))
        return out

    @triton.autotune(configs=_configs(), key=["N_CTX_Q", "N_CTX_K", "BLOCK_D", "CAUSAL"])
    @triton.jit
    def _attn_fwd(
        Q, K, V, Out, Lse,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        stride_lz, stride_lh, stride_lm,
        H, N_CTX_Q, N_CTX_K,
        Q_OFFSET, K_OFFSET,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_base = Q + off_b * stride_qz + off_h * stride_qh
        k_base = K + off_b * stride_kz + off_h * stride_kh
        v_base = V + off_b * stride_vz + off_h * stride_vh

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < N_CTX_Q
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Running softmax state, fp32 regardless of input dtype -- same choice
        # as the reference implementation, and for the same reason.
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Global positions. Causal masking is expressed against these, never
        # against local indices -- under a striped layout they differ.
        q_pos = offs_m + Q_OFFSET

        # Under causal masking, keys beyond this query block's last position are
        # entirely in the future. Stopping early is where the triangular saving
        # actually happens.
        if CAUSAL:
            hi = tl.minimum(N_CTX_K, (start_m + 1) * BLOCK_M + Q_OFFSET - K_OFFSET)
        else:
            hi = N_CTX_K

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            kv_mask = offs_n[:, None] < N_CTX_K

            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # Out-of-range keys must not contribute.
            qk = tl.where(offs_n[None, :] < N_CTX_K, qk, float("-inf"))
            if CAUSAL:
                k_pos = offs_n + K_OFFSET
                qk = tl.where(q_pos[:, None] >= k_pos[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            # A row whose every key was masked gives m_ij = -inf, and
            # exp(-inf - -inf) is NaN. Pin those rows so every exp below
            # evaluates to exp(-inf) = 0 and the update becomes a no-op. Same
            # guard as block_update in the reference; it is not optional.
            m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_ij

        # Rows that saw nothing stay at zero rather than dividing by zero.
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_safe[:, None]

        o_ptrs = (Out + off_b * stride_oz + off_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

        # lse in natural log, matching lse.merge's expectation. Empty rows carry
        # -inf so merge() treats them as contributing nothing.
        lse = tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf"))
        l_ptrs = Lse + off_b * stride_lz + off_h * stride_lh + offs_m * stride_lm
        tl.store(l_ptrs, lse, mask=offs_m < N_CTX_Q)


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    q_offset: int = 0,
    k_offset: int = 0,
) -> PartialAttention:
    """Triton flash attention over one (query chunk, key chunk) pair.

    Same signature and same contract as the unfused backend, so the dispatcher
    can substitute either and the correctness test can diff them directly.
    """
    if not available():  # pragma: no cover
        raise RuntimeError("Triton backend unavailable on this machine")

    b, h, sq, d = q.shape
    sk = k.shape[2]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    if d & (d - 1):
        raise ValueError(f"head_dim {d} must be a power of two for this kernel")

    out = torch.empty_like(q)
    lse = torch.empty((b, h, sq), device=q.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(sq, meta["BLOCK_M"]), b * h)  # noqa: E731

    _attn_fwd[grid](
        q, k, v, out, lse,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        h, sq, sk,
        q_offset, k_offset,
        BLOCK_D=d,
        CAUSAL=causal,
    )

    # Record what the autotuner picked. The contract suite fails if it is always
    # candidate zero, which would mean the tile sizes were effectively hardcoded
    # and Triton's main advantage unused.
    try:
        best = _attn_fwd.best_config
        key = f"sq{sq}_sk{sk}_d{d}_causal{int(causal)}"
        LAST_CONFIG[key] = {
            "config": str(best),
            "chosen_index": next(
                (i for i, c in enumerate(_configs()) if str(c) == str(best)), -1
            ),
        }
    except Exception:  # pragma: no cover - autotuner internals vary by version
        pass

    return PartialAttention(out=out, lse=lse)
