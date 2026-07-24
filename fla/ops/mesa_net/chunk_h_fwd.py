# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp2, safe_dot
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h_init'] is not None,
    'STORE_FINAL_STATE': lambda args: args['h_final'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_fwd_kernel_h(
    k,
    v,
    beta,
    g,
    h,
    h_kv,
    h_init,
    h_kv_init,
    h_final,
    h_kv_final,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_kv = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        desc_h0 = make_tensor_descriptor(h_init + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_h = desc_h0.load([i_k * BK, i_v * BV]).to(tl.float32)
        desc_h_kv0 = make_tensor_descriptor(h_kv_init + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_h_kv = desc_h_kv0.load([i_k * BK, i_v * BV]).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // (BS // BT)
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_k2 = make_tensor_descriptor(k + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_beta = tl.load(beta + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        desc_h = make_tensor_descriptor(h + o_h, [K, V], [V, 1], [BK, BV])
        desc_h_kv = make_tensor_descriptor(h_kv + o_h, [K, V], [V, 1], [BK, BV])

        if i_t % (BS // BT) == 0:
            desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))
            desc_h_kv.store([i_k * BK, i_v * BV], b_h_kv.to(desc_h_kv.dtype))

        # [BK, BT]
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_k2 = desc_k2.load([i_t * BT, i_v * BV])
        # [BT, BV]
        b_v = desc_v.load([i_t * BT, i_v * BV])
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_h *= exp2(b_g_last)
        b_h_kv *= exp2(b_g_last)
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_k_decay = ((b_k * exp2(b_g_last - b_g)[:, None]) * b_beta[:, None]).to(b_k2.dtype)
        b_h += safe_dot(tl.trans(b_k_decay), b_k2)
        b_h_kv += safe_dot(tl.trans(b_k_decay), b_v.to(b_k2.dtype))

    if STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(h_final + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_ht.store([i_k * BK, i_v * BV], b_h.to(desc_ht.dtype))
        desc_h_kv_final = make_tensor_descriptor(h_kv_final + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_h_kv_final.store([i_k * BK, i_v * BV], b_h_kv.to(desc_h_kv_final.dtype))


def chunk_mesa_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h_init: torch.Tensor,
    h_kv_init: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K == V, "K must be equal to V for now"
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    h_kv = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    h_final = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    h_kv_final = k.new_empty(N, H, K, V, dtype=torch.float)

    def grid(meta): return (triton.cdiv(K, 64), triton.cdiv(V, 64), N * H)

    chunk_mesa_net_fwd_kernel_h[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        h=h,
        h_kv=h_kv,
        h_init=h_init,
        h_kv_init=h_kv_init,
        h_final=h_final,
        h_kv_final=h_kv_final,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=64,
        BV=64,
    )
    return h, h_kv, h_final, h_kv_final
