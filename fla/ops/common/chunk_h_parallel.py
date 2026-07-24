# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Fully parallelized state passing.
"""


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_parallel(
    k,
    v,
    h,
    g,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    # i_b: batch index
    # i_h: head index
    # i_n: sequence index
    # i_t: chunk index within current sequence
    # i_tg: (global) chunk index across all sequences
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * H + i_h

    desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
    desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
    desc_h = make_tensor_descriptor(h + (i_tg * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])

    if i_t == 0:
        if USE_INITIAL_STATE:
            desc_h0 = make_tensor_descriptor(h0 + i_nh * K*V, [K, V], [V, 1], [BK, BV])
            b_h = desc_h0.load([i_k * BK, i_v * BV]).to(tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))

    # [BK, BT]
    b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
    # [BT, BV]
    b_v = desc_v.load([i_t * BT, i_v * BV])

    last_idx = min(i_t * BT + BT, T) - 1
    # scalar decay
    if USE_G:
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

    # vector decay, h = Diag(gk) @ h
    if USE_GK:
        desc_gk = make_tensor_descriptor(gk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)

        b_gk = tl.trans(desc_gk.load([i_t * BT, i_k * BK]))
        b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

    # vector decay, h = h @ Diag(gv)
    if USE_GV:
        desc_gv = make_tensor_descriptor(gv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

        b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
        b_gv = desc_gv.load([i_t * BT, i_v * BV])
        b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

    b_h = tl.dot(b_k, b_v)
    if i_t < NT - 1:
        desc_h = make_tensor_descriptor(h + ((i_tg + 1) * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))
    elif STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(ht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_ht.store([i_k * BK, i_v * BV], b_h.to(desc_ht.dtype))


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    h,
    g,
    gk,
    gv,
    kvt,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        desc_h = make_tensor_descriptor(h + ((boh + i_t) * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        b_h += desc_h.load([i_k * BK, i_v * BV]).to(tl.float32)
        if i_t > 0:
            desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))

        last_idx = min(i_t * BT + BT, T) - 1
        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_h *= exp(b_g_last)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= exp(b_gk_last)[:, None]

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= exp(b_gv_last)[None, :]

    if STORE_FINAL_STATE:
        desc_kvt = make_tensor_descriptor(kvt + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_ht = make_tensor_descriptor(ht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_h += desc_kvt.load([i_k * BK, i_v * BV]).to(tl.float32)
        desc_ht.store([i_k * BK, i_v * BV], b_h.to(desc_ht.dtype))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_parallel(
    q,
    g,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * HQ + i_hq

    desc_q = make_tensor_descriptor(q + (bos*HQ + i_hq) * K, [T, K], [HQ*K, 1], [BT, BK])
    desc_do = make_tensor_descriptor(do + (bos*HQ + i_hq) * V, [T, V], [HQ*V, 1], [BT, BV])
    desc_dh = make_tensor_descriptor(dh + (i_tg * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])

    if i_t == NT - 1:
        if USE_FINAL_STATE_GRADIENT:
            desc_dht = make_tensor_descriptor(dht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
            b_dh = desc_dht.load([i_k * BK, i_v * BV]).to(tl.float32)
        else:
            b_dh = tl.zeros([BK, BV], dtype=tl.float32)
        desc_dh.store([i_k * BK, i_v * BV], b_dh.to(desc_dh.dtype))

    # [BK, BT]
    b_q = tl.trans(desc_q.load([i_t * BT, i_k * BK]))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_do = desc_do.load([i_t * BT, i_v * BV])

    if USE_G:
        p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)

    if USE_GK:
        desc_gk = make_tensor_descriptor(gk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_gk = tl.trans(desc_gk.load([i_t * BT, i_k * BK]))
        b_q = (b_q * exp(b_gk)).to(b_q.dtype)

    if USE_GV:
        desc_gv = make_tensor_descriptor(gv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_gv = desc_gv.load([i_t * BT, i_v * BV])
        b_do = (b_do * exp(b_gv)).to(b_do.dtype)

    b_dh = tl.dot(b_q, b_do)
    if i_t > 0:
        desc_dh = make_tensor_descriptor(dh + ((i_tg - 1) * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        desc_dh.store([i_k * BK, i_v * BV], b_dh.to(desc_dh.dtype))
    elif STORE_INITIAL_STATE_GRADIENT:
        desc_dh0 = make_tensor_descriptor(dh0 + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_dh0.store([i_k * BK, i_v * BV], b_dh.to(desc_dh0.dtype))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_reduction(
    g,
    gk,
    gv,
    dh,
    doq0,
    dh0,
    cu_seqlens,
    chunk_offsets,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        desc_dh = make_tensor_descriptor(dh + ((boh+i_t) * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        b_dh += desc_dh.load([i_k * BK, i_v * BV]).to(tl.float32)
        if i_t < NT - 1:
            desc_dh.store([i_k * BK, i_v * BV], b_dh.to(desc_dh.dtype))

        last_idx = min(i_t * BT + BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_dh *= exp(b_g_last)

        if USE_GK:
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_dh *= exp(b_gk_last)[:, None]

        if USE_GV:
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_dh *= exp(b_gv_last)[None, :]

    if STORE_INITIAL_STATE_GRADIENT:
        desc_doq0 = make_tensor_descriptor(doq0 + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_dh0 = make_tensor_descriptor(dh0 + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_dh += desc_doq0.load([i_k * BK, i_v * BV]).to(tl.float32)
        desc_dh0.store([i_k * BK, i_v * BV], b_dh.to(desc_dh0.dtype))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    states_in_fp32: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']) * triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_h_parallel[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    kvt, ht = ht, (torch.empty_like(ht) if output_final_state else None)
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](
        h=h,
        g=g,
        gk=gk,
        gv=gv,
        kvt=kvt,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    h = h.to(k.dtype) if not states_in_fp32 else h
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    states_in_fp32: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    NG = HQ // H

    dh = k.new_empty(B, NT, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']) * triton.cdiv(V, meta['BV']), NT, B * HQ)
    chunk_bwd_kernel_dh_parallel[grid](
        q=q,
        g=g,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )

    doq0, dh0 = dh0, (torch.empty_like(dh0) if dh0 is not None else None)
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * HQ)
    chunk_bwd_kernel_dh_reduction[grid](
        g=g,
        gk=gk,
        gv=gv,
        dh=dh,
        doq0=doq0,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    dh = dh.to(q.dtype) if not states_in_fp32 else dh
    return dh, dh0
