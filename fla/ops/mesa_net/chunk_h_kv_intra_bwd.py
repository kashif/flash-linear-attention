# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.mesa_net.chunk_h_kv_intra_bwd_separate import chunk_mesa_net_h_kv_bwd_intra_separate_fn
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel(
    q_star,
    k,
    v,
    beta,
    h_kv,
    g,
    do,
    dh_kv,
    dq,
    dk_beta,
    dg,
    dv,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    desc_q = make_tensor_descriptor(q_star, [T, K], [H*K, 1], [BT, BK])
    desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, BK])
    desc_v = make_tensor_descriptor(v, [T, V], [H*V, 1], [BT, BV])
    desc_do = make_tensor_descriptor(do, [T, V], [H*V, 1], [BT, BV])
    desc_h = make_tensor_descriptor(h_kv, [K, V], [V, 1], [BK, BV])
    desc_dh = make_tensor_descriptor(dh_kv, [K, V], [V, 1], [BK, BV])

    b_q = desc_q.load([i_t * BT, 0])
    b_k = desc_k.load([i_t * BT, 0])
    b_v = desc_v.load([i_t * BT, 0])
    b_g = tl.load(g + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_beta = tl.load(beta + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_do = desc_do.load([i_t * BT, 0])
    b_h = tl.trans(desc_h.load([0, 0]))
    b_dh = tl.trans(desc_dh.load([0, 0]))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    # calculation
    b_dg_last += tl.sum(b_h * b_dh)
    b_dg_last *= exp2(b_g_last)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp2(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m

    b_ds = tl.dot(b_do, tl.trans(b_v))
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)

    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)

    b_g_exp_q = exp2(b_g)
    b_g_exp_k = tl.where(m_t, exp2(-b_g + b_g_last), 0)
    b_ds = b_ds * b_m
    b_dq += tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dg += tl.sum(b_dq * b_q, axis=1)
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
    b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    desc_dq = make_tensor_descriptor(dq, [T, K], [H*K, 1], [BT, BK])
    desc_dk = make_tensor_descriptor(dk_beta, [T, K], [H*K, 1], [BT, BK])
    desc_dv = make_tensor_descriptor(dv, [T, V], [H*V, 1], [BT, BV])
    desc_dq.store([i_t * BT, 0], b_dq.to(desc_dq.dtype))
    desc_dk.store([i_t * BT, 0], b_dk.to(desc_dk.dtype))
    desc_dv.store([i_t * BT, 0], b_dv.to(desc_dv.dtype))
    tl.store(dg + (i_t * BT + tl.arange(0, BT)) * H, b_dg.to((dg).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def chunk_mesa_net_h_kv_bwd_intra_fn(
    q_star,
    k,
    v,
    beta,
    h_kv,
    dh_kv,
    g,
    do,
    cu_seqlens,
    chunk_size=64,
    chunk_indices: torch.LongTensor | None = None,
):
    # share memory is not large enough for a single fused kernel
    if not check_shared_mem('ampere'):
        return chunk_mesa_net_h_kv_bwd_intra_separate_fn(
            q_star=q_star,
            k=k,
            v=v,
            beta=beta,
            h_kv=h_kv,
            dh_kv=dh_kv,
            g=g,
            do=do,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    dq = torch.empty_like(q_star, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    grid = (NT, B * H)
    chunk_mesa_net_h_kv_bwd_intra_kernel[grid](
        q_star=q_star,
        k=k,
        v=v,
        beta=beta,
        h_kv=h_kv,
        g=g,
        do=do,
        dh_kv=dh_kv,
        dq=dq,
        dk_beta=dk,
        dg=dg,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, dk, dv, dg
