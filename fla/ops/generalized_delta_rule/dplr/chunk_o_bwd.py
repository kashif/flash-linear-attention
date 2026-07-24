# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]

BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BV', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dAu(
    v,
    do,
    v_new,
    A_qb,
    dA_qk,
    dA_qb,
    dv_new,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA_qb = tl.zeros([BT, BT], dtype=tl.float32)

    desc_A_qb = make_tensor_descriptor(A_qb + (bos * H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])

    b_A_qb = desc_A_qb.load([i_t * BT, 0])
    # causal mask
    b_A_qb = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_A_qb, 0.).to(b_A_qb.dtype)

    for i_v in range(tl.cdiv(V, BV)):
        desc_do = make_tensor_descriptor(do + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_v_new = make_tensor_descriptor(v_new + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_dv_new = make_tensor_descriptor(dv_new + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = tl.trans(desc_v.load([i_t * BT, i_v * BV]))
        b_do = desc_do.load([i_t * BT, i_v * BV])
        b_v_new = tl.trans(desc_v_new.load([i_t * BT, i_v * BV]))
        b_dA_qk += tl.dot(b_do, b_v)
        b_dA_qb += tl.dot(b_do, b_v_new)
        b_dv_new = tl.dot(tl.trans(b_A_qb), b_do)
        # for recurrent
        desc_dv_new.store([i_t * BT, i_v * BV], b_dv_new.to(desc_dv_new.dtype))

    desc_dA_qk = make_tensor_descriptor(dA_qk + (bos * H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_dA_qb = make_tensor_descriptor(dA_qb + (bos * H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA_qk = tl.where(m_s, b_dA_qk * scale, 0.)
    desc_dA_qk.store([i_t * BT, 0], b_dA_qk.to(desc_dA_qk.dtype))
    b_dA_qb = tl.where(m_s, b_dA_qb * scale, 0.)
    desc_dA_qb.store([i_t * BT, 0], b_dA_qb.to(desc_dA_qb.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit
def chunk_dplr_bwd_o_kernel(
    v,
    v_new,
    h,
    do,
    dh,
    dk,
    db,
    w,
    dq,
    dv,
    dw,
    gk,
    dgk_last,
    k,
    b,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    # offset calculation
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    dk += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    w += (bos * H + i_h) * K

    dgk_last += (i_tg * H + i_h) * K
    gk += (bos * H + i_h) * K

    stride_qk = H*K
    stride_vo = H*V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_db = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk_last = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v, [T, V], [stride_vo, 1], [BT, BV])
        desc_v_new = make_tensor_descriptor(v_new, [T, V], [stride_vo, 1], [BT, BV])
        desc_do = make_tensor_descriptor(do, [T, V], [stride_vo, 1], [BT, BV])
        desc_h = make_tensor_descriptor(h, [K, V], [V, 1], [BK, BV])
        desc_dh = make_tensor_descriptor(dh, [K, V], [V, 1], [BK, BV])
        # [BT, BV]
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_v_new = desc_v_new.load([i_t * BT, i_v * BV])
        b_do = desc_do.load([i_t * BT, i_v * BV])
        # [BV, BK]
        b_h = tl.trans(desc_h.load([i_k * BK, i_v * BV]))
        b_dh = tl.trans(desc_dh.load([i_k * BK, i_v * BV]))
        b_dgk_last += tl.sum((b_h * b_dh).to(tl.float32), axis=0)

        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        b_db += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
        desc_dv = make_tensor_descriptor(dv, [T, V], [stride_vo, 1], [BT, BV])
        b_dv = desc_dv.load([i_t * BT, i_v * BV])
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    m_k = (i_k*BK+tl.arange(0, BK)) < K
    last_idx = min(i_t * BT + BT, T) - 1
    b_gk_last = tl.load(gk + last_idx * stride_qk + i_k*BK + tl.arange(0, BK), mask=m_k, other=float('-inf'))
    b_dgk_last *= exp2(b_gk_last)
    desc_k = make_tensor_descriptor(k, [T, K], [stride_qk, 1], [BT, BK])
    desc_b = make_tensor_descriptor(b, [T, K], [stride_qk, 1], [BT, BK])
    b_k = desc_k.load([i_t * BT, i_k * BK])
    b_b = desc_b.load([i_t * BT, i_k * BK])
    b_dgk_last += tl.sum(b_k * b_dk, axis=0)
    b_dgk_last += tl.sum(b_b * b_db, axis=0)
    tl.store(dgk_last + tl.arange(0, BK) + i_k * BK, b_dgk_last, mask=m_k)

    desc_dw = make_tensor_descriptor(dw, [T, K], [stride_qk, 1], [BT, BK])
    desc_dk = make_tensor_descriptor(dk, [T, K], [stride_qk, 1], [BT, BK])
    desc_db = make_tensor_descriptor(db, [T, K], [stride_qk, 1], [BT, BK])
    desc_dq = make_tensor_descriptor(dq, [T, K], [stride_qk, 1], [BT, BK])
    desc_dw.store([i_t * BT, i_k * BK], b_dw.to(desc_dw.dtype))
    desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))
    desc_db.store([i_t * BT, i_k * BK], b_db.to(desc_db.dtype))
    desc_dq.store([i_t * BT, i_k * BK], b_dq.to(desc_dq.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
        for BK in BK_LIST
        for BV in BK_LIST
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def chunk_dplr_bwd_kernel_dv(
    A_qk,
    kg,
    do,
    dv,
    dh,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    A_qk += (bos * H + i_h) * BT
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    dh += (i_tg * H + i_h) * K*V

    stride_qk = H*K
    stride_vo = H*V
    stride_A = H*BT

    for i_k in range(tl.cdiv(K, BK)):
        desc_dh = make_tensor_descriptor(dh, [K, V], [V, 1], [BK, BV])
        desc_kg = make_tensor_descriptor(kg, [T, K], [stride_qk, 1], [BT, BK])
        b_dh = desc_dh.load([i_k * BK, i_v * BV])
        b_kg = desc_kg.load([i_t * BT, i_k * BK])
        b_dv += tl.dot(b_kg, b_dh.to(b_kg.dtype))

    desc_Aqk = make_tensor_descriptor(A_qk, [T, BT], [stride_A, 1], [BT, BT])
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], tl.trans(desc_Aqk.load([i_t * BT, 0])), 0)
    desc_do = make_tensor_descriptor(do, [T, V], [stride_vo, 1], [BT, BV])
    desc_dv = make_tensor_descriptor(dv, [T, V], [stride_vo, 1], [BT, BV])
    b_do = desc_do.load([i_t * BT, i_v * BV])
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))


def chunk_dplr_bwd_dv(
    A_qk: torch.Tensor,
    kg: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_bwd_kernel_dv[grid](
        A_qk=A_qk,
        kg=kg,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


def chunk_dplr_bwd_o(
    k: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    scale: float = 1.0,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *w.shape, v.shape[-1]

    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = min(max(triton.next_power_of_2(K), 16), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    BV = min(max(triton.next_power_of_2(V), 16), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    grid = (NK, NT, B * H)

    dgk_last = torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)

    chunk_dplr_bwd_o_kernel[grid](
        k=k,
        b=b,
        v=v,
        v_new=v_new,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        db=db,
        dgk_last=dgk_last,
        w=w,
        dv=dv,
        dw=dw,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_dAu(
    v: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    A_qb: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, V = v.shape
    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if check_shared_mem('ampere'):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem('ada'):  # 4090
        BV = min(max(triton.next_power_of_2(V), 16), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    dA_qk = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](
        v=v,
        do=do,
        v_new=v_new,
        A_qb=A_qb,
        dA_qk=dA_qk,
        dA_qb=dA_qb,
        dv_new=dv_new,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dv_new, dA_qk, dA_qb
