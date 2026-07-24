# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem

# Blackwell can select unstable Triton configs for prepare_wy_repr_bwd_kernel
# during autotuning (see #913). Restrict it to the config that has been
# validated on B200 until the wider config space is re-validated.
PREPARE_WY_REPR_BWD_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]
PREPARE_WY_REPR_BWD_NUM_STAGES = [4] if IS_NVIDIA_BLACKWELL else [2, 3, 4]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    b_b = tl.load(beta + bos*HV + i_h + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

    desc_A = make_tensor_descriptor(A + (bos*HV + i_h) * BT, [T, BT], [HV*BT, 1], [BT, BT])
    b_A = desc_A.load([i_t * BT, 0])

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*HV + i_h) * V, [T, V], [HV*V, 1], [BT, BV])
        desc_u = make_tensor_descriptor(u + (bos*HV + i_h) * V, [T, V], [HV*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        desc_u.store([i_t * BT, i_v * BV], b_u.to(desc_u.dtype))

    if USE_G:
        pass
        b_g = exp2(tl.load(g + (bos*HV + i_h) + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0))

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h // (HV // H)) * K, [T, K], [H*K, 1], [BT, BK])
        desc_w = make_tensor_descriptor(w + (bos*HV + i_h) * K, [T, K], [HV*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in PREPARE_WY_REPR_BWD_NUM_WARPS
        for num_stages in PREPARE_WY_REPR_BWD_NUM_STAGES
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    desc_A = make_tensor_descriptor(A + (bos*HV + i_h) * BT, [T, BT], [HV*BT, 1], [BT, BT])

    b_b = tl.load(beta + (bos*HV + i_h) + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.trans(desc_A.load([i_t * BT, 0]))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        pass
        b_g = tl.load(g + (bos*HV + i_h) + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
        b_g_exp = exp2(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h // (HV // H)) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dk = make_tensor_descriptor(dk + (bos*HV + i_h) * K, [T, K], [HV*K, 1], [BT, BK])
        desc_dw = make_tensor_descriptor(dw + (bos*HV + i_h) * K, [T, K], [HV*K, 1], [BT, BK])
        # [BT, BK]
        b_k = desc_k.load([i_t * BT, i_k * BK])
        if USE_G:
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = b_k * b_b[:, None]
        b_dw = desc_dw.load([i_t * BT, i_k * BK])

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkbg * b_k, 1)
        desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*HV + i_h) * V, [T, V], [HV*V, 1], [BT, BV])
        desc_dv = make_tensor_descriptor(dv + (bos*HV + i_h) * V, [T, V], [HV*V, 1], [BT, BV])
        desc_du = make_tensor_descriptor(du + (bos*HV + i_h) * V, [T, V], [HV*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = desc_du.load([i_t * BT, i_v * BV])
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    if USE_G:
        b_dA *= exp2(b_g[:, None] - b_g[None, :])

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    tl.debug_barrier()
    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h // (HV // H)) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dk = make_tensor_descriptor(dk + (bos*HV + i_h) * K, [T, K], [HV*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_kt = tl.trans(b_k)
        b_kb = b_k * b_b[:, None]

        b_A += tl.dot(b_k, b_kt)
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))
        b_dk += desc_dk.load([i_t * BT, i_k * BK])

        desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))
    tl.store(db + (bos*HV + i_h) + (i_t * BT + tl.arange(0, BT)) * HV, b_db.to((db + (bos*HV + i_h)).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)

    b_A *= b_b[:, None]
    if USE_G:
        b_AdA = b_dA * b_A
        b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
        tl.store(dg + (bos*HV + i_h) + (i_t * BT + tl.arange(0, BT)) * HV, b_dg.to((dg + (bos*HV + i_h)).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


@dispatch('gated_delta_rule')
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


@dispatch('gated_delta_rule')
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


fwd_recompute_w_u = recompute_w_u_fwd
bwd_prepare_wy_repr = prepare_wy_repr_bwd
