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
from fla.utils import autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_G': lambda args: args['g'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_comba_pkt_fwd_kernel(
    k,
    p,
    beta,
    g0,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_beta = tl.load(beta + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_p = make_tensor_descriptor(p + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_p = desc_p.load([i_t * BT, i_k * BK])
        b_pb = b_p * b_beta[:, None]
        b_A += tl.dot(b_pb.to(b_k.dtype), tl.trans(b_k))

    if USE_G:
        pass
        b_g0 = tl.load(g0 + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
        b_g = tl.load(g + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
        b_A = b_A * exp2(b_g0[:, None] - b_g[None, :])

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    desc_A = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [BT*H, 1], [BT, BT])
    desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))


def chunk_scaled_dot_comba_pkt_fwd(
    k: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    g0: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Compute beta \mathcal{A}(i-1/j) * P * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        p (torch.Tensor):
            The auxiliary key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g0 (torch.Tensor):
            The cumulative sum minus the original one of the gate tensor of shape `[B, T, H]`.
            Default: None
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_comba_pkt_fwd_kernel[(NT, B * H)](
        k=k,
        p=p,
        beta=beta,
        g0=g0,
        g=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return A


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    p,
    beta,
    g0,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    dp,
    dbeta,
    dg0,
    dg,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    desc_A = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])

    b_A = tl.trans(desc_A.load([i_t * BT, 0]))
    b_beta = tl.load(beta + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_g0 = tl.load(g0 + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_g0_exp = exp2(b_g0)
    b_g = tl.load(g + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg0 = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_p = make_tensor_descriptor(p + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dp = make_tensor_descriptor(dp + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dw = make_tensor_descriptor(dw + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_p = desc_p.load([i_t * BT, i_k * BK])
        b_p_beta_g0 = (b_p * b_beta[:, None] * b_g0_exp[:, None]).to(b_p.dtype)
        b_dw = desc_dw.load([i_t * BT, i_k * BK])
        b_dA += tl.dot(b_dw, tl.trans(b_p_beta_g0))
        b_dp_beta_g0 = tl.dot(b_A, b_dw)
        b_dp = b_dp_beta_g0 * b_beta[:, None] * b_g0_exp[:, None]
        b_dbeta += tl.sum(b_dp_beta_g0 * b_p * b_g0_exp[:, None], 1)
        b_dg0 += tl.sum(b_dp * b_p, 1)
        desc_dp.store([i_t * BT, i_k * BK], b_dp.to(desc_dp.dtype))

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_dv = make_tensor_descriptor(dv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_du = make_tensor_descriptor(du + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = desc_du.load([i_t * BT, i_v * BV])
        b_dA += tl.dot(b_du, tl.trans(b_v_beta))
        b_dv_beta = tl.dot(b_A, b_du)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA * exp2(b_g0[:, None] - b_g[None, :]), 0).to(k.dtype.element_ty)
    b_dA = b_dA.to(k.dtype.element_ty)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_p = make_tensor_descriptor(p + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dk = make_tensor_descriptor(dk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dp = make_tensor_descriptor(dp + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_p = desc_p.load([i_t * BT, i_k * BK])
        b_dp = desc_dp.load([i_t * BT, i_k * BK])
        b_p_beta = (b_p * b_beta[:, None]).to(b_p.dtype)
        b_A += tl.dot(b_p_beta, tl.trans(b_k))
        b_dp_beta = tl.dot(b_dA, b_k)
        b_dbeta += tl.sum(b_dp_beta * b_p, 1)
        b_dk = tl.dot(tl.trans(b_dA), b_p_beta)
        b_dp += b_dp_beta * b_beta[:, None]
        desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))
        desc_dp.store([i_t * BT, i_k * BK], b_dp.to(desc_dp.dtype))

    b_dA_A = b_dA * b_A
    b_dg0 += tl.sum(b_dA_A, axis=1)
    b_dg = - tl.sum(b_dA_A, axis=0)
    tl.store(dg + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, b_dg.to((dg + (bos*H + i_h)).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)
    tl.store(dg0 + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, b_dg0.to((dg0 + (bos*H + i_h)).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)
    tl.store(dbeta + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, b_dbeta.to((dbeta + (bos*H + i_h)).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
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
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    desc_A = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    b_beta = tl.load(beta + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_A = desc_A.load([i_t * BT, 0])
    b_g = exp2(tl.load(g + (bos*H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0))

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_u = make_tensor_descriptor(u + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        desc_u.store([i_t * BT, i_v * BV], b_u.to(desc_u.dtype))

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_w = make_tensor_descriptor(w + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    recompute_w_u_fwd_kernel[(NT, B*H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
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
    return w, u


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g0: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dp = torch.empty_like(p)
    dbeta = torch.empty_like(beta)
    dg0 = torch.empty_like(g0)
    dg = torch.empty_like(g)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        p=p,
        beta=beta,
        g0=g0,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dp=dp,
        dbeta=dbeta,
        dg0=dg0,
        dg=dg,
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
    return dk, dv, dp, dbeta, dg0, dg
