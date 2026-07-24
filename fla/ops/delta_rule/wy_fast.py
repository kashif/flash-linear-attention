# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import safe_dot
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

# Triton miscompiles `prepare_wy_repr_bwd_kernel` with num_warps=4 on Hopper
# (sm_90) for BT=64: it produces incorrect dk/dbeta (and can raise an illegal
# memory access), while num_warps=2 is correct (see #984). Restrict the Hopper
# autotune configs to num_warps=2 until the upstream compiler issue is resolved.
NUM_WARPS = [2] if IS_NVIDIA_HOPPER else [2, 4, 8]


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

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_u = make_tensor_descriptor(u + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False)
        desc_u.store([i_t * BT, i_v * BV], (b_u).to(desc_u.dtype))

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_w = make_tensor_descriptor(w + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    A,
    dw,
    du,
    dk,
    dv,
    dbeta,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    desc_A = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])

    b_beta = tl.load(beta + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_A = tl.trans(desc_A.load([i_t * BT, 0]))

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_dv = make_tensor_descriptor(dv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_du = make_tensor_descriptor(du + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])

        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = desc_du.load([i_t * BT, i_v * BV])
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)

        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dk = make_tensor_descriptor(dk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dw = make_tensor_descriptor(dw + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_dw = desc_dw.load([i_t * BT, i_k * BK])
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
        b_dk_beta = tl.dot(b_A, b_dw, allow_tf32=False)
        b_dk = b_dk_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)

        desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))

    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA, 0).to(k.dtype.element_ty)

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dk = make_tensor_descriptor(dk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_dk = desc_dk.load([i_t * BT, i_k * BK])
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += safe_dot(tl.trans(b_dA), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))

    tl.store(dbeta + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_dbeta.to((dbeta + bos*H + i_h).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def prepare_wy_repr_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    recompute_w_u_fwd_kernel[(NT, B*H)](
        k,
        v,
        beta,
        w,
        u,
        A,
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
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k,
        v,
        beta,
        A,
        dw,
        du,
        dk,
        dv,
        dbeta,
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
    return dk, dv, dbeta


fwd_prepare_wy_repr = prepare_wy_repr_fwd

bwd_prepare_wy_repr = prepare_wy_repr_bwd

fwd_recompute_w_u = recompute_w_u_fwd
