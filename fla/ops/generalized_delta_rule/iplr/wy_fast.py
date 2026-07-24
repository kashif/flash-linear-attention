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
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32(
    a,
    b,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,  # dummy placeholder
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

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_a = make_tensor_descriptor(a + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_b = make_tensor_descriptor(b + (bos * H + i_h) * K, [T, K], [K*H, 1], [BT, BK])
        b_a = desc_a.load([i_t * BT, i_k * BK])
        b_b = tl.trans(desc_b.load([i_t * BT, i_k * BK]))
        b_A += tl.dot(b_a, b_b)

    b_A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]

    desc_A = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
    a,
    b,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
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

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_a1 = make_tensor_descriptor(a + (bos * H + i_h) * K, [T, K], [H*K, 1], [BC, BK])
        desc_a2 = make_tensor_descriptor(a + (bos * H + i_h) * K, [T, K], [H*K, 1], [BC, BK])
        desc_b1 = make_tensor_descriptor(b + (bos * H + i_h) * K, [T, K], [K*H, 1], [BC, BK])
        desc_b2 = make_tensor_descriptor(b + (bos * H + i_h) * K, [T, K], [K*H, 1], [BC, BK])
        b_a1 = desc_a1.load([i_t * BT, i_k * BK])
        b_a2 = desc_a2.load([i_t * BT + BC, i_k * BK])
        b_b1 = tl.trans(desc_b1.load([i_t * BT, i_k * BK]))
        b_b2 = tl.trans(desc_b2.load([i_t * BT + BC, i_k * BK]))
        b_A += tl.dot(b_a1, b_b1, allow_tf32=False)
        b_A2 += tl.dot(b_a2, b_b2, allow_tf32=False)
        b_A3 += tl.dot(b_a2, b_b1, allow_tf32=False)

    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3, allow_tf32=False), b_A, allow_tf32=False)

    desc_A1 = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A2 = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A3 = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A4 = make_tensor_descriptor(A + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A1.store([i_t * BT, 0], b_A.to(desc_A1.dtype))
    desc_A2.store([i_t * BT + BC, BC], b_A2.to(desc_A2.dtype))
    desc_A3.store([i_t * BT + BC, 0], b_A3.to(desc_A3.dtype))
    # causal mask
    desc_A4.store([i_t * BT, BC], tl.zeros([BC, BC], dtype=tl.float32).to(desc_A4.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(
    w,
    u,
    a,
    k,
    v,
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

    b_A = desc_A.load([i_t * BT, 0])
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_k = make_tensor_descriptor(k + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_a = make_tensor_descriptor(a + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_w = make_tensor_descriptor(w + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_a = desc_a.load([i_t * BT, i_k * BK])
        b_w = tl.dot(b_A, b_a)
        b_Aak += tl.dot(b_a, tl.trans(b_k))
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype))

    b_Aak = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_Aak, 0)
    b_Aak = b_Aak.to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_u = make_tensor_descriptor(u + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_v = tl.dot(b_Aak, b_v).to(v.dtype.element_ty)
        b_u = tl.dot(b_A, b_v)
        desc_u.store([i_t * BT, i_v * BV], b_u.to(desc_u.dtype))


def prepare_wy_repr_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = a.shape
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    BK = min(max(triton.next_power_of_2(K), 16), 64)

    A = torch.empty(B, T, H, BT, device=a.device, dtype=a.dtype)
    fwd_fn = prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else prepare_wy_repr_fwd_kernel_chunk32

    fwd_fn[(NT, B * H)](
        a=a,
        b=b,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BC=BC,
    )
    w, u = wu_fwd(
        a=a,
        v=v,
        k=k,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return w, u, A


def wu_fwd(
    a: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *a.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    u = torch.empty_like(v)
    w = torch.empty_like(a)
    wu_fwd_kernel[(NT, B*H)](
        a=a,
        v=v,
        w=w,
        u=u,
        A=A,
        k=k,
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


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
