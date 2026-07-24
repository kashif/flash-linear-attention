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
from fla.ops.utils.op import gather
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_GATHER_SUPPORTED, autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,  # placeholder, do not delete
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
    desc_Aab = make_tensor_descriptor(A_ab + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_Aab_inv = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    b_A_ab = desc_Aab.load([i_t * BT, 0])
    b_A_ab = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A_ab, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    desc_Aab_inv.store([i_t * BT, 0], b_A_ab.to(desc_Aab_inv.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr = IS_GATHER_SUPPORTED,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    desc_A1 = make_tensor_descriptor(A_ab + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A2 = make_tensor_descriptor(A_ab + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A3 = make_tensor_descriptor(A_ab + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A_inv1 = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A_inv2 = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A_inv3 = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_A_inv4 = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])

    b_A = desc_A1.load([i_t * BT, 0])
    b_A2 = desc_A2.load([i_t * BT + BC, BC])
    b_A3 = desc_A3.load([i_t * BT + BC, 0])
    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BC], i, dtype=tl.int16)
            # [1, BK] -> [BK]
            b_a = tl.sum(gather(b_A, row_idx, axis=0), 0)
            b_a2 = tl.sum(gather(b_A2, row_idx, axis=0), 0)
        else:
            mask = tl.arange(0, BC) == i
            b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
            b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        mask = tl.arange(0, BC) == i
        # b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        # b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    # tl.debug_barrier()
    desc_A_inv1.store([i_t * BT, 0], b_A.to(desc_A_inv1.dtype, fp_downcast_rounding="rtne"))
    desc_A_inv2.store([i_t * BT + BC, BC], b_A2.to(desc_A_inv2.dtype, fp_downcast_rounding="rtne"))
    desc_A_inv3.store([i_t * BT + BC, 0], b_A3.to(desc_A_inv3.dtype, fp_downcast_rounding="rtne"))
    # causal mask
    desc_A_inv4.store([i_t * BT, BC], tl.zeros([BC, BC], dtype=tl.float32).to(desc_A_inv4.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(
    w,
    u,
    ag,
    v,
    A_ab_inv,
    A_ak,
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
    o_s = tl.arange(0, BT)

    desc_A_ab_inv = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_A_ak = make_tensor_descriptor(A_ak + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])

    b_Aab_inv = desc_A_ab_inv.load([i_t * BT, 0])
    b_Aak = desc_A_ak.load([i_t * BT, 0])
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    # let's use tf32 here
    b_Aak = tl.dot(b_Aab_inv, b_Aak)
    # (SY 01/04) should be bf16 or tf32? To verify.
    b_Aak = b_Aak.to(v.dtype.element_ty, fp_downcast_rounding="rtne")
    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding="rtne")

    for i_k in range(tl.cdiv(K, BK)):
        desc_ag = make_tensor_descriptor(ag + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_w = make_tensor_descriptor(w + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_ag = desc_ag.load([i_t * BT, i_k * BK])
        b_w = tl.dot(b_Aab_inv, b_ag)  # both bf16 or fp16
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype, fp_downcast_rounding="rtne"))

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_u = make_tensor_descriptor(u + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_u = tl.dot(b_Aak, b_v)  # both bf16 or fp16
        desc_u.store([i_t * BT, i_v * BV], b_u.to(desc_u.dtype, fp_downcast_rounding="rtne"))


def wu_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)

    w = torch.empty_like(ag)
    u = torch.empty_like(v)
    wu_fwd_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
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


def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, _ = ag.shape
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    fwd_fn = prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else prepare_wy_repr_fwd_kernel_chunk32
    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
    )
    w, u = wu_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )
    return w, u, A_ab_inv


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
