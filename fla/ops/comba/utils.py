# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_fwd_kernel(
    g,
    g0,
    g1,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
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

    # [BT]
    b_g = tl.load(g + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0).to(tl.float32)
    if HAS_SCALE:
        b_g = b_g * scale
    b_g1 = tl.cumsum(b_g, axis=0)
    b_g0 = b_g1 - b_g
    tl.store(g0 + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_g0.to((g0 + bos*H + i_h).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)
    tl.store(g1 + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_g1.to((g1 + bos*H + i_h).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def chunk_comba_cumsum_scalar_fwd(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g0, g1 = torch.empty_like(g, dtype=output_dtype or g.dtype), torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_comba_cumsum_scalar_fwd_kernel[grid](
        g,
        g0,
        g1,
        scale,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return g0, g1


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_bwd_kernel(
    dg0,
    dgr,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
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

    # [BT]
    """
    b_dg:   1,2,3,4
    b_dg0:  0,1,2,3
    b_temp: 0,1,3,6
    b_dz:   6
    b_dgr:  6,5,3,0
    """
    b_dg0 = tl.load(dg0 + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0).to(tl.float32)
    b_temp = tl.cumsum(b_dg0, axis=0)
    b_dz = tl.sum(b_dg0, axis=0)
    b_dgr = -b_temp + b_dz[None]
    tl.store(dgr + bos*H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_dgr.to((dgr + bos*H + i_h).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def chunk_comba_cumsum_scalar_bwd(
    dg0: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H = dg0.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dg = torch.empty_like(dg0, dtype=output_dtype or dg0.dtype)
    grid = (NT, B * H)
    chunk_comba_cumsum_scalar_bwd_kernel[grid](
        dg0,
        dg,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return dg
