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
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]

BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BK_LIST
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_o(
    qg,
    v,
    v_new,
    A_qk,
    A_qb,
    h,
    o,
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

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_qg = make_tensor_descriptor(qg + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_h = make_tensor_descriptor(h + (i_tg * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        b_qg = desc_qg.load([i_t * BT, i_k * BK])
        b_h = desc_h.load([i_k * BK, i_v * BV])
        b_o += tl.dot(b_qg, b_h)

    desc_Aqk = make_tensor_descriptor(A_qk + (bos * H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_Aqb = make_tensor_descriptor(A_qb + (bos * H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
    desc_v_new = make_tensor_descriptor(v_new + (bos * H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
    desc_o = make_tensor_descriptor(o + (bos * H + i_h) * V, [T, V], [H*V, 1], [BT, BV])

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_Aqk = desc_Aqk.load([i_t * BT, 0])
    b_Aqb = desc_Aqb.load([i_t * BT, 0])
    b_Aqk = tl.where(m_s, b_Aqk, 0)
    b_Aqb = tl.where(m_s, b_Aqb, 0)
    b_v = desc_v.load([i_t * BT, i_v * BV])
    b_v_new = desc_v_new.load([i_t * BT, i_v * BV])
    b_o = b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_v_new.dtype), b_v_new)
    desc_o.store([i_t * BT, i_v * BV], b_o.to(desc_o.dtype))


def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_fwd_kernel_o[grid](
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o
