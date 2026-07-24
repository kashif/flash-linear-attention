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
from fla.utils import IS_INTEL_ALCHEMIST, autotune_cache_kwargs, check_shared_mem

# https://github.com/intel/intel-xpu-backend-for-triton/issues/3449
triton_config = {'grf_mode': 'large'} if IS_INTEL_ALCHEMIST else {}


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    A_ab_inv,
    A_ak,
    ag,
    v,
    dw,
    du,
    dv,
    dv0,
    dag,
    dAak,
    dAab,
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

    desc_Aak_t = make_tensor_descriptor(A_ak + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_Aab_inv_t = make_tensor_descriptor(A_ab_inv + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_dAak = make_tensor_descriptor(dAak + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_dAab = make_tensor_descriptor(dAab + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])

    b_A_ab_inv_t = tl.trans(desc_Aab_inv_t.load([i_t * BT, 0]))
    b_A_ak_t = tl.trans(desc_Aak_t.load([i_t * BT, 0]))
    b_A_ak_t = tl.where(tl.arange(0, BT)[:, None] < tl.arange(0, BT)[None, :], b_A_ak_t, 0)
    b_A_ab_inv_t = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A_ab_inv_t, 0)
    b_A_tmp_t = tl.dot(b_A_ak_t, b_A_ab_inv_t).to(v.dtype.element_ty)
    b_dA_tmp = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_dv = make_tensor_descriptor(dv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_dv0 = make_tensor_descriptor(dv0 + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        desc_du = make_tensor_descriptor(du + (bos*H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_du = desc_du.load([i_t * BT, i_v * BV])
        b_dA_tmp += tl.dot(b_du.to(b_v.dtype), tl.trans(b_v))
        b_dv0 = desc_dv0.load([i_t * BT, i_v * BV])
        b_dv = b_dv0 + tl.dot(b_A_tmp_t, b_du)
        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))

    m_i = tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :]
    b_dA_tmp = tl.where(m_i, b_dA_tmp, 0)
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp)
    b_dA_ak = tl.where(m_i, b_dA_ak, 0)
    desc_dAak.store([i_t * BT, 0], b_dA_ak)
    b_dA_ab_inv = tl.dot(b_dA_tmp, b_A_ak_t)

    for i_k in range(tl.cdiv(K, BK)):
        desc_ag = make_tensor_descriptor(ag + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dag = make_tensor_descriptor(dag + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_dw = make_tensor_descriptor(dw + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        b_ag = desc_ag.load([i_t * BT, i_k * BK])
        b_dw = desc_dw.load([i_t * BT, i_k * BK])
        b_dA_ab_inv += tl.dot(b_dw, tl.trans(b_ag))
        b_dag = tl.dot(b_A_ab_inv_t.to(b_dw.dtype), b_dw)
        desc_dag.store([i_t * BT, i_k * BK], b_dag.to(desc_dag.dtype))

    # if we know dL/dA^(-1), for dL/dA, we can use the following formula:
    # dL/dA = -(A^(-1))^T @ (dL/dA^(-1)) @ (A^(-1))^T
    # in the fwd pass we use fwd substitution to calculate (I-lower(A_ab))^-1.
    # denote A = I - lower(A_ab), B = A^-1
    # in the backward pass.
    # dL/dA = -(B)^T @ (dL/dB) @ B^T
    # dL/dA_ab = lower(B^T @ dL/dB @ B^T)
    b_dA_ab_inv = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dA_ab_inv, 0)
    b_dA_ab_inv = tl.dot(b_A_ab_inv_t, b_dA_ab_inv)
    b_dA_ab_inv = tl.dot(b_dA_ab_inv, b_A_ab_inv_t)
    b_dA_ab_inv = tl.where(m_i, b_dA_ab_inv, 0)
    desc_dAab.store([i_t * BT, 0], b_dA_ab_inv)


def chunk_dplr_bwd_wy(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv, A_ak, v, ag, dw, du])
    B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64) if check_shared_mem() else min(max(triton.next_power_of_2(V), 16), 32)

    dA_ab = torch.empty_like(A_ab_inv, dtype=torch.float)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)

    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        ag=ag,
        v=v,
        dw=dw,
        du=du,
        dv=dv,
        dv0=dv0,
        dag=dag,
        dAak=dA_ak,
        dAab=dA_ab,
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
    return dA_ab, dA_ak, dv, dag
