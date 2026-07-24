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
from fla.ops.utils.op import exp
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_inter(
    q,
    k,
    h,
    gv,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
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

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_q = make_tensor_descriptor(q + (bos * HQ + i_hq) * K, [T, K], [HQ*K, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k + (bos * H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
        desc_h = make_tensor_descriptor(h + (i_tg * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])

        # [BT, BK]
        b_q = desc_q.load([i_t * BT, i_k * BK])
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
        # [BK, BV]
        b_h = desc_h.load([i_k * BK, i_v * BV])
        # [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k)
    desc_g = make_tensor_descriptor(gv + (bos * H + i_h) * V, [T, V], [H*V, 1], [BT, BV])
    desc_o = make_tensor_descriptor(o + (bos * HQ + i_hq) * V, [T, V], [HQ*V, 1], [BT, BV])
    desc_A = make_tensor_descriptor(A + (bos * HQ + i_hq) * BT, [T, BT], [HQ*BT, 1], [BT, BT])
    # [BT, BV]
    b_g = desc_g.load([i_t * BT, i_v * BV])
    b_o = b_o * exp(b_g)
    desc_o.store([i_t * BT, i_v * BV], b_o.to(desc_o.dtype))

    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_intra(
    v,
    gv,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    desc_g = make_tensor_descriptor(gv + (bos * H + i_h) * V, [T, V], [H*V, 1], [BC, BV])
    p_gn = gv + (bos + min(i_t * BT + i_i * BC, T)) * H*V + i_h * V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        desc_A = make_tensor_descriptor(A + (bos*HQ+i_hq) * BT, [T, BT], [HQ*BT, 1], [BC, BC])
        desc_v = make_tensor_descriptor(v + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        desc_gv = make_tensor_descriptor(gv + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        # [BC, BV]
        b_v = desc_v.load([i_t * BT + i_j * BC, i_v * BV])
        b_gv = desc_gv.load([i_t * BT + i_j * BC, i_v * BV])
        b_vg = (b_v * exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = desc_A.load([i_t*BT+i_i*BC, i_j * BC])
        b_o += tl.dot(b_A, b_vg)
    # [BC, BV]
    b_g = desc_g.load([i_t * BT + i_i * BC, i_v * BV])
    b_o *= exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * HQ*BT + i_hq * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_v = v + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        p_gv = gv + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * exp(b_g - b_gv[None, :])
        # avoid 0 * inf = inf
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    desc_o = make_tensor_descriptor(o + (bos*HQ + i_hq) * V, [T, V], [HQ*V, 1], [BC, BV])
    b_o += desc_o.load([i_t * BT + i_i * BC, i_v * BV])
    desc_o.store([i_t * BT + i_i * BC, i_v * BV], b_o.to(desc_o.dtype))


def chunk_oja_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[2]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H

    o = v.new_empty(B, T, HQ, V)
    A = q.new_empty(B, T, HQ, BT)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HQ)
    chunk_oja_fwd_inter[grid](
        q,
        k,
        h,
        gv,
        o,
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
    )

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT * NC, B * HQ)
    chunk_oja_fwd_intra[grid](
        v,
        gv,
        o,
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        num_warps=4,
        num_stages=2
    )
    return A, o


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=["BT"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dA(
    v,
    gv,
    do,
    dA,
    chunk_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    # [BC, BC]
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        desc_v = make_tensor_descriptor(v + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        desc_gv = make_tensor_descriptor(gv + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        p_gn = gv + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
        desc_g = make_tensor_descriptor(gv + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        desc_do = make_tensor_descriptor(do + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        # [BC, BV]
        b_g = desc_g.load([i_t*BT + i_i*BC, i_v*BV])
        b_do = desc_do.load([i_t*BT + i_i*BC, i_v*BV])
        b_do = (b_do * exp(b_g - b_gn[None, :]) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.trans(desc_v.load([i_t*BT + i_j*BC, i_v * BV]))
        b_gv = tl.trans(desc_gv.load([i_t*BT + i_j*BC, i_v * BV]))
        b_vg = (b_v * exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        desc_g = make_tensor_descriptor(gv + (bos*H + i_h) * V, [T, V], [H*V, 1], [BC, BV])
        desc_do = make_tensor_descriptor(do + (bos*H + i_h) * V, [T, V], [H*V, 1], [BC, BV])
        p_v = v + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        p_gv = gv + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        # [BC, BV]
        b_g = desc_g.load([i_t*BT + i_i*BC, i_v*BV])
        b_do = desc_do.load([i_t*BT + i_i*BC, i_v*BV]) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            # [BV,]
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            # [BC,]
            b_dAj = tl.sum(b_do * b_v[None, :] * exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += H*V
            p_gv += H*V
        b_dA = tl.where(m_dA, b_dA, 0.)

    desc_dA = make_tensor_descriptor(dA+((i_v*all+bos)*H+i_h)*BT, [T, BT], [H*BT, 1], [BC, BC])
    desc_dA.store([i_t*BT+i_i*BC, i_j*BC], b_dA.to(dA.dtype.element_ty))


def chunk_oja_bwd_dA(
    v: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NV = triton.cdiv(V, BV)

    dA = v.new_empty(NV, B, T, H, BT)
    # 计算dA
    grid = (NV, NT * NC * NC, B * H)
    chunk_oja_bwd_kernel_dA[grid](
        v,
        gv,
        do,
        dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
    )
    dA = dA.sum(0, dtype=dA.dtype)

    return dA


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dqk(
    q,
    k,
    h,
    gv,
    A,
    dq,
    dk,
    dA,
    do,
    scale,
    cu_seqlens,
    chunk_indices,
    B,
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
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    # [B, T, H, BT]
    desc_q = make_tensor_descriptor(q + (bos*H+i_h) * K, [T, K], [H*K, 1], [BT, BK])
    desc_k = make_tensor_descriptor(k + (bos*H+i_h) * K, [T, K], [H*K, 1], [BT, BK])
    desc_A = make_tensor_descriptor(A + ((i_k*all+bos)*H+i_h)*BT, [T, BT], [H*BT, 1], [BT, BT])
    b_q = desc_q.load([i_t * BT, i_k * BK])
    b_k = desc_k.load([i_t * BT, i_k * BK])

    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    # 先计算do对应的dq
    for i_v in range(tl.cdiv(V, BV)):
        desc_h = make_tensor_descriptor(h + (i_tg * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        desc_do = make_tensor_descriptor(do + (bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_gv = make_tensor_descriptor(gv + (bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        b_h = tl.trans(desc_h.load([i_k * BK, i_v * BV]))
        b_do = desc_do.load([i_t * BT, i_v * BV])
        b_gv = desc_gv.load([i_t * BT, i_v * BV])
        b_do = (b_do * exp(b_gv) * scale).to(b_do.dtype)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))

    # 接着计算dA对应的dq, dk
    desc_dA = make_tensor_descriptor(dA + (bos*H + i_h) * BT, [T, BT], [H*BT, 1], [BT, BT])
    desc_dq = make_tensor_descriptor(dq + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
    desc_dk = make_tensor_descriptor(dk + (bos*H + i_h) * K, [T, K], [H*K, 1], [BT, BK])
    # [BT, BT]
    b_dA = desc_dA.load([i_t * BT, 0])
    # [BT, BK]
    b_dq += tl.dot(b_dA.to(b_q.dtype), b_k)
    b_dk = tl.dot(tl.trans(b_dA).to(b_q.dtype), b_q)

    desc_dq.store([i_t * BT, i_k * BK], b_dq.to(desc_dq.dtype))
    desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))


def chunk_oja_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gv: torch.Tensor,
    dA: torch.Tensor,
    do: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, K, V = *q.shape, gv.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    A = dA.new_empty(NK, B, T, H, BT)
    # 计算dA
    grid = (NK, NT, B * H)
    chunk_oja_bwd_kernel_dqk[grid](
        q,
        k,
        h,
        gv,
        A,
        dq,
        dk,
        dA,
        do,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV
    )

    A = A.sum(0, dtype=A.dtype)

    return A, dq, dk


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dv_o(
    v,
    g,
    o,
    A,
    do,
    dv,
    dv2,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    desc_gv = make_tensor_descriptor(g + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T)-1)*H*V + i_h*V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = desc_gv.load([i_t * BT + i_i * BC, i_v * BV])
    b_dvg = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        desc_g = make_tensor_descriptor(g + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        desc_A = make_tensor_descriptor(A + (bos*H+i_h) * BT, [T, BT], [H*BT, 1], [BC, BC])
        desc_do = make_tensor_descriptor(do + (bos*H+i_h) * V, [T, V], [H*V, 1], [BC, BV])
        # [BC, BV]
        b_g = desc_g.load([i_t * BT + i_j * BC, i_v * BV])
        b_do = desc_do.load([i_t*BT + i_j*BC, i_v*BV]) * exp(b_g - b_gn[None, :])
        # [BC, BC]
        b_A = tl.trans(desc_A.load([i_t*BT + i_j*BC, i_i*BC]))
        # [BC, BV]
        b_dvg += tl.dot(b_A, b_do.to(b_A.dtype))
    b_dv = b_dvg * exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)

    p_g = g + (bos + i_t * BT + i_i * BC) * H*V + i_h * V + o_v
    p_A = A + (bos + i_t*BT + i_i*BC) * H*BT + i_h * BT + o_c
    p_do = do + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_A = tl.load(p_A)
        # [BV,]
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += H * V
        p_A += H * BT
        p_do += H * V
    desc_o = make_tensor_descriptor(o + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    desc_v = make_tensor_descriptor(v + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    desc_do = make_tensor_descriptor(do + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    desc_dv = make_tensor_descriptor(dv + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    desc_dv2 = make_tensor_descriptor(dv2 + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
    desc_dg = make_tensor_descriptor(dg + (bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])

    b_o = desc_o.load([i_t*BT + i_i*BC, i_v*BV]).to(tl.float32)
    b_v = desc_v.load([i_t*BT + i_i*BC, i_v*BV]).to(tl.float32)
    b_do = desc_do.load([i_t*BT + i_i*BC, i_v*BV]).to(tl.float32)
    b_dv = b_dv + desc_dv.load([i_t*BT + i_i*BC, i_v*BV]).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    desc_dv2.store([i_t*BT + i_i*BC, i_v*BV], b_dv.to(desc_dv.dtype))
    desc_dg.store([i_t*BT + i_i*BC, i_v*BV], b_dg.to(desc_dg.dtype))


def chunk_oja_bwd_dv_o(
    v: torch.Tensor,
    gv: torch.Tensor,
    o: torch.Tensor,
    A: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    dv2 = torch.empty_like(v, dtype=torch.float)
    dgv = torch.empty_like(gv)
    # 计算dA
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT * NC, B * H)
    chunk_oja_bwd_kernel_dv_o[grid](
        v=v,
        g=gv,
        o=o,
        A=A,
        do=do,
        dv=dv,
        dv2=dv2,
        dg=dgv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        num_warps=4,
        num_stages=2
    )
    return dv2, dgv
