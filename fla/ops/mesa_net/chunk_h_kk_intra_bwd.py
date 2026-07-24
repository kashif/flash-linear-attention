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


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kk_bwd_intra_kernel(
    k,
    beta,
    h,
    dh,
    g,
    q_star,
    dq,
    dk,
    dg,
    dbeta,
    dk_beta,
    dlamb,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
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
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # offset calculation
    q_star += (bos * H + i_h) * V
    dq += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    k += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dlamb += (i_tg * H + i_h).to(tl.int64) * K
    beta += (bos * H + i_h)
    dbeta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    b_g = tl.load(g + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_gk = tl.where(m_t, exp2(b_g_last - b_g), 0)

    desc_q_star = make_tensor_descriptor(q_star, [T, V], [H*V, 1], [BT, BV])
    b_q_star = desc_q_star.load([i_t * BT, 0])
    desc_dq = make_tensor_descriptor(dq, [T, V], [H*V, 1], [BT, BV])
    b_dq = desc_dq.load([i_t * BT, 0])
    b_dlamb = -tl.sum(b_q_star * b_dq, axis=0)
    desc_dlamb = make_tensor_descriptor(dlamb, [K], [1], [BK])
    desc_dlamb.store([0], b_dlamb.to(desc_dlamb.dtype))

    desc_h = make_tensor_descriptor(h, [K, V], [V, 1], [BK, BV])
    desc_dh = make_tensor_descriptor(dh, [K, V], [V, 1], [BK, BV])
    b_beta = tl.load(beta + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, BK])
    b_v = desc_k.load([i_t * BT, 0])
    b_k = (b_v * b_beta[:, None]).to(b_v.dtype)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp2(b_g[:, None] - b_g[None, :]), 0)
    b_s = tl.dot(b_q_star, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_dq, tl.trans(b_v))
    b_dv += tl.dot(tl.trans(b_s.to(b_dq.dtype)), b_dq)
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_ds = b_ds * b_m
    b_dk += tl.dot(tl.trans(b_ds.to(b_q_star.dtype)), b_q_star)

    b_h = tl.trans(desc_h.load([0, 0]))
    b_dg += tl.sum(tl.dot(b_dq, tl.trans(b_h)) * exp2(b_g)[:, None] * b_q_star, axis=1)
    b_dh = tl.trans(desc_dh.load([0, 0]))
    b_dk2 = tl.dot(b_v, b_dh.to(b_v.dtype)) * b_gk[:, None]
    b_dg -= tl.sum(b_dk2 * b_k, axis=1)
    b_dg_last += tl.sum(b_dk2 * b_k)
    b_dk += b_dk2
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_gk[:, None]
    b_dh = b_dh * b_h
    b_dg_last += tl.sum(b_dh) * exp2(b_g_last)

    desc_dk_beta = make_tensor_descriptor(dk_beta, [T, K], [H*K, 1], [BT, BK])
    b_dk -= desc_dk_beta.load([i_t * BT, 0])
    b_dbeta = tl.sum(b_dk * b_v, axis=1)
    b_dk = b_dk * b_beta[:, None] + b_dv
    b_dk = -b_dk

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    desc_dk = make_tensor_descriptor(dk, [T, K], [H*K, 1], [BT, BK])
    desc_dk.store([i_t * BT, 0], b_dk.to(desc_dk.dtype))
    tl.store(dg + (i_t * BT + tl.arange(0, BT)) * H, -b_dg.to((dg).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)
    tl.store(dbeta + (i_t * BT + tl.arange(0, BT)) * H, -b_dbeta.to((dbeta).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def chunk_mesa_net_h_kk_bwd_intra_fn(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    q_star: torch.Tensor,
    dq: torch.Tensor,
    dk_beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K = k.shape
    V = K
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # CONST_TILING = 64
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dlamb = torch.empty(B, NT, H, K, dtype=torch.float32, device=k.device)
    grid = (NT, B * H)

    chunk_mesa_net_h_kk_bwd_intra_kernel[grid](
        k=k,
        h=h,
        dh=dh,
        g=g,
        q_star=q_star,
        beta=beta,
        dbeta=dbeta,
        dq=dq,
        dk=dk,
        dk_beta=dk_beta,
        dg=dg,
        dlamb=dlamb,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dlamb = dlamb.sum([0, 1])
    return dk, dg, dlamb, dbeta
