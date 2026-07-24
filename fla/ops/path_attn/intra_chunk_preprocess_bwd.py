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
from fla.utils import check_shared_mem


# episold
@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_bwd_kernel(
    q, k, w, w2, beta,
    AT,
    dA_local, dq, dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T,
    offsets, indices,
    HQ: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_beta = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dT = tl.zeros([BT, BT], dtype=tl.float32)

    desc_q = make_tensor_descriptor(q + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    desc_k = make_tensor_descriptor(k + (bos * H + i_h) * K, [T, K], [K*H, 1], [BT, BK])
    desc_w = make_tensor_descriptor(w + (bos * H + i_h) * K, [T, K], [K*H, 1], [BT, BK])
    desc_w2 = make_tensor_descriptor(w2 + (bos * H + i_h) * K, [T, K], [K*H, 1], [BT, BK])
    desc_T = make_tensor_descriptor(AT + (bos * H + i_h) * BT, [T, BT], [BT*H, 1], [BT, BT])
    b_w = desc_w.load([i_t * BT, 0])
    b_Twb = desc_w2.load([i_t * BT, 0])
    b_beta = tl.load(beta + (bos * H + i_h) + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_q = desc_q.load([i_t * BT, 0])
    b_k = desc_k.load([i_t * BT, 0])
    b_T = desc_T.load([i_t * BT, 0])
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)

    o_i = tl.arange(0, BT)
    b_qw = tl.where(o_i[:, None] >= o_i[None, :], tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, tl.trans(b_k)), 0).to(b_k.dtype)
    b_Twbk = tl.dot(b_T, b_wbk).to(b_w.dtype)

    desc_dA_local = make_tensor_descriptor(dA_local + (bos * HQ + i_hq) * BT, [T, BT], [BT*HQ, 1], [BT, BT])
    b_dA_local = desc_dA_local.load([i_t * BT, 0])

    # # Twb part qw part.
    desc_dq = make_tensor_descriptor(dq + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    b_dq = desc_dq.load([i_t * BT, 0])

    desc_dw1 = make_tensor_descriptor(dw1 + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    b_dw += desc_dw1.load([i_t * BT, 0])

    b_dqw = -tl.dot(b_dA_local, tl.trans(b_Twbk)) - tl.dot(b_dq.to(b_Twb.dtype), tl.trans(b_Twb))
    desc_dw2 = make_tensor_descriptor(dw2 + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    b_dTwb = -tl.dot(tl.trans(b_qw), b_dq) + desc_dw2.load([i_t * BT, 0])
    b_dT += tl.dot(b_dTwb.to(b_w_beta.dtype), tl.trans(b_w_beta))
    b_dw_beta += tl.dot(tl.trans(b_T), b_dTwb.to(b_T.dtype))

    b_dqw = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dqw, 0)
    b_dq += tl.dot(b_dA_local.to(b_k.dtype), b_k)
    b_dq += tl.dot(b_dqw.to(b_w.dtype), b_w)
    b_dw += tl.dot(tl.trans(b_dqw.to(b_q.dtype)), b_q)
    desc_q_new = make_tensor_descriptor(dq_new + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    desc_q_new.store([i_t * BT, 0], b_dq.to(dq_new.dtype.element_ty))

    # Twbk part
    desc_dk = make_tensor_descriptor(dk + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    b_dk = desc_dk.load([i_t * BT, 0])
    b_dTwbk = -tl.dot(tl.trans(b_qw), b_dA_local.to(b_qw.dtype)) - tl.dot(b_w, tl.trans(b_dk.to(b_w.dtype)))
    b_dw -= tl.dot(b_Twbk, b_dk.to(b_w.dtype))
    b_dT += tl.dot(b_dTwbk.to(b_wbk.dtype), tl.trans(b_wbk))
    b_dwbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(tl.trans(b_T), b_dTwbk.to(b_T.dtype)), 0).to(b_w.dtype)
    b_dw_beta += tl.dot(b_dwbk, b_k)

    b_dk += tl.dot(tl.trans(b_dwbk), b_w_beta)
    b_dk += tl.dot(tl.trans(b_dA_local), b_q)
    desc_dk_new = make_tensor_descriptor(dk_new + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    desc_dk_new.store([i_t * BT, 0], b_dk.to(dk_new.dtype.element_ty))

    # matrix inverse's gradient
    desc_T = make_tensor_descriptor(AT + (bos * H + i_h) * BT, [T, BT], [BT*H, 1], [BT, BT])
    b_Tt = tl.trans(desc_T.load([i_t * BT, 0]))
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0).to(b_w.dtype)
    b_dT = tl.dot(b_Tt, b_dT).to(b_w.dtype)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0).to(b_k.dtype)

    b_dw_beta += tl.dot(b_dT, b_w)
    b_dw += tl.dot(tl.trans(b_dT), b_w_beta)
    b_dw += b_dw_beta * b_beta[:, None]
    b_dbeta = tl.sum(b_dw_beta * b_w, axis=1)

    desc_dw = make_tensor_descriptor(dw + (bos * HQ + i_hq) * K, [T, K], [K*HQ, 1], [BT, BK])
    desc_dw.store([i_t * BT, 0], b_dw.to(dw.dtype.element_ty))
    tl.store(dbeta + (bos * HQ + i_hq) + (i_t * BT + tl.arange(0, BT)) * HQ, b_dbeta.to(dbeta.dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)


def intra_chunk_preprocess_bwd_fn(q, k, w, w2, beta,
                                  dq, dk, dA_local,
                                  dw1, dw2,
                                  A, L, D, do, scale, cu_seqlens=None,
                                  chunk_indices: torch.LongTensor | None = None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    # better precision because h would be of norm smaller than 1 anyways

    dbeta = torch.empty(B, T, HQ, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dw = torch.empty(B, T, HQ, K, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dk_new = torch.empty_like(dk, dtype=k.dtype if G == 1 else torch.float32)  # float32 reduction
    dq_new = torch.empty_like(dq, dtype=q.dtype)

    intra_chunk_preprocess_bwd_kernel[grid](
        q=q, k=k, w=w, w2=w2, beta=beta,
        AT=A,
        dA_local=dA_local, dq=dq, dq_new=dq_new, dk=dk, dk_new=dk_new, dw=dw, dbeta=dbeta, dw1=dw1, dw2=dw2, T=T,
        offsets=cu_seqlens, indices=indices,
        HQ=HQ, G=G, H=H,
        K=K, BT=BT, BK=triton.next_power_of_2(K),
        num_stages=3 if check_shared_mem('hopper') else 1,
    )
    return dq_new, dk_new, dbeta, dw
