# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.modules.layernorm import group_norm
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_HOPPER, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

NUM_WARPS = [1, 2] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_fwd_kernel(
    q,
    k,
    v,
    eta,
    w,
    b,
    o,
    scale,
    eps,
    h0,
    hb0,
    ht,
    hbt,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        desc_h0 = make_tensor_descriptor(h0 + i_nh * K * V, [K, V], [V, 1], [BK, BV])
        b_h = desc_h0.load([0, 0]).to(tl.float32)
    if USE_INITIAL_STATE_B:
        desc_hb0 = make_tensor_descriptor(hb0 + i_nh * V, [V], [1], [BV])
        b_hb = desc_hb0.load([0]).to(tl.float32)

    for i_t in range(NT):
        desc_q = make_tensor_descriptor(q+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_v = make_tensor_descriptor(v+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_o = make_tensor_descriptor(o+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t*BT, 0]))
        # [BT, BV]
        b_v = desc_v.load([i_t*BT, 0])

        # [BT, BV]
        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V

        # [BT, BK]
        b_q = desc_q.load([i_t*BT, 0])
        # [BT]
        b_e = tl.load(eta+(bos*H+i_h) + (i_t*BT + tl.arange(0, BT)) * H, mask=(i_t*BT + tl.arange(0, BT)) < T, other=0)
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BT]
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        b_A = tl.where(m_A, b_A, 0)
        b_Ae = tl.where(m_A, b_e[:, None], 0.0)

        b_o = - tl.dot(b_e[:, None] * b_A.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += b_hb[None, :] - tl.dot(b_Ae.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        b_e_last = tl.load(p_e_last)
        b_h = b_h - tl.dot(b_e_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_e_last * b_v2.to(b_k.dtype), axis=0)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)
        b_hb = tl.where((v_i < V), b_hb, 0.)
        desc_o.store([i_t*BT, 0], b_o.to(desc_o.dtype))

    if STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(ht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_hbt = make_tensor_descriptor(hbt + i_nh * V, [V], [1], [BV])
        desc_ht.store([0, 0], b_h.to(desc_ht.dtype))
        desc_hbt.store([0], b_hb.to(desc_hbt.dtype))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_h(
    k,
    v,
    v2,
    x,
    y,
    r,
    w,
    b,
    eta,
    h0,
    hb0,
    h,
    do,
    dq,
    scale,
    eps,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        desc_h0 = make_tensor_descriptor(h0 + i_nh * K * V, [K, V], [V, 1], [BK, BV])
        b_h = desc_h0.load([0, 0]).to(tl.float32)
    if USE_INITIAL_STATE_B:
        desc_hb0 = make_tensor_descriptor(hb0 + i_nh * V, [V], [1], [BV])
        b_hb = desc_hb0.load([0]).to(tl.float32)

    for i_t in range(NT):
        desc_h = make_tensor_descriptor(h+((boh+i_t)*H+i_h)*K*V, [K, V], [V, 1], [BK, BV])
        desc_k = make_tensor_descriptor(k+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_v = make_tensor_descriptor(v+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_v2 = make_tensor_descriptor(v2+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_x = make_tensor_descriptor(x+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_y = make_tensor_descriptor(y+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_r = make_tensor_descriptor(r+bos*H+i_h, [T, 1], [H, 1], [BT, 1])
        desc_dq = make_tensor_descriptor(dq+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_do = make_tensor_descriptor(do+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        desc_h.store([0, 0], b_h.to(desc_h.dtype))
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t*BT, 0]))
        # [BT, BV]
        b_v = desc_v.load([i_t*BT, 0])

        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V
        desc_x.store([i_t*BT, 0], b_kh_hat.to(desc_x.dtype))
        desc_y.store([i_t*BT, 0], b_v.to(desc_y.dtype))
        desc_r.store([i_t*BT, 0], rstd.to(desc_r.dtype))
        desc_v2.store([i_t*BT, 0], b_v2.to(desc_v2.dtype))

        b_e = tl.load(eta+(bos*H+i_h) + (i_t*BT + tl.arange(0, BT)) * H, mask=(i_t*BT + tl.arange(0, BT)) < T, other=0)
        b_do = desc_do.load([i_t*BT, 0])

        b_v2 = tl.where((v_i < V)[None, :], b_v2, 0.)
        b_ds = tl.dot(b_do, tl.trans(b_v2).to(b_do.dtype))
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq = tl.dot(b_do, tl.trans(b_h).to(b_do.dtype))
        b_dq -= tl.dot(b_ds, tl.trans(b_k)) * b_e[:, None]
        b_dq *= scale

        b_e_last = tl.load(p_e_last)
        b_h = b_h - tl.dot(b_e_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_e_last * b_v2.to(b_k.dtype), axis=0)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)
        b_hb = tl.where((v_i < V), b_hb, 0.)
        desc_dq.store([i_t*BT, 0], b_dq.to(desc_dq.dtype))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['dhb0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_STATE_GRADIENT_B': lambda args: args['dhbt'] is not None,
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
def fused_chunk_ttt_linear_bwd_kernel_dh(
    q,
    k,
    v,
    v2,
    x,
    y,
    r,
    w,
    b,
    eta,
    h,
    dht,
    dhbt,
    dh0,
    dhb0,
    do,
    dk,
    dv,
    de,
    dw,
    db,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        desc_dht = make_tensor_descriptor(dht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_dh += desc_dht.load([0, 0])
    if USE_FINAL_STATE_GRADIENT_B:
        desc_dhbt = make_tensor_descriptor(dhbt + i_nh * V, [V], [1], [BV])
        b_dhb += desc_dhbt.load([0])

    # [BV]
    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    m_A_t = o_i[:, None] <= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)
    b_dw = tl.zeros([BV], dtype=b_w.dtype)
    b_db = tl.zeros([BV], dtype=b_b.dtype)
    desc_dw = make_tensor_descriptor(dw + i_nh * V, [V], [1], [BV])
    desc_db = make_tensor_descriptor(db + i_nh * V, [V], [1], [BV])

    for i_t in range(NT - 1, -1, -1):
        desc_h = make_tensor_descriptor(h+((boh+i_t)*H+i_h)*K*V, [K, V], [V, 1], [BK, BV])
        desc_q = make_tensor_descriptor(q+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_v = make_tensor_descriptor(v+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_v2 = make_tensor_descriptor(v2+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_x = make_tensor_descriptor(x+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_y = make_tensor_descriptor(y+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_r = make_tensor_descriptor(r+bos*H+i_h, [T, 1], [H, 1], [BT, 1])
        desc_dv = make_tensor_descriptor(dv+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        desc_dk = make_tensor_descriptor(dk+(bos*H+i_h)*K, [T, K], [H*K, 1], [BT, BK])
        desc_do = make_tensor_descriptor(do+(bos*H+i_h)*V, [T, V], [H*V, 1], [BT, BV])
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_q = tl.trans(desc_q.load([i_t*BT, 0]))
        b_k = desc_k.load([i_t*BT, 0])
        b_e = tl.load(eta+(bos*H+i_h) + (i_t*BT + tl.arange(0, BT)) * H, mask=(i_t*BT + tl.arange(0, BT)) < T, other=0)
        b_do = desc_do.load([i_t*BT, 0])
        b_e_last = tl.load(p_e_last)
        b_A = tl.dot(b_k, b_q)
        b_A = - tl.where(m_A_t, b_A * scale * b_e[None, :], 0).to(do.dtype.element_ty)
        b_Ae = - tl.where(m_A_t, b_e[None, :], 0).to(do.dtype.element_ty)
        b_dv_new = tl.dot(b_A.to(b_do.dtype), b_do) + tl.dot(b_Ae.to(b_do.dtype), b_do)
        b_dv_new -= tl.dot(b_e_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_e_last * b_dhb.to(b_k.dtype)[None, :]

        b_v2 = desc_v2.load([i_t*BT, 0]).to(b_k.dtype)
        b_x = desc_x.load([i_t*BT, 0]).to(b_k.dtype)
        b_y = desc_y.load([i_t*BT, 0]).to(b_k.dtype)
        b_rstd = desc_r.load([i_t*BT, 0]).to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v2.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = desc_v.load([i_t*BT, 0])
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        b_h = tl.trans(desc_h.load([0, 0]))
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((v_i < V)[None, :] * (o_i < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)

        b_ds = tl.dot(b_do, tl.trans(b_v2))
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
        mask = (o_i == i_last)
        b_dk -= b_e_last * tl.dot(b_v2, tl.trans(b_dh).to(b_v2.dtype))
        b_dk -= tl.dot(tl.trans(b_ds), tl.trans(b_q) * b_e[:, None])
        b_de = mask * tl.sum(- b_dh * tl.trans(tl.dot(tl.trans(b_v2), b_k))).to(b_k.dtype)
        b_de -= mask * tl.sum(b_dhb * tl.sum(b_v2, axis=0)).to(b_k.dtype)
        b_de -= tl.sum(tl.dot(b_ds, b_k) * tl.trans(b_q).to(b_k.dtype), axis=1)
        b_de -= tl.sum(b_ds, axis=1)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((v_i < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((v_i < V), b_dhb, 0.)

        desc_dk.store([i_t*BT, 0], b_dk.to(desc_dk.dtype))
        desc_dv.store([i_t*BT, 0], b_dv.to(desc_dv.dtype))
        tl.store(de+(bos*H+i_h) + (i_t*BT + tl.arange(0, BT)) * H, b_de.to((de+(bos*H+i_h)).dtype.element_ty), mask=(i_t*BT + tl.arange(0, BT)) < T)
    desc_dw.store([0], b_dw.to(desc_dw.dtype))
    desc_db.store([0], b_db.to(desc_db.dtype))

    if USE_INITIAL_STATE:
        desc_dh0 = make_tensor_descriptor(dh0+i_nh*K*V, [K, V], [V, 1], [BK, BV])
        desc_dh0.store([0, 0], b_dh.to(desc_dh0.dtype))
    if USE_INITIAL_STATE_B:
        desc_dhb0 = make_tensor_descriptor(dhb0+i_nh*V, [V], [1], [BV])
        desc_dhb0.store([0], b_dhb.to(desc_dhb0.dtype))


def fused_chunk_ttt_linear_bwd_h(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    do: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N, NT = B, triton.cdiv(T, BT)
    BK, BV = max(triton.next_power_of_2(K), 16), max(triton.next_power_of_2(V), 16)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."

    h = k.new_empty(B, NT, H, K, V)
    r = v.new_empty(B, T, H, 1, dtype=torch.float32)
    v2 = torch.empty_like(v)
    x = torch.empty_like(v)
    y = torch.empty_like(v)
    dq = torch.empty_like(q)

    grid = (N * H,)
    fused_chunk_ttt_linear_bwd_kernel_h[grid](
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=r,
        w=w,
        b=b,
        eta=eta,
        h0=initial_state,
        hb0=initial_state_bias,
        h=h,
        do=do,
        dq=dq,
        scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, h, v2, x, y, r


def fused_chunk_ttt_linear_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    h: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dhbt: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N = B
    BK, BV = max(triton.next_power_of_2(K), 16), max(triton.next_power_of_2(V), 16)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."

    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    dhb0 = torch.empty_like(initial_state_bias, dtype=torch.float32) if initial_state_bias is not None else None
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    de = torch.empty_like(eta)
    dw = w.new_empty(B, H, V)
    db = b.new_empty(B, H, V)

    grid = (N * H,)
    fused_chunk_ttt_linear_bwd_kernel_dh[grid](
        q=q,
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=r,
        w=w,
        b=b,
        eta=eta,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dh0=dh0,
        dhb0=dhb0,
        do=do,
        dk=dk,
        dv=dv,
        de=de,
        dw=dw,
        db=db,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dw = dw.sum(dim=0)
    db = db.sum(dim=0)
    return dk, dv, de, dw, db, dh0, dhb0


def fused_chunk_ttt_linear_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    initial_state: torch.Tensor,
    initial_state_bias: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    BT: int = 16,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = max(triton.next_power_of_2(K), 16), max(triton.next_power_of_2(V), 16)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."
    o = torch.empty_like(v)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_state_bias = k.new_empty(N, H, 1, V, dtype=torch.float32) if output_final_state else None

    grid = (N * H,)
    fused_chunk_ttt_linear_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        eta=eta,
        w=w,
        b=b,
        o=o,
        scale=scale,
        eps=eps,
        h0=initial_state,
        hb0=initial_state_bias,
        ht=final_state,
        hbt=final_state_bias,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return o, final_state, final_state_bias


def fused_chunk_ttt_linear_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    do: torch.Tensor,
    dht: torch.Tensor,
    dhbt: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    dq, h, v2, x, y, rstd = fused_chunk_ttt_linear_bwd_h(
        q=q,
        k=k,
        v=v,
        w=w,
        b=b,
        eta=eta,
        scale=scale,
        eps=eps,
        do=do,
        BT=BT,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        cu_seqlens=cu_seqlens,
    )
    dk, dv, de, dw, db, dh0, dhb0 = fused_chunk_ttt_linear_bwd_dh(
        q=q,
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=rstd,
        w=w,
        b=b,
        eta=eta,
        scale=scale,
        h=h,
        do=do,
        dht=dht,
        dhbt=dhbt,
        BT=BT,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, de, dw, db, dh0, dhb0


class FusedChunkTTTLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, b, BT, eta, scale, eps, initial_state,
                initial_state_bias, output_final_state, cu_seqlens):
        o, final_state, final_state_bias = fused_chunk_ttt_linear_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            b=b,
            eta=eta,
            scale=scale,
            eps=eps,
            BT=BT,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, eta, w, b, initial_state, initial_state_bias)
        ctx.BT = BT
        ctx.scale = scale
        ctx.eps = eps
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state, final_state_bias

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dhbt):
        q, k, v, eta, w, b, initial_state, initial_state_bias = ctx.saved_tensors
        dq, dk, dv, de, dw, db, dh0, dhb0 = fused_chunk_ttt_linear_bwd(
            q=q,
            k=k,
            v=v,
            w=w,
            b=b,
            eta=eta,
            scale=ctx.scale,
            eps=ctx.eps,
            do=do,
            dht=dht,
            dhbt=dhbt,
            BT=ctx.BT,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), dw.to(w), db.to(b), None, de.to(eta), None, None, dh0, dhb0, None, None


def norm_residual(x, weight, bias, eps):
    # GroupNorm and Residual
    B, T, H, D = x.shape
    x += group_norm(
        x.reshape(B, T, -1).clone(),
        weight=weight.reshape(-1).clone(),
        bias=bias.reshape(-1).clone(),
        eps=eps,
        num_groups=H,
    ).reshape(x.shape)
    return x


def fused_chunk_ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: torch.Tensor | None = None,
    initial_state_bias: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        w (torch.Tensor):
            layer norm weight of shape `[H, V]`.
        b (torch.Tensor):
            layer norm bias of shape `[H, V]`.
        eta (torch.Tensor):
            Learning rate for hidden state, of shape `[B, T, H, 1]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        chunk_size (int):
            Chunk size. Default: `16`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]`. Default: `None`.
        initial_state_bias (Optional[torch.Tensor]):
            Initial state bias of shape `[N, H, 1, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
        final_state_bias (torch.Tensor):
            Final state bias of shape `[N, H, 1, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    o, final_state, final_state_bias = FusedChunkTTTLinearFunction.apply(
        q,
        k,
        v,
        w,
        b,
        chunk_size,
        eta,
        scale,
        eps,
        initial_state,
        initial_state_bias,
        output_final_state,
        cu_seqlens,
    )
    o = norm_residual(o, w, b, eps)
    return o, final_state, final_state_bias
