# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import (
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    autotune_cache_kwargs,
    check_shared_mem,
)

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]

# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            desc_h0_1 = make_tensor_descriptor(h0, [V, K], [K, 1], [BV, 64])
        else:
            desc_h0_1 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
        b_h1 += desc_h0_1.load([0, i_v * BV]).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                desc_h0_2 = make_tensor_descriptor(h0, [V, K], [K, 1], [BV, 64])
            else:
                desc_h0_2 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h2 += desc_h0_2.load([64, i_v * BV]).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                desc_h0_3 = make_tensor_descriptor(h0, [V, K], [K, 1], [BV, 64])
            else:
                desc_h0_3 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h3 += desc_h0_3.load([128, i_v * BV]).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                desc_h0_4 = make_tensor_descriptor(h0, [V, K], [K, 1], [BV, 64])
            else:
                desc_h0_4 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h4 += desc_h0_4.load([192, i_v * BV]).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if STATE_V_FIRST:
            desc_h1 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [V, K], [K, 1], [BV, 64])
        else:
            desc_h1 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [K, V], [V, 1], [64, BV])
        desc_h1.store([0, i_v * BV], b_h1.to(desc_h1.dtype))
        if K > 64:
            if STATE_V_FIRST:
                desc_h2 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_h2 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [K, V], [V, 1], [64, BV])
            desc_h2.store([64, i_v * BV], b_h2.to(desc_h2.dtype))
        if K > 128:
            if STATE_V_FIRST:
                desc_h3 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_h3 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [K, V], [V, 1], [64, BV])
            desc_h3.store([128, i_v * BV], b_h3.to(desc_h3.dtype))
        if K > 192:
            if STATE_V_FIRST:
                desc_h4 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_h4 = make_tensor_descriptor(h + i_t_int64 * HV*K*V, [K, V], [V, 1], [64, BV])
            desc_h4.store([192, i_v * BV], b_h4.to(desc_h4.dtype))

        desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
        b_w = desc_w.load([i_t * BT, 0])
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 64])
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 128])
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 192])
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        desc_v = make_tensor_descriptor(v, [T, V], [HV*V, 1], [BT, BV])
        b_v = desc_v.load([i_t * BT, i_v * BV]) - b_v

        if SAVE_NEW_VALUE:
            desc_v = make_tensor_descriptor(v_new, [T, V], [HV*V, 1], [BT, BV])
            desc_v.store([i_t * BT, i_v * BV], b_v.to(desc_v.dtype))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            b_g = tl.load(g + bos * HV + i_h + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
        b_k = tl.trans(desc_k.load([i_t * BT, 0]))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 64]))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 128]))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 192]))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            desc_ht = make_tensor_descriptor(ht, [V, K], [K, 1], [BV, 64])
        else:
            desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
        desc_ht.store([0, i_v * BV], b_h1.to(desc_ht.dtype))
        if K > 64:
            if STATE_V_FIRST:
                desc_ht = make_tensor_descriptor(ht, [V, K], [K, 1], [BV, 64])
            else:
                desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([64, i_v * BV], b_h2.to(desc_ht.dtype))
        if K > 128:
            if STATE_V_FIRST:
                desc_ht = make_tensor_descriptor(ht, [V, K], [K, 1], [BV, 64])
            else:
                desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([128, i_v * BV], b_h3.to(desc_ht.dtype))
        if K > 192:
            if STATE_V_FIRST:
                desc_ht = make_tensor_descriptor(ht, [V, K], [K, 1], [BV, 64])
            else:
                desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([192, i_v * BV], b_h4.to(desc_ht.dtype))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BV', 'USE_G', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            desc_dht1 = make_tensor_descriptor(dht, [V, K], [K, 1], [BV, 64])
        else:
            desc_dht1 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
        b_dh1 += desc_dht1.load([0, i_v * BV])
        if K > 64:
            if STATE_V_FIRST:
                desc_dht2 = make_tensor_descriptor(dht, [V, K], [K, 1], [BV, 64])
            else:
                desc_dht2 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh2 += desc_dht2.load([64, i_v * BV])
        if K > 128:
            if STATE_V_FIRST:
                desc_dht3 = make_tensor_descriptor(dht, [V, K], [K, 1], [BV, 64])
            else:
                desc_dht3 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh3 += desc_dht3.load([128, i_v * BV])
        if K > 192:
            if STATE_V_FIRST:
                desc_dht4 = make_tensor_descriptor(dht, [V, K], [K, 1], [BV, 64])
            else:
                desc_dht4 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh4 += desc_dht4.load([192, i_v * BV])

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        if STATE_V_FIRST:
            desc_dh1 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [V, K], [K, 1], [BV, 64])
        else:
            desc_dh1 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [K, V], [V, 1], [64, BV])
        desc_dh1.store([0, i_v * BV], b_dh1.to(desc_dh1.dtype))
        if K > 64:
            if STATE_V_FIRST:
                desc_dh2 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh2 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [K, V], [V, 1], [64, BV])
            desc_dh2.store([64, i_v * BV], b_dh2.to(desc_dh2.dtype))
        if K > 128:
            if STATE_V_FIRST:
                desc_dh3 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh3 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [K, V], [V, 1], [64, BV])
            desc_dh3.store([128, i_v * BV], b_dh3.to(desc_dh3.dtype))
        if K > 192:
            if STATE_V_FIRST:
                desc_dh4 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh4 = make_tensor_descriptor(dh + i_t_int64*HV*K*V, [K, V], [V, 1], [64, BV])
            desc_dh4.store([192, i_v * BV], b_dh4.to(desc_dh4.dtype))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            b_g = tl.load(g + bos * HV + i_h + (i_t * BT + tl.arange(0, BT)) * HV, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)
        desc_dv = make_tensor_descriptor(dv, [T, V], [HV*V, 1], [BT, BV])
        desc_dv2 = make_tensor_descriptor(dv2, [T, V], [HV*V, 1], [BT, BV])
        desc_do = make_tensor_descriptor(do, [T, V], [HV*V, 1], [BT, BV])

        b_do = desc_do.load([i_t * BT, i_v * BV])

        # Update dv
        desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
        b_k = desc_k.load([i_t * BT, 0])
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * HV*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 64])
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * HV*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 128])
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * HV*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 192])
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * HV*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += desc_dv.load([i_t * BT, i_v * BV])

        desc_dv2.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))
        # Update dh
        desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
        desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BT, 64])
        b_w = tl.trans(desc_w.load([i_t * BT, 0]))
        b_q = tl.trans(desc_q.load([i_t * BT, 0]))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if STATE_V_FIRST:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 64]))
            b_w = tl.trans(desc_w.load([i_t * BT, 64]))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if STATE_V_FIRST:
                b_dh2 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 128]))
            b_w = tl.trans(desc_w.load([i_t * BT, 128]))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if STATE_V_FIRST:
                b_dh3 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [HV*K, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 192]))
            b_w = tl.trans(desc_w.load([i_t * BT, 192]))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if STATE_V_FIRST:
                b_dh4 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            desc_dh0 = make_tensor_descriptor(dh0, [V, K], [K, 1], [BV, 64])
        else:
            desc_dh0 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
        desc_dh0.store([0, i_v * BV], b_dh1.to(desc_dh0.dtype))
        if K > 64:
            if STATE_V_FIRST:
                desc_dh1 = make_tensor_descriptor(dh0, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh1 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh1.store([64, i_v * BV], b_dh2.to(desc_dh1.dtype))
        if K > 128:
            if STATE_V_FIRST:
                desc_dh2 = make_tensor_descriptor(dh0, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh2 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh2.store([128, i_v * BV], b_dh3.to(desc_dh2.dtype))
        if K > 192:
            if STATE_V_FIRST:
                desc_dh3 = make_tensor_descriptor(dh0, [V, K], [K, 1], [BV, 64])
            else:
                desc_dh3 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh3.store([192, i_v * BV], b_dh4.to(desc_dh3.dtype))


@dispatch('common')
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


@dispatch('common')
def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return dh, dh0, dv2
