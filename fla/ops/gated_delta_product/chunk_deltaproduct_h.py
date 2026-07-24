# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,  # number of delta products
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv(T // num_householder, BT)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K*V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        desc_h0_1 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
        b_h1 += desc_h0_1.load([0, i_v * BV]).to(tl.float32)
        if K > 64:
            desc_h0_2 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h2 += desc_h0_2.load([64, i_v * BV]).to(tl.float32)
        if K > 128:
            desc_h0_3 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h3 += desc_h0_3.load([128, i_v * BV]).to(tl.float32)
        if K > 192:
            desc_h0_4 = make_tensor_descriptor(h0, [K, V], [V, 1], [64, BV])
            b_h4 += desc_h0_4.load([192, i_v * BV]).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if i_t % num_householder == 0:
            i_t_true = i_t // num_householder
            desc_h1 = make_tensor_descriptor(h + i_t_true * stride_h, [K, V], [V, 1], [64, BV])
            desc_h1.store([0, i_v * BV], b_h1.to(desc_h1.dtype))
            if K > 64:
                desc_h2 = make_tensor_descriptor(h + i_t_true * stride_h, [K, V], [V, 1], [64, BV])
                desc_h2.store([64, i_v * BV], b_h2.to(desc_h2.dtype))
            if K > 128:
                desc_h3 = make_tensor_descriptor(h + i_t_true * stride_h, [K, V], [V, 1], [64, BV])
                desc_h3.store([128, i_v * BV], b_h3.to(desc_h3.dtype))
            if K > 192:
                desc_h4 = make_tensor_descriptor(h + i_t_true * stride_h, [K, V], [V, 1], [64, BV])
                desc_h4.store([192, i_v * BV], b_h4.to(desc_h4.dtype))

        desc_v = make_tensor_descriptor(v, [T, V], [stride_v, 1], [BT, BV])
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
                                    (BT, BV), (1, 0)) if SAVE_NEW_VALUE else None
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
        b_w = desc_w.load([i_t * BT, 0])
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 64])
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 128])
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_w = desc_w.load([i_t * BT, 192])
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + desc_v.load([i_t * BT, i_v * BV])

        if SAVE_NEW_VALUE:
            desc_v_new = make_tensor_descriptor(v_new, [T, V], [stride_v, 1], [BT, BV])
            desc_v_new.store([i_t * BT, i_v * BV], b_v_new.to(desc_v_new.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_v_new = b_v_new * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
        b_k = tl.trans(desc_k.load([i_t * BT, 0]))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 64]))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 128]))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = tl.trans(desc_k.load([i_t * BT, 192]))
            b_h4 += tl.dot(b_k, b_v_new)
    # epilogue
    if STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
        desc_ht.store([0, i_v * BV], b_h1.to(desc_ht.dtype))
        if K > 64:
            desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([64, i_v * BV], b_h2.to(desc_ht.dtype))
        if K > 128:
            desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([128, i_v * BV], b_h3.to(desc_ht.dtype))
        if K > 192:
            desc_ht = make_tensor_descriptor(ht, [K, V], [V, 1], [64, BV])
            desc_ht.store([192, i_v * BV], b_h4.to(desc_ht.dtype))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
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
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    dh += (boh * H + i_h) * K*V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        desc_dht1 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
        b_dh1 += desc_dht1.load([0, i_v * BV])
        if K > 64:
            desc_dht2 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh2 += desc_dht2.load([64, i_v * BV])
        if K > 128:
            desc_dht3 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh3 += desc_dht3.load([128, i_v * BV])
        if K > 192:
            desc_dht4 = make_tensor_descriptor(dht, [K, V], [V, 1], [64, BV])
            b_dh4 += desc_dht4.load([192, i_v * BV])

    for i_t in range(NT - 1, -1, -1):
        desc_dh1 = make_tensor_descriptor(dh + i_t*stride_h, [K, V], [V, 1], [64, BV])
        desc_dh1.store([0, i_v * BV], b_dh1.to(desc_dh1.dtype))
        if K > 64:
            desc_dh2 = make_tensor_descriptor(dh + i_t*stride_h, [K, V], [V, 1], [64, BV])
            desc_dh2.store([64, i_v * BV], b_dh2.to(desc_dh2.dtype))
        if K > 128:
            desc_dh3 = make_tensor_descriptor(dh + i_t*stride_h, [K, V], [V, 1], [64, BV])
            desc_dh3.store([128, i_v * BV], b_dh3.to(desc_dh3.dtype))
        if K > 192:
            desc_dh4 = make_tensor_descriptor(dh + i_t*stride_h, [K, V], [V, 1], [64, BV])
            desc_dh4.store([192, i_v * BV], b_dh4.to(desc_dh4.dtype))

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp2(bg_last)
            b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_g_exp = exp2(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None

        desc_dv = make_tensor_descriptor(dv, [T, V], [stride_v, 1], [BT, BV])
        desc_wo = make_tensor_descriptor(do, [T, V], [stride_v, 1], [BT, BV])
        desc_dv2 = make_tensor_descriptor(dv2, [T, V], [stride_v, 1], [BT, BV])

        b_wo = desc_wo.load([i_t * BT, i_v * BV])
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        # Update dv
        desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
        b_k = desc_k.load([i_t * BT, 0])
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 64])
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 128])
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            desc_k = make_tensor_descriptor(k, [T, K], [stride_k, 1], [BT, 64])
            b_k = desc_k.load([i_t * BT, 192])
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += desc_dv.load([i_t * BT, i_v * BV])

        desc_dv2.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))
        # Update dh
        desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
        desc_q = make_tensor_descriptor(q, [T, K], [stride_k, 1], [BT, 64])
        b_w = tl.trans(desc_w.load([i_t * BT, 0]))
        b_q = tl.trans(desc_q.load([i_t * BT, 0]))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh1 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            desc_q = make_tensor_descriptor(q, [T, K], [stride_k, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 64]))
            b_w = tl.trans(desc_w.load([i_t * BT, 64]))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            desc_q = make_tensor_descriptor(q, [T, K], [stride_k, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 128]))
            b_w = tl.trans(desc_w.load([i_t * BT, 128]))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            desc_q = make_tensor_descriptor(q, [T, K], [stride_k, 1], [BT, 64])
            desc_w = make_tensor_descriptor(w, [T, K], [stride_k, 1], [BT, 64])
            b_q = tl.trans(desc_q.load([i_t * BT, 192]))
            b_w = tl.trans(desc_w.load([i_t * BT, 192]))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        desc_dh0 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
        desc_dh0.store([0, i_v * BV], b_dh1.to(desc_dh0.dtype))
        if K > 64:
            desc_dh1 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh1.store([64, i_v * BV], b_dh2.to(desc_dh1.dtype))
        if K > 128:
            desc_dh2 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh2.store([128, i_v * BV], b_dh3.to(desc_dh2.dtype))
        if K > 192:
            desc_dh3 = make_tensor_descriptor(dh0, [K, V], [V, 1], [64, BV])
            desc_dh3.store([192, i_v * BV], b_dh4.to(desc_dh3.dtype))


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    num_householder: int = 1,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    assert T % num_householder == 0, "T must be divisible by num_householder"
    T_true = T // num_householder
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T_true, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - \
            1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens // num_householder, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_householder=num_householder,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state


def chunk_gated_delta_product_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]

    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
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
        K=K,
        V=V,
        BT=BT,
    )
    return dh, dh0, dv2
