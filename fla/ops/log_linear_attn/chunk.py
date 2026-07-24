# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import reduce

from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.op import exp
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

BLOCK_K = 64


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    },
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BLOCK_K}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    llut,
    o,
    h0,
    ht,
    offsets,
    new_offsets,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    MIN_LEVEL: tl.constexpr,
    MAX_LEVEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    desc_llut = make_tensor_descriptor(llut, [BT, BT], [BT, 1], [BT, BT])
    b_llut = desc_llut.load([0, 0])
    # parallel over sequences and heads
    i_k = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_i = tl.arange(0, BT)

    # For hierarchical masking
    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1
    i_idx = o_i[:, None]  # BT x 1
    j_idx = o_i[None, :]  # 1 x BT

    # This is not great.
    # See issue: https://github.com/triton-lang/triton/discussions/1313
    KV_0_CREATED = MIN_LEVEL <= 1 and MAX_LEVEL >= 0
    KV_1_CREATED = MIN_LEVEL <= 2 and MAX_LEVEL >= 0
    KV_2_CREATED = MIN_LEVEL <= 3 and MAX_LEVEL >= 1
    KV_3_CREATED = MIN_LEVEL <= 4 and MAX_LEVEL >= 2
    KV_4_CREATED = MIN_LEVEL <= 5 and MAX_LEVEL >= 3
    KV_5_CREATED = MIN_LEVEL <= 6 and MAX_LEVEL >= 4
    KV_6_CREATED = MIN_LEVEL <= 7 and MAX_LEVEL >= 5
    KV_7_CREATED = MIN_LEVEL <= 8 and MAX_LEVEL >= 6
    KV_8_CREATED = MIN_LEVEL <= 9 and MAX_LEVEL >= 7
    KV_9_CREATED = MIN_LEVEL <= 10 and MAX_LEVEL >= 8
    KV_10_CREATED = MIN_LEVEL <= 11 and MAX_LEVEL >= 9
    KV_11_CREATED = MIN_LEVEL <= 12 and MAX_LEVEL >= 10

    kv_0 = tl.zeros([BK, V], dtype=tl.float32)
    kv_1 = tl.zeros([BK, V], dtype=tl.float32)
    kv_2 = tl.zeros([BK, V], dtype=tl.float32)
    kv_3 = tl.zeros([BK, V], dtype=tl.float32)
    kv_4 = tl.zeros([BK, V], dtype=tl.float32)
    kv_5 = tl.zeros([BK, V], dtype=tl.float32)
    kv_6 = tl.zeros([BK, V], dtype=tl.float32)
    kv_7 = tl.zeros([BK, V], dtype=tl.float32)
    kv_8 = tl.zeros([BK, V], dtype=tl.float32)
    kv_9 = tl.zeros([BK, V], dtype=tl.float32)
    kv_10 = tl.zeros([BK, V], dtype=tl.float32)
    kv_11 = tl.zeros([BK, V], dtype=tl.float32)

    offset = 0  # total number to cached tokens
    first_chunk_index = 0  # next chunk index to compute
    if USE_INITIAL_STATE:
        offset = tl.load(offsets + i_n)

        first_chunk_index = offset // BT

        if KV_0_CREATED and (first_chunk_index & 1 > 0):
            desc_kv_0 = make_tensor_descriptor(h0 + ((i_n * L_IN + 0) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_0 = desc_kv_0.load([i_k * BK, 0])
        if KV_1_CREATED and (first_chunk_index & 2 > 0):
            desc_kv_1 = make_tensor_descriptor(h0 + ((i_n * L_IN + 1) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_1 = desc_kv_1.load([i_k * BK, 0])
        if KV_2_CREATED and (first_chunk_index & 4 > 0):
            desc_kv_2 = make_tensor_descriptor(h0 + ((i_n * L_IN + 2) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_2 = desc_kv_2.load([i_k * BK, 0])
        if KV_3_CREATED and (first_chunk_index & 8 > 0):
            desc_kv_3 = make_tensor_descriptor(h0 + ((i_n * L_IN + 3) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_3 = desc_kv_3.load([i_k * BK, 0])
        if KV_4_CREATED and (first_chunk_index & 16 > 0):
            desc_kv_4 = make_tensor_descriptor(h0 + ((i_n * L_IN + 4) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_4 = desc_kv_4.load([i_k * BK, 0])
        if KV_5_CREATED and (first_chunk_index & 32 > 0):
            desc_kv_5 = make_tensor_descriptor(h0 + ((i_n * L_IN + 5) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_5 = desc_kv_5.load([i_k * BK, 0])
        if KV_6_CREATED and (first_chunk_index & 64 > 0):
            desc_kv_6 = make_tensor_descriptor(h0 + ((i_n * L_IN + 6) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_6 = desc_kv_6.load([i_k * BK, 0])
        if KV_7_CREATED and (first_chunk_index & 128 > 0):
            desc_kv_7 = make_tensor_descriptor(h0 + ((i_n * L_IN + 7) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_7 = desc_kv_7.load([i_k * BK, 0])
        if KV_8_CREATED and (first_chunk_index & 256 > 0):
            desc_kv_8 = make_tensor_descriptor(h0 + ((i_n * L_IN + 8) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_8 = desc_kv_8.load([i_k * BK, 0])
        if KV_9_CREATED and (first_chunk_index & 512 > 0):
            desc_kv_9 = make_tensor_descriptor(h0 + ((i_n * L_IN + 9) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_9 = desc_kv_9.load([i_k * BK, 0])
        if KV_10_CREATED and (first_chunk_index & 1024 > 0):
            desc_kv_10 = make_tensor_descriptor(h0 + ((i_n * L_IN + 10) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_10 = desc_kv_10.load([i_k * BK, 0])
        if KV_11_CREATED and (first_chunk_index & 2048 > 0):
            desc_kv_11 = make_tensor_descriptor(h0 + ((i_n * L_IN + 11) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            kv_11 = desc_kv_11.load([i_k * BK, 0])

    NT = tl.cdiv(T, BT)
    output_offset = -1 * (offset % BT)
    for i_t in range(NT):
        b_h_ptrs = level_scales + ((bos + i_t * BT + i_idx) * H + i_h) * L + b_llut
        b_h = tl.load(b_h_ptrs, mask=i_idx >= j_idx)

        desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, BK])
        desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
        desc_o = make_tensor_descriptor(o + ((bos * H + i_h) * (K // BK) + i_k) * V, [T, V], [H * (K // BK) * V, 1], [BT, V])

        b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
        b_q = desc_q.load([i_t * BT, i_k * BK])
        b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))

        m_t = i_t * BT + o_i < T
        b_s = tl.where((i_idx >= j_idx) & m_t[:, None] & m_t[None, :], tl.exp(b_g[:, None] - b_g[None, :]), 0)
        b_s = (tl.dot(b_q, b_k) * b_s).to(b_q.dtype) * b_h

        b_v = desc_v.load([i_t * BT, 0])
        b_o = tl.zeros((BT, V), dtype=tl.float32)
        if MIN_LEVEL == 0:
            b_o += tl.dot(b_s, b_v)

        chunk_index = (
            first_chunk_index + i_t
        )  # index of the chunk over the entire sequence, including the offset

        if MIN_LEVEL <= 0 and MAX_LEVEL >= 0:
            if chunk_index & 1:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels])
                b_o += tl.dot((b_l * b_q), kv_0.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 1 and MAX_LEVEL >= 1:
            if chunk_index & 2:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 1])
                b_o += tl.dot((b_l * b_q), kv_1.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 2 and MAX_LEVEL >= 2:
            if chunk_index & 4:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 2])
                b_o += tl.dot((b_l * b_q), kv_2.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 3 and MAX_LEVEL >= 3:
            if chunk_index & 8:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 3])
                b_o += tl.dot((b_l * b_q), kv_3.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 4 and MAX_LEVEL >= 4:
            if chunk_index & 16:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 4])
                b_o += tl.dot((b_l * b_q), kv_4.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 5 and MAX_LEVEL >= 5:
            if chunk_index & 32:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 5])
                b_o += tl.dot((b_l * b_q), kv_5.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 6 and MAX_LEVEL >= 6:
            if chunk_index & 64:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 6])
                b_o += tl.dot((b_l * b_q), kv_6.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 7 and MAX_LEVEL >= 7:
            if chunk_index & 128:  # 8192 - 16384
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 7])
                b_o += tl.dot((b_l * b_q), kv_7.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 8 and MAX_LEVEL >= 8:
            if chunk_index & 256:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 8])
                b_o += tl.dot((b_l * b_q), kv_8.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 9 and MAX_LEVEL >= 9:
            if chunk_index & 512:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 9])
                b_o += tl.dot((b_l * b_q), kv_9.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 10 and MAX_LEVEL >= 10:
            if chunk_index & 1024:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 10])
                b_o += tl.dot((b_l * b_q), kv_10.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 11 and MAX_LEVEL >= 11:
            if chunk_index & 2048:
                desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
                b_l = desc_l.load([i_t * BT, num_intra_levels + 11])
                b_o += tl.dot((b_l * b_q), kv_11.to(b_q.dtype)) * tl.exp(b_g)[:, None]

        desc_o.store([i_t * BT + output_offset, 0], b_o.to(desc_o.dtype))

        if i_t < NT - 1 or T % BT == 0:
            # Only apply the state update if the last chunk is a full chunk.
            # Otherwise, it needs to be included in the next kernel call.

            # update the recurrent states
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            if KV_0_CREATED:
                kv_0 *= tl.exp(b_g_last)
            if KV_1_CREATED:
                kv_1 *= tl.exp(b_g_last)
            if KV_2_CREATED:
                kv_2 *= tl.exp(b_g_last)
            if KV_3_CREATED:
                kv_3 *= tl.exp(b_g_last)
            if KV_4_CREATED:
                kv_4 *= tl.exp(b_g_last)
            if KV_5_CREATED:
                kv_5 *= tl.exp(b_g_last)
            if KV_6_CREATED:
                kv_6 *= tl.exp(b_g_last)
            if KV_7_CREATED:
                kv_7 *= tl.exp(b_g_last)
            if KV_8_CREATED:
                kv_8 *= tl.exp(b_g_last)
            if KV_9_CREATED:
                kv_9 *= tl.exp(b_g_last)
            if KV_10_CREATED:
                kv_10 *= tl.exp(b_g_last)
            if KV_11_CREATED:
                kv_11 *= tl.exp(b_g_last)

            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)
            if MIN_LEVEL <= 1:
                kv_0 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 2:
                kv_1 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 3:
                kv_2 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 4:
                kv_3 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 5:
                kv_4 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 6:
                kv_5 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 7:
                kv_6 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 8:
                kv_7 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 9:
                kv_8 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 10:
                kv_9 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 11:
                kv_10 += tl.dot(b_k, b_v)

            check_value = (~chunk_index & (chunk_index + 1)) - 1

            if MIN_LEVEL <= 1 and MAX_LEVEL >= 0:
                if check_value & 1:
                    kv_1 += kv_0
                    kv_0 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 2 and MAX_LEVEL >= 1:
                if check_value & 2:
                    kv_2 += kv_1
                    kv_1 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 3 and MAX_LEVEL >= 2:
                if check_value & 4:
                    kv_3 += kv_2
                    kv_2 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 4 and MAX_LEVEL >= 3:
                if check_value & 8:
                    kv_4 += kv_3
                    kv_3 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 5 and MAX_LEVEL >= 4:
                if check_value & 16:
                    kv_5 += kv_4
                    kv_4 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 6 and MAX_LEVEL >= 5:
                if check_value & 32:
                    kv_6 += kv_5
                    kv_5 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 7 and MAX_LEVEL >= 6:
                if check_value & 64:
                    kv_7 += kv_6
                    kv_6 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 8 and MAX_LEVEL >= 7:
                if check_value & 128:
                    kv_8 += kv_7
                    kv_7 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 9 and MAX_LEVEL >= 8:
                if check_value & 256:
                    kv_9 += kv_8
                    kv_8 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 10 and MAX_LEVEL >= 9:
                if check_value & 512:
                    kv_10 += kv_9
                    kv_9 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 11 and MAX_LEVEL >= 10:
                if check_value & 1024:
                    kv_11 += kv_10
                    kv_10 = tl.zeros([BK, V], dtype=tl.float32)

    chunk_index = offset // BT + T // BT

    if STORE_FINAL_STATE:
        if (MIN_LEVEL <= 0 and MAX_LEVEL >= 0) and (chunk_index & 1 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 0) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_0)
        if (MIN_LEVEL <= 1 and MAX_LEVEL >= 1) and (chunk_index & 2 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 1) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_1)
        if (MIN_LEVEL <= 2 and MAX_LEVEL >= 2) and (chunk_index & 4 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 2) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_2)
        if (MIN_LEVEL <= 3 and MAX_LEVEL >= 3) and (chunk_index & 8 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 3) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_3)
        if (MIN_LEVEL <= 4 and MAX_LEVEL >= 4) and (chunk_index & 16 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 4) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_4)
        if (MIN_LEVEL <= 5 and MAX_LEVEL >= 5) and (chunk_index & 32 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 5) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_5)
        if (MIN_LEVEL <= 6 and MAX_LEVEL >= 6) and (chunk_index & 64 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 6) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_6)
        if (MIN_LEVEL <= 7 and MAX_LEVEL >= 7) and (chunk_index & 128 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 7) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_7)
        if (MIN_LEVEL <= 8 and MAX_LEVEL >= 8) and (chunk_index & 256 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 8) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_8)
        if (MIN_LEVEL <= 9 and MAX_LEVEL >= 9) and (chunk_index & 512 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 9) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_9)
        if (MIN_LEVEL <= 10 and MAX_LEVEL >= 10) and (chunk_index & 1024 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 10) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_10)
        if (MIN_LEVEL <= 11 and MAX_LEVEL >= 11) and (chunk_index & 2048 > 0):
            desc_kv = make_tensor_descriptor(ht + ((i_n * L_OUT + 11) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
            desc_kv.store([i_k * BK, 0], kv_11)

        tl.store(new_offsets + i_n, (offset // BT) * BT + T)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def copy_input_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    cu_seqlens,
    q_prev,
    k_prev,
    v_prev,
    g_prev,
    level_scales_prev,
    offsets,
    q_new,
    k_new,
    v_new,
    g_new,
    level_scales_new,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # parallel over sequences and heads
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    offset = tl.load(offsets + i_n)
    input_offset = -1 * (offset % BT)

    NT = tl.cdiv(T, BT)

    for i_t in range(NT):
        pass
        desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, K])
        desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
        desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
        desc_q_new = make_tensor_descriptor(q_new + bos * K, [T, K], [K, 1], [BT, K])
        desc_k_new = make_tensor_descriptor(k_new + bos * K, [T, K], [K, 1], [BT, K])
        desc_v_new = make_tensor_descriptor(v_new + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])

        b_g = tl.load(g + bos * H + i_h + (i_t * BT + input_offset + tl.arange(0, BT)) * H, mask=(i_t * BT + input_offset + tl.arange(0, BT)) < T, other=0)
        b_q = desc_q.load([i_t * BT + input_offset, 0])
        b_k = desc_k.load([i_t * BT + input_offset, 0])
        b_v = desc_v.load([i_t * BT + input_offset, 0])

        if i_t == 0:
            pass
            desc_q_prev = make_tensor_descriptor(q_prev + i_n * BT * K, [BT, K], [K, 1], [BT, K])
            desc_k_prev = make_tensor_descriptor(k_prev + i_n * BT * K, [BT, K], [K, 1], [BT, K])
            desc_v_prev = make_tensor_descriptor(v_prev + (i_n * BT * H + i_h) * V, [BT, V], [H * V, 1], [BT, V])

            b_g += tl.load(g_prev + i_n * BT * H + i_h + (0 + tl.arange(0, BT)) * H, mask=(0 + tl.arange(0, BT)) < BT, other=0)
            b_q += desc_q_prev.load([0, 0])
            b_k += desc_k_prev.load([0, 0])
            b_v += desc_v_prev.load([0, 0])

        tl.store(g_new + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_g, mask=(i_t * BT + tl.arange(0, BT)) < T)
        desc_q_new.store([i_t * BT, 0], b_q)
        desc_k_new.store([i_t * BT, 0], b_k)
        desc_v_new.store([i_t * BT, 0], b_v)

        for i in range(L):
            desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
            desc_l_new = make_tensor_descriptor(level_scales_new + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
            b_l = desc_l.load([i_t * BT + input_offset, i])
            if i_t == 0:
                desc_l_prev = make_tensor_descriptor(level_scales_prev + (i_n * BT * H + i_h) * L, [BT, L], [H * L, 1], [BT, 1])
                b_l += desc_l_prev.load([0, i])
            desc_l_new.store([i_t * BT, i], b_l)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    },
)
@triton.jit(do_not_specialize=["T"])
def copy_last_chunk_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    cu_seqlens,
    q_prev,
    k_prev,
    v_prev,
    g_prev,
    level_scales_prev,
    offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # parallel over sequences and heads
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    seq_offset = (T // BT) * BT

    desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, K])
    desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
    desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
    desc_q_prev = make_tensor_descriptor(q_prev + i_n * BT * K, [BT, K], [K, 1], [BT, K])
    desc_k_prev = make_tensor_descriptor(k_prev + i_n * BT * K, [BT, K], [K, 1], [BT, K])
    desc_v_prev = make_tensor_descriptor(v_prev + (i_n * BT * H + i_h) * V, [BT, V], [H * V, 1], [BT, V])

    tl.store(g_prev + i_n * BT * H + i_h + (0 + tl.arange(0, BT)) * H, tl.load(g + bos * H + i_h + (seq_offset + tl.arange(0, BT)) * H, mask=(seq_offset + tl.arange(0, BT)) < T, other=0), mask=(0 + tl.arange(0, BT)) < BT)
    desc_q_prev.store([0, 0], desc_q.load([seq_offset, 0]))
    desc_k_prev.store([0, 0], desc_k.load([seq_offset, 0]))
    desc_v_prev.store([0, 0], desc_v.load([seq_offset, 0]))

    for i in range(L):
        desc_l = make_tensor_descriptor(level_scales + (bos * H + i_h) * L, [T, L], [H * L, 1], [BT, 1])
        desc_l_prev = make_tensor_descriptor(level_scales_prev + (i_n * BT * H + i_h) * L, [BT, L], [H * L, 1], [BT, 1])
        desc_l_prev.store([0, i], desc_l.load([seq_offset, i]))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dh", "dg_last"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dhg(
    do,
    q,
    g,
    l,
    h_l,
    dh,
    dg_last,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # parallel over batches and heads
    i_k = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    b_dh = tl.zeros([BK, V], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        desc_dh = make_tensor_descriptor(dh + ((i_n * NT + i_t) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])
        b_dh_old = desc_dh.load([i_k * BK, 0])

        if (i_t & (1 << ell)) == 0:  # store the chunk
            desc_dh.store([i_k * BK, 0], b_dh.to(desc_dh.dtype) + b_dh_old)
            # if you are about the transition to compute, reset to zeros
            if i_t > 0 and ((i_t - 1) & (1 << ell)) > 0:
                b_dh = tl.zeros([BK, V], dtype=tl.float32)
        if i_t & (1 << ell):
            desc_h = make_tensor_descriptor(h_l + ((i_n * NT + i_t) * H + i_h) * K * V, [K, V], [V, 1], [BK, V])

            b_h = desc_h.load([i_k * BK, 0])
            p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
            tl.atomic_add(p_dg_last, tl.sum(b_h * (b_dh + b_dh_old)))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.exp(tl.load(g + bos * H + last_idx * H + i_h))
        b_dh *= b_g_last
        if i_t & (1 << ell):  # compute this chunk
            desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, BK])
            desc_do = make_tensor_descriptor(do + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
            b_l = tl.load(l + (bos * H + i_h) * L + num_intra_levels + ell + (i_t * BT + tl.arange(0, BT)) * H * L, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_q = tl.trans(desc_q.load([i_t * BT, i_k * BK]))
            b_q = (b_q * (tl.exp(b_g) * b_l)[None, :]).to(b_q.dtype)
            b_do = desc_do.load([i_t * BT, 0])

            b_s = tl.dot(b_q, b_do).to(b_q.dtype)
            b_dh += b_s


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dq", "dg"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_hdqgl(
    do,
    q,
    k,
    v,
    g,
    l,
    h_l,
    dq,
    dg,
    dl,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # parallel over batches and heads
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    b_h = tl.zeros([V, K], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT)):
        pass
        if i_t & (1 << ell):  # compute and store derivatives
            desc_do = make_tensor_descriptor(do + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
            desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, K])
            desc_dq = make_tensor_descriptor(dq + (bos * H + i_h) * K, [T, K], [H * K, 1], [BT, K])
            desc_h = make_tensor_descriptor(h_l + ((i_n * NT + i_t) * H + i_h) * K * V, [K, V], [V, 1], [K, V])

            b_do = desc_do.load([i_t * BT, 0])
            b_q = desc_q.load([i_t * BT, 0])
            b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_l = tl.load(l + (bos * H + i_h) * L + num_intra_levels + ell + (i_t * BT + tl.arange(0, BT)) * H * L, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

            b_dlq = tl.exp(b_g)[:, None] * tl.dot(b_do, b_h.to(b_do.dtype))

            b_dl = tl.sum(b_dlq * b_q, axis=1)
            b_dg = b_l * b_dl

            desc_h.store([0, 0], tl.trans(b_h))
            b_dq_old = desc_dq.load([i_t * BT, 0])
            desc_dq.store([i_t * BT, 0], (b_l[:, None] * b_dlq).to(desc_dq.dtype) + b_dq_old)
            tl.store(dl + (bos * H + i_h) * L + num_intra_levels + ell + (i_t * BT + tl.arange(0, BT)) * H * L, b_dl.to((dl + (bos * H + i_h) * L + num_intra_levels + ell).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)
            b_dg_old = tl.load(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            tl.store(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_dg.to((dg + bos * H + i_h).dtype.element_ty) + b_dg_old, mask=(i_t * BT + tl.arange(0, BT)) < T)
            if ((i_t + 1) & (1 << ell)) == 0:
                b_h = tl.zeros([V, K], dtype=tl.float32)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        b_h *= tl.exp(b_g_last)
        if (i_t & (1 << ell)) == 0:  # update the state
            desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
            desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
            b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
            b_k = desc_k.load([i_t * BT, 0])
            b_v = tl.trans(desc_v.load([i_t * BT, 0]))
            b_k = (b_k * tl.exp(b_g_last - b_g)[:, None]).to(b_k.dtype)
            b_h += tl.dot(b_v, b_k)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dk", "dg", "dg_last"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dkg(
    dh,
    k,
    v,
    g,
    dg_last,
    dk,
    dg,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T

    desc_dh = make_tensor_descriptor(dh + ((i_n * NT + i_t) * H + i_h) * K * V, [K, V], [V, 1], [K, V])
    desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
    desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
    desc_dk = make_tensor_descriptor(dk + (bos * H + i_h) * K, [T, K], [H * K, 1], [BT, K])

    b_dh = tl.trans(desc_dh.load([0, 0]))
    b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_v = desc_v.load([i_t * BT, 0])
    b_k = desc_k.load([i_t * BT, 0])
    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
    b_dg_last = tl.load(p_dg_last)

    b_dg_last *= tl.exp(b_g_last)
    b_dk = tl.where(m_t, exp(b_g_last - b_g), 0)[:, None] * tl.dot(b_v, b_dh).to(b_v.dtype)
    b_dg = tl.load(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_dg -= tl.sum(b_k * b_dk, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)

    b_dg = tl.where(o_i < BT - 1, b_dg, b_dg + b_dg_last)

    tl.store(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_dg, mask=(i_t * BT + tl.arange(0, BT)) < T)
    desc_dk.store([i_t * BT, 0], b_dk.to(desc_dk.dtype))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dv"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dv(
    dh,
    k,
    g,
    dv,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    desc_dh = make_tensor_descriptor(dh + ((i_n * NT + i_t) * H + i_h) * K * V, [K, V], [V, 1], [K, V])
    desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
    desc_dv = make_tensor_descriptor(dv + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])

    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)

    b_dh = desc_dh.load([0, 0])
    b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_k = desc_k.load([i_t * BT, 0])
    b_dv = tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None] * tl.dot(b_k, b_dh).to(b_k.dtype)
    desc_dv.store([i_t * BT, 0], b_dv.to(desc_dv.dtype))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dl", "dq", "dk", "dv", "dg"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_diag(
    do,
    q,
    k,
    v,
    g,
    l,
    llut,
    mask,
    dq,
    dk,
    dv,
    dg,
    dl,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    desc_llut = make_tensor_descriptor(llut, [BT, BT], [BT, 1], [BT, BT])
    b_llut = desc_llut.load([0, 0])
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_i = tl.arange(0, BT)
    i_idx = o_i[:, None]  # BT x 1
    j_idx = o_i[None, :]  # 1 x BT

    b_h_ptrs = l + ((bos + i_t * BT + i_idx) * H + i_h) * L + b_llut
    b_h = tl.load(b_h_ptrs, mask=i_idx >= j_idx)

    desc_q = make_tensor_descriptor(q + bos * K, [T, K], [K, 1], [BT, K])
    desc_k = make_tensor_descriptor(k + bos * K, [T, K], [K, 1], [BT, K])
    desc_v = make_tensor_descriptor(v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
    desc_do = make_tensor_descriptor(do + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])
    desc_dq = make_tensor_descriptor(dq + (bos * H + i_h) * K, [T, K], [H * K, 1], [BT, K])
    desc_dk = make_tensor_descriptor(dk + (bos * H + i_h) * K, [T, K], [H * K, 1], [BT, K])
    desc_dv = make_tensor_descriptor(dv + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, V])

    b_g = tl.load(g + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)
    b_q = tl.trans(desc_q.load([i_t * BT, 0]))
    b_k = desc_k.load([i_t * BT, 0])
    b_v = tl.trans(desc_v.load([i_t * BT, 0]))
    b_do = desc_do.load([i_t * BT, 0])
    b_dq = desc_dq.load([i_t * BT, 0])
    b_dk = desc_dk.load([i_t * BT, 0])
    b_dv = desc_dv.load([i_t * BT, 0])
    b_dg = tl.load(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, mask=(i_t * BT + tl.arange(0, BT)) < T, other=0)

    b_s = (tl.dot(b_k, b_q)).to(b_q.dtype)
    # Apply causal and padding masks
    m_t = i_t * BT + o_i < T
    b_a = tl.where((i_idx >= j_idx) & m_t[:, None] & m_t[None, :], tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_dv += tl.dot((b_s * tl.trans(b_a * b_h)).to(b_do.dtype), b_do)
    b_ds = tl.dot(b_do, b_v) * b_a
    b_dl = b_ds * tl.trans(b_s)
    b_dg += tl.sum(b_dl * b_h, axis=1)
    b_dg -= tl.sum(b_dl * b_h, axis=0)
    b_ds = (b_ds * b_h).to(b_k.dtype)
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.trans(tl.dot(b_q, b_ds))

    desc_dv.store([i_t * BT, 0], b_dv.to(desc_dv.dtype))
    desc_dq.store([i_t * BT, 0], b_dq.to(desc_dq.dtype))
    desc_dk.store([i_t * BT, 0], b_dk.to(desc_dk.dtype))
    tl.store(dg + bos * H + i_h + (i_t * BT + tl.arange(0, BT)) * H, b_dg.to((dg + bos * H + i_h).dtype.element_ty), mask=(i_t * BT + tl.arange(0, BT)) < T)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i in range(num_intra_levels):
        desc_mask = make_tensor_descriptor(mask + i * (BT * BT), [BT, BT], [BT, 1], [BT, BT])
        b_mask = desc_mask.load([0, 0])
        dl_i = tl.sum(tl.where(b_mask == 1, b_dl, 0), axis=1)
        tl.store(dl + (bos * H + i_h) * L + i + (i_t * BT + tl.arange(0, BT)) * H * L, dl_i, mask=(i_t * BT + tl.arange(0, BT)) < T)


def construct_binary_level_mask(level, T):
    if level == 0:
        return torch.diag(torch.ones(T, dtype=torch.bool))

    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T))

    mask = torch.where(
        torch.logical_and(
            torch.logical_and(
                indices[:, 0] % (1 << level) >= (1 << (level - 1)),
                indices[:, 1] + (1 << (level - 1))
                >= indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
            ),
            indices[:, 1] < indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
        ).view(T, T),
        1,
        0,
    )

    return mask


def level_lut(BT, device):
    lut = torch.zeros((BT, BT), dtype=torch.int32, device=device)
    for level in range(1, ceil_log(BT, 2) + 1):
        mask = construct_binary_level_mask(level, BT).to(device)
        lut = torch.where(mask.to(torch.bool), level, lut)
    return lut


def masks(BT, device):
    masks = []
    for level in range(0, ceil_log(BT, 2) + 1):
        mask = construct_binary_level_mask(level, BT).to(device).to(torch.int32)
        masks.append(mask)
    return torch.stack(masks)


def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


@dataclass
class LogLinearAttentionState:
    ht: torch.Tensor
    offsets: torch.Tensor
    q_prev: torch.Tensor
    k_prev: torch.Tensor
    v_prev: torch.Tensor
    g_prev: torch.Tensor
    level_scales_prev: torch.Tensor


class ChunkLogLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        level_scales,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = level_scales.shape

        if G != 1:
            raise ValueError("Group dimension must be 1.")

        if not math.log2(V).is_integer():
            raise ValueError(
                "Head dimension must be a power of two. Please pad the head dimension to the next power of two.",
            )

        if K % BLOCK_K != 0:
            raise ValueError(f"State dimension must be divisible by {BLOCK_K}.")

        if triton.__version__ > "3.2.0":
            warnings.warn("Triton>3.2.0 detected, which is known to have worse performance. "
                          "For optimal performance, it is recommended to install Triton==3.2.0 (if possible).")

        BT = 64  # chunk size

        h0 = initial_state.ht if initial_state is not None else None
        offsets = initial_state.offsets if initial_state is not None else None

        if cu_seqlens is None:
            NT = ceil_div(T + (torch.max(offsets) if offsets is not None else 0), BT)
            MAX_LEVEL = ceil_log(NT, 2) - 1
        else:
            NT = max(
                [
                    ceil_div(
                        cu_seqlens[i + 1]
                        - cu_seqlens[i]
                        + (offsets[i] if offsets is not None else 0),
                        BT,
                    )
                    for i in range(len(cu_seqlens) - 1)
                ],
            )
            MAX_LEVEL = ceil_log(NT, 2) - 1
            B = len(cu_seqlens) - 1

        if MAX_LEVEL > 10:
            raise ValueError("Sequence length must be less than 2**17")

        S0 = B if cu_seqlens is None else 1
        o = torch.zeros(
            (S0, T, H, (K // BLOCK_K), V),
            dtype=v.dtype,
            device=v.device,
        )

        if initial_state is not None:
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens + F.pad(torch.cumsum(offsets % BT), (1, 0))
            else:
                assert (offsets == offsets[0]).all()
                T += offsets[0].item() % BT
            S1 = cu_seqlens[-1] if cu_seqlens is not None else T
            q_new = torch.zeros((S0, S1, G, K), dtype=q.dtype, device=q.device)
            k_new = torch.zeros((S0, S1, G, K), dtype=k.dtype, device=k.device)
            v_new = torch.zeros((S0, S1, H, V), dtype=v.dtype, device=v.device)
            g_new = torch.zeros((S0, S1, H), dtype=g.dtype, device=g.device)
            level_scales_new = torch.zeros((S0, S1, H, L), dtype=level_scales.dtype, device=level_scales.device)

            copy_input_kernel[(B * H,)](
                q=q,
                k=k,
                v=v,
                g=g,
                level_scales=level_scales,
                cu_seqlens=cu_seqlens,
                q_prev=initial_state.q_prev,
                k_prev=initial_state.k_prev,
                v_prev=initial_state.v_prev,
                g_prev=initial_state.g_prev,
                level_scales_prev=initial_state.level_scales_prev,
                q_new=q_new,
                k_new=k_new,
                v_new=v_new,
                g_new=g_new,
                level_scales_new=level_scales_new,
                offsets=offsets,
                T=T,
                H=H,
                K=K,
                V=V,
                L=L,
                BT=BT,
            )
            q = q_new
            k = k_new
            v = v_new
            g = g_new
            level_scales = level_scales_new

        # Store one extra level (MAX_LEVEL + 2) in case the length is multiple of 2
        ht = (
            torch.zeros((B, MAX_LEVEL + 2, H, K, V), dtype=torch.float, device=v.device)
            if output_final_state
            else None
        )

        new_offsets = torch.zeros((B,), dtype=torch.int32, device=v.device)
        g = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)

        def grid(meta):
            return (triton.cdiv(K, meta["BK"]), B * H)

        l_in = h0.shape[1] if initial_state is not None else None
        l_out = ht.shape[1] if output_final_state else None

        ctx.llut = level_lut(BT, v.device)

        chunkwise_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            g=g,
            level_scales=level_scales,
            llut=ctx.llut,
            o=o,
            h0=h0,
            ht=ht,
            offsets=offsets,
            new_offsets=new_offsets,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
            L_IN=l_in,
            L_OUT=l_out,
            MIN_LEVEL=0,
            MAX_LEVEL=MAX_LEVEL,
        )

        ctx.save_for_backward(q, k, v, g, level_scales, initial_state, cu_seqlens)
        ctx.chunk_size = BT

        if output_final_state:
            q_prev = torch.zeros((B, BT, G, K), dtype=q.dtype, device=q.device)
            k_prev = torch.zeros((B, BT, G, K), dtype=k.dtype, device=k.device)
            v_prev = torch.zeros((B, BT, H, V), dtype=v.dtype, device=v.device)
            g_prev = torch.zeros((B, BT, H), dtype=g.dtype, device=g.device)
            level_scales_prev = torch.zeros((B, BT, H, L), dtype=level_scales.dtype, device=level_scales.device)

            copy_last_chunk_kernel[(B * H,)](
                q=q,
                k=k,
                v=v,
                g=g,
                level_scales=level_scales,
                cu_seqlens=cu_seqlens,
                q_prev=q_prev,
                k_prev=k_prev,
                v_prev=v_prev,
                g_prev=g_prev,
                level_scales_prev=level_scales_prev,
                offsets=new_offsets,
                T=T,
                H=H,
                K=K,
                V=V,
                L=L,
                BT=BT,
            )

            final_state = LogLinearAttentionState(
                ht=ht,
                offsets=new_offsets,
                q_prev=q_prev,
                k_prev=k_prev,
                v_prev=v_prev,
                g_prev=g_prev,
                level_scales_prev=level_scales_prev,
            )
            return o.sum(dim=-2), final_state

        return o.sum(dim=-2), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        if triton.__version__ < "3.1.0":
            raise ValueError("Triton>=3.1.0 is required")

        q, k, v, g, level_scales, initial_state, cu_seqlens = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        llut = ctx.llut
        mask = masks(chunk_size, v.device)

        if initial_state is not None:
            raise NotImplementedError(
                "Backward pass is not implemented for log-linear attention with a prefilled kernel.",
            )

        B, T, G, K = k.shape
        assert G == 1, "Multi-head attention is not supported"
        _, _, H, V = v.shape
        _, _, _, L = level_scales.shape
        BT = chunk_size
        if cu_seqlens is not None:
            NT = max(
                [
                    ceil_div(cu_seqlens[i + 1] - cu_seqlens[i], BT)
                    for i in range(len(cu_seqlens) - 1)
                ],
            )
        else:
            NT = ceil_div(T, BT)

        if cu_seqlens is not None:
            B = len(cu_seqlens) - 1

        dh = torch.zeros((B, NT, H, K, V), dtype=v.dtype, device=v.device)
        dq = torch.zeros((B if cu_seqlens is None else 1, T, H, K), dtype=v.dtype, device=v.device)
        dk = torch.zeros((B if cu_seqlens is None else 1, T, H, K), dtype=v.dtype, device=v.device)
        dv = torch.zeros_like(v)
        dg = torch.zeros(g.shape, dtype=torch.float, device=v.device)
        dl = torch.zeros(level_scales.shape, dtype=torch.float, device=v.device)
        h_l = torch.zeros((B, NT, H, K, V), dtype=torch.float, device=v.device)
        dg_last = torch.zeros((B, NT, H), dtype=torch.float, device=v.device)
        do = do.to(v.dtype)

        grid = (B * H,)

        def grid_f(meta):
            return (triton.cdiv(K, meta["BK"]), B * H)

        grid_t = (NT, B * H)

        num_inter_chunk_levels = ceil_log(NT, 2)
        for ell in range(num_inter_chunk_levels - 1, -1, -1):
            chunkwise_bwd_kernel_hdqgl[grid](
                do=do,
                q=q,
                k=k,
                v=v,
                g=g,
                l=level_scales,
                h_l=h_l,
                dq=dq,
                dg=dg,
                dl=dl,
                cu_seqlens=cu_seqlens,
                ell=ell,
                T=T,
                H=H,
                K=K,
                V=V,
                L=L,
                BT=BT,
                NT=NT,
            )
            chunkwise_bwd_kernel_dhg[grid_f](
                do=do,
                q=q,
                g=g,
                l=level_scales,
                h_l=h_l,
                dh=dh,
                dg_last=dg_last,
                cu_seqlens=cu_seqlens,
                ell=ell,
                T=T,
                H=H,
                K=K,
                V=V,
                L=L,
                BT=BT,
                NT=NT,
            )

        chunkwise_bwd_kernel_dkg[grid_t](
            dh=dh,
            k=k,
            v=v,
            g=g,
            dg_last=dg_last,
            dk=dk,
            dg=dg,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
            NT=NT,
        )

        chunkwise_bwd_kernel_dv[grid_t](
            dh=dh,
            k=k,
            g=g,
            dv=dv,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
            NT=NT,
        )

        chunkwise_bwd_kernel_diag[grid_t](
            do=do,
            q=q,
            k=k,
            v=v,
            g=g,
            l=level_scales,
            llut=llut,
            mask=mask,
            dq=dq,
            dk=dk,
            dv=dv,
            dg=dg,
            dl=dl,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
        )

        dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens).to(g.dtype)

        dq = reduce(dq, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        dk = reduce(dk, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        return dq, dk, dv, dg, dl, None, None, None


@torch.compiler.disable
def chunk_log_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    level_scales: torch.Tensor,
    initial_state: LogLinearAttentionState | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
        level_scales (torch.Tensor):
            Scales for each level of shape `[B, T, H, L]`.
        initial_state (Optional[LogLinearAttentionState]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of type `LogLinearAttentionState` if `output_final_state=True` else `None`.

    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )

    o, final_state = ChunkLogLinearAttentionFunction.apply(
        q,
        k,
        v,
        g,
        level_scales,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
    return o, final_state
