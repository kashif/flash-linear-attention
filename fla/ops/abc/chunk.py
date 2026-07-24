# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import softmax_bwd, softmax_fwd
from fla.ops.utils.logcumsumexp import logcumsumexp_fwd_kernel
from fla.ops.utils.op import exp
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import input_guard


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_h(
    k,
    v,
    z,
    h,
    h0,
    ht,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        desc_h = make_tensor_descriptor(h0 + i_bh * K * V, [K, V], [V, 1], [BK, BV])
        b_h += desc_h.load([i_k * BK, i_v * BV]).to(tl.float32)
    if NORMK:
        desc_z0 = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
    else:
        desc_z0 = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
    b_zp = desc_z0.load([i_v * BV]).to(tl.float32)
    for i_t in range(NT):
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_h = make_tensor_descriptor(h + i_bh * NT*K*V + i_t * K * V, [K, V], [V, 1], [BK, BV])

        desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
        # [BT, BV]
        b_v = desc_v.load([i_t * BT, i_v * BV])
        if NORMK:
            desc_zc = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
            # [BK,]
            b_zc = desc_zc.load([(i_t * BT + BT - 1) * K + i_k * BK])
            b_r, b_zp = exp(b_zp - b_zc), b_zc
            # [BK, BV]
            b_h = b_h * b_r[:, None]
            b_k = exp(b_k - b_zc[:, None]).to(b_k.dtype)
        else:
            desc_zc = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
            # [BV,]
            b_zc = desc_zc.load([(i_t * BT + BT - 1) * V + i_v * BV])
            b_r, b_zp = exp(b_zp - b_zc), b_zc
            # [BK, BV]
            b_h = b_h * b_r[None, :]
            b_v = exp(b_v - b_zc[None, :]).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        desc_h = make_tensor_descriptor(ht + i_bh * K * V, [K, V], [V, 1], [BK, BV])
        desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_intra_K(
    v,
    z,
    o,
    A,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BC, BV])
    desc_zn = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
    # [BV,]
    b_zn = desc_zn.load([(i_t * BT + i_i * BC) * V + i_v * BV])
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BC, BC])
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        # [BC, BV]
        b_v = desc_v.load([i_t * BT + i_j * BC, i_v * BV])
        # [BC, BC]
        b_A = desc_A.load([i_t * BT + i_i * BC, i_j * BC])
        b_o += tl.dot(b_A, exp(b_v - b_zn[None, :]).to(b_v.dtype), allow_tf32=False)
    b_z = desc_z.load([i_t * BT + i_i * BC, i_v * BV])
    b_o *= exp(b_zn[None, :] - b_z)

    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T * V], [1], [BV])
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = desc_v.load([(i_t * BT + i_i * BC + j) * V + i_v * BV]).to(tl.float32)
        # [BC, BV]
        # avoid 0 * inf = inf
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * exp(b_v[None, :] - b_z), 0)
    desc_o = make_tensor_descriptor(o + i_bh * T*V, [T, V], [V, 1], [BC, BV])
    desc_o.store([i_t * BT + i_i * BC, i_v * BV], b_o.to(desc_o.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_K(
    q,
    k,
    z,
    h,
    o,
    A,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_h = make_tensor_descriptor(h + i_bh * NT*K*V + i_t * K * V, [K, V], [V, 1], [BK, BV])

        # [BT, BK]
        b_q = desc_q.load([i_t * BT, i_k * BK])
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
        # [BK, BV]
        b_h = desc_h.load([i_k * BK, i_v * BV])
        # [BT, BV]
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k, allow_tf32=False)
    desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    desc_o = make_tensor_descriptor(o + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    # [BT, BV]
    b_z = desc_z.load([i_t * BT, i_v * BV])
    # [BT, BV]
    desc_zp = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
    b_zp = desc_zp.load([i_p * V + i_v * BV])
    b_o = b_o * exp(b_zp[None, :] - b_z)
    desc_o.store([i_t * BT, i_v * BV], b_o.to(desc_o.dtype))

    desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_intra_V(
    q,
    k,
    z,
    A,
    scale,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_A = make_tensor_descriptor(A + (i_k*n_bh+i_bh)*T*BT, [T, BT], [BT, 1], [BC, BC])
        desc_zn = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
        # [BK,]
        b_zn = desc_zn.load([(i_t * BT + i_i * BC) * K + i_k * BK])
        # [BC, BK]
        b_q = desc_q.load([i_t * BT + i_i * BC, i_k * BK])
        b_z = desc_z.load([i_t * BT + i_i * BC, i_k * BK])
        b_q = (b_q * exp(b_zn[None, :] - b_z) * scale).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.trans(desc_k.load([i_t * BT + i_j * BC, i_k * BK]))
        b_k = exp(b_k - b_zn[:, None]).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        desc_A.store([i_t * BT + i_i * BC, i_j * BC], b_A.to(A.dtype.element_ty))
    elif i_i == i_j:
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        # [BC, BK]
        b_q = desc_q.load([i_t * BT + i_i * BC, i_k * BK])
        b_z = desc_z.load([i_t * BT + i_i * BC, i_k * BK])

        o_i = tl.arange(0, BC)
        o_bk = tl.arange(0, BK)
        base_k = k + i_bh * T*K + (i_t * BT + i_j * BC) * K + i_k * BK
        m_bk = o_bk < BK
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BK,]
            b_k = tl.load(base_k + j * K + o_bk, mask=m_bk, other=0).to(tl.float32)
            # [BC,]
            b_A = tl.sum(b_q * exp(b_k[None, :] - b_z) * scale, 1)
            b_A = tl.where(o_i >= j, b_A, 0.)
            tl.store(A + o_A + j, b_A.to(b_q.dtype), mask=m_A)


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_V(
    q,
    v,
    z,
    h,
    o,
    A,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_h = make_tensor_descriptor(h + i_bh * NT*K*V + i_t * K * V, [K, V], [V, 1], [BK, BV])
        desc_zp = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])

        # [BT, BK]
        b_q = desc_q.load([i_t * BT, i_k * BK])
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_z = desc_z.load([i_t * BT, i_k * BK])
        # [BT, BK]
        b_zp = desc_zp.load([i_p * K + i_k * BK])
        b_q = (b_q * exp(b_zp[None, :] - b_z)).to(b_q.dtype)
        # [BK, BV]
        b_h = desc_h.load([i_k * BK, i_v * BV])
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
    desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    desc_o = make_tensor_descriptor(o + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    # [BT, BV]
    b_v = desc_v.load([i_t * BT, i_v * BV])
    # [BT, BT]
    b_A = desc_A.load([i_t * BT, 0])
    b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)
    desc_o.store([i_t * BT, i_v * BV], b_o.to(desc_o.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_dh(
    q,
    z,
    do,
    dh,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.full([BK if NORMK else BV], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        i_p = tl.maximum(i_t * BT - 1, 0)
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_dh = make_tensor_descriptor(dh + i_bh * NT*K*V + i_t * K*V, [K, V], [V, 1], [BK, BV])

        # [BK, BT]
        b_q = tl.trans(desc_q.load([i_t * BT, i_k * BK]))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = desc_do.load([i_t * BT, i_v * BV])

        desc_dh.store([i_k * BK, i_v * BV], b_dh.to(desc_dh.dtype))
        if NORMK:
            desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BT, BK])
            desc_zc = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
            # [BK,]
            b_zc = desc_zc.load([i_p * K + i_k * BK])
            b_r, b_zp = exp(b_zc - b_zp), b_zc
            # [BK, BT]
            b_z = tl.trans(desc_z.load([i_t * BT, i_k * BK]))
            b_q = (b_q * exp(b_zc[:, None] - b_z)).to(b_q.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[:, None]
        else:
            desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BT, BV])
            desc_zc = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
            # [BV,]
            b_zc = desc_zc.load([i_p * V + i_v * BV])
            b_r, b_zp = exp(b_zc - b_zp), b_zc
            # [BT, BV]
            b_z = desc_z.load([i_t * BT, i_v * BV])
            b_do = (b_do * exp(b_zc[None, :] - b_z)).to(b_do.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[None, :]
        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_V(
    k,
    v,
    z,
    h,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)

    desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_zc = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
    desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])

    # [BK,]
    b_zc = desc_zc.load([(i_t * BT + BT - 1) * K + i_k * BK])
    # [BT, BK]
    b_k = desc_k.load([i_t * BT, i_k * BK])
    b_k = exp(b_k - b_zc[None, :]).to(b_k.dtype)
    # [BT, BT]
    b_A = tl.trans(desc_A.load([i_t * BT, 0]))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_h = make_tensor_descriptor(h + i_bh * NT*K*V + i_t * V * K, [K, V], [V, 1], [BK, BV])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_dh = make_tensor_descriptor(dh + i_bh * NT*K*V + i_t * K*V, [K, V], [V, 1], [BK, BV])
        desc_dv = make_tensor_descriptor(dv + (i_k*n_bh+i_bh) * T*V, [T, V], [V, 1], [BT, BV])

        # [BT, BV]
        b_v = desc_v.load([i_t * BT, i_v * BV])
        # [BV, BK]
        b_h = tl.trans(desc_h.load([i_k * BK, i_v * BV]))
        # [BT, BV]
        b_do = desc_do.load([i_t * BT, i_v * BV])
        # [BK, BV]
        b_dh = desc_dh.load([i_k * BK, i_v * BV])

        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
        if i_k == 0:
            b_dv += tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)
        b_do = (b_do * scale).to(b_do.dtype)
        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))
        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
    desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_zp = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
    # [BK,]
    b_zp = desc_zp.load([i_p * K + i_k * BK])
    # [BT, BK]
    b_z = desc_z.load([i_t * BT, i_k * BK])
    b_z = exp(b_zp[None, :] - b_z)
    # [BT, BK]
    b_dq = b_dq * b_z
    b_dk = b_dk * b_k

    desc_dq = make_tensor_descriptor(dq + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_dk = make_tensor_descriptor(dk + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_dA = make_tensor_descriptor(dA + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    desc_dq.store([i_t * BT, i_k * BK], b_dq.to(desc_dq.dtype))
    desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        desc_dA.store([i_t * BT, 0], b_dA.to(desc_dA.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_intra_V(
    q,
    k,
    z,
    dA,
    dq,
    dk,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BC, BK])
    desc_zn = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
    # [BK,]
    b_zn = desc_zn.load([(i_t * BT + i_i * BC) * K + i_k * BK])
    # [BC, BK]
    b_z = desc_z.load([i_t * BT + i_i * BC, i_k * BK])
    b_zq = exp(b_zn[None, :] - b_z)
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_dA = make_tensor_descriptor(dA + i_bh * T * BT, [T, BT], [BT, 1], [BC, BC])
        # [BC, BK]
        b_k = desc_k.load([i_t * BT + i_j * BC, i_k * BK])
        b_kz = exp(b_k - b_zn[None, :]).to(b_k.dtype)
        # [BC, BC]
        b_dA = desc_dA.load([i_t * BT + i_i * BC, i_j * BC])
        # [BC, BK]
        b_dq += tl.dot(b_dA, b_kz, allow_tf32=False)
    b_dq *= b_zq

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        desc_kj = make_tensor_descriptor(k + i_bh * T*K, [T * K], [1], [BK])
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = desc_kj.load([(i_t * BT + i_i*BC+j) * K + i_k * BK]).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dq += tl.where(m_i, b_dA[:, None] * exp(b_kj[None, :] - b_z), 0.)
    desc_dq = make_tensor_descriptor(dq + i_bh * T*K, [T, K], [K, 1], [BC, BK])
    desc_dq.store([i_t * BT + i_i * BC, i_k * BK], b_dq.to(desc_dq.dtype))

    tl.debug_barrier()
    desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BC, BK])
    desc_zn = make_tensor_descriptor(z + i_bh * T*K, [T*K], [1], [BK])
    # [BK,]
    b_zn = desc_zn.load([(i_t * BT + i_i * BC + BC - 1) * K + i_k * BK])
    # [BC, BK]
    b_k = desc_k.load([i_t * BT + i_i * BC, i_k * BK])
    b_kz = exp(b_k - b_zn[None, :])
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_z = make_tensor_descriptor(z + i_bh * T*K, [T, K], [K, 1], [BC, BK])
        desc_dA = make_tensor_descriptor(dA + i_bh * T * BT, [T, BT], [BT, 1], [BC, BC])
        # [BC, BK]
        b_q = desc_q.load([i_t * BT + i_j * BC, i_k * BK])
        b_z = desc_z.load([i_t * BT + i_j * BC, i_k * BK])
        b_qz = (b_q * exp(b_zn[None, :] - b_z)).to(b_q.dtype)
        # [BC, BC]
        b_dA = desc_dA.load([i_t * BT + i_j * BC, i_i * BC])
        # [BC, BK]
        b_dk += tl.dot(tl.trans(b_dA), b_qz, allow_tf32=False)
    b_dk *= b_kz

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        desc_qj = make_tensor_descriptor(q + i_bh * T*K, [T * K], [1], [BK])
        desc_zj = make_tensor_descriptor(z + i_bh * T*K, [T * K], [1], [BK])
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = desc_qj.load([(i_t * BT + i_i * BC + j) * K + i_k * BK]).to(tl.float32)
        b_zj = desc_zj.load([(i_t * BT + i_i * BC + j) * K + i_k * BK]).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * exp(b_k - b_zj[None, :]), 0.)
    desc_dk = make_tensor_descriptor(dk + i_bh * T*K, [T, K], [K, 1], [BC, BK])
    desc_dk.store([i_t * BT + i_i * BC, i_k * BK], b_dk.to(desc_dk.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_intra_K(
    v,
    z,
    do,
    dA,
    scale,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        desc_zn = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        desc_dA = make_tensor_descriptor(dA+(i_bh+i_v*n_bh)*T*BT, [T, BT], [BT, 1], [BC, BC])
        # [BV,]
        b_zn = desc_zn.load([(i_t * BT + i_i * BC) * V + i_v * BV])
        # [BC, BV]
        b_z = desc_z.load([i_t * BT + i_i * BC, i_v * BV])
        b_do = desc_do.load([i_t * BT + i_i * BC, i_v * BV])
        b_do = (b_do * exp(b_zn[None, :] - b_z) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.trans(desc_v.load([i_t * BT + i_j * BC, i_v * BV]))
        b_v = exp(b_v - b_zn[:, None]).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_v, allow_tf32=False)
        desc_dA.store([i_t * BT + i_i * BC, i_j * BC], b_dA.to(dA.dtype.element_ty))
    elif i_i == i_j:
        desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        # [BC, BV]
        b_z = desc_z.load([i_t * BT + i_i * BC, i_v * BV])
        b_do = desc_do.load([i_t * BT + i_i * BC, i_v * BV]) * scale

        o_i = tl.arange(0, BC)
        o_bv = tl.arange(0, BV)
        base_v = v + i_bh * T*V + (i_t * BT + i_j * BC) * V + i_v * BV
        m_bv = o_bv < BV
        o_A = (i_bh + i_v * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BV,]
            b_v = tl.load(base_v + j * V + o_bv, mask=m_bv, other=0).to(tl.float32)
            # [BC,]
            b_dA = tl.sum(b_do * exp(b_v[None, :] - b_z), 1)
            b_dA = tl.where(o_i >= j, b_dA, 0)
            tl.store(dA + o_A + j, b_dA.to(b_do.dtype), mask=m_A)


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_K(
    q,
    k,
    v,
    z,
    h,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_A = make_tensor_descriptor(A + (i_k*n_bh+i_bh) * T * BT, [T, BT], [BT, 1], [BT, BT])

    # [BT, BK]
    b_q = desc_q.load([i_t * BT, i_k * BK])
    b_k = desc_k.load([i_t * BT, i_k * BK])
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k), allow_tf32=False)
    b_A = tl.where(m_s, b_A, 0.)
    desc_A.store([i_t * BT, 0], b_A.to(desc_A.dtype))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_zp = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
        desc_zc = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
        desc_h = make_tensor_descriptor(h + i_bh * NT*K*V + i_t * K*V, [K, V], [V, 1], [BK, BV])

        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BT, BV])
        desc_dh = make_tensor_descriptor(dh + i_bh * NT*K*V + i_t * K*V, [K, V], [V, 1], [BK, BV])
        desc_dv = make_tensor_descriptor(dv + (i_k*n_bh+i_bh) * T*V, [T, V], [V, 1], [BT, BV])

        # [BV,]
        b_zp = desc_zp.load([i_p * V + i_v * BV])
        b_zc = desc_zc.load([(i_t * BT + BT - 1) * V + i_v * BV])
        # [BT, BV]
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_v = exp(b_v - b_zc[None, :]).to(b_v.dtype)
        b_z = desc_z.load([i_t * BT, i_v * BV])
        b_z = exp(b_zp[None, :] - b_z)
        # [BV, BK]
        b_h = tl.trans(desc_h.load([i_k * BK, i_v * BV]))
        # [BT, BV]
        b_do = desc_do.load([i_t * BT, i_v * BV])
        b_do = (b_do * b_z * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = desc_dh.load([i_k * BK, i_v * BV])

        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = b_v * tl.dot(b_k, b_dh, allow_tf32=False)
        desc_dv.store([i_t * BT, i_v * BV], b_dv.to(desc_dv.dtype))
    desc_dA = make_tensor_descriptor(dA + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    # [BT, BT]
    b_dA = desc_dA.load([i_t * BT, 0])
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, allow_tf32=False)

    desc_dq = make_tensor_descriptor(dq + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_dk = make_tensor_descriptor(dk + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_dq.store([i_t * BT, i_k * BK], b_dq.to(desc_dq.dtype))
    desc_dk.store([i_t * BT, i_k * BK], b_dk.to(desc_dk.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_intra_KV(
    v,
    z,
    A,
    do,
    dv,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BC, BV])
    desc_zn = make_tensor_descriptor(z + i_bh * T*V, [T*V], [1], [BV])
    # [BV,]
    b_zn = desc_zn.load([(i_t * BT + i_i * BC + BC - 1) * V + i_v * BV])
    # [BC, BV]
    b_v = desc_v.load([i_t * BT + i_i * BC, i_v * BV])
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        desc_z = make_tensor_descriptor(z + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BC, BC])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T, V], [V, 1], [BC, BV])
        # [BC, BV]
        b_z = desc_z.load([i_t * BT + i_j * BC, i_v * BV])
        b_do = desc_do.load([i_t * BT + i_j * BC, i_v * BV])
        b_do = (b_do * exp(b_zn[None, :] - b_z)).to(b_do.dtype)
        # [BC, BC]
        b_A = tl.trans(desc_A.load([i_t * BT + i_j * BC, i_i * BC]))
        b_dv += tl.dot(b_A, b_do, allow_tf32=False)
    b_dv *= exp(b_v - b_zn[None, :])

    o_i = tl.arange(0, BC)
    for j in range(0, BC):
        desc_z = make_tensor_descriptor(z + i_bh * T*V, [T * V], [1], [BV])
        desc_A = make_tensor_descriptor(A + i_bh * T * BT, [T * BT], [1], [BC])
        desc_do = make_tensor_descriptor(do + i_bh * T*V, [T * V], [1], [BV])
        # [BC,]
        b_A = desc_A.load([(i_t * BT + i_i * BC + j) * BT + i_i * BC])
        # [BV,]
        b_z = desc_z.load([(i_t * BT + i_i * BC + j) * V + i_v * BV])
        b_do = desc_do.load([(i_t * BT + i_i * BC + j) * V + i_v * BV])
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_v - b_z[None, :]) * b_A[:, None] * b_do[None, :], 0.)
    desc_dv = make_tensor_descriptor(dv + i_bh * T*V, [T, V], [V, 1], [BC, BV])
    desc_dv.store([i_t * BT + i_i * BC, i_v * BV], b_dv.to(desc_dv.dtype))


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_rcum_inter(
    s,
    z,
    ss,
    doo,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)

    b_sp = tl.zeros([BS], dtype=tl.float32)
    b_zp = tl.full([BS], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        desc_s = make_tensor_descriptor(s + i_bh * T*S, [T, S], [S, 1], [BT, BS])
        desc_z = make_tensor_descriptor(z + i_bh * T*S, [T, S], [S, 1], [BT, BS])
        desc_zc = make_tensor_descriptor(z + i_bh * T*S, [T*S], [1], [BS])
        desc_ss = make_tensor_descriptor(ss + i_bh * T*S, [T, S], [S, 1], [BT, BS])
        desc_doo = make_tensor_descriptor(doo + i_bh * T*S, [T, S], [S, 1], [BT, BS])
        # [BS,]
        b_zc = desc_zc.load([(i_t * BT) * S + i_m * BS])
        # [BT, BS]
        b_s = desc_s.load([i_t * BT, i_m * BS])
        b_z = desc_z.load([i_t * BT, i_m * BS])
        b_ss = desc_ss.load([i_t * BT, i_m * BS])

        b_doo = exp(b_s - b_zp[None, :]) * b_sp[None, :]
        desc_doo.store([i_t * BT, i_m * BS], b_doo.to(desc_doo.dtype))
        # [BS,]
        b_sp = b_sp * exp(b_zc - b_zp) + tl.sum(b_ss * exp(b_zc[None, :] - b_z), 0)
        b_zp = b_zc


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_rcum_intra(
    s,
    z,
    ss,
    doo,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
):
    i_s, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    o_i = tl.arange(0, BC)
    m_o = tl.full([BC, BC], 1., dtype=tl.float32)

    desc_s = make_tensor_descriptor(s + i_bh * T*S, [T, S], [S, 1], [BC, BS])
    desc_zn = make_tensor_descriptor(z + i_bh * T*S, [T*S], [1], [BS])
    desc_doo = make_tensor_descriptor(doo + i_bh * T*S, [T, S], [S, 1], [BC, BS])
    # [BC, BS]
    b_s = desc_s.load([i_t * BT + i_i * BC, i_s * BS])
    # [BS,]
    b_zn = desc_zn.load([(i_t * BT + i_i * BC + BC - 1) * S + i_s * BS])

    b_doo = tl.zeros([BC, BS], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        desc_z = make_tensor_descriptor(z + i_bh * T*S, [T, S], [S, 1], [BC, BS])
        desc_ss = make_tensor_descriptor(ss + i_bh * T*S, [T, S], [S, 1], [BC, BS])
        # [BC, BS]
        b_z = desc_z.load([i_t * BT + i_j * BC, i_s * BS])
        b_ss = desc_ss.load([i_t * BT + i_j * BC, i_s * BS])
        # [BC, BS]
        b_doo += b_ss * exp(b_zn[None, :] - b_z)
    b_doo = exp(b_s - b_zn[None, :]) * tl.dot(m_o.to(b_s.dtype), b_doo.to(b_s.dtype), allow_tf32=False)

    for j in range(0, BC):
        desc_z = make_tensor_descriptor(z + i_bh * T*S, [T*S], [1], [BS])
        desc_ss = make_tensor_descriptor(ss + i_bh * T*S, [T*S], [1], [BS])
        # [BS,]
        b_z = desc_z.load([(i_t * BT + i_i * BC + j) * S + i_s * BS])
        b_ss = desc_ss.load([(i_t * BT + i_i * BC + j) * S + i_s * BS])
        # [BC, BS]
        m_i = o_i[:, None] <= j
        b_doo += tl.where(m_i, exp(b_s - b_z[None, :]) * b_ss[None, :], 0.)
    b_doo += desc_doo.load([i_t * BT + i_i * BC, i_s * BS])
    desc_doo.store([i_t * BT + i_i * BC, i_s * BS], b_doo.to(desc_doo.dtype))


class ChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, s, initial_state, output_final_state):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NV, NM = triton.cdiv(V, BV), triton.cdiv(M, BM)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_pre(s, B, H, T, S):
            # keep cummulative normalizer in fp32
            z = torch.empty_like(s, dtype=torch.float)
            grid = (B * H,)
            logcumsumexp_fwd_kernel[grid](
                s, z,
                T=T, S=S,
            )
            return z

        def fwd_inner(q, k, v, z, B, H, T, K, V, BT, BK, BV, NT, normk=False, h0=None, ht=None):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NV, NK, B * H)
            chunk_abc_fwd_kernel_h[grid](
                k, v, z, h, h0, ht,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                NORMK=normk,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=ht is not None,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return h

        final_state = None
        if output_final_state:
            final_state = (q.new_empty(B, H, K, M, dtype=torch.float),
                           q.new_empty(B, H, M, V, dtype=torch.float))

        z = fwd_pre(s, B, H, T, M)
        scale = K ** -0.5
        hk = fwd_inner(
            q=q, k=k, v=s, z=z,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            normk=False,
            h0=initial_state[0] if initial_state is not None else None,
            ht=final_state[0] if final_state is not None else None,
        )
        ok1 = torch.empty_like(s)
        Ak = q.new_empty(B, H, T, BT)
        grid = (NM, NT, B * H)
        chunk_abc_fwd_kernel_K[grid](
            q, k, z, hk, ok1, Ak,
            scale=scale,
            T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ok0 = torch.empty_like(s)
        grid = (NM, NT * NC, B * H)
        chunk_abc_fwd_kernel_intra_K[grid](
            s, z, ok0, Ak,
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages,
        )
        ok = ok0.add_(ok1)

        scale = 1.
        # p is kept in fp32 for safe softmax backward
        p = softmax_fwd(ok, dtype=torch.float)
        qv = p.to(q.dtype)

        scale = 1.
        hv = fwd_inner(
            q=qv, k=s, v=v, z=z,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
            normk=True,
            h0=initial_state[1] if initial_state is not None else None,
            ht=final_state[1] if final_state is not None else None,
        )
        Av = q.new_zeros(NM, B, H, T, BT)
        grid = (NM, NT * NC * NC, B * H)
        chunk_abc_fwd_kernel_intra_V[grid](
            qv, s, z, Av,
            scale=scale,
            T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages,
        )
        Av = Av.sum(0)
        ov = torch.empty_like(v)
        grid = (NV, NT, B * H)
        chunk_abc_fwd_kernel_V[grid](
            qv, v, z, hv, ov, Av,
            scale=scale,
            T=T,
            K=M,
            V=V,
            BT=BT,
            BK=BM,
            BV=BV,
            NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, s, z, ok, p, hk, hv, Av)
        ctx.BT = BT
        return ov, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dov, dht=None):
        q, k, v, s, z, ok, p, hk, hv, Av = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK, NM = triton.cdiv(K, BK), triton.cdiv(M, BM)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def bwd_inner(q, z, do, B, H, T, K, V, BT, BK, BV, NT, scale, normk=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_abc_bwd_kernel_dh[grid](
                q, z, do, dh,
                scale=scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                NORMK=normk,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return dh

        def bwd_post(s, z, ss, B, H, T, S, BT, BC, BS, NT, NC, NS):
            doo = torch.empty_like(s)
            grid = (NS, B * H)
            chunk_abc_bwd_kernel_rcum_inter[grid](
                s, z, ss, doo,
                T=T, S=S, BT=BT, BS=BS, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grid = (NS, NT * NC, B * H)
            chunk_abc_bwd_kernel_rcum_intra[grid](
                s, z, ss, doo,
                T=T, S=S, BT=BT, BC=BC, BS=BS, NC=NC,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return doo

        scale = 1.
        qv = p.to(q.dtype)
        dhv = bwd_inner(
            qv, z, dov,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
            scale=scale,
            normk=True,
        )
        dp1 = torch.empty_like(p)
        dsv1 = torch.empty_like(s, dtype=torch.float)
        dv = v.new_empty(NM, *v.shape)
        dAv = q.new_zeros(B, H, T, BT)
        grid = (NM, NT, B * H)
        chunk_abc_bwd_kernel_V[grid](
            s, v, z, hv, Av, dov, dhv, dp1, dsv1, dv, dAv,
            scale=scale,
            T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dv = dv.sum(0)
        dp0 = torch.empty_like(p)
        dsv0 = s.new_zeros(s.shape, dtype=torch.float)
        grid = (NM, NT * NC, B * H)
        chunk_abc_bwd_kernel_intra_V[grid](
            qv, s, z, dAv, dp0, dsv0,
            T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages,
        )
        dp = dp1.add_(dp0)
        dsv = dsv1.add_(dsv0)

        # softmax gradient, equivalent to:
        # dok = p * (dp - (p * dp).sum(-1, True))
        dok = softmax_bwd(p, dp, dtype=ok.dtype)

        scale = K ** -0.5
        dhk = bwd_inner(
            q, z, dok,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            scale=scale,
            normk=False,
        )
        dAk = q.new_zeros(NM, B, H, T, BT)
        grid = (NM, NT * NC * NC, B * H)
        chunk_abc_bwd_kernel_intra_K[grid](
            s, z, dok, dAk,
            scale=scale,
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages,
        )
        dAk = dAk.sum(0)

        Ak = q.new_zeros(NK, B, H, T, BT)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dsk1 = s.new_empty(NK, *s.shape, dtype=torch.float)
        grid = (NK, NT, B * H)
        chunk_abc_bwd_kernel_K[grid](
            q, k, s, z, hk, Ak, dok, dhk, dq, dk, dsk1, dAk,
            scale=scale,
            T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        Ak = Ak.sum(0)
        dsk1 = dsk1.sum(0)
        dsk0 = torch.empty_like(s, dtype=torch.float)
        grid = (NM, NT * NC, B * H)
        chunk_abc_bwd_kernel_intra_KV[grid](
            s, z, Ak, dok, dsk0,
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages,
        )
        ds = dsv.add_(dsk1.add_(dsk0))
        ds -= bwd_post(s, z, ok * dok + p * dp, B, H, T, M, BT, BC, BM, NT, NC, NM)
        ds = ds.to(s.dtype)
        return dq, dk, dv, ds, None, None


@torch.compiler.disable
def chunk_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    initial_state: tuple[torch.Tensor] | None = None,
    output_final_state: bool = False,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, T, H, M]`.
        initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]]):
            Initial states of shape `[B, H, K, M]` and `[B, H, M, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, M]` and `[B, H, M, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, M]` and `[B, H, M, V]` if `output_final_state=True` else `None`.
    """
    if not head_first:
        q, k, v, s = map(lambda x: x.transpose(1, 2), (q, k, v, s))
    o, final_state = ChunkABCFunction.apply(q, k, v, s, initial_state, output_final_state)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
