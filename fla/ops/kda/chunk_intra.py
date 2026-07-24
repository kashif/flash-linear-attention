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
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2, gather
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_GATHER_SUPPORTED, IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "HV", "K", "BT", "BC", "NC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        desc_k0 = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
        desc_g0 = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
        b_k0 = desc_k0.load([i_tc0, i_k * BK]).to(tl.float32)
        b_g0 = desc_g0.load([i_tc0, i_k * BK]).to(tl.float32)

        if i_tc1 < T:
            desc_q1 = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
            desc_k1 = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
            desc_g1 = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
            # [BC, BK]
            b_q1 = desc_q1.load([i_tc1, i_k * BK]).to(tl.float32)
            b_k1 = desc_k1.load([i_tc1, i_k * BK]).to(tl.float32)
            b_g1 = desc_g1.load([i_tc1, i_k * BK]).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if NC >= 3 and i_tc2 < T:
                desc_q2 = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
                desc_k2 = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
                desc_g2 = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
                # [BC, BK]
                b_q2 = desc_q2.load([i_tc2, i_k * BK]).to(tl.float32)
                b_k2 = desc_k2.load([i_tc2, i_k * BK]).to(tl.float32)
                b_g2 = desc_g2.load([i_tc2, i_k * BK]).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if NC >= 4 and i_tc3 < T:
                    desc_q3 = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
                    desc_k3 = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
                    desc_g3 = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
                    # [BC, BK]
                    b_q3 = desc_q3.load([i_tc3, i_k * BK]).to(tl.float32)
                    b_k3 = desc_k3.load([i_tc3, i_k * BK]).to(tl.float32)
                    b_g3 = desc_g3.load([i_tc3, i_k * BK]).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        desc_Aqk10 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk10.store([i_tc1, 0], (b_Aqk10 * scale).to(Aqk.dtype.element_ty))

        b_b1 = tl.load(beta + bos * HV + i_hv + (i_tc1 + tl.arange(0, BC)) * HV, mask=(i_tc1 + tl.arange(0, BC)) < T, other=0).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3 and i_tc2 < T:
        desc_Aqk20 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk21 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk20.store([i_tc2, 0], (b_Aqk20 * scale).to(Aqk.dtype.element_ty))
        desc_Aqk21.store([i_tc2, BC], (b_Aqk21 * scale).to(Aqk.dtype.element_ty))

        b_b2 = tl.load(beta + bos * HV + i_hv + (i_tc2 + tl.arange(0, BC)) * HV, mask=(i_tc2 + tl.arange(0, BC)) < T, other=0).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if NC >= 4 and i_tc3 < T:
        desc_Aqk30 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk31 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk32 = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Aqk30.store([i_tc3, 0], (b_Aqk30 * scale).to(Aqk.dtype.element_ty))
        desc_Aqk31.store([i_tc3, BC], (b_Aqk31 * scale).to(Aqk.dtype.element_ty))
        desc_Aqk32.store([i_tc3, 2*BC], (b_Aqk32 * scale).to(Aqk.dtype.element_ty))

        b_b3 = tl.load(beta + bos * HV + i_hv + (i_tc3 + tl.arange(0, BC)) * HV, mask=(i_tc3 + tl.arange(0, BC)) < T, other=0).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    desc_Akk00 = make_tensor_descriptor(Akkd, [T, BC], [HV*BC, 1], [BC, BC])
    desc_Akk11 = make_tensor_descriptor(Akkd, [T, BC], [HV*BC, 1], [BC, BC])
    b_Ai00 = desc_Akk00.load([i_tc0, 0]).to(tl.float32)
    b_Ai11 = desc_Akk11.load([i_tc1, 0]).to(tl.float32)
    if NC >= 3:
        desc_Akk22 = make_tensor_descriptor(Akkd, [T, BC], [HV*BC, 1], [BC, BC])
        b_Ai22 = desc_Akk22.load([i_tc2, 0]).to(tl.float32)
    if NC >= 4:
        desc_Akk33 = make_tensor_descriptor(Akkd, [T, BC], [HV*BC, 1], [BC, BC])
        b_Ai33 = desc_Akk33.load([i_tc3, 0]).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        if NC >= 3:
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        if NC >= 4:
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        if NC >= 3:
            b_Ai22 += m_I
        if NC >= 4:
            b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )

    ################################################################################
    # store full Akk_inv to Akk
    ################################################################################

    desc_Akk00 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
    desc_Akk10 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
    desc_Akk11 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])

    desc_Akk00.store([i_tc0, 0], b_Ai00.to(Akk.dtype.element_ty))
    desc_Akk10.store([i_tc1, 0], b_Ai10.to(Akk.dtype.element_ty))
    desc_Akk11.store([i_tc1, BC], b_Ai11.to(Akk.dtype.element_ty))
    if NC >= 3:
        desc_Akk20 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk21 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk22 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk20.store([i_tc2, 0], b_Ai20.to(Akk.dtype.element_ty))
        desc_Akk21.store([i_tc2, BC], b_Ai21.to(Akk.dtype.element_ty))
        desc_Akk22.store([i_tc2, 2*BC], b_Ai22.to(Akk.dtype.element_ty))
    if NC >= 4:
        desc_Akk30 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk31 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk32 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk33 = make_tensor_descriptor(Akk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_Akk30.store([i_tc3, 0], b_Ai30.to(Akk.dtype.element_ty))
        desc_Akk31.store([i_tc3, BC], b_Ai31.to(Akk.dtype.element_ty))
        desc_Akk32.store([i_tc3, 2*BC], b_Ai32.to(Akk.dtype.element_ty))
        desc_Akk33.store([i_tc3, 3*BC], b_Ai33.to(Akk.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT', 'HV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * all + bos) * HV + i_hv

    desc_g = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
    b_g = desc_g.load([i_ti, i_k * BK]).to(tl.float32)

    b_b = tl.load(beta + (i_ti + tl.arange(0, BC)) * HV, mask=(i_ti + tl.arange(0, BC)) < T, other=0)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
            desc_gk = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
            desc_dAqk = make_tensor_descriptor(dAqk, [T, BT], [HV*BT, 1], [BC, BC])
            desc_dAkk = make_tensor_descriptor(dAkk, [T, BT], [HV*BT, 1], [BC, BC])
            # [BC, BK]
            b_k = desc_k.load([i_t * BT + i_j * BC, i_k * BK])
            b_gk = desc_gk.load([i_t * BT + i_j * BC, i_k * BK])
            b_kg = b_k * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = desc_dAqk.load([i_ti, i_j * BC])
            b_dAkk = desc_dAkk.load([i_ti, i_j * BC])
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * HV*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k

    desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
    desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
    b_q = desc_q.load([i_ti, i_k * BK])
    b_k = desc_k.load([i_ti, i_k * BK])

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        desc_dAqk = make_tensor_descriptor(dAqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_dAkk = make_tensor_descriptor(dAkk, [T, BT], [HV*BT, 1], [BC, BC])
        b_dAqk_diag_qk = desc_dAqk.load([i_ti, i_i * BC]).to(tl.float32)
        b_dAkk_diag_qk = desc_dAkk.load([i_ti, i_i * BC]).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H*K
            p_gkj += HV*K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    desc_dq = make_tensor_descriptor(dq, [T, K], [HV*K, 1], [BC, BK])
    desc_dq2 = make_tensor_descriptor(dq2, [T, K], [HV*K, 1], [BC, BK])

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + desc_dq.load([i_ti, i_k * BK])
    desc_dq2.store([i_ti, i_k * BK], b_dq2.to(desc_dq2.dtype))
    tl.store(db + (i_ti + tl.arange(0, BC)) * HV, b_db.to((db).dtype.element_ty), mask=(i_ti + tl.arange(0, BC)) < T)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
            desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
            desc_gk = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])
            desc_dAqk = make_tensor_descriptor(dAqk, [T, BT], [HV*BT, 1], [BC, BC])
            desc_dAkk = make_tensor_descriptor(dAkk, [T, BT], [HV*BT, 1], [BC, BC])
            # [BC]
            b_b = tl.load(beta + (i_t * BT + i_j * BC + tl.arange(0, BC)) * HV, mask=(i_t * BT + i_j * BC + tl.arange(0, BC)) < T, other=0)
            # [BC, BK]
            b_q = desc_q.load([i_t*BT+i_j*BC, i_k*BK])
            b_kb = desc_k.load([i_t * BT + i_j * BC, i_k * BK]) * b_b[:, None]
            b_gk = desc_gk.load([i_t * BT + i_j * BC, i_k*BK]).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.trans(desc_dAqk.load([i_t * BT + i_j * BC, i_i * BC]))
            b_dAkk = tl.trans(desc_dAkk.load([i_t * BT + i_j * BC, i_i * BC]))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * HV*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
        b_q = desc_q.load([i_ti, i_k * BK])
        b_b = tl.load(beta + (i_ti + tl.arange(0, BC)) * HV, mask=(i_ti + tl.arange(0, BC)) < T, other=0)

        desc_dAqk = make_tensor_descriptor(dAqk, [T, BT], [HV*BT, 1], [BC, BC])
        desc_dAkk = make_tensor_descriptor(dAkk, [T, BT], [HV*BT, 1], [BC, BC])
        b_dAqk_diag_kk = tl.trans(desc_dAqk.load([i_ti, i_i * BC])).to(tl.float32)
        b_dAkk_diag_kk = tl.trans(desc_dAkk.load([i_ti, i_i * BC])).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        # ensure numerical stability
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b[:, None] * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * HV*BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV*BT)
            # [BK,]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj += HV*K
            p_bj += HV
    desc_dk = make_tensor_descriptor(dk, [T, K], [HV*K, 1], [BC, BK])
    desc_dk2 = make_tensor_descriptor(dk2, [T, K], [HV*K, 1], [BC, BK])
    desc_dg = make_tensor_descriptor(dg, [T, K], [HV*K, 1], [BC, BK])
    desc_dg2 = make_tensor_descriptor(dg2, [T, K], [HV*K, 1], [BC, BK])

    b_dg2 += (b_dk2 - b_dkt) * b_k + desc_dg.load([i_ti, i_k * BK])
    b_dk2 += desc_dk.load([i_ti, i_k * BK])
    b_dk2 += b_dkt

    desc_dk2.store([i_ti, i_k * BK], b_dk2.to(desc_dk2.dtype))
    desc_dg2.store([i_ti, i_k * BK], b_dg2.to(desc_dg2.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    desc_q = make_tensor_descriptor(q, [T, K], [H*K, 1], [BC, BK])
    desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BC, BK])
    desc_g = make_tensor_descriptor(g, [T, K], [HV*K, 1], [BC, BK])


    b_q = desc_q.load([i_ti, 0])
    b_k = desc_k.load([i_ti, 0])
    b_g = desc_g.load([i_ti, 0])
    b_beta = tl.load(beta + (i_ti + tl.arange(0, BC)) * HV, mask=(i_ti + tl.arange(0, BC)) < T, other=0)

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        # caculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    desc_Aqk = make_tensor_descriptor(Aqk, [T, BT], [HV*BT, 1], [BC, BC])
    desc_Akk = make_tensor_descriptor(Akk, [T, BC], [HV*BC, 1], [BC, BC])
    desc_Aqk.store([i_ti, i_i * BC], b_Aqk.to(Aqk.dtype.element_ty))
    desc_Akk.store([i_ti, 0], b_Akk.to(Akk.dtype.element_ty))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * HV*BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    desc_Akk.store([i_ti, 0], b_Ai.to(Akk.dtype.element_ty))


@dispatch('kda')
def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    # Step 1: Run token_parallel first to compute diagonal blocks into Akkd (fp32)
    # Step 1: compute diagonal blocks into Akk_diag (fp32)
    if safe_gate:
        grid = (NT, NC, B * HV)
        BK = triton.next_power_of_2(K)
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_SAFE_GATE=safe_gate,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


@dispatch('kda')
def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC, NT, B * HV)
    chunk_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg
