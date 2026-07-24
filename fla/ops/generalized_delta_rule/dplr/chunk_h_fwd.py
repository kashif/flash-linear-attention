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
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_h(
    kg,
    v,
    w,
    bg,
    u,
    v_new,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    o_k = i_k * BK + tl.arange(0, BK)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        desc_h0 = make_tensor_descriptor(h0 + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        b_h = desc_h0.load([i_k * BK, i_v * BV]).to(tl.float32)

    for i_t in range(NT):
        desc_h = make_tensor_descriptor(h + ((boh + i_t) * H + i_h) * K*V, [K, V], [V, 1], [BK, BV])
        desc_h.store([i_k * BK, i_v * BV], b_h.to(desc_h.dtype))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            desc_kg = make_tensor_descriptor(kg+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_bg = make_tensor_descriptor(bg+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_w = make_tensor_descriptor(w+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_v = make_tensor_descriptor(v+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            desc_u = make_tensor_descriptor(u+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            desc_v_new = make_tensor_descriptor(v_new+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            # [BK, BC]
            b_kg = tl.trans(desc_kg.load([i_t * BT + i_c * BC, i_k * BK]))
            b_v = desc_v.load([i_t * BT + i_c * BC, i_v * BV])
            b_w = desc_w.load([i_t * BT + i_c * BC, i_k * BK])
            b_bg = tl.trans(desc_bg.load([i_t * BT + i_c * BC, i_k * BK]))
            b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + desc_u.load([i_t * BT + i_c * BC, i_v * BV])
            b_hc += tl.dot(b_kg, b_v)
            b_hc += tl.dot(b_bg.to(b_hc.dtype), b_v2)
            desc_v_new.store([i_t*BT+i_c*BC, i_v * BV], b_v2.to(desc_v_new.dtype))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k, mask=o_k < K).to(tl.float32)
        b_h *= exp2(b_g_last[:, None])
        b_h += b_hc

    if STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(ht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_ht.store([i_k * BK, i_v * BV], b_h.to(desc_ht.dtype, fp_downcast_rounding="rtne"))


def chunk_dplr_fwd_h(
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BK = max(triton.next_power_of_2(K), 16)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size

    if check_shared_mem('hopper', kg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', kg.device.index):  # A100
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = kg.new_empty(B, NT, H, K, V)
    final_state = kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)
    chunk_dplr_fwd_kernel_h[grid](
        kg=kg,
        v=v,
        w=w,
        bg=bg,
        u=u,
        v_new=v_new,
        h=h,
        gk=gk,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
    )
    return h, v_new, final_state
