# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule.iplr.wy_fast import prepare_wy_repr_fwd
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4] + ([] if check_shared_mem('hopper') else [8])
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_generalized_iplr_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    b,
    u,
    v_new,
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
            desc_k = make_tensor_descriptor(k+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_b = make_tensor_descriptor(b+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_d = make_tensor_descriptor(d+(bos*H+i_h)*K, [T, K], [H*K, 1], [BC, BK])
            desc_v = make_tensor_descriptor(v+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            desc_u = make_tensor_descriptor(u+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            desc_v_new = make_tensor_descriptor(v_new+(bos*H+i_h)*V, [T, V], [H*V, 1], [BC, BV])
            # [BK, BC]
            b_k = tl.trans(desc_k.load([i_t * BT + i_c * BC, i_k * BK]))
            b_v = desc_v.load([i_t * BT + i_c * BC, i_v * BV])
            b_d = desc_d.load([i_t * BT + i_c * BC, i_k * BK])
            b_b = tl.trans(desc_b.load([i_t * BT + i_c * BC, i_k * BK]))
            b_v2 = tl.dot(b_d, b_h.to(b_d.dtype)) + desc_u.load([i_t * BT + i_c * BC, i_v * BV])
            b_hc += tl.dot(b_k, b_v)
            b_hc += tl.dot(b_b, b_v2.to(b_k.dtype))
            desc_v_new.store([i_t*BT+i_c*BC, i_v * BV], b_v2.to(desc_v_new.dtype))
        b_h += b_hc

    if STORE_FINAL_STATE:
        desc_ht = make_tensor_descriptor(ht + i_nh * K*V, [K, V], [V, 1], [BK, BV])
        desc_ht.store([i_k * BK, i_v * BV], b_h.to(desc_ht.dtype))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_generalized_iplr_delta_rule_fwd_kernel_o(
    q,
    k,
    v,
    u,
    b,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    u += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    stride_qk = H*K
    stride_vo = H*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        desc_q = make_tensor_descriptor(q, [T, K], [stride_qk, 1], [BT, BK])
        desc_k = make_tensor_descriptor(k, [T, K], [stride_qk, 1], [BT, BK])
        desc_h = make_tensor_descriptor(h, [K, V], [V, 1], [BK, BV])
        desc_b = make_tensor_descriptor(b, [T, K], [stride_qk, 1], [BT, BK])
        # [BT, BK]
        b_q = desc_q.load([i_t * BT, i_k * BK])
        # [BK, BT]
        b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
        b_b = tl.trans(desc_b.load([i_t * BT, i_k * BK]))
        # [BK, BV]
        b_h = desc_h.load([i_k * BK, i_v * BV])
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_Aqk += tl.dot(b_q, b_k)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_Aqb += tl.dot(b_q, b_b)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_Aqk = tl.where(m_A, b_Aqk, 0)
    b_Aqb = tl.where(m_A, b_Aqb, 0)

    desc_v = make_tensor_descriptor(v, [T, V], [stride_vo, 1], [BT, BV])
    desc_u = make_tensor_descriptor(u, [T, V], [stride_vo, 1], [BT, BV])
    desc_o = make_tensor_descriptor(o, [T, V], [stride_vo, 1], [BT, BV])
    b_v = desc_v.load([i_t * BT, i_v * BV])
    b_u = desc_u.load([i_t * BT, i_v * BV])
    b_o = (b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_u.dtype), b_u)) * scale
    desc_o.store([i_t * BT, i_v * BV], b_o.to(desc_o.dtype))


def chunk_generalized_iplr_delta_rule_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)

    def grid(meta): return (
        triton.cdiv(V, meta['BV']),
        NT,
        B * H,
    )
    chunk_generalized_iplr_delta_rule_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        u=v_new,
        b=b,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def chunk_generalized_iplr_delta_rule_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
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

    if check_shared_mem('hopper', k.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', k.device.index):  # A100
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)

    chunk_generalized_iplr_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=v,
        d=w,
        b=b,
        u=u,
        v_new=v_new,
        h=h,
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


def chunk_generalized_iplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    w, u, _ = prepare_wy_repr_fwd(
        a=a,
        b=b,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    h, v_new, final_state = chunk_generalized_iplr_delta_rule_fwd_h(
        k=k,
        v=v,
        b=b,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    o = chunk_generalized_iplr_delta_rule_fwd_o(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        b=b,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return o, final_state


class ChunkGeneralizedIPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_size: int | None = None,
    ):
        if chunk_size is None:
            chunk_size = 64
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        o, final_state = chunk_generalized_iplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        raise NotImplementedError(
            "Backward pass for ChunkGeneralizedIPLRDeltaRuleFunction is not implemented yet. "
            "Stay tuned!",
        )


@torch.compiler.disable
def chunk_iplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
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
        a (torch.Tensor):
            activations of shape `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, T, H, K]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            CPU copy of `cu_seqlens` to avoid unnecessary device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    chunk_size = kwargs.pop('chunk_size', None)
    if chunk_size is not None and chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"`chunk_size` must be a power of 2, got {chunk_size}.")
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
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkGeneralizedIPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size,
    )
    return o, final_state
