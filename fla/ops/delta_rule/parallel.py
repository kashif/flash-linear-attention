# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.delta_rule.wy_fast import fwd_prepare_T
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'K', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    A,
    q_new,
    k_new,
    A_local,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    b_q = (desc_q.load([i_t * BT, 0]) * scale).to(desc_q.dtype)
    b_k = desc_k.load([i_t * BT, 0])
    b_v = desc_v.load([i_t * BT, 0])

    desc_T = make_tensor_descriptor(A + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    b_T = desc_T.load([i_t * BT, 0])

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(b_k.dtype)

    desc_beta = make_tensor_descriptor(beta + i_bh * T, [T], [1], [BT])
    b_beta = desc_beta.load([i_t * BT])
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)

    if OUTPUT_ATTENTIONS:
        desc_a = make_tensor_descriptor(A_local + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
        desc_a.store([i_t * BT, 0], b_qkT.to(desc_a.dtype))

    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    desc_o = make_tensor_descriptor(o + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    desc_o.store([i_t * BT, 0], tl.dot(b_qkT, b_v).to(desc_o.dtype))

    desc_q_new = make_tensor_descriptor(q_new + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    desc_q_new.store([i_t * BT, 0], (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(desc_q_new.dtype))

    desc_k_new = make_tensor_descriptor(k_new + i_bh * T*K, [T, K], [K, 1], [BT, BK])
    b_k_new = b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
    desc_k_new.store([i_t * BT, 0], b_k_new.to(desc_k_new.dtype))


def chunk_transform_qk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    chunk_size: int,
    output_attentions: bool,
):
    B, H, T, K = k.shape
    BT = chunk_size
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    grid = (triton.cdiv(T, BT), B*H)
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    chunk_transform_qk_fwd_kernel[grid](
        q,
        k,
        v,
        beta,
        o,
        A,
        q_new,
        k_new,
        A_local,
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        OUTPUT_ATTENTIONS=output_attentions,
    )
    return q_new, k_new, o, A_local


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def save_intra_chunk_attn(
    A,
    A_local,
    T,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    desc_A = make_tensor_descriptor(A + i_bh * T * T, [T, T], [T, 1], [BT, BT])
    desc_A_local = make_tensor_descriptor(A_local + i_bh * T * BT, [T, BT], [BT, 1], [BT, BT])
    b_A_local = desc_A_local.load([i_t * BT, 0])
    desc_A.store([i_t * BT, i_t * BT], b_A_local.to(desc_A.dtype))


@triton.heuristics({
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_delta_rule_fwd_kernel(
    q,
    k,
    k2,  # original k
    v,
    beta,
    o,
    o_new,
    attn,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    desc_q = make_tensor_descriptor(q + i_bh * T*K, [T, K], [K, 1], [BT, BK])

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += desc_q.load([i_t * BT, 0])

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    desc_o = make_tensor_descriptor(o + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    b_o += desc_o.load([i_t * BT, 0])

    # As opposed to Flashattention, this kernel requires scanning the KV blocks from right to left
    # Q block and K block have overlap.
    # masks required
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BS, BK])
        desc_k2 = make_tensor_descriptor(k2 + i_bh * T*K, [T, K], [K, 1], [BS, BK])
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BS, BV])
        desc_beta = make_tensor_descriptor(beta + i_bh * T, [T], [1], [BS])
        # [BK, BS]
        b_k = tl.trans(desc_k.load([offset, 0]))
        # [BS, BV]
        b_v = desc_v.load([offset, 0])
        # [BS]
        b_beta = desc_beta.load([offset])
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_s = tl.where(m_s[:, None], b_s, 0)

        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (desc_k2.load([offset, 0]) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            desc_a = make_tensor_descriptor(attn + i_bh * T * T, [T, T], [T, 1], [BT, BS])
            desc_a.store([i_t * BT, offset], b_s.to(desc_a.dtype))

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for offset in range(i_t * BT - BS, -BS, -BS):
        desc_k = make_tensor_descriptor(k + i_bh * T*K, [T, K], [K, 1], [BS, BK])
        desc_v = make_tensor_descriptor(v + i_bh * T*V, [T, V], [V, 1], [BS, BV])
        desc_beta = make_tensor_descriptor(beta + i_bh * T, [T], [1], [BS])
        desc_k2 = make_tensor_descriptor(k2 + i_bh * T*K, [T, K], [K, 1], [BS, BK])

        # [BK, BS]
        b_k = tl.trans(desc_k.load([offset, 0]))
        # [BS, BV]
        b_v = desc_v.load([offset, 0])
        # [BS]
        b_beta = desc_beta.load([offset])
        # [BT, BS]
        b_s = (tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False))
        # [BT, BV]
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (desc_k2.load([offset, 0]) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False).to(b_q.dtype)

        if OUTPUT_ATTENTIONS:
            desc_a = make_tensor_descriptor(attn + i_bh * T * T, [T, T], [T, 1], [BT, BS])
            desc_a.store([i_t * BT, offset], b_s.to(desc_a.dtype))

    desc_o_new = make_tensor_descriptor(o_new + i_bh * T*V, [T, V], [V, 1], [BT, BV])
    desc_o_new.store([i_t*BT, 0], b_o.to(desc_o.dtype))


class ParallelDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, scale, output_attentions):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 32
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0

        A = fwd_prepare_T(k, beta, BS)
        attn = q.new_zeros(B, H, T, T) if output_attentions else None
        q_new, k_new, o, A_local = chunk_transform_qk_fwd(
            q,
            k,
            v,
            beta,
            A,
            scale,
            BS,
            output_attentions,
        )

        num_stages = 3 if K <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(T, BT), B * H)
        o_new = torch.empty_like(o)

        parallel_delta_rule_fwd_kernel[grid](
            q=q_new,
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            o=o,
            o_new=o_new,
            attn=attn,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        if output_attentions:
            grid = (triton.cdiv(T, BS), B * H)
            save_intra_chunk_attn[grid](
                A=attn,
                A_local=A_local,
                T=T,
                BT=BS,
            )
        return o_new.to(q.dtype), attn

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, d_attn=None):
        raise NotImplementedError('Backward pass is not implemented. Stay tuned!')


def parallel_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape `[B, H, T, T]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    o, attn = ParallelDeltaRuleFunction.apply(q, k, v, beta, scale, output_attentions)
    return o, attn


def naive_delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    b, h, l, d_k = q.shape
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A
