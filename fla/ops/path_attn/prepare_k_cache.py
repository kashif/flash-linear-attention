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


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel_prepare_k_cache(
    k, k_new, w1, w2,
    offsets, indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    # constants
    desc_k = make_tensor_descriptor(k, [T, K], [H*K, 1], [BT, BK])
    b_k = tl.zeros([BT, BK], dtype=tl.float32)
    b_k += desc_k.load([i_t * BT, 0])
    for k_block_idx in range(i_t + 1, tl.cdiv(T, BT)):
        desc_w1 = make_tensor_descriptor(w1, [T, K], [H*K, 1], [BT, BK])
        desc_w2 = make_tensor_descriptor(w2, [T, K], [H*K, 1], [BT, BK])
        b_w1 = desc_w1.load([k_block_idx * BT, 0])
        b_w2 = desc_w2.load([k_block_idx * BT, 0])
        b_A = tl.dot(b_k.to(b_w2.dtype), tl.trans(b_w2))
        b_k = b_k - tl.dot(b_A.to(b_w1.dtype), b_w1)

    desc_k_new = make_tensor_descriptor(k_new, [T, K], [H*K, 1], [BT, BK])
    desc_k_new.store([i_t * BT, 0], b_k.to(desc_k_new.dtype))


def prepare_k_cache_fn(k, w1, w2, cu_seqlens, BS, use_cache=False, chunk_indices: torch.LongTensor | None = None):
    if not use_cache:
        return None
    else:
        B, T, H, K = k.shape
        k_new = torch.empty_like(k)
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        indices = chunk_indices
        NT = triton.cdiv(T, BS) if cu_seqlens is None else len(indices)
        grid = (NT, B * H)
        parallel_path_fwd_kernel_prepare_k_cache[grid](
            k=k,
            k_new=k_new,
            w1=w1,
            w2=w2,
            offsets=cu_seqlens,
            indices=indices,
            H=H,
            T=T,
            K=K,
            BT=BS,
            BK=triton.next_power_of_2(K),
        )
        return k_new
