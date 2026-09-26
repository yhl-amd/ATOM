# SPDX-License-Identifier: MIT
"""EP transport over epx: two one-way hops per dispatch, one per combine.

epx (see epx_kernels.hip) exchanges per-destination counts first, so every source
writes its rows at a precomputed offset: the receive buffer is a dense prefix of
`recv_count` rows in (source rank, token) order, ids < 0 are dropped for free, and
combine pushes the expert output (plain device memory) back to the source for a
local reduction. The wire format is MXFP8, the same activation quantization AITER
applies inside fused_moe on this path.
"""

from typing import Any

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.mori_prepare_finalize import (
    MoriPrepareAndFinalize,
    _mxfp8_quant,
)
from atom.utils import envs
from atom.utils.forward_context import enable_pad_rows_device


class EpxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    def __init__(self, epx_op: Any, max_tokens_per_rank: int, num_dispatchers: int):
        super().__init__()
        self._op = epx_op
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        # Same pad-row masking as the MoRI path: epx drops ids < 0 by construction.
        self._mask_pad_rows = envs.ATOM_MORI_MASK_PAD_ROWS
        if self._mask_pad_rows:
            enable_pad_rows_device(
                max_tokens_per_rank, torch.device("cuda", torch.cuda.current_device())
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    # The receive prefix, the AITER sentinel column and the stale-tail mask are the
    # same contract as MoRI's arena, so reuse MoRI's implementations.
    mask_pad_topk_ids = MoriPrepareAndFinalize.mask_pad_topk_ids
    adapt_routing_for_fused_moe = MoriPrepareAndFinalize.adapt_routing_for_fused_moe

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config=None,
        quant_type=None,
    ) -> mk.PrepareResultType:
        assert not apply_router_weight_on_input
        a1q, scale = _mxfp8_quant(a1)
        recv_x, recv_w, recv_s, recv_ids, recv_count = self._op.dispatch(
            a1q, scale, topk_ids.to(torch.int32), topk_weights.to(torch.float32)
        )
        meta = mk.ExpertTokensMetadata(
            expert_num_tokens=recv_count, expert_num_tokens_cpu=None
        )
        return recv_x, recv_s, meta, recv_ids, recv_w

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        return self._op.combine(
            fused_expert_output,
            topk_ids.shape[0],
            max_recv_rows=fused_expert_output.shape[0],
        )


def make_epx_prepare_finalize(moe, ep_group) -> EpxPrepareAndFinalize:
    """One epx op per process (every MoE layer shares it), like MoRI's cached handle."""
    global _EPX_OP
    if _EPX_OP is None:
        from epx import EpxOp

        _EPX_OP = EpxOp(
            rank=ep_group.rank_in_group,
            world_size=ep_group.world_size,
            hidden=moe.hidden_dim,
            max_tokens_per_rank=moe.max_num_tokens,
            experts_per_rank=moe.num_local_experts,
            topk=moe.experts_per_token,
            x_bytes_per_elem=1,
            scale_bytes=moe.hidden_dim // 32,
            out_dtype=moe.in_dtype,
            group=ep_group.cpu_group,
        )
    return EpxPrepareAndFinalize(
        _EPX_OP,
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=ep_group.world_size,
    )


_EPX_OP = None
