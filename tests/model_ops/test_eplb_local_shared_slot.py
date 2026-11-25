# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# EPLB without redundant experts and a staged shared expert: each rank keeps
# its shared slot (LOCAL_REPLICA) and EPLB places the routed experts only.

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fused_moe.expert_layout import MoEExpertLayout, SharedExpertMode

try:
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "requires full atom import env")


def _layout(redundant=0, eplb_enabled=True, stays=True):
    return MoEExpertLayout.make(
        num_routed=384,
        num_fused_shared_experts=1,
        num_configured_redundant=redundant,
        ep_size=8,
        use_all2all=True,
        eplb_enabled=eplb_enabled,
        shared_stays_local=stays,
    )


def test_staged_shared_keeps_its_slot_under_eplb_without_redundancy():
    layout = _layout()
    assert layout.mode is SharedExpertMode.LOCAL_REPLICA
    assert not layout.shared_is_routed
    assert layout.num_routed_physical == 384
    assert layout.num_physical == 392
    assert layout.physical_per_rank == 49
    assert layout.routed_physical_per_rank == 48


def test_redundant_experts_still_route_the_shared_expert():
    # Redundant slots are where EPLB replicates the shared expert; a staged
    # shared expert still refuses that layout at load.
    assert _layout(redundant=8).mode is SharedExpertMode.EPLB_ROUTED


def test_other_layers_keep_the_routed_shared_expert_under_eplb():
    assert _layout(stays=False).mode is SharedExpertMode.EPLB_ROUTED
    assert _layout(eplb_enabled=False).mode is SharedExpertMode.LOCAL_REPLICA


def _layer(layout, ep_rank=0):
    ep = 8
    per_rank = layout.routed_physical_per_rank
    routed = torch.full((layout.num_routed_physical,), -1, dtype=torch.int32)
    routed[ep_rank * per_rank : (ep_rank + 1) * per_rank] = torch.arange(
        per_rank, dtype=torch.int32
    )
    expert_map = torch.cat(
        (
            routed,
            torch.tensor([per_rank], dtype=torch.int32),  # the shared slot
            torch.tensor([-1], dtype=torch.int32),  # topK's sentinel
        )
    )
    mask = torch.zeros(layout.num_physical, dtype=torch.int32)
    mask[ep_rank * 49 : ep_rank * 49 + 49] = 1
    return SimpleNamespace(
        expert_layout=layout,
        global_num_experts=layout.num_physical,
        num_redundant_experts=layout.num_redundant,
        ep_size=ep,
        ep_rank=ep_rank,
        local_num_experts=per_rank + 1,
        expert_map=expert_map,
        expert_mask=mask,
    )


def test_eplb_places_the_routed_experts_only():
    assert eplb.eplb_num_physical(_layer(_layout())) == 384
    routed = _layer(_layout(stays=False))
    assert eplb.eplb_num_physical(routed) == routed.global_num_experts


def test_weight_views_drop_the_shared_slot():
    layer = _layer(_layout())
    w = torch.arange(49 * 6, dtype=torch.float32).view(49, 6)

    class _Method:
        def get_eplb_weight_views(self, layer):
            return [w]

    layer.quant_method = _Method()
    manager = object.__new__(eplb.EPLBManager)
    manager.live_metadata = SimpleNamespace(num_local_physical_experts=48)
    (view,) = manager._collect_expert_weight_tensors(layer)
    assert view.shape == (48, 6)
    assert view.data_ptr() == w.data_ptr()  # a view: migration writes in place


def test_bind_keeps_the_shared_slot_and_the_dispatch_mask():
    layer = _layer(_layout())
    mask = layer.expert_mask.clone()
    new_map = torch.full((384,), -1, dtype=torch.int32)
    new_map[100:148] = torch.arange(48, dtype=torch.int32)  # moved routed experts
    manager = object.__new__(eplb.EPLBManager)
    maps = torch.full((4, 384), -1, dtype=torch.int32)
    maps[3] = new_map
    manager.live_metadata = SimpleNamespace(num_physical_experts=384, expert_map=maps)
    manager._moe_layers = {3: layer}
    manager._expert_map_tails = {}
    manager._bind_layer_expert_maps()
    assert layer.expert_map.numel() == 386
    assert torch.equal(layer.expert_map[:384], new_map)
    assert layer.expert_map[384:].tolist() == [48, -1]
    assert torch.equal(layer.expert_mask, mask)


def test_staging_refuses_a_routed_shared_expert():
    from atom.model_ops.fused_moe import shared_expert_staging

    layer = SimpleNamespace(
        prefix="model.layers.3.mlp.experts",
        expert_layout=_layout(redundant=8),
    )
    with pytest.raises(NotImplementedError, match="EPLB redundant experts"):
        shared_expert_staging.SharedExpertStaging(
            layer, None, 4096, 512, shared_expert_spec=None
        )
