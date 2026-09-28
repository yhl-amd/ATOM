# SPDX-License-Identifier: MIT
"""ATOM_MEGA_HYBRID: every rank must pick the same pipeline for a forward.

Host-side routing only: MegaMoEV2 and the epx modular kernel are replaced by
fakes (see test_mega_decode_capacity for the Mega fixture). Nothing here runs
a GPU kernel or checks numerics.
"""

import pytest
import torch
from test_mega_decode_capacity import LARGE_CAPACITY, _built, _context, runtime

from atom.model_ops.fused_moe import flydsl_mega_experts as mega

__all__ = ["runtime"]

MAX_EPX = 1024


@pytest.mark.parametrize(
    ("tokens", "route"),
    [(1, "mega"), (112, "mega"), (128, "mega"), (129, "epx"), (224, "epx")]
    + [(MAX_EPX, "epx"), (MAX_EPX + 1, "mega"), (16384, "mega")],
)
def test_unified_forwards_route_by_row_count(tokens, route):
    assert (
        mega.select_hybrid_route(
            _context(tokens), max_epx_tokens=MAX_EPX, tbo_active=False
        )
        == route
    )


@pytest.mark.parametrize("draft", [False, True])
def test_only_group_agreed_shapes_leave_mega(draft):
    # A rank-local count must never pick the transport: peers could disagree.
    split = _context(224, running_tokens_are_unified=False, is_draft=draft)
    assert (
        mega.select_hybrid_route(split, max_epx_tokens=MAX_EPX, tbo_active=False)
        == "mega"
    )
    assert (
        mega.select_hybrid_route(_context(224), max_epx_tokens=MAX_EPX, tbo_active=True)
        == "mega"
    )
    assert mega.select_hybrid_route(None, max_epx_tokens=MAX_EPX, tbo_active=False) == (
        "mega"
    )


class _Recorder:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __call__(self, **kwargs):
        self.calls.append(self.name)
        return torch.full_like(kwargs["hidden_states"], len(self.calls))


class _MegaFused:
    """MegaFusedExperts' attributes, forwarding straight to run_mega_moe."""

    def __init__(self, layer):
        self._layer, self._model_dim, self._inter_dim = layer, 8, 8
        self._mtpr, self._quant = LARGE_CAPACITY, "a8w4"

    def __call__(self, *, hidden_states, topk_weights, topk_ids, **_):
        return mega.run_mega_moe(
            self._layer,
            hidden_states,
            topk_weights,
            topk_ids,
            model_dim=8,
            inter_dim=8,
            experts=384,
            topk=6,
            mtpr=self._mtpr,
            swiglu_limit=0.0,
        )


def _hybrid(runtime, calls, *, max_epx=MAX_EPX):
    layer = runtime.layer
    hybrid = mega.MegaHybridFusedExperts(
        layer, _MegaFused(layer), _Recorder("epx", calls), max_epx_tokens=max_epx
    )

    def call(context, rows=None):
        runtime.forward_context.context = context
        rows = context.running_tokens if rows is None else rows
        return hybrid(
            hidden_states=torch.zeros(rows, 8, dtype=torch.bfloat16),
            topk_weights=torch.ones(rows, 6, dtype=torch.float32),
            topk_ids=torch.zeros(rows, 6, dtype=torch.int64),
            global_num_experts=384,
        )

    return call


def test_epx_forward_first_still_allocates_mega_before_capture(runtime):
    calls = []
    call = _hybrid(runtime, calls)

    call(_context(224))
    assert calls == ["epx"]
    # Both Mega capacities exist before any Mega forward, in the fixed order.
    assert _built(runtime) == [LARGE_CAPACITY, 128]

    runtime.capturing = True
    call(_context(112))
    call(_context(224, is_draft=True))
    assert calls == ["epx", "epx"]
    assert [e for e in runtime.events if e[0] == "forward"] == [("forward", 0, 128)]


def test_capture_never_allocates_from_the_epx_leg(runtime):
    calls = []
    call = _hybrid(runtime, calls)

    runtime.capturing = True
    call(_context(224))
    assert calls == ["epx"]
    assert _built(runtime) == []


def test_large_and_split_forwards_use_mega_compact(runtime):
    calls = []
    call = _hybrid(runtime, calls)

    call(_context(4096))
    call(_context(512, running_tokens_are_unified=False), rows=300)
    assert calls == []
    assert [e[2] for e in runtime.events if e[0] == "forward"] == [
        LARGE_CAPACITY,
        LARGE_CAPACITY,
    ]


def test_epx_leg_refuses_rows_past_its_arena(runtime):
    calls = []
    call = _hybrid(runtime, calls, max_epx=256)

    with pytest.raises(ValueError, match="epx capacity"):
        call(_context(224), rows=300)


def test_shared_weight_layout_matches_mega_shuffle():
    """The aliasing in MegaHybridMxfp4MoEMethod relies on this identity."""
    try:
        from aiter.ops import shuffle
    except ImportError as exc:
        from import_guard import skip_if_dependency_missing

        skip_if_dependency_missing(exc, "requires aiter")

    g = torch.Generator().manual_seed(0)
    experts, inter, hidden = 2, 256, 512
    w13 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=g)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=g)
    s13 = torch.randint(0, 256, (experts * 2 * inter, hidden // 32), generator=g)
    s2 = torch.randint(0, 256, (experts * hidden, inter // 32), generator=g)
    w13, w2, s13, s2 = (t.to(torch.uint8) for t in (w13, w2, s13, s2))

    # Mxfp4MoEMethod branch C on gfx950 with ATOM_MOE_GU_ITLV=1 ...
    std = (
        shuffle.shuffle_weight(
            w13, layout=(16, 16), is_guinterleave=True, gate_up=True
        ),
        shuffle.shuffle_weight(
            w2, layout=(16, 16), is_guinterleave=True, gate_up=False
        ),
        shuffle.shuffle_scale(s13, experts, is_guinterleave=True, gate_up=True),
        shuffle.shuffle_scale(s2, experts, is_guinterleave=True, gate_up=False),
    )
    # ... and build_mega_weights.
    ref = (
        shuffle.shuffle_weight_a16w4(w13, 16, True),
        shuffle.shuffle_weight_a16w4(w2, 16, False),
        shuffle.shuffle_scale_a16w4(s13, experts, True),
        shuffle.shuffle_scale_a16w4(s2, experts, False),
    )
    for a, b in zip(std, ref):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
