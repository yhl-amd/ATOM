# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded replay for DeepSeek-V4: the scheduler side.

A hit on the compressed history that reaches further than the nearest state
checkpoint by more than the window's receptive field keeps the whole hit and
recomputes only that field. These tests pin the admission decision, where the
forward starts, which blocks get published, and where checkpoints may land.

Geometry: block 4, window 3, three trunk layers, so the replay is
3 + 2 * 2 = 7 tokens.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer

BLOCK = 4
REPLAY = 3 + 2 * 2


def _hf(**overrides):
    fields = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "sliding_window": 3,
        "num_hidden_layers": 3,
        "compress_ratios": [4, 128, 4, 0],
        "num_nextn_predict_layers": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _bm(monkeypatch, replay=True, **overrides):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1" if replay else "0")
    fields = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 64,
        "enable_prefix_caching": True,
        "pool_entries": {"state": 8},
        "pool_entries_per_req": {"state": 1},
        # No grid and no demand rung: nothing places a checkpoint unless a test
        # asks for one, so every hit is "compressed history, no state".
        "state_checkpoint_interval_tokens": -1,
        "state_checkpoint_demand": False,
        "prefill_context_parallel_size": 1,
        "hf_config": _hf(),
    }
    fields.update(overrides)
    spec = PagedStateCheckpointSpec(10, 25, "replay-test", image_bytes=25)
    return BlockManager(
        MockConfig(**fields),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )


def _seq(n, base=100):
    return Sequence(list(range(base, base + n)), BLOCK, has_per_req_cache=True)


def _prefill_and_finish(bm, seq):
    """Admit `seq`, publish its whole prompt as one forward, and free it: its
    blocks stay hash-indexed (the compressed history) with no checkpoint."""
    hit = bm.can_allocate(seq)
    assert hit >= 0
    assert bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_tokens - seq.num_cached_tokens)
    bm.deallocate(seq)


def test_replay_length_is_the_window_receptive_field(monkeypatch):
    assert _bm(monkeypatch).state_replay_tokens == REPLAY


def test_mtp_counts_as_one_more_layer(monkeypatch):
    bm = _bm(
        monkeypatch,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
    )
    assert bm.state_replay_tokens == 3 + 3 * 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"hf_config": _hf(architectures=["DeepseekV41ForCausalLM"])},
        {"hf_config": _hf(compress_ratios=[1, 2])},
        {"speculative_config": SimpleNamespace(method="dspark")},
        {"prefill_context_parallel_size": 2},
        # A load that does not know about the replay would write the claimed
        # blocks it reads; only LMCache MP offload is taught to.
        {"kv_transfer_config": {"kv_connector": "nixl", "kv_role": "kv_both"}},
    ],
)
def test_off_where_the_receptive_field_argument_does_not_hold(monkeypatch, overrides):
    assert _bm(monkeypatch, **overrides).state_replay_tokens == 0


def test_off_by_default(monkeypatch):
    assert _bm(monkeypatch, replay=False).state_replay_tokens == 0


# --- LMCache MP: the KV-only reach -----------------------------------------
_MP = {"kv_transfer_config": {"kv_connector": "lmcache_mp", "kv_role": "offload"}}


def _mp_bm(monkeypatch, min_load=0):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", str(min_load))
    return _bm(monkeypatch, **_MP)


def _mp_seq(bm, reach_with_state, reach_kv_only, n=60):
    """A 60-token prompt whose first 32 are in HBM with no checkpoint, and
    whose LMCache lookups answered `reach_with_state` / `reach_kv_only`."""
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(n)
    seq.offload_joint.kv_prefix_tokens = reach_with_state
    seq.offload_joint.kv_only_tokens = reach_kv_only
    return seq


def test_lmcache_mp_offload_keeps_replay_on(monkeypatch):
    assert _mp_bm(monkeypatch).state_replay_tokens == REPLAY


def test_a_kv_only_reach_past_hbm_loads_then_replays(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    hit = bm.can_allocate(seq)
    assert hit == 8  # the whole HBM prefix is claimed
    assert seq.replay_kv_load
    assert (seq.replay_load_start, seq.replay_start, seq.replay_end) == (
        32,
        48 - REPLAY,
        48,
    )
    assert bm.allocate(seq, hit)
    # The load runs first, from the HBM hit; nothing is published past it yet.
    assert seq.num_cached_tokens == 32
    assert seq.replay_kv_load
    bm.begin_loaded_replay(seq)
    assert not seq.replay_kv_load
    assert seq.num_cached_tokens == 48 - REPLAY
    assert seq.num_hashed_tokens == 48


def test_a_failed_kv_load_replays_below_the_hbm_hit(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.fall_back_to_hbm_replay(seq)
    assert not seq.replay_kv_load
    assert (seq.replay_start, seq.replay_end) == (32 - REPLAY, 32)
    assert seq.num_cached_tokens == 32 - REPLAY


def test_a_state_carrying_hit_close_to_the_kv_reach_is_loaded_as_is(monkeypatch):
    """44 with its image beats replaying to 48: 48 - 7 < 44."""
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)
    assert bm.can_allocate(seq) == 0
    assert seq.replay_end == 0 and not seq.replay_kv_load


def test_a_kv_reach_too_small_to_load_replays_from_hbm(monkeypatch):
    """The connector skips loads under OFFLOAD_MIN_LOAD_TOKENS, so the
    decision must not count on one."""
    bm = _mp_bm(monkeypatch, min_load=100)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.can_allocate(seq) == 8
    assert not seq.replay_kv_load
    assert (seq.replay_start, seq.replay_end) == (32 - REPLAY, 32)


def test_clear_replay_forgets_the_load(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.deallocate(seq)
    assert not seq.replay_kv_load and seq.replay_load_start == 0


def test_a_far_hit_without_a_checkpoint_replays(monkeypatch):
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    hit = bm.can_allocate(seq)
    # 8 published blocks match; no checkpoint anywhere, so the gate alone
    # would admit 0. 32 > 7, so the hit is kept and the last 7 replayed.
    assert hit == 8
    assert (seq.replay_start, seq.replay_end) == (32 - REPLAY, 32)
    assert seq.num_compressed_hit_blocks == 8
    assert bm.prefill_start_tokens(seq, hit) == 32 - REPLAY
    assert bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 32 - REPLAY
    assert seq.num_hashed_tokens == 32
    # A fresh slot, no restore queued: the state comes from the replay.
    assert seq.state_slot >= 0
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)


def test_the_replay_publishes_only_from_the_hit_on(monkeypatch):
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    hit = bm.can_allocate(seq)
    assert bm.allocate(seq, hit)
    shared = list(seq.block_table[:8])
    before = [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared]
    bm.hash_blocks(seq, seq.num_tokens - seq.num_cached_tokens)
    assert [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared] == before
    # The two new blocks are published and chain onto the hit.
    assert all(bm.kv.block(b).hash != -1 for b in seq.block_table[8:10])
    assert seq.num_hashed_tokens == 40


def test_a_near_hit_is_left_to_the_checkpoint_gate(monkeypatch):
    """Two blocks (8 tokens) beat a 7-token replay; one block does not."""
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(4))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 0
    assert seq.replay_end == 0
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(8))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 2
    assert (seq.replay_start, seq.replay_end) == (1, 8)


def test_off_the_gate_decides_as_before(monkeypatch):
    bm = _bm(monkeypatch, replay=False)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.can_allocate(seq) == 0
    assert seq.replay_end == 0


def test_no_checkpoint_inside_the_replay(monkeypatch):
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    seq.checkpoint_end_pos = 28  # pretend an anchor fell inside the replay
    assert bm.checkpointers_at(seq, 28) == []
    assert bm.checkpoint_cut(seq, seq.num_cached_tokens, 40, record=False) != 28


def test_deallocate_and_disown_clear_the_replay(monkeypatch):
    bm = _bm(monkeypatch)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    assert bm.disown_claimed_prefix(seq)
    assert (seq.replay_start, seq.replay_end) == (0, 0)
    bm.deallocate(seq)
    seq2 = _seq(40)
    assert bm.allocate(seq2, bm.can_allocate(seq2))
    bm.deallocate(seq2)
    assert (seq2.replay_start, seq2.replay_end) == (0, 0)
