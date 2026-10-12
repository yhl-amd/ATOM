# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP native-state loads that bring the PAGE KV alone, for a replay.

A lookup reports how far the KV reaches *with* a state image at its end (the
hit); a bounded replay (`Sequence.replay_kv_load`) can use the KV alone. With
replay on, the same lookup also reports how far the PAGE KV alone reaches (the
reach) and keeps it locked. These tests drive that one lookup through its whole
life: answered, weighed, turned into a load with no image or released, closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload.mp import deployment
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp.lookup import (
    OBJECT_GROUPS_CONFIG,
    REACH_OBJECT_GROUPS_CONFIG,
    _MPLookupClient,
)
from atom.kv_transfer.offload.mp.native_state_scheduler import (
    NativeStateLMCacheMPConnectorScheduler,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_STORE_CHANNEL,
    PAGE_OBJECT_GROUP,
    STATE_OBJECT_GROUP,
)
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.sequence import Sequence

CHUNK = 8
REPLAY = 6
SESSION = "atom-offload-dp0:1"


@dataclass(frozen=True)
class _Key:
    token_ids: tuple
    start: int
    end: int
    request_id: str
    worker_id: int | None
    request_configs: dict | None = field(default=None, compare=False)


class _Future:
    def __init__(self, value=None):
        self.value = value

    def query(self):
        return True

    def result(self, timeout=None):
        return self.value


class _Client:
    """Answers `[chunks, kv_chunks]` to a lookup that asks for the PAGE reach.

    `legacy=True` is a server that does not know the option: the hit alone.
    """

    def __init__(self, chunks, kv_chunks, legacy=False):
        self.chunks = chunks
        self.kv_chunks = kv_chunks
        self.legacy = legacy
        self.keys = []

    def lookup(self, key, tp_size):
        self.keys.append(key)
        return _Future()

    def query_prefetch_status(self, request_id):
        [key] = [k for k in self.keys if k.request_id == request_id][-1:]
        configs = key.request_configs or {}
        if configs.get(REACH_OBJECT_GROUPS_CONFIG) and not self.legacy:
            return _Future([self.chunks, self.kv_chunks])
        return _Future(self.chunks)


class _Adapter:
    lmcache_tokens_per_chunk = CHUNK

    def __init__(self, client):
        self._client = client
        self._parallel = SimpleNamespace(tp_size=1)
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}
        self.freed = []
        self.ended = []

    def _create_key(self, token_ids, start, end, request_id, worker_id):
        return _Key(tuple(token_ids[start:end]), start, end, request_id, worker_id)

    def maybe_submit_lookup_request(self, request_id, token_ids):
        raise AssertionError("no synchronous lookup: it cannot ask for the reach")

    def check_lookup_result(self, request_id):
        return self._lookup_results.get(request_id)

    def free_lookup_locks(self, **kwargs):
        self.freed.append(kwargs)

    def cleanup_lookup_result(self, request_id):
        self._pending_lookups.discard(request_id)
        self._lookup_results.pop(request_id, None)

    def end_session(self, request_id):
        self.ended.append(request_id)

    def shutdown(self):
        pass


def _freed(adapter):
    """`(start, end, groups or None)` per release, in order."""
    return [
        (
            f["start"],
            f["end"],
            (f.get("request_configs") or {}).get(OBJECT_GROUPS_CONFIG),
        )
        for f in adapter.freed
    ]


def _scheduler(
    monkeypatch,
    *,
    chunks,
    kv_chunks,
    replay=REPLAY,
    min_load=0,
    legacy=False,
    page_only=False,
    images=None,
):
    """`images`: ATOM_LMCACHE_MP_STATE_IMAGES as given ("unset" for none);
    by default "0" with `page_only` and "1" without."""
    monkeypatch.setenv("OFFLOAD_MIN_SAVE_TOKENS", "0")
    if images is None:
        images = "0" if page_only else "1"
    if images == "unset":
        monkeypatch.delenv("ATOM_LMCACHE_MP_STATE_IMAGES", raising=False)
    else:
        monkeypatch.setenv("ATOM_LMCACHE_MP_STATE_IMAGES", images)
    monkeypatch.setattr(
        deployment.offcfg,
        "build_lmcache_config",
        lambda kvc: SimpleNamespace(
            chunk_size=kvc["kv_connector_extra_config"]["lmcache.chunk_size"]
        ),
    )
    adapter = _Adapter(_Client(chunks, kv_chunks, legacy=legacy))
    monkeypatch.setattr(
        mp_scheduler, "_make_scheduler_adapter", lambda config, **_: adapter
    )
    config = SimpleNamespace(
        kv_cache_block_size=4,
        kv_transfer_config={
            "kv_role": "offload",
            "kv_connector_extra_config": {"lmcache.chunk_size": CHUNK},
        },
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    scheduler = NativeStateLMCacheMPConnectorScheduler(config)
    checkpoints = PagedStateCheckpointCoordinator(
        BlockPool(30),
        PagedStateCheckpointSpec(10, 50, "kv-only-test", image_bytes=25),
        enabled=True,
    )
    fallbacks = []
    manager = SimpleNamespace(
        paged_state_checkpoints=checkpoints,
        enable_prefix_caching=True,
        hash_block_size=4,
        compute_hash=BlockManager.compute_hash,
        state_replay_tokens=replay,
        fall_back_to_hbm_replay=fallbacks.append,
    )
    manager._hash_block_tokens = lambda seq, index: BlockManager._hash_block_tokens(
        manager, seq, index
    )
    manager.prefix_hash_chain = lambda seq, hashes, blocks: BlockManager._chain_to(
        manager, seq, hashes, blocks
    )
    scheduler.bind_block_manager(manager)
    scheduler._min_load_tokens = min_load
    return scheduler, checkpoints, adapter, fallbacks


def _seq(count=40):
    seq = Sequence(list(range(count)), 4, id=1, has_per_req_cache=True)
    seq.state_slots = [5]
    seq.block_table = list(range((count + 3) // 4))
    return seq


def _look_up(scheduler, seq):
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    assert not scheduler.lookup_pending(seq)
    return scheduler.get_num_new_matched_tokens(seq)


def _admit_kv_only(scheduler, seq, *, hbm):
    """What `BlockManager._replay_hit` and `allocate` leave for a KV-only load."""
    _look_up(scheduler, seq)
    reach = seq.offload_joint.kv_only_tokens
    seq.num_cached_tokens = hbm
    seq.replay_kv_load = True
    seq.replay_load_start = hbm
    seq.replay_end = reach
    seq.replay_start = reach - REPLAY
    return reach


def test_one_lookup_asks_for_the_page_reach():
    adapter = _Adapter(_Client(chunks=1, kv_chunks=3))
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector_extra_config": {}},
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    client = _MPLookupClient(adapter, config=config, timeout=1.0, poll_interval=0.01)
    client.ask_reach((PAGE_OBJECT_GROUP,))
    assert client.submit(list(range(32)), "r")
    [key] = adapter._client.keys
    assert key.request_configs == {REACH_OBJECT_GROUPS_CONFIG: [PAGE_OBJECT_GROUP]}
    assert client.poll("r")
    assert client.lookup(list(range(32)), "r") == 1 * CHUNK
    assert client.hit_tokens("r") == 1 * CHUNK
    assert client.reach_tokens("r") == 3 * CHUNK
    # Its locks run to the reach, so that is what goes back.
    client.clear_lookup_status("r")
    assert [(f["start"], f["end"]) for f in adapter.freed] == [(0, 3 * CHUNK)]


def test_one_lookup_records_the_hit_and_the_reach(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    assert _look_up(scheduler, seq) == (1 * CHUNK, True)
    assert [k.request_id for k in adapter._client.keys] == [SESSION]
    assert seq.offload_joint.kv_only_tokens == 4 * CHUNK


def test_no_reach_asked_with_replay_off(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4, replay=0)
    seq = _seq()
    _look_up(scheduler, seq)
    [key] = adapter._client.keys
    assert not (key.request_configs or {}).get(REACH_OBJECT_GROUPS_CONFIG)
    assert seq.offload_joint.kv_only_tokens == 0


def test_a_server_without_the_option_reports_no_reach_past_the_hit(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4, legacy=True)
    seq = _seq()
    _look_up(scheduler, seq)
    assert seq.offload_joint.kv_only_tokens == 1 * CHUNK


def test_a_kv_only_load_reads_page_alone_and_restores_nothing(monkeypatch):
    scheduler, checkpoints, adapter, fallbacks = _scheduler(
        monkeypatch, chunks=1, kv_chunks=4
    )
    seq = _seq()
    reach = _admit_kv_only(scheduler, seq, hbm=8)
    assert reach == 32
    free_before = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    # The lookup's locks back the load: nothing went back at admission.
    assert adapter.freed == []
    assert scheduler.should_park_for_load_after_alloc(seq)
    assert fallbacks == []
    # No image units are reserved for it.
    assert checkpoints.store.pool.num_free == free_before
    [request] = scheduler.build_connector_meta().requests
    assert request.native_state.kv_only
    assert request.native_state.unit_ids == ()
    assert request.native_state.boundary_tokens == reach
    assert request.load_spec.hbm_cached_tokens == 8
    assert request.load_spec.lmcache_cached_tokens == reach
    # HBM covers [0, 8); the retrieve reads PAGE [8, 32) and nothing of the
    # image, whose lock in that range goes back here.
    assert _freed(adapter) == [(0, 8, None), (8, 32, [STATE_OBJECT_GROUP])]
    operation = seq._load_operation
    assert isinstance(operation, LoadOperationId)
    assert scheduler.load_finished(operation) is True
    # Nothing was adopted as a checkpoint: there was no image.
    assert checkpoints.store.pool.num_free == free_before
    scheduler.request_finished(seq)
    assert len(adapter.freed) == 2


def test_an_ordinary_load_hands_back_the_reach_past_its_hit(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=2, kv_chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    seq.num_cached_tokens = 0  # no replay: the image-carrying hit loads
    scheduler.update_state_after_alloc(seq)
    assert adapter.freed == []
    assert scheduler.should_park_for_load_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert not request.native_state.kv_only
    # PAGE stayed locked to the reach, 32; this load reads to its hit, 16.
    assert _freed(adapter) == [(16, 32, None)]


def test_a_refused_kv_only_load_falls_back_and_returns_its_locks(monkeypatch):
    scheduler, _, adapter, fallbacks = _scheduler(
        monkeypatch, chunks=1, kv_chunks=4, min_load=10_000
    )
    seq = _seq()
    _admit_kv_only(scheduler, seq, hbm=8)
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert fallbacks == [seq]
    assert _freed(adapter) == [(0, 32, None)]


def test_an_unused_reach_returns_its_locks_at_admission(monkeypatch):
    # No image anywhere (hit 0), so no load is armed; the reach is not used.
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=4)
    seq = _seq()
    assert _look_up(scheduler, seq) == (0, False)
    assert adapter.freed == []
    seq.num_cached_tokens = 0  # an ordinary admission: no replay chosen
    scheduler.update_state_after_alloc(seq)
    assert _freed(adapter) == [(0, 32, None)]


def test_an_ordinary_hit_inside_hbm_keeps_the_reach_until_admission(monkeypatch):
    # HBM already holds the image-carrying hit, so no load is armed and the
    # base would release the lookup at once; the PAGE KV past it is exactly
    # what a replay can still use, so its locks stay until admission.
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    seq.num_cached_tokens = CHUNK
    assert _look_up(scheduler, seq) == (0, False)
    assert seq.offload_joint.kv_only_tokens == 4 * CHUNK
    assert adapter.freed == []
    scheduler.update_state_after_alloc(seq)
    assert _freed(adapter) == [(0, 32, None)]


def test_a_replay_from_hbm_never_loads(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    seq.num_cached_tokens = 3  # replay_start, below claimed blocks
    seq.replay_end, seq.replay_start = 16, 3
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert scheduler.build_connector_meta().requests == []
    assert _freed(adapter) == [(0, 32, None)]


def test_asking_again_first_returns_the_earlier_locks(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    assert adapter.freed == []
    # A fresh lookup for the same request (its remembered hit ran out, say)
    # would replace the server's record of what the first one locked.
    scheduler._fresh_tier_lookup(seq, str(seq.id))
    assert _freed(adapter) == [(0, 32, None)]
    assert len(adapter._client.keys) == 2


def _admit_on_a_remembered_hit(scheduler, seq):
    """Asked in one step, admitted in a later one on the remembered answer."""
    assert _look_up(scheduler, seq) == (0, False)
    # Not admitted this step (no room, say).
    assert scheduler.build_connector_meta().requests == []
    return _admit_kv_only(scheduler, seq, hbm=8)


def test_a_kv_only_load_admitted_on_a_remembered_hit_still_loads(monkeypatch):
    # No image anywhere (the hit is 0), PAGE KV to 32.
    scheduler, _, adapter, fallbacks = _scheduler(monkeypatch, chunks=0, kv_chunks=4)
    seq = _seq()
    assert _admit_on_a_remembered_hit(scheduler, seq) == 32
    assert len(adapter._client.keys) == 1  # answered from memory, not asked again
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    # The dispatch confirms the reach its lookup still holds, not a new hit,
    # which can never promise KV past the last image.
    [request] = scheduler.build_connector_meta().requests
    assert request.native_state.kv_only
    assert request.load_spec.lmcache_cached_tokens == 32
    assert _freed(adapter) == [(0, 8, None), (8, 32, [STATE_OBJECT_GROUP])]
    assert fallbacks == []
    output = scheduler.process_completions(KVConnectorOutput())
    assert output.failed_loading == set()


def test_a_parked_load_dropped_at_dispatch_wakes_its_request(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=4)
    seq = _seq()
    _admit_on_a_remembered_hit(scheduler, seq)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    # The lookup no longer holds the reach the load was aimed at.
    monkeypatch.setattr(scheduler._lookup_client, "reach_tokens", lambda _id: 16)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler._native_loads == {}
    # Parked, with no transfer left to report: woken as a failed load, which
    # the scheduler turns into a replay below the HBM hit.
    output = scheduler.process_completions(KVConnectorOutput())
    assert output.failed_loading == {seq.id}
    assert scheduler.process_completions(KVConnectorOutput()).failed_loading == set()


def test_an_abort_while_waiting_returns_the_reach(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    scheduler.request_finished(seq)
    assert (0, 32, None) in _freed(adapter)


# -- ATOM_LMCACHE_MP_STATE_IMAGES=0: PAGE alone, every hit a replay ----------


def test_page_only_needs_a_replay(monkeypatch):
    with pytest.raises(ValueError, match="ATOM_DSV4_STATE_REPLAY"):
        _scheduler(monkeypatch, chunks=0, kv_chunks=0, replay=0, page_only=True)


def test_by_default_a_replay_stores_no_images(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=0, images="unset")
    assert scheduler._page_only


def test_by_default_no_replay_keeps_the_images(monkeypatch):
    scheduler, _, _, _ = _scheduler(
        monkeypatch, chunks=0, kv_chunks=0, replay=0, images="unset"
    )
    assert not scheduler._page_only


def test_an_unknown_images_setting_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="expected auto, 1 or 0"):
        _scheduler(monkeypatch, chunks=0, kv_chunks=0, images="sometimes")


def test_a_page_only_save_stores_page_alone_with_no_checkpoint(monkeypatch):
    scheduler, checkpoints, _, _ = _scheduler(
        monkeypatch, chunks=0, kv_chunks=0, page_only=True
    )
    seq = _seq()
    seq.num_cached_tokens = 36  # computed so far; no checkpoint anywhere
    free = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    # Up to the computed chunk boundary, with no image waited for or pinned.
    assert request.native_state.kv_only
    assert request.native_state.unit_ids == ()
    assert request.native_state.boundary_tokens == 32
    assert request.token_ids == list(seq.token_ids[:32])
    assert request.save_spec.skip_leading_tokens == 0
    assert checkpoints.store.pool.num_free == free
    operation = request.save_operation
    scheduler.process_completions(
        KVConnectorOutput(
            finished_saving={operation},
            connector_completions={
                ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, operation, True)
            },
        )
    )
    assert scheduler._native_saves == {}
    assert str(seq.id) not in scheduler._save_inflight
    assert checkpoints.store.pool.num_free == free


def test_page_only_never_restores_an_image(monkeypatch):
    # A hit carrying an image cannot come from this layout, but if a server
    # reported one the load would read units nothing ever wrote.
    scheduler, checkpoints, _, _ = _scheduler(
        monkeypatch, chunks=2, kv_chunks=2, page_only=True
    )
    seq = _seq()
    _look_up(scheduler, seq)
    seq.num_cached_tokens = 0
    free = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert scheduler._native_loads == {}
    assert checkpoints.store.pool.num_free == free


def test_a_page_only_save_starts_at_the_page_reach(monkeypatch):
    # The lookup's hit needs an image, so it is 0 here; saving from it would
    # store the whole prompt again. LMCache holds the PAGE KV to the reach.
    scheduler, _, adapter, _ = _scheduler(
        monkeypatch, chunks=0, kv_chunks=3, page_only=True
    )
    seq = _seq()
    _look_up(scheduler, seq)
    assert seq.offload_joint.kv_only_tokens == 24
    seq.num_cached_tokens = 36  # HBM covers the reach: nothing loads
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec is None
    assert request.save_spec.skip_leading_tokens == 24
    assert request.native_state.boundary_tokens == 32
    # The lookup's locks, to the reach, went back at admission.
    assert (0, 24, None) in _freed(adapter)


def test_a_page_only_readmission_keeps_the_earlier_reach_as_its_floor(monkeypatch):
    # A request admitted on a remembered answer asks nothing new.
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=0, kv_chunks=0, page_only=True)
    seq = _seq()
    seq._mp_page_reach = 16
    seq.num_cached_tokens = 36
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.save_spec.skip_leading_tokens == 16
