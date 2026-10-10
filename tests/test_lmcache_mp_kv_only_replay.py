# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP native-state loads that bring the PAGE KV alone, for a replay.

The ordinary lookup reports how far the KV reaches *with* a state image at its
end; a bounded replay (`Sequence.replay_kv_load`) can use the KV alone. These
tests drive the second, PAGE-only lookup through its whole life: submitted
beside the first, weighed, turned into a load with no image, and closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from atom.kv_transfer.disaggregation.types import LoadOperationId
from atom.kv_transfer.offload.mp import deployment
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp.lookup import (
    OBJECT_GROUPS_CONFIG,
    kv_only_lookup_id,
)
from atom.kv_transfer.offload.mp.native_state_scheduler import (
    NativeStateLMCacheMPConnectorScheduler,
)
from atom.kv_transfer.offload.mp.native_state_worker import PAGE_OBJECT_GROUP
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.sequence import Sequence

CHUNK = 8
REPLAY = 6


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
    """Answers a PAGE-only lookup with `kv_chunks`, any other with `chunks`."""

    def __init__(self, chunks, kv_chunks):
        self.chunks = chunks
        self.kv_chunks = kv_chunks
        self.keys = {}

    def lookup(self, key, tp_size):
        self.keys[key.request_id] = key
        return _Future()

    def query_prefetch_status(self, request_id):
        configs = self.keys[request_id].request_configs or {}
        page_only = configs.get(OBJECT_GROUPS_CONFIG) == [PAGE_OBJECT_GROUP]
        return _Future(self.kv_chunks if page_only else self.chunks)


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
        assert request_id in self._pending_lookups, "no second, synchronous lookup"

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


def _scheduler(monkeypatch, *, chunks, kv_chunks, replay=REPLAY, min_load=0):
    monkeypatch.setenv("OFFLOAD_MIN_SAVE_TOKENS", "0")
    monkeypatch.setattr(
        deployment.offcfg,
        "build_lmcache_config",
        lambda kvc: SimpleNamespace(
            chunk_size=kvc["kv_connector_extra_config"]["lmcache.chunk_size"]
        ),
    )
    adapter = _Adapter(_Client(chunks, kv_chunks))
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


def _admit_kv_only(scheduler, seq, *, hbm):
    """What `BlockManager._replay_hit` and `allocate` leave for a KV-only load."""
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    assert not scheduler.lookup_pending(seq)
    scheduler.get_num_new_matched_tokens(seq)
    reach = seq.offload_joint.kv_only_tokens
    seq.num_cached_tokens = hbm
    seq.replay_kv_load = True
    seq.replay_load_start = hbm
    seq.replay_end = reach
    seq.replay_start = reach - REPLAY
    return reach


def test_kv_only_lookup_restricts_the_key_to_page():
    from atom.kv_transfer.offload.mp.lookup import _MPLookupClient

    adapter = _Adapter(_Client(chunks=1, kv_chunks=3))
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector_extra_config": {}},
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    client = _MPLookupClient(adapter, config=config, timeout=1.0, poll_interval=0.01)
    assert client.submit(list(range(32)), "r:kv", object_groups=(PAGE_OBJECT_GROUP,))
    [key] = adapter._client.keys.values()
    assert key.request_configs == {OBJECT_GROUPS_CONFIG: [PAGE_OBJECT_GROUP]}
    assert not client.has_answer("r:kv")
    assert client.poll("r:kv")
    assert client.has_answer("r:kv")
    assert client.lookup(list(range(32)), "r:kv") == 3 * CHUNK


def test_both_lookups_run_and_the_reach_is_recorded(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    assert set(adapter._client.keys) == {
        deployment._mp_session_id(scheduler._config, "1"),
        deployment._mp_session_id(scheduler._config, kv_only_lookup_id("1")),
    }
    # The ordinary answer is unchanged (the scheduler records it as
    # `kv_prefix_tokens`); the KV-only reach rides along beside it.
    assert scheduler.get_num_new_matched_tokens(seq) == (1 * CHUNK, True)
    assert seq.offload_joint.kv_only_tokens == 4 * CHUNK


def test_no_kv_only_lookup_with_replay_off(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4, replay=0)
    seq = _seq()
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    assert len(adapter._client.keys) == 1
    scheduler.get_num_new_matched_tokens(seq)
    assert seq.offload_joint.kv_only_tokens == 0


def test_a_kv_only_load_reads_page_alone_and_restores_nothing(monkeypatch):
    scheduler, checkpoints, adapter, fallbacks = _scheduler(
        monkeypatch, chunks=1, kv_chunks=4
    )
    seq = _seq()
    reach = _admit_kv_only(scheduler, seq, hbm=8)
    assert reach == 32
    free_before = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    # The ordinary lookup's locks went back at once.
    state_session = deployment._mp_session_id(scheduler._config, "1")
    assert {"start": 0, "end": 8, "request_id": state_session}.items() <= (
        adapter.freed[0].items()
    )
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
    # The KV-only lookup handed its range to the retrieve, keeping [8, 32).
    kv_session = deployment._mp_session_id(scheduler._config, kv_only_lookup_id("1"))
    assert any(
        f["request_id"] == kv_session and (f["start"], f["end"]) == (0, 8)
        for f in adapter.freed
    )
    operation = seq._load_operation
    assert isinstance(operation, LoadOperationId)
    assert scheduler.load_finished(operation) is True
    # Nothing was adopted as a checkpoint: there was no image.
    assert checkpoints.store.pool.num_free == free_before
    scheduler.request_finished(seq)
    assert kv_session in adapter.ended


def test_a_refused_kv_only_load_falls_back_and_returns_its_locks(monkeypatch):
    scheduler, _, adapter, fallbacks = _scheduler(
        monkeypatch, chunks=1, kv_chunks=4, min_load=10_000
    )
    seq = _seq()
    _admit_kv_only(scheduler, seq, hbm=8)
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert fallbacks == [seq]
    kv_session = deployment._mp_session_id(scheduler._config, kv_only_lookup_id("1"))
    assert any(
        f["request_id"] == kv_session and (f["start"], f["end"]) == (0, 32)
        for f in adapter.freed
    )


def test_an_unused_kv_only_lookup_returns_its_locks_at_admission(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    scheduler.get_num_new_matched_tokens(seq)
    seq.num_cached_tokens = 0  # an ordinary admission: no replay chosen
    scheduler.update_state_after_alloc(seq)
    kv_session = deployment._mp_session_id(scheduler._config, kv_only_lookup_id("1"))
    assert any(
        f["request_id"] == kv_session and (f["start"], f["end"]) == (0, 32)
        for f in adapter.freed
    )


def test_a_replay_from_hbm_never_loads(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=1, kv_chunks=4)
    seq = _seq()
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    scheduler.get_num_new_matched_tokens(seq)
    seq.num_cached_tokens = 3  # replay_start, below claimed blocks
    seq.replay_end, seq.replay_start = 16, 3
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert scheduler.build_connector_meta().requests == []


def test_a_kv_only_reach_below_the_state_reach_is_not_trusted(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=3, kv_chunks=1)
    seq = _seq()
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    scheduler.get_num_new_matched_tokens(seq)
    assert seq.offload_joint.kv_only_tokens == 0
