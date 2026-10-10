# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PAGE eviction order between state checkpoints and cached history blocks.

Default: a fresh block spends the coldest cached history block, and checkpoints
are touched only once none is left. `evict_before_history`: the coldest
checkpoint goes first.
"""

import array

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    READY,
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
)


def _store(num_units=8):
    pool = BlockPool(num_units)
    # Two units per checkpoint image.
    spec = PagedStateCheckpointSpec(
        page_unit_bytes=10, slot_bytes=20, layout_id="evict-test", image_bytes=20
    )
    return pool, PageUnitCheckpointStore(pool, spec)


def _ready(store, prefix_hash):
    assert store.begin_store(prefix_hash, src_slot=0) is not None
    store.complete_inflight()
    cid = next(c for c, r in store.records.items() if r.prefix_hash == prefix_hash)
    assert store.records[cid].state == READY
    return cid


def _cached_history(pool, n, base=1000):
    """`n` blocks that held published history and were freed: cached, reusable."""
    ids = []
    for i in range(n):
        block_id = pool.pop()
        pool.allocate(block_id)
        pool.publish(block_id, base + i, array.array("i", [i]))
        ids.append(block_id)
    for block_id in ids:
        pool.free(block_id)
    return ids


def _full_pool(evict_first):
    """8 units: two checkpoints (4 units) and 4 cached history blocks, none vacant."""
    pool, store = _store()
    store.evict_before_history = evict_first
    _ready(store, 11)
    _ready(store, 22)
    _cached_history(pool, 4)
    assert pool.num_free == 4 and pool.num_reusable_free == 4
    return pool, store


def test_by_default_history_goes_before_any_checkpoint():
    pool, store = _full_pool(evict_first=False)
    assert store.ensure_free_units(1)
    block_id = pool.pop()
    pool.allocate(block_id)
    assert pool.blocks_evicted == 1
    assert store.contains(11) and store.contains(22)
    assert pool.lookup(1000) == -1  # the coldest history block was spent


def test_evict_first_spends_the_coldest_checkpoint_and_keeps_history():
    pool, store = _full_pool(evict_first=True)
    assert store.ensure_free_units(1)
    block_id = pool.pop()
    pool.allocate(block_id)
    assert pool.blocks_evicted == 0
    assert not store.contains(11) and store.contains(22)
    assert all(pool.lookup(1000 + i) != -1 for i in range(4))


def test_evict_first_falls_back_to_history_when_no_checkpoint_is_left():
    pool, store = _full_pool(evict_first=True)
    for _ in range(5):  # 4 checkpoint units, then one history block
        assert store.ensure_free_units(1)
        pool.allocate(pool.pop())
    assert not store.contains(11) and not store.contains(22)
    assert pool.blocks_evicted == 1


def test_a_new_checkpoint_spends_an_old_one_rather_than_history():
    pool, store = _full_pool(evict_first=True)
    _ready(store, 33)
    assert not store.contains(11) and store.contains(22) and store.contains(33)
    assert pool.blocks_evicted == 0
