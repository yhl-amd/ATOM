# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable

import numpy as np
import xxhash

from atom.config import Config
from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from atom.kv_cache.hybrid import HybridKvCacheTables
from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence


def _make_block_stored(
    hashes: list[int],
    tokens: list[int],
    parent: int | None,
    block_size: int,
    medium: str = MEDIUM_GPU,
) -> BlockStored:
    """Construct a BlockStored event from a coalesced run of new blocks."""
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=block_size,
        medium=medium,
    )


def _make_block_removed(hashes: list[int]) -> BlockRemoved:
    return BlockRemoved(block_hashes=hashes, medium=MEDIUM_GPU)


def _make_all_cleared() -> AllBlocksCleared:
    return AllBlocksCleared()


class BlockManager:
    """Architecture-neutral paged primary KV-cache manager.

    Owns the generic control plane: chained content hashing, primary block
    lifecycle, per-request state slots, and KV events. DSV4's compressed / SWA
    / CSA sidecar lifecycle lives in ``Dsv4KvCacheManager``, which overrides the
    protected ``_*`` seams below. The base defaults reproduce the pure dense
    path exactly (no window pool, no arena), so ``BlockManager`` is usable on
    its own for dense models.
    """

    @staticmethod
    def _make_primary_free_list(
        capacity: int, *, initially_backed: bool
    ) -> PooledFreeList:
        return PooledFreeList(capacity, initially_backed=initially_backed)

    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.block_size = block_size
        self.dcp_world_size = getattr(config, "decode_context_parallel_size", 1)
        # dcp_rank is always 0 here: BlockManager runs only on the scheduler
        # (rank 0). DCP rank is used only to compute local token counts for
        # memory reservation; the actual per-rank routing is done in the workers.
        self.dcp_rank = 0

        # Null sidecar members expected by the manager contract (a dense manager
        # answers ``.arena is None``). Subclasses that add elastic paging replace
        # these before the free list is sized.
        self.arena = None
        self._arena_on = False
        # The logical id space defaults to the fixed physical block count; a
        # subclass with an elastic arena sizes it to the arena's capacity.
        n_logical = self._logical_capacity(config, num_blocks)
        self.blocks: list[Block] = [Block(i) for i in range(n_logical)]
        self.hash_to_block_id: dict[int, int] = dict()
        # Free logical ids. Arena OFF (dense): block_id IS its fixed physical
        # slot (always "backed"), the unbacked pool stays empty and
        # free_block_ids holds every id. A subclass may split the pool into
        # backed / unbacked ids when it borrows physical pages elastically.
        self._free_list = self._make_primary_free_list(
            n_logical, initially_backed=not self._arena_on
        )
        # Migration aliases for existing diagnostics/tests. Allocation mechanics
        # are owned by PooledFreeList; these names no longer implement queues.
        self.free_block_ids = self._free_list.backed_ids
        self.free_block_ids_set = self._free_list.backed_set
        self._unbacked_free_ids = self._free_list.unbacked_ids
        self._unbacked_free_set = self._free_list.unbacked_set
        self.used_block_ids = self._free_list.used_ids
        self.enable_prefix_caching = config.enable_prefix_caching

        kv_events = getattr(config, "kv_events_config", None)
        self._events_enabled: bool = bool(kv_events and kv_events.enable)
        self._event_log: list[KVCacheEvent] | None = (
            [] if self._events_enabled else None
        )
        # Per-request cache slot pool. Used by attention types with a
        # stateful per-request buffer (GDN recurrent state, V4 compressor
        # state). The backing tensor is pre-allocated by ModelRunner sized
        # to max_num_seqs and excluded from `num_kvcache_blocks` at sizing
        # time, so admission only needs a free slot index from this list.
        # Each slot group contains slots_per_req() contiguous tensor indices
        # (1 for stateless / + num_spec for spec-decoding-aware variants).
        num_per_req_cache_groups: int = getattr(config, "num_per_req_cache_groups", 0)
        self.free_per_req_cache_groups: list[int] = list(
            range(num_per_req_cache_groups)
        )

        # Sidecar pools (windowed SWA, CSA boundary snapshot) are built by
        # subclasses that need them; the base has none.
        self._init_sidecars(config, block_size)

    # ---------------- Architecture seams (base = dense no-ops) ---------------- #

    def _logical_capacity(self, config: Config, num_blocks: int) -> int:
        """Number of logical block ids. Dense: the fixed physical block count.

        A subclass with an elastic arena builds the arena here (setting
        ``self.arena`` / ``self._arena_on``) and returns its max capacity.
        """
        return num_blocks

    def _init_sidecars(self, config: Config, block_size: int) -> None:
        """Build window / boundary-state sidecar pools. Dense: none."""
        del config, block_size

    def _back_block(self, block_id: int) -> None:
        """Ensure physical backing for a freshly allocated block. Dense: no-op
        (block_id is its own fixed physical slot)."""
        del block_id

    def _primary_has_free(self, n: int) -> bool:
        """Whether ``n`` primary blocks can be admitted. Dense: free slots."""
        return len(self.free_block_ids_set) >= n

    def _bounded_window_hit(
        self, seq: Sequence, compressed_hit: int, block_hashes: list[int]
    ) -> int:
        """Shrink the compressed hit to what the window pool can also serve.
        Dense: no window, so the full compressed hit stands."""
        del seq, block_hashes
        return compressed_hit

    def _window_has_free(self, n: int) -> bool:
        """Whether the window pool can admit ``n`` blocks. Dense: always."""
        del n
        return True

    def _window_admission_blocks(self, seq: Sequence) -> int:
        """Windowed peak the pool must admit for ``seq``. Dense: none."""
        del seq
        return 0

    def _window_claim_cached(
        self,
        seq: Sequence,
        i: int,
        h: int,
        token_ids: list[int],
        num_cached_blocks: int,
    ) -> None:
        """Claim / placeholder the window block parallel to a cached primary
        block. Dense: no-op."""
        del seq, i, h, token_ids, num_cached_blocks

    def _window_alloc_new(self, seq: Sequence) -> None:
        """Add a window placeholder parallel to a fresh primary block. Dense:
        no-op."""
        del seq

    def _set_boundary_state(self, seq: Sequence, num_cached_blocks: int) -> None:
        """Record the CSA boundary-state restore source. Dense: no-op."""
        del seq, num_cached_blocks

    def _publish_window_hash(
        self, seq: Sequence, i: int, h: int, token_ids: list[int]
    ) -> None:
        """Publish the parallel window block under a content hash. Dense: no-op."""
        del seq, i, h, token_ids

    def _release_window(self, seq: Sequence) -> None:
        """Release window blocks and boundary state for ``seq``. Dense: no-op."""
        del seq

    def _window_append_new(self, seq: Sequence) -> None:
        """Append a decode-step window block in lockstep. Dense: no-op."""
        del seq

    def _window_free_out(self, seq: Sequence, seq_len: int) -> None:
        """Reclaim window blocks that fell out of the window. Dense: no-op."""
        del seq, seq_len

    @property
    def swa_enabled(self) -> bool:
        """Whether a sliding-window pool is active. Dense: never."""
        return False

    def build_batch_tables(self, seqs: Iterable[Sequence]) -> HybridKvCacheTables:
        """Physical batch tables. Dense: empty (logical == physical)."""
        return HybridKvCacheTables.empty(num_sequences=sum(1 for _ in seqs))

    def materialize_window(self, seq: Sequence, seq_len: int) -> None:
        del seq, seq_len

    def ensure_window_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        del seq, num_cached_tokens, num_new_tokens

    def finish_prefill_chunk(self, seq: Sequence) -> None:
        del seq

    # ---------------- Metrics / block access ---------------- #

    @property
    def num_total_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free_per_req_cache_groups(self) -> int:
        return len(self.free_per_req_cache_groups)

    def kv_usage(self) -> float:
        return len(self.used_block_ids) / len(self.blocks) if self.blocks else 0.0

    def get_block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # ---------------- Primary block lifecycle ---------------- #

    def _drop_hash_if_owner(self, block_id: int) -> None:
        """Evict the stale prefix-cache hash entry a block still owns, emitting
        a BlockRemoved event. ATOM's eviction is lazy: a freed block keeps its
        hash until its slot is reused, so this is the true eviction event."""
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            if self._event_log is not None:
                self._event_log.append(_make_block_removed([block.hash]))

    def _pop_free_block(self) -> int:
        """Pop the next available free block id. Prefer a BACKED-free id (reuse
        its held page, no borrow) before an UNBACKED-free one. Arena off:
        `_unbacked_free_ids` is empty (unchanged)."""
        return self._free_list.pop()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Evict stale hash entry before resetting (lazy eviction fires here).
        self._drop_hash_if_owner(block_id)
        block.reset()
        self._free_list.mark_used(block_id)
        # Ensure physical backing (no-op dense; borrows arena pages in DSV4).
        self._back_block(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        assert self._free_list.deallocate(block_id), f"block {block_id} not in use"

    def _dcp_num_blocks(self, seq_len: int) -> int:
        if self.dcp_world_size <= 1:
            return (seq_len + self.block_size - 1) // self.block_size
        from atom.model_ops.dcp_ops import get_dcp_local_seq_lens

        local_len = get_dcp_local_seq_lens(
            np.array([seq_len]), self.dcp_world_size, self.dcp_rank
        )[0]
        return int((local_len + self.block_size - 1) // self.block_size)

    def _effective_block_size(self):
        return self.block_size * self.dcp_world_size

    def can_allocate(self, seq: Sequence) -> int:
        """Return number of cache-hit blocks (>=0) if seq fits, else -1.

        The hit count is the contiguous run of cache hits starting at the
        prompt's first block. On the first miss we break: subsequent blocks
        cannot match either (hash is chained, so a divergent token breaks the
        chain for the rest of the prompt). The last block is never considered
        for reuse — prefill must forward at least one block to produce
        sampler logits, so it always comes from the free pool.

        Caller (scheduler) passes the returned hit count to `allocate()`,
        avoiding a second hash pass.
        """
        # State cache (mamba / V4 compressor ring) has its own pre-allocated
        # tensor; admission only needs a free slot index, not extra paged
        # blocks. See `allocate()` for the budget reasoning.
        if seq.has_per_req_cache and not self.free_per_req_cache_groups:
            return -1
        if not self.enable_prefix_caching:
            if not self._primary_has_free(self._dcp_num_blocks(len(seq))):
                return -1
            # Window admission: only the per-request windowed peak (filled
            # incrementally + window-freed), not the whole prompt. No-op / True
            # when there is no window pool.
            if not self._window_has_free(self._window_admission_blocks(seq)):
                return -1
            return 0
        # Step 1: compressed prefix (CSA/HCA/indexer share the block hash and
        # read the WHOLE history, so this stays a full front-to-back chained
        # match). Record each block's hash for the window scan below.
        h = -1
        compressed_hit = 0
        block_hashes: list[int] = []
        compressed_block_ids: list[int] = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            block_hashes.append(h)
            compressed_block_ids.append(block_id)
            compressed_hit += 1
        # Step 2: shrink the compressed hit to the largest boundary the window
        # pool can also serve (a local sliding window may have evicted the tail
        # even when the compressed prefix is present). No-op when there is no
        # window pool. The assert guards the caller's compressed-prefix invariant
        # (every counted hit block must have a materialized physical id).
        assert (
            len(compressed_block_ids) >= compressed_hit
        ), "missing physical ids for compressed prefix hit"
        num_cached_blocks = self._bounded_window_hit(seq, compressed_hit, block_hashes)
        # Instrumentation: record the pre-gate compressed hit so CacheStats can
        # separate reuse lost to the window tail gate (compressed_hit -
        # num_cached_blocks) from reuse lost to compressed eviction.
        seq.num_compressed_hit_blocks = compressed_hit
        # Free-pool demand: blocks we actually reuse minus those already used
        # (shared ref); blocks we drop from the hit become fresh → counted.
        num_new_blocks = seq.num_blocks
        for i in range(num_cached_blocks):
            if self.hash_to_block_id[block_hashes[i]] in self.used_block_ids:
                num_new_blocks -= 1
        if not self._primary_has_free(num_new_blocks):
            return -1
        # Window new-block demand is bounded by the windowed peak (filled
        # incrementally + window-freed), not the full new-block count. No-op /
        # True when there is no window pool.
        if not self._window_has_free(
            min(num_new_blocks, self._window_admission_blocks(seq))
        ):
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0):
        """Allocate blocks for `seq`. `num_cached_blocks` is the hit count
        returned by `can_allocate` (0 if caller didn't call it).

        Hash registration is deferred to hash_blocks(), called from
        scheduler.postprocess() once the forward has computed each block's
        KV. This keeps the manager correct under future chunked-prefill
        scheduling: a block spanning multiple steps must not be published as
        a hash until fully filled.
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # Cache hit on a free-pool block — claim without _allocate_block
                # (whose reset() would evict the hash entry and destroy the
                # cache for everyone).
                assert block.ref_count == 0
                block.ref_count = 1
                # A cache hit lands only on a BACKED block (its KV is still
                # resident); discard from the free pool so the id leaves cleanly.
                self._free_list.mark_used(block_id)
            seq.block_table.append(block_id)
            # Window bookkeeping parallel to this cached block (no-op dense).
            self._window_claim_cached(seq, i, h, token_ids, num_cached_blocks)
        # Boundary-state restore source (no-op dense).
        self._set_boundary_state(seq, num_cached_blocks)
        for _ in range(num_cached_blocks, self._dcp_num_blocks(len(seq))):
            block_id = self._pop_free_block()
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
            # Window placeholder parallel to this fresh block (no-op dense).
            self._window_alloc_new(seq)
        seq.num_cached_tokens = num_cached_blocks * self.block_size

        # Per-request cache: claim one slot index from the pre-allocated
        # state tensor (e.g. GDN mamba_k_cache, V4 compressor state + SWA
        # ring). The state tensor's memory was already excluded from
        # `num_kvcache_blocks` in ModelRunner._compute_kv_budget(), so
        # admitting a seq adds no further paged-block cost. The slot cap
        # (`free_per_req_cache_groups` size = `max_num_seqs`) is the sole
        # admission bound for state cache.
        if seq.has_per_req_cache:
            seq.per_req_cache_group = self.free_per_req_cache_groups.pop()

    def hash_blocks(self, seq: Sequence, num_new_tokens: int) -> None:
        """Register hashes for blocks finalized by the most recent step.

        Called from scheduler.postprocess() after the forward completes, so a
        block's hash is only published once its KV is actually computed. The
        `[start, end)` range covers blocks fully filled by this step:
          start = first block whose first token was at num_cached_tokens
          end   = first block not yet fully filled (excludes the partial one)
        Caller passes `num_new_tokens` = tokens forwarded in this step. For
        single-shot prefill that's `seq.num_tokens - seq.num_cached_tokens`;
        chunked prefill will pass the per-chunk count.
        """
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + num_new_tokens) // self.block_size
        if start >= end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            # Publish the parallel window block under the same content hash so
            # cross-request hits can reuse its sliding-window KV (no-op dense).
            self._publish_window_hash(seq, i, h, token_ids)
            if record:
                store_run_hashes.append(h)
                store_run_tokens.extend(token_ids)
        if record and store_run_hashes:
            self._event_log.append(
                _make_block_stored(
                    store_run_hashes,
                    store_run_tokens,
                    store_run_parent,
                    self.block_size,
                )
            )

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # Release window blocks + boundary state (no-op dense).
        self._release_window(seq)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            self.free_per_req_cache_groups.append(seq.per_req_cache_group)
            seq.per_req_cache_group = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        needed_blocks = (seq_len + num_new_tokens + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        if not self._primary_has_free(new_blocks_needed):
            return False
        if not self._window_has_free(new_blocks_needed):  # True when no window
            return False
        return True

    def may_append(self, seq: Sequence, num_new_tokens: int = 1):
        # Note: in disaggregated (P/D) mode the scheduler skips this call on
        # the first decode step after remote prefill, because blocks were
        # already allocated during the KV transfer phase.
        block_table = seq.block_table
        seq_len = len(seq)
        # Check if we need to allocate a new block
        # When len(seq) % block_size == 1, we need a new block for the next token
        # When block_size == 1, every token needs a new block
        ebs = self._effective_block_size()
        if 0 < seq_len % ebs <= num_new_tokens or ebs == 1:
            needed_blocks = (seq_len + ebs - 1) // ebs
            while len(block_table) < needed_blocks:
                # Decode-generated blocks: token not finalized yet (depends on
                # sampling / speculative verification), so we cannot compute a
                # correct hash here.  Just allocate the block without hashing.
                block_id = self._pop_free_block()
                self._allocate_block(block_id)
                block_table.append(block_id)
                self._window_append_new(seq)  # lockstep window block (no-op dense)
        # Reclaim window blocks that just fell out of the window (no-op dense).
        self._window_free_out(seq, len(seq))

    # ---------------- KV event API ---------------- #

    def take_events(self) -> list[KVCacheEvent]:
        """Drain and return events accumulated since the last call."""
        if self._event_log is None or not self._event_log:
            return []
        self._event_log, events = [], self._event_log
        return events

    def clear_cache(self) -> None:
        """Drop every prefix-cache entry. Used by `/reset_prefix_cache`-style
        admin APIs. Does NOT touch blocks currently held by live sequences —
        they remain valid via their block_table refs, just unhashable for
        future requests."""
        self.hash_to_block_id.clear()
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = []
        if self._event_log is not None:
            self._event_log.append(_make_all_cleared())

    @property
    def kv_events_enabled(self) -> bool:
        """True iff KV events are being recorded."""
        return self._event_log is not None

    def record_remote_store(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_block_hash: int | None = None,
    ) -> None:
        """Emit a BlockStored(medium=REMOTE) for blocks received from a remote
        KV transfer producer (Mooncake/MoriIO decode side). Called by the
        KVConnector worker once the transfer completes so external KV-cache
        consumers (LMCache, etc.) can track remote-resident blocks."""
        if self._event_log is None or not block_hashes:
            return
        self._event_log.append(
            _make_block_stored(
                block_hashes,
                token_ids,
                parent_block_hash,
                self.block_size,
                medium=MEDIUM_REMOTE,
            )
        )
