"""DeepSeek-V4 compressed/SWA/arena KV-cache manager.

Owns everything the architecture-neutral ``BlockManager`` deliberately does
not: the unified-KV chunk arena, the sliding-window pool, CSA boundary-state
sourcing, and the three-way (compress / SWA / CSA) elastic lending. It plugs
into the base through the protected ``_*`` seams and inherits the generic
primary-block lifecycle (chained hash, ref counting, KV events) unchanged.
"""

from collections.abc import Iterable

from atom.config import Config
from atom.kv_cache.dsv4.arena import Dsv4UnifiedArena
from atom.kv_cache.dsv4.batch_tables import build_dsv4_batch_tables
from atom.kv_cache.dsv4.swa_pool import Dsv4SwaPool
from atom.kv_cache.hybrid import HybridKvCacheTables
from atom.kv_cache.pools.chunk_arena import ArenaEmpty
from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence
from atom.utils import envs


class Dsv4CompressedPool(PooledFreeList):
    """Backed/unbacked logical IDs coordinated with the DSV4 chunk arena.

    Hashing, events, and sibling eviction intentionally stay in the manager;
    this class only names the shared ``PooledFreeList`` mechanism for DSV4.
    """


class Dsv4KvCacheManager(BlockManager):
    """Coordinate compressed blocks, paged SWA, CSA sources, and arena lending."""

    @staticmethod
    def _make_primary_free_list(
        capacity: int, *, initially_backed: bool
    ) -> Dsv4CompressedPool:
        return Dsv4CompressedPool(capacity, initially_backed=initially_backed)

    # ---------------- Construction seams ---------------- #

    def _logical_capacity(self, config: Config, num_blocks: int) -> int:
        """Unified-KV chunk arena (ATOM_V4_UNIFIED_KV_ARENA): elastic SWA/
        compressed split. When on, a compressed block_id is a LOGICAL id and
        the arena maps it to per-group physical pages; the logical id space is
        sized to the arena's max compressed capacity so it can grow into
        borrowed SWA chunks. Off -> block_id is its fixed physical slot."""
        self.arena = self._build_arena(config, self.block_size)
        self._arena_on = self.arena is not None and self.arena.enabled
        return self.arena.max_compressed_blocks() if self._arena_on else num_blocks

    def _init_sidecars(self, config: Config, block_size: int) -> None:
        """Build the sliding-window pool (sole prefix-cache sidecar owner now
        that the CSA boundary snapshot is fused into the SWA chunk) and wire it
        to the shared arena for pool-driven three-way lending."""
        _spec = getattr(config, "speculative_config", None)
        _mtp_k = int(getattr(_spec, "num_speculative_tokens", 0) or 0) if _spec else 0
        _num_swa = getattr(config, "num_swa_blocks", 0)
        # Under the unified-KV arena, size the SWA logical id space to the arena's
        # max SWA capacity so SWA can grow into borrowed compressed chunks; else
        # the fixed num_swa pool.
        if self.arena is not None and self.arena.enabled:
            _num_swa = max(_num_swa, self.arena.max_swa_blocks())
        self.swa = Dsv4SwaPool(
            num_blocks=_num_swa,
            window=getattr(config, "swa_window_size", 0),
            block_size=block_size,
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 0),
            mtp_k=_mtp_k,
            full_retain=envs.ATOM_SWA_FULL_RETAIN,
            retention_interval=envs.ATOM_SWA_RETENTION_INTERVAL,
            checkpoint_frac=envs.ATOM_SWA_CHECKPOINT_FRAC,
        )
        # CSA boundary snapshot is fused into the SWA chunk (feat/csa-swa-fusion):
        # it has no separate page pool — capture writes into the block's SWA chunk
        # and retention rides the SWA pin. This flag only gates whether the
        # capture/restore plans are built; exposed via requires_csa_boundary_state.
        self._require_csa_boundary_state = bool(
            getattr(config, "enable_v4_csa_prefix_state_cache", False)
        )
        # Wire the arena into SWA so SWA + compress borrow chunks from the shared
        # arena and reclaim from a sibling under pressure. CSA rides the SWA
        # chunk, so no separate wiring.
        if self.arena is not None and self.arena.enabled:
            self.swa.attach_arena(self.arena, self._evict_cold_for_borrow)

    @staticmethod
    def _build_arena(config: Config, block_size: int):
        """Construct the unified-KV arena from ModelRunner-provided group specs
        when ATOM_V4_UNIFIED_KV_ARENA is on; None otherwise (fixed two-pool)."""
        if not envs.ATOM_V4_UNIFIED_KV_ARENA:
            return None
        specs = getattr(config, "arena_group_specs", None)
        if not specs:
            return None
        return Dsv4UnifiedArena(block_size=block_size, group_specs=list(specs))

    # ---------------- Allocation seams ---------------- #

    def _back_block(self, block_id: int) -> None:
        self._arena_alloc_compressed(block_id)

    def _primary_has_free(self, n: int) -> bool:
        return self._has_free_compressed(n)

    def _bounded_window_hit(
        self, seq: Sequence, compressed_hit: int, block_hashes: list[int]
    ) -> int:
        # Under CSA-into-SWA fusion the CSA boundary rides the same SWA block, so
        # this one gate covers CSA too — a boundary whose SWA window is present
        # has its fused CSA state present.
        return self.swa.bounded_hit(seq, compressed_hit, block_hashes)

    def _window_has_free(self, n: int) -> bool:
        return self.swa.has_free(n)

    def _window_admission_blocks(self, seq: Sequence) -> int:
        return self.swa.admission_blocks(seq)

    def _window_claim_cached(
        self,
        seq: Sequence,
        i: int,
        h: int,
        token_ids: list[int],
        num_cached_blocks: int,
    ) -> None:
        # SWA tail-gate: only the trailing window before the hit boundary is
        # SWA-reused; earlier (out-of-window) blocks are never read by the
        # resumed forward → placeholder (-1), keeping swa_block_table aligned
        # with block_table. swa_hit_start == boundary - swa_tail_blocks on a
        # full-window hit, 0 on a short/partial hit.
        swa_hit_start = max(0, num_cached_blocks - self.swa.tail_blocks)
        if i < swa_hit_start:
            self.swa.alloc_placeholder(seq)  # out of window: never read → -1
        else:
            self.swa.claim_cached(seq, h, token_ids)  # trailing window: reuse

    def _window_alloc_new(self, seq: Sequence) -> None:
        # -1 placeholder keeps swa_block_table the same length as block_table;
        # ensure_for_tokens fills the current chunk's window slots before each
        # forward, free_after_prefill_chunk releases out-of-window ones.
        self.swa.alloc_placeholder(seq)

    def _set_boundary_state(self, seq: Sequence, num_cached_blocks: int) -> None:
        # Fused CSA (feat/csa-swa-fusion): the restore source is the terminal
        # cached block's LOGICAL c4 swa id — its physical SWA chunk (content-
        # addressed, retention-pinned) holds the captured boundary in its fused
        # tail segment. bounded_hit already guaranteed that block's SWA window is
        # present, so its swa id is live (>= 0). The scheduler translates this to
        # the c4 physical swa page for the restore kernel; no separate pool pin is
        # needed (the SWA reuse claim + retention pin already protect the chunk).
        seq.csa_boundary_state_block_id = -1
        if (
            self.requires_csa_boundary_state
            and num_cached_blocks
            and len(seq.swa_block_table) >= num_cached_blocks
        ):
            swa_id = seq.swa_block_table[num_cached_blocks - 1]
            seq.csa_boundary_state_block_id = (
                int(swa_id) if swa_id is not None and swa_id >= 0 else -1
            )

    def _publish_window_hash(
        self, seq: Sequence, i: int, h: int, token_ids: list[int]
    ) -> None:
        # Publishing the SWA block under this hash ALSO publishes its fused CSA
        # boundary (feat/csa-swa-fusion): the capture kernel wrote the boundary
        # into this block's SWA chunk during the forward, so a later prefix hit
        # that reuses the content-addressed SWA block restores the boundary for
        # free — no separate CSA publish needed.
        self.swa.publish_hash(seq, i, h, token_ids)

    def _release_window(self, seq: Sequence) -> None:
        self.swa.release(seq)  # release SWA blocks + clear swa_block_table
        seq.csa_boundary_state_block_id = -1

    def _window_append_new(self, seq: Sequence) -> None:
        self.swa.append_new(seq)

    def _window_free_out(self, seq: Sequence, seq_len: int) -> None:
        self.swa.free_out_of_window(seq, seq_len)

    @property
    def swa_enabled(self) -> bool:
        return self.swa.enabled

    @property
    def requires_csa_boundary_state(self) -> bool:
        """Whether CSA boundary-state capture/restore plans should be built.

        Kept as a property (not a bare read) as a seam: a future non-SWA sidecar
        that does not ride the SWA pin would gate its own lifecycle here.
        """
        return self._require_csa_boundary_state

    def build_batch_tables(self, seqs: Iterable[Sequence]) -> HybridKvCacheTables:
        seqs = list(seqs)
        block_tables = [seq.block_table for seq in seqs if seq.block_table]
        swa_block_tables = [seq.swa_block_table for seq in seqs if seq.block_table]
        boundary_sources = [
            int(getattr(seq, "csa_boundary_state_block_id", -1)) for seq in seqs
        ]
        return build_dsv4_batch_tables(
            arena=self.arena,
            block_tables=block_tables,
            swa_block_tables=swa_block_tables,
            v4_csa_boundary_source_ids=boundary_sources,
        )

    def materialize_window(self, seq: Sequence, seq_len: int) -> None:
        self.swa.materialize_window(seq, seq_len)

    def ensure_window_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        self.swa.ensure_for_tokens(seq, num_cached_tokens, num_new_tokens)

    def finish_prefill_chunk(self, seq: Sequence) -> None:
        self.swa.free_after_prefill_chunk(seq)

    # ---------------- Arena three-way lending (DSV4-only) ---------------- #

    def _evict_cold_for_borrow(self) -> bool:
        """Free one arena page by evicting the coldest ref-0 page of either owner
        (compressed / SWA), so a starved owner can borrow the chunk. Tries each in
        turn; returns True if one was evicted (progress). Pool-driven lending.
        (CSA has no owner of its own — it rides the SWA chunk.)"""
        return self._evict_cold_compressed() or self.swa.evict_cold_for_arena()

    def _evict_cold_compressed(self) -> bool:
        """Truly evict the coldest ref-0 compressed block (drop hash + return its
        arena pages) so the SWA pool can borrow the freed chunk. Returns False
        when no ref-0 compressed block is available. Pool-driven lending."""
        if self.arena is None:
            return False
        # free_block_ids holds only BACKED-free ids (all evictable candidates),
        # so no unbacked-skip dance is needed. Pop the coldest ref-0 backed id,
        # drop its hash + return its arena page, and move the id to the UNBACKED
        # free pool so it stays reusable (re-borrows a page on reuse) instead of
        # leaking out of circulation.
        while True:
            block_id = self._free_list.pop_backed()
            if block_id is None:
                return False
            block = self.blocks[block_id]
            if block.ref_count != 0:
                # Should not happen (used ids are not backed-free); self-heal
                # rather than spin on a stale queue entry.
                continue
            self._drop_hash_if_owner(block_id)
            block.reset()
            self.arena.free_compressed(block_id)
            self._free_list.move_to_unbacked(block_id)
            return True

    def _arena_alloc_compressed(self, block_id: int) -> None:
        """Back a compressed block with arena pages, evicting a cold sibling
        (SWA/CSA/compressed) on starvation and retrying (pool-driven three-way
        lending)."""
        if self.arena is None or self.arena.is_compressed_backed(block_id):
            return
        while True:
            try:
                self.arena.alloc_compressed(block_id)
                return
            except ArenaEmpty:
                if not self._evict_cold_for_borrow():
                    raise

    def _has_free_compressed(self, n: int) -> bool:
        """Whether ``n`` compressed blocks can be admitted. Off: free logical
        slots. On: enough free logical ids AND enough physical placement:
        reusing a BACKED-free id costs no new page, so only the shortfall beyond
        the backed-free ids must be backed by arena free pages + pages reclaimable
        from the SWA pool (one evicted SWA block frees one chunk per group, worth
        the tightest group's pages/chunk). Allocation pops backed-free first
        (`_pop_free_block`), so this accounting is sound."""
        if not self._arena_on:
            return len(self.free_block_ids_set) >= n
        backed_free = len(self.free_block_ids_set)
        total_free = backed_free + len(self._unbacked_free_set)
        if total_free < n:
            return False
        backable = (
            self.arena.compressed_available()
            + self.swa.num_evictable() * self.arena.compress_pages_per_chunk()
        )
        # backed_free ids reuse their held page (0 new pages); the remaining
        # (n - backed_free) must draw a page from `backable`.
        return backed_free + backable >= n
