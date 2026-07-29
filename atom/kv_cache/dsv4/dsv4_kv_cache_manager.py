"""DeepSeek-V4 compressed/SWA/arena KV-cache manager."""

from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.block_manager import BlockManager


class Dsv4CompressedPool(PooledFreeList):
    """Backed/unbacked logical IDs coordinated with the DSV4 chunk arena.

    Hashing, events, and sibling eviction intentionally stay in the manager;
    this class only names the shared ``PooledFreeList`` mechanism for DSV4.
    """


class Dsv4KvCacheManager(BlockManager):
    """Coordinate compressed blocks, paged SWA, CSA sources, and arena lending.

    The primary-cache lifecycle (chained hash, primary blocks, per-request
    slots, KV events) is inherited directly from ``BlockManager``; there is no
    intermediate base layer.
    """

    @staticmethod
    def _make_primary_free_list(
        capacity: int, *, initially_backed: bool
    ) -> Dsv4CompressedPool:
        return Dsv4CompressedPool(capacity, initially_backed=initially_backed)

    def ids_conserved(self) -> bool:
        """Public invariant helper for repeated arena borrow/return tests."""
        return self._free_list.ids_conserved() and self.swa._free_list.ids_conserved()
