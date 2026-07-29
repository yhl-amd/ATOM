"""Dense single-pool KV-cache manager."""

from atom.model_engine.block_manager import BlockManager


class DenseKvCacheManager(BlockManager):
    """Primary paged-cache manager with no DSV4 sidecar lifecycle.

    ``BlockManager`` is already the pure dense manager — it builds no arena or
    window pool and its window/batch-table seams are no-ops — so this is just a
    named factory target. It never reads DSV4 geometry from the config, so a
    stale ``num_swa_blocks`` / ``v4_arena_group_specs`` cannot bring a sidecar
    to life (``.arena`` stays ``None``, ``swa_enabled`` stays ``False``).
    """
