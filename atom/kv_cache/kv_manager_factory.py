"""KV-cache manager construction without scheduler/backend coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atom.kv_cache.kv_manager_protocol import KvCacheManager


def make_kv_cache_manager(config: Any) -> "KvCacheManager":
    """Construct the manager the attention builder selected.

    ``kv_manager_kind`` is the single authority: the model's attention builder
    declares it via ``compute_kv_pool_layout`` (``"dense"`` by default,
    ``"dsv4"`` for the DSV4 family), ModelRunner copies it onto the Config, and
    EngineCore propagates it over IPC. We never sniff geometry fields to guess
    the kind — a missing value simply falls back to the neutral default.
    """
    manager_kind = getattr(config, "kv_manager_kind", None) or "dense"

    if manager_kind == "dense":
        from atom.kv_cache.dense_kv_cache_manager import DenseKvCacheManager

        return DenseKvCacheManager(config)
    if manager_kind == "dsv4" or str(manager_kind).startswith("dsv4_"):
        from atom.kv_cache.dsv4.kv_cache_manager import Dsv4KvCacheManager

        return Dsv4KvCacheManager(config)
    raise ValueError(f"unknown KV cache manager kind: {manager_kind!r}")
