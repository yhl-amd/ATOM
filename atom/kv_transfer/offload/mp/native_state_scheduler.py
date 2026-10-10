# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded transfers of native READY checkpoint images over LMCache MP."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
    StateStoreOperationId,
)
from atom.kv_transfer.offload._offload_common import (
    max_pending_saves,
    validated_kv_role,
)
from atom.kv_transfer.offload.metadata import LoadSpec, NativeStateTransfer
from atom.kv_transfer.offload.mp.deployment import _mp_session_id, _validate_mp_config
from atom.kv_transfer.offload.mp.lookup import kv_only_lookup_id
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_STORE_CHANNEL,
    PAGE_OBJECT_GROUP,
    require_native_state_server,
)
from atom.kv_transfer.offload.mp.scheduler import LMCacheMPConnectorScheduler
from atom.model_engine.page_unit_checkpoint import SuspendedCheckpointRestore
from atom.utils import envs

logger = logging.getLogger("atom")

_MAX_SAVE_ATTEMPTS = 3


@dataclass
class _NativeSave:
    seq: Any
    source: StateStoreOperationId
    saved_before: int
    boundary: int


@dataclass
class _NativeLoad:
    seq: Any
    transfer: NativeStateTransfer
    hbm_cached_tokens: int
    local_restore: SuspendedCheckpointRestore | None = None
    dispatched: bool = False


class NativeStateLMCacheMPConnectorScheduler(LMCacheMPConnectorScheduler):
    """Pair PAGE KV with an exact native state image under one operation ID.

    Native sources are acquired only after save admission. Loads reserve raw
    PAGE units after request allocation and before parking; their release waits
    for the worker's terminal transfer-and-restore report. Neither side recycles
    these units in response to elapsed time.
    """

    _supports_early_block_release = True

    def __init__(self, config: Any) -> None:
        # The layout is known only after BlockManager builds the native pool.
        # Defer connecting so scheduler and workers use the same namespace.
        _validate_mp_config(config)
        self._config = config
        self.kv_role = validated_kv_role(
            getattr(config, "kv_transfer_config", {}) or {}
        )
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self._block_manager = None

    def bind_block_manager(self, block_manager: Any) -> None:
        if self._block_manager is block_manager:
            return
        if self._block_manager is not None:
            raise RuntimeError(
                "native-state LMCache MP scheduler is already bound to a block manager"
            )
        coordinator = getattr(block_manager, "paged_state_checkpoints", None)
        if coordinator is None:
            raise ValueError(
                "native-state LMCache MP requires PAGE checkpoint geometry"
            )
        super().__init__(self._config, checkpoint_spec=coordinator.store.spec)
        try:
            require_native_state_server(self._mp_adapter, self._config)
            self._hash_block_size = int(block_manager.hash_block_size)
            # The shortest prefix worth storing, in absolute tokens. With the
            # default equal to OFFLOAD_MIN_LOAD_TOKENS a shorter prefix could
            # never be loaded back. Normal and late saves apply the same rule.
            self._min_save_tokens = envs.OFFLOAD_MIN_SAVE_TOKENS
            if self.chunk_size % self._hash_block_size:
                raise ValueError(
                    "native-state LMCache MP chunk size must align to native "
                    "hash blocks"
                )
            self._max_pending_saves = max_pending_saves(
                getattr(self._config, "kv_transfer_config", {}) or {},
                envs.OFFLOAD_COPY_WORKERS,
            )
        except Exception:
            shutdown = getattr(self._mp_adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise
        self._block_manager = block_manager
        self._checkpoints = coordinator
        self._native_saves: dict[SaveOperationId, _NativeSave] = {}
        self._native_loads: dict[LoadOperationId, _NativeLoad] = {}
        self._native_load_operations: dict[str, LoadOperationId] = {}
        # One retry record per tracked request: a newer boundary can retry the
        # complete unsaved prefix, but a permanently failing boundary cannot
        # hold a finished request forever.
        self._save_failures: dict[str, tuple[Any, int, int]] = {}
        self._retired_requests: dict[str, Any] = {}
        # Bounded replay (`Sequence.replay_kv_load`): tokens one replays, or 0
        # when it is off. With it on, each request also asks how far the PAGE
        # KV alone reaches, which a replay can resume past the last image.
        self._replay_tokens = int(getattr(block_manager, "state_replay_tokens", 0))
        # sid -> (seq, KV-only reach in tokens, or None while it is asked).
        self._kv_reach: dict[str, tuple[Any, int | None]] = {}
        # Requests whose KV-only lookup session must be ended with them.
        self._kv_sessions: set[str] = set()
        # Requests parked for a load the next dispatch has yet to send, and
        # those whose load it dropped instead: nothing will ever report on
        # those, so `process_completions` wakes them as failed loads.
        self._parked_loads: dict[str, Any] = {}
        self._dropped_parked_loads: set[Any] = set()

    def can_partially_deallocate_state(self, seq: Any) -> bool:
        """The checkpoint coordinator owns state sources past request free.

        Native-state saves never read the request's active state slot after
        admission: ``acquire_checkpoint_source`` gives each emitted save an
        independent checkpoint PAGE lease. Waiting candidates likewise refer
        to READY checkpoint content and acquire that lease only when admitted.
        """
        return (
            bool(getattr(seq, "has_per_req_cache", False))
            and self._block_manager is not None
            and getattr(self, "_checkpoints", None)
            is getattr(self._block_manager, "paged_state_checkpoints", None)
        )

    def _boundary_hash(self, seq: Any, boundary: int) -> int:
        count = boundary // self._hash_block_size
        if boundary <= 0 or boundary % self._hash_block_size:
            raise ValueError("native checkpoint boundary must align to hash blocks")
        chain = getattr(seq, "block_hashes", ())
        if len(chain) >= count:
            return int(chain[count - 1])
        # Some checkpoint producers do not reserve midstep checkpoints, so
        # BlockManager may leave Sequence.block_hashes empty. Extend the chain
        # through BlockManager itself -- its seed (`seq.cache_seed`, which
        # multimodal requests set), algorithm and token slices -- caching only
        # this request's immutable prompt chain.
        chain = getattr(seq, "_mp_checkpoint_hashes", None) or []
        if len(chain) < count:
            chain = self._block_manager.prefix_hash_chain(seq, chain, count)
            seq._mp_checkpoint_hashes = chain
        return int(chain[count - 1])

    def _lookup_token_ids(self, seq: Any) -> list[int]:
        # Truncate the QUERY, not its answer: MP recurrent/window keys and read
        # locks belong to the requested endpoint. A full-prompt query followed
        # by rounding the result would pair the earlier KV with later state.
        boundary = self._chunk_floor(max(0, int(seq.num_prompt_tokens) - 1))
        return list(seq.token_ids[:boundary])

    def _lookup_len(self, seq: Any) -> int:
        return self._chunk_floor(max(0, int(seq.num_prompt_tokens) - 1))

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        if not getattr(seq, "has_per_req_cache", False):
            return 0, False
        previous = self._native_load_operations.get(str(seq.id))
        if previous is not None:
            return 0, False
        matched = super().get_num_new_matched_tokens(seq)
        seq.offload_joint.kv_only_tokens = self._kv_only_reach(seq)
        return matched

    # -- KV-only reach, for a bounded replay -------------------------------
    #
    # The ordinary lookup reports the longest prefix whose PAGE KV is present
    # AND whose state image sits at its end. A replay needs only the first:
    # it rebuilds the window state from the KV. So with replay on, a second
    # lookup over the PAGE object group alone runs beside the first, under its
    # own session and locks, and `BlockManager._replay_hit` weighs its answer.

    def prefetch_lookups(self, seqs) -> None:
        # The scheduler hands over an iterator (a slice of `waiting`), which
        # the base would exhaust before the loop below sees a single request.
        seqs = list(seqs)
        super().prefetch_lookups(seqs)
        if not self._replay_tokens:
            return
        client = self._lookup_client
        submitted = client.pending_ids()
        for seq in seqs:
            sid = str(seq.id)
            entry = self._kv_reach.get(sid)
            if entry is not None and entry[0] is seq:
                continue
            # Alongside the ordinary lookup only: one never sent (the request
            # was admitted on a remembered hit, say) has nothing to beat.
            if sid not in submitted or not getattr(seq, "has_per_req_cache", False):
                continue
            if entry is not None:
                self._drop_kv_reach(sid)
            if client.submit(
                self._lookup_token_ids(seq),
                kv_only_lookup_id(sid),
                object_groups=(PAGE_OBJECT_GROUP,),
            ):
                self._kv_reach[sid] = (seq, None)
                self._kv_sessions.add(sid)

    def lookup_pending(self, seq: Any) -> bool:
        if super().lookup_pending(seq):
            return True
        sid = str(seq.id)
        entry = self._kv_reach.get(sid)
        if entry is None or entry[0] is not seq or entry[1] is not None:
            return False
        kv_id = kv_only_lookup_id(sid)
        client = self._lookup_client
        if not client.is_pending(kv_id) or client.poll(kv_id):
            self._lookup_defer_since.pop(kv_id, None)
            return False
        # Waiting for it is bounded by the same window as the ordinary lookup.
        now = time.monotonic()
        first = self._lookup_defer_since.setdefault(kv_id, now)
        return now - first <= envs.OFFLOAD_LOOKUP_DEFER_S

    def _kv_only_reach(self, seq: Any) -> int:
        """This request's KV-only reach in tokens, or 0 if unknown."""
        sid = str(seq.id)
        entry = self._kv_reach.get(sid)
        if entry is None or entry[0] is not seq:
            return 0
        if entry[1] is None:
            kv_id = kv_only_lookup_id(sid)
            client = self._lookup_client
            if client.is_pending(kv_id) and not client.poll(kv_id):
                return 0  # still unanswered: decide without it
            if not client.has_answer(kv_id):
                self._drop_kv_reach(sid)  # the lookup failed
                return 0
            reach = int(client.lookup(self._lookup_token_ids(seq), kv_id) or 0)
            # The ordinary lookup's answer this step, before any cap.
            with_state = int(self._last_tier_hit(seq, sid) or 0)
            if reach < with_state:
                # PAGE alone cannot reach less than PAGE plus the image: the
                # lookup did not cover the group this assumes is PAGE.
                logger.warning(
                    "LMCache MP KV-only lookup for %s reached %d < %d with "
                    "state; object group %d is not PAGE? Not using it.",
                    sid,
                    reach,
                    with_state,
                    PAGE_OBJECT_GROUP,
                )
                reach = 0
            entry = (seq, reach)
            self._kv_reach[sid] = entry
        return entry[1]

    def _drop_kv_reach(self, sid: str) -> None:
        """Give back a KV-only lookup no load will use, and its locks."""
        entry = self._kv_reach.pop(sid, None)
        if entry is None:
            return
        kv_id = kv_only_lookup_id(sid)
        self._lookup_defer_since.pop(kv_id, None)
        if entry[1] is None:
            self._lookup_client.discard(kv_id)  # in flight or never consumed
        else:
            self._lookup_client.clear_lookup_status(kv_id)

    def update_state_after_alloc(self, seq: Any) -> None:
        sid = str(seq.id)
        entry = self._kv_reach.get(sid)
        if seq.replay_kv_load and entry is not None and entry[0] is seq:
            # The KV-only lookup loads instead of the ordinary one: release
            # that one's locks, and aim the load at the KV-only reach. Its own
            # locks pass to the retrieve (`build_connector_meta`).
            del self._kv_reach[sid]
            self._lookup_client.clear_lookup_status(sid)
            reach = int(seq.replay_end)
            self._load_specs[sid] = LoadSpec(
                hbm_cached_tokens=int(seq.num_cached_tokens),
                lmcache_cached_tokens=reach,
            )
            self._hit_save_floors[sid] = reach
        else:
            self._drop_kv_reach(sid)
            if getattr(seq, "replay_kv_load", False):
                # Nothing here backs that load (a stale reach from an earlier
                # admission): replay from the HBM hit instead.
                self._block_manager.fall_back_to_hbm_replay(seq)
        super().update_state_after_alloc(seq)

    def should_park_for_load_after_alloc(self, seq: Any) -> bool:
        parks = super().should_park_for_load_after_alloc(seq)
        if parks:
            self._parked_loads[str(seq.id)] = seq
        elif getattr(seq, "replay_kv_load", False):
            # The load was refused (and `_clear_pending_load` gave its locks
            # back): the slot is fresh and the KV past the HBM hit is missing,
            # so replay below the HBM hit.
            self._block_manager.fall_back_to_hbm_replay(seq)
        return parks

    def _ensure_lookup_pin(self, seq: Any, sid: str, spec) -> bool:
        operation = self._native_load_operations.get(sid)
        lease = self._native_loads.get(operation)
        if lease is None or lease.seq is not seq or not lease.transfer.kv_only:
            return super()._ensure_lookup_pin(seq, sid, spec)
        # A KV-only load reads under the KV-only lookup, whose locks nothing
        # has released since it answered (no dispatch unpins it). The ordinary
        # lookup asks for an image at the end as well, so asking it again --
        # what the base does for a load admitted on a remembered hit -- falls
        # short of any reach past the last image and drops every such load.
        hit = self._lookup_client.hit_tokens(kv_only_lookup_id(sid))
        return hit is not None and int(hit) >= int(spec.lmcache_cached_tokens)

    def _save_frontier(self, seq: Any) -> int:
        if not getattr(seq, "has_per_req_cache", False):
            return 0
        frontier = super()._save_frontier(seq)
        failed = self._save_failures.get(str(seq.id))
        exhausted = (
            failed[1]
            if failed is not None
            and failed[0] is seq
            and failed[2] >= _MAX_SAVE_ATTEMPTS
            else 0
        )
        floor = self._save_floor(exhausted)
        # Every tracked request asks every step. The answer changes only when
        # its frontier moves or one of ITS boundary checkpoints is published or
        # dropped, so rescan only then -- not whenever any request's
        # checkpoint traffic moves the store.
        store = self._checkpoints.store
        memo = getattr(seq, "_mp_save_frontier_memo", None)
        if memo is not None and memo[:2] == (frontier, floor):
            changed = store.changed_since(memo[2])
            boundaries = getattr(seq, "_mp_boundary_by_hash", {})
            if changed is not None and not any(
                floor < boundaries.get(prefix_hash, 0) <= frontier
                for prefix_hash in changed
            ):
                seq._mp_save_frontier_memo = (
                    frontier,
                    floor,
                    store.generation,
                    memo[3],
                )
                return memo[3]
        found = 0
        boundaries = getattr(seq, "_mp_boundary_by_hash", None)
        if boundaries is None:
            boundaries = seq._mp_boundary_by_hash = {}
        for boundary in range(frontier, floor, -self.chunk_size):
            prefix_hash = self._boundary_hash(seq, boundary)
            boundaries[prefix_hash] = boundary
            if self._checkpoints.contains(prefix_hash):
                found = boundary
                break
        seq._mp_save_frontier_memo = (frontier, floor, store.generation, found)
        return found

    def _save_floor(self, lower: int) -> int:
        """Exclusive lower bound for boundaries worth storing (never 0)."""
        return max(lower, self._min_save_tokens - 1, 0)

    def _late_save_frontier(self, seq: Any, saved: int, available: int) -> int:
        available = self._chunk_floor(available)
        for boundary in range(available, self._save_floor(saved), -self.chunk_size):
            if self._checkpoints.contains(self._boundary_hash(seq, boundary)):
                return boundary
        return saved

    def _may_emit_save(self) -> bool:
        return len(self._save_inflight) < self._max_pending_saves

    def _build_save_request(
        self, seq, saved, aligned, operation, block_ids, is_last_prefill
    ):
        if not self._may_emit_save():
            return None
        prefix_hash = self._boundary_hash(seq, aligned)
        source = self._checkpoints.acquire_checkpoint_source(
            prefix_hash, max_inflight=self._max_pending_saves
        )
        if source is None:
            return None
        state_operation, unit_ids = source
        try:
            request = super()._build_save_request(
                seq, saved, aligned, operation, block_ids, is_last_prefill
            )
            request.native_state = NativeStateTransfer(unit_ids, aligned, prefix_hash)
            self._native_saves[operation] = _NativeSave(
                seq, state_operation, saved, aligned
            )
        except BaseException:
            # Nothing was dispatched: return the lease the same way a settled
            # save does, or it leaks for the process lifetime (the pin is never
            # timeout-reclaimable).
            self._checkpoints.release_offload_store_source(state_operation)
            self._checkpoints.settle_offload_store(state_operation)
            raise
        return request

    def _complete_native_save(
        self, operation: SaveOperationId, *, succeeded: bool
    ) -> None:
        lease = self._native_saves.pop(operation, None)
        if lease is None:
            return
        # A terminal MP event is a backstop source fence even if an older
        # server did not provide per-chunk milestones, or those notifications
        # were coalesced/lost before reaching the scheduler.
        self._source_group_finished(
            SaveSourceGroupId(operation, ((lease.saved_before, lease.boundary),))
        )
        # The image pin is released on the terminal only: no PAGE chunk
        # milestone proves the server has finished reading the STATE groups.
        self._checkpoints.release_offload_store_source(lease.source)
        self._checkpoints.settle_offload_store(lease.source)
        sid = str(operation.req_id)
        if succeeded:
            self._save_failures.pop(sid, None)
        else:
            failed = self._save_failures.get(sid)
            attempts = (
                failed[2] + 1
                if failed is not None and failed[:2] == (lease.seq, lease.boundary)
                else 1
            )
            self._save_failures[sid] = (lease.seq, lease.boundary, attempts)
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is lease.seq:
                entry[1] = min(int(entry[1]), lease.saved_before)
        self._store_finished(operation, succeeded=succeeded)

    def save_finished(self, req_id: Any) -> None:
        if isinstance(req_id, SaveOperationId):
            self._complete_native_save(req_id, succeeded=True)

    def connector_completion(self, completion: ConnectorCompletion) -> bool | None:
        if completion.channel != NATIVE_STATE_MP_STORE_CHANNEL:
            return super().connector_completion(completion)
        if not isinstance(completion.operation_id, SaveOperationId):
            return False
        self._complete_native_save(
            completion.operation_id, succeeded=completion.succeeded
        )
        return True

    def _decide_load_after_alloc(self, seq: Any, load_spec):
        if getattr(seq, "replay_end", 0) and not getattr(seq, "replay_kv_load", False):
            # A replay from the HBM hit loads nothing: `num_cached_tokens` sits
            # below blocks it claimed, and a load from there would write them.
            hbm = int(seq.num_cached_tokens)
            lmc = int(load_spec.lmcache_cached_tokens)
            chunk = int(self.chunk_size)
            return False, "replay_from_hbm", hbm, lmc, lmc - hbm, chunk
        decision = super()._decide_load_after_alloc(seq, load_spec)
        should_load, _reason, hbm, lmc, need, chunk = decision
        if not should_load:
            return decision
        if not getattr(seq, "has_per_req_cache", False) or seq.state_slot < 0:
            return False, "native_destination_missing", hbm, lmc, need, chunk
        if lmc % chunk or lmc >= int(seq.num_prompt_tokens):
            return False, "native_boundary_unaligned", hbm, lmc, need, chunk
        sid = str(seq.id)
        operation = self._native_load_operations.get(sid)
        if operation is not None:
            lease = self._native_loads[operation]
            if lease.seq is seq:
                return decision
            return False, "native_load_id_busy", hbm, lmc, need, chunk
        prefix_hash = self._boundary_hash(seq, lmc)
        operation = LoadOperationId(seq.id, self._load_nonce)
        self._load_nonce += 1
        if getattr(seq, "replay_kv_load", False):
            if lmc != int(seq.replay_end):
                return False, "replay_reach_moved", hbm, lmc, need, chunk
            # PAGE alone: no image units, nothing restored. The replay that
            # follows rebuilds the window state into the fresh slot.
            self._native_loads[operation] = _NativeLoad(
                seq,
                NativeStateTransfer((), lmc, prefix_hash, kv_only=True),
                hbm,
            )
            self._native_load_operations[sid] = operation
            self._active_load_operations[sid] = (seq, operation)
            seq._load_operation = operation
            return decision
        # No byte budget on top of the PAGE pool: one load per request bounds
        # these by the admitted batch, and a pool without room for an image
        # refuses here. A refusal recomputes the prefix, which costs far more
        # than the image's few PAGE units, and a completed image is adopted as
        # a READY checkpoint that later hits reuse.
        units = self._checkpoints.reserve_transfer_units(operation)
        if units is None:
            return False, "native_state_units", hbm, lmc, need, chunk
        local_restore = None
        try:
            local_restore = self._checkpoints.suspend_queued_restore(
                int(seq.state_slot)
            )
            self._native_loads[operation] = _NativeLoad(
                seq,
                NativeStateTransfer(
                    units,
                    lmc,
                    prefix_hash,
                    destination_slot=int(seq.state_slot),
                ),
                hbm,
                local_restore,
            )
        except BaseException:
            if local_restore is not None:
                self._checkpoints.resume_suspended_restore(local_restore)
            self._checkpoints.release_transfer_units(operation)
            raise
        self._native_load_operations[sid] = operation
        self._active_load_operations[sid] = (seq, operation)
        seq._load_operation = operation
        return decision

    def _new_load_operation(self, seq: Any) -> LoadOperationId:
        operation = self._native_load_operations[str(seq.id)]
        self._native_loads[operation].dispatched = True
        # The scheduler publishes the loaded PAGE prefix only after terminal
        # success. Its hash chain must exist before suffix prefill checkpoints.
        seq.offload_load_start_tokens = self._native_loads[operation].hbm_cached_tokens
        return operation

    def build_connector_meta(self):
        metadata = super().build_connector_meta()
        # Every load parked since the last dispatch was sent or dropped just
        # now. A dropped one leaves its request parked with no transfer to
        # report on it.
        sent = {
            request.req_id
            for request in metadata.requests
            if request.load_operation is not None
        }
        for seq in self._parked_loads.values():
            if seq.id not in sent and not self._has_active_load(seq):
                logger.warning(
                    "LMCache MP load for parked request %s was dropped at "
                    "dispatch; waking it to prefill",
                    seq.id,
                )
                self._dropped_parked_loads.add(seq.id)
        self._parked_loads.clear()
        for request in metadata.requests:
            if request.load_operation is not None:
                transfer = self._native_loads[request.load_operation].transfer
                request.native_state = transfer
                if transfer.kv_only:
                    # The base handed the ordinary lookup's (already released)
                    # range over; the retrieve reads under the KV-only one.
                    self._lookup_client.prepare_retrieve(
                        kv_only_lookup_id(str(request.req_id)),
                        int(request.load_spec.hbm_cached_tokens),
                        int(transfer.boundary_tokens),
                    )
        return metadata

    def _release_native_load(
        self, operation: LoadOperationId, *, release_units: bool = True
    ) -> _NativeLoad | None:
        lease = self._native_loads.pop(operation, None)
        if lease is not None:
            if release_units:
                self._checkpoints.release_transfer_units(operation)
            sid = str(operation.req_id)
            if self._native_load_operations.get(sid) == operation:
                del self._native_load_operations[sid]
        return lease

    def _clear_pending_load(self, sid: str) -> None:
        operation = self._native_load_operations.get(sid)
        lease = self._native_loads.get(operation)
        if lease is not None and not lease.dispatched:
            if lease.local_restore is not None:
                self._checkpoints.resume_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation)
            if self._active_load_operations.get(sid) == (lease.seq, operation):
                self._active_load_operations.pop(sid, None)
            if getattr(lease.seq, "_load_operation", None) == operation:
                delattr(lease.seq, "_load_operation")
        if self._replay_tokens:
            # A KV-only load dropped before its retrieve: its lookup's locks
            # come back too. A no-op once the retrieve owns them, or with none.
            self._lookup_client.clear_lookup_status(kv_only_lookup_id(sid))
        super()._clear_pending_load(sid)

    def _finish_native_load(self, operation: Any, *, succeeded: bool) -> bool:
        if (
            not isinstance(operation, LoadOperationId)
            or operation not in self._native_loads
        ):
            return False
        lease = self._native_loads[operation]
        if lease.transfer.kv_only:
            # No image, no units: only the KV-only lookup's retrieve to close.
            self._release_native_load(operation, release_units=False)
            self._lookup_client.complete_retrieve(
                kv_only_lookup_id(str(operation.req_id)), succeeded=succeeded
            )
        elif succeeded:
            if not self._checkpoints.adopt_transfer_units(
                operation, lease.transfer.prefix_hash
            ):
                # Another request published the same image first. Ours was
                # released; the SLOT already holds identical restored state.
                logger.debug(
                    "Native restore for %s deduplicated against a READY image",
                    operation,
                )
            if lease.local_restore is not None:
                self._checkpoints.release_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation, release_units=False)
        else:
            if lease.local_restore is not None:
                self._checkpoints.resume_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation)
        finished = (
            super().load_finished(operation)
            if succeeded
            else super().load_failed(operation)
        )
        if self._retired_requests.get(str(operation.req_id)) is lease.seq:
            self.request_finished(lease.seq)
        return finished

    def load_finished(self, req_id: Any) -> bool:
        return self._finish_native_load(req_id, succeeded=True)

    def load_failed(self, req_id: Any) -> bool:
        return self._finish_native_load(req_id, succeeded=False)

    def _has_active_load(self, seq: Any) -> bool:
        # The native lease also retains a cancelled/reused request lifecycle
        # whose generic request-ID entry may have been replaced.
        return any(
            lease.seq is seq for lease in self._native_loads.values()
        ) or super()._has_active_load(seq)

    def cancel_pending_load(self, seq: Any) -> None:
        operation = self._native_load_operations.get(str(seq.id))
        lease = self._native_loads.get(operation)
        if lease is not None and lease.seq is seq and lease.dispatched:
            return  # A request cancellation does not cancel an MP DMA.
        super().cancel_pending_load(seq)

    def request_finished(self, seq: Any) -> None:
        sid = str(seq.id)
        self._retired_requests[sid] = seq
        operation = self._native_load_operations.get(sid)
        lease = self._native_loads.get(operation)
        if lease is not None and lease.seq is seq and lease.dispatched:
            return
        super().request_finished(seq)
        self._finish_retired_request(sid)
        # Its KV-only lookup, if one was never used (aborted while waiting),
        # and its session -- which no save ever runs under, so it can end now.
        self._drop_kv_reach(sid)
        if sid in self._kv_sessions:
            self._kv_sessions.discard(sid)
            try:
                self._mp_adapter.end_session(
                    _mp_session_id(self._config, kv_only_lookup_id(sid))
                )
            except Exception:
                logger.warning(
                    "LMCache MP end_session failed for the KV-only lookup of %s",
                    sid,
                    exc_info=True,
                )

    def _finish_retired_request(self, sid: str) -> None:
        super()._finish_retired_request(sid)
        seq = self._retired_requests.get(sid)
        if seq is not None and not self.should_defer_free(seq):
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is seq:
                self._save_tracker.pop(sid, None)
            self._save_failures.pop(sid, None)
            self._retired_requests.pop(sid, None)

    def has_pending_work(self) -> bool:
        return (
            bool(self._native_loads)
            or bool(self._retired_requests)
            or super().has_pending_work()
        )

    def process_completions(self, output):
        output = super().process_completions(output)
        # A retired request may have waited behind admission with an unpinned
        # READY candidate. If eviction spends it before dispatch, no worker
        # completion exists to wake deferred-free cleanup. Report that local
        # terminal condition only for this same request lifecycle, after all
        # actual transfers and every remaining candidate are gone.
        for sid, seq in list(self._retired_requests.items()):
            entry = self._save_tracker.get(sid)
            lifecycle = self._load_lifecycles.get(sid)
            if (entry is not None and entry[0] is not seq) or (
                lifecycle is not None and lifecycle is not seq
            ):
                continue
            if not self.should_defer_free(seq):
                output.finished_saving.add(seq.id)
                self._finish_retired_request(sid)
        # The scheduler wakes a failed load to prefill (a KV-only one to
        # replay below its HBM hit), exactly as it would a transfer that ran.
        output.failed_loading.update(self._dropped_parked_loads)
        self._dropped_parked_loads.clear()
        return output

    def _live_transfers(self) -> set[Any]:
        live = super()._live_transfers()
        # A native save holds its checkpoint pin and state budget until its
        # STORE terminal; nothing but that report may release them.
        live.update(self._native_saves)
        live.update(
            operation
            for operation, lease in self._native_loads.items()
            if lease.dispatched
        )
        return live


__all__ = ["NativeStateLMCacheMPConnectorScheduler"]
